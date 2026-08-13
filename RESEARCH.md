# Agent Runtime 机制研究

> 阶段一：LLM + API。本文梳理 Agent Runtime 的核心机制与执行策略（Execution Strategy）的演进，
> 作为本 Lab 设计 `Task -> Policy -> Execution Strategy` 管线的依据。
> 研究素材包括公开文献与 pi（earendil-works/pi，一个真实的 TypeScript Agent Runtime）的实现。

## 1. 什么是 Agent Runtime

Agent Runtime 是承载"LLM 循环"的运行时，职责大致是：

1. **对话/状态管理**：维护 messages、工具调用结果、上下文窗口内的状态（session）。
2. **LLM 调用循环（agent loop）**：`LLM -> 输出 -> (需要工具? 执行工具 -> 回填 Observation -> 再调 LLM) -> 最终答案`。
3. **工具（Tools）**：向 LLM 暴露可执行能力（bash、文件读写、搜索、计算器……），并校验/执行/回填结果。
4. **上下文管理**：prompt 组装、压缩（compaction/summarization）、分叉（branch）。
5. **可观测性**：trace、telemetry，记录每步 LLM 调用、工具调用、耗时、token。

pi 的实现对应关系（`packages/agent/src/harness/`）：

| 机制 | pi 中的实现 |
|---|---|
| Agent loop | `reducer.ts`（667 行，核心状态归约循环） |
| Session/状态 | `session/state.ts`、`session/context.ts` |
| 消息/上下文 | `messages.ts`、`session/jsonl/`（持久化） |
| 压缩 | `compaction/`（branch summarization） |
| 工具 | `tools/` + 应用层 `core/tools/*`（bash/edit/read/grep/…） |
| 遥测 | `telemetry.ts` |
| 技能/提示模板 | `skills.ts`、`prompt-templates.ts` |

## 2. 执行策略（Execution Strategy）谱系

按"LLM 被调用的方式/循环深度"分类：

### 2.1 Direct（单次调用）
- 一次 `LLM(user)` -> answer。无工具、无循环。
- 适用：简单问答、格式转换、不需要外部信息的任务。
- 成本最低，延迟最低，但能力上限就是模型本身。

### 2.2 ReAct（Reason + Act 交替循环）
- 论文：*ReAct: Synergizing Reasoning and Acting in Language Models*（Yao et al., 2022）。
- 循环：`Thought -> Action(tool) -> Observation -> … -> Final Answer`。
- 关键点：推理与行动交错，Observation 回填上下文，让 LLM 依据工具结果继续推理。
- 适用：需要工具（计算、搜索、代码执行）的单线程任务。
- 弱点：循环深度有限（max_steps）、上下文随步数膨胀、长链任务易漂移。

### 2.3 Plan-and-Execute / 规划式
- 先规划（Plan：拆成子步骤），再逐个执行，执行后视情况调整计划。
- 代表：Plan-and-Solve、LLM+P、BabyAGI / AutoGPT 一系。
- 比 ReAct 更擅长长任务，但规划质量依赖模型，且计划与执行分离会丢失中间推理信息。

### 2.4 Subagent（子代理 / 多代理协作）
- 父代理把子任务**委托**给一个全新的、独立上下文的子代理执行，取回结果后合成。
- 上下文隔离：子任务不会污染主上下文（不占用主上下文 token）。
- 代表范式：Multi-agent（AutoGen、LangGraph、Claude Code 的 subagent）、层级委托（hierarchical）。
- 适用：并行子任务、长文档处理（每个子代理只读自己负责的片段）、隔离危险操作。
- 代价：多次 LLM 调用、需要"分解 + 合成"两层开销、结果一致性需要协调。

### 2.5 其他
- **Reflexion**：执行后带反馈重试，把失败经验写回 prompt。
- **Self-Consistency / Best-of-N**：多次采样取共识。
- **MCP/工具路由**：策略本身不作为循环，而是按需挂载不同工具集。

## 3. Policy：如何决定策略

`Task -> Policy -> Execution Strategy` 中的 Policy 层回答："这个任务该用哪种执行策略？"

候选实现：

1. **规则启发式**：关键词/正则/复杂度特征 -> 策略。可解释、零成本，但泛化差。
2. **LLM 分类**：把任务文本交给 LLM，输出结构化 Policy（复杂度、所需工具、策略、理由）。
   - 优点：理解自然语言意图，可扩展到任意工具集。
   - 缺点：多一次 LLM 调用（约 +1 次推理成本）；分类错误会把任务导向错误策略。
3. **混合**：规则做快速通道（明显简单任务直接 direct），不确定的交给 LLM 分类。
4. **学习式（后期）**：用 trace 数据训练/校准策略，形成闭环。这是本 Lab 的远期方向——`trace.py`
   采集的数据就是训练策略分类器的样本。

本 Lab v1 采用 **LLM 分类 + 规则 fallback**（无 key 或分类失败时走启发式）。

## 4. 策略选择的关键权衡

| 维度 | Direct | ReAct | Subagent |
|---|---|---|---|
| LLM 调用次数 | 1 | 2..N | 2..N（分解 + 每个子任务 + 合成） |
| 工具能力 | 无 | 有 | 有（子代理各自可用） |
| 上下文隔离 | - | 无（单上下文膨胀） | 有（子任务独立上下文） |
| 适用任务 | 简单问答 | 单线程需工具 | 可分解/并行/长文档 |
| 失败模式 | 模型知识不够 | 循环漂移/超步数 | 分解错误/合成丢失 |

经验法则（v1 初始版）：**计算/搜索类 -> ReAct；可分解多步/长内容 -> Subagent；其余简单问答 -> Direct**。
这条法则本身是 Lab 要探索和验证的对象，而不是结论。

## 5. 与 pi 的对照（务实参考）

- pi 的 agent loop 是"工具调用协议"驱动的 ReAct 变体：模型发结构化 tool_call，runtime 执行并回填，
  循环直到无 tool_call。我们没有从头发明，而是复刻这个最小闭环。
- pi 有 compaction（长上下文压缩）与 branch（分叉），本 Lab 阶段一不做，列入 roadmap。
- pi 的 subagent 通过扩展机制（handoff）实现，本 Lab v1 用"新上下文 + 复用工具注册表"的简化版。

## 6. 阶段一范围（v1）

- LLM + API：OpenAI 兼容端点（env 配置），最小 `LLM` 封装。
- 三条执行策略：direct / react / subagent。
- 两个工具：calculator（安全 AST 求值）、search（本地语料占位，后续换真实搜索 API）。
- Policy 层：LLM 分类 + 规则 fallback。
- trace 层：结构化记录策略选择、每次 LLM/工具调用、耗时；为后续策略评估与学习式 Policy 铺路。

## 7. Roadmap（后续阶段）

- v2：真实搜索 API（Tavily/SerpAPI）、流式输出、并行 subagent。
- v3：Plan-and-Execute 策略、Reflexion 重试。
- v4：用 trace 数据离线评估各策略（同一任务集跑三个策略对比）。
- v5：学习式 Policy（用 trace 样本训练策略选择器），形成 Task->Policy 闭环。
