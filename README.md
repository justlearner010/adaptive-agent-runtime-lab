# Adaptive Agent Runtime Lab

> [English](README.en.md) | 中文

**这是什么**：一个**研究型实验项目**（不是生产系统）。用可控实验回答一个问题——
**"AI 任务该用哪种执行策略（直接答 / 边推理边用工具 / 拆给子代理）才又对又省？"**
为此搭了一条最小 Agent 运行时管线（`Task -> Policy -> Execution Strategy`），并用自制评测集反复测量，
用测量结果驱动设计决策。

**给谁看**：对"Agent 运行时如何选择执行策略"感兴趣的研究者、想参与实验的协作者。

**怎么读（新人 5 分钟路线）**：
1. [RESEARCH.md](RESEARCH.md) — 背景：Agent Runtime 机制与三种执行策略的谱系（10 分钟）
2. 本文件「实验定位」——我们做了什么、进行到哪、结论与限制
3. [eval/EXPERIMENT-001..004](eval/EXPERIMENT-001.md) — 实验史（按需深读）
4. [v3-design.md](v3-design.md) — 当前实验设计（v3：SubAgent 何时超过 ReAct）

探索 Agent 运行时如何根据任务类型选择不同的执行策略（Execution Strategy）。

核心管线：**Task -> Policy -> Execution Strategy**

```
task ──▶ task_analyzer (Policy) ──▶ router ──▶ executor
                                              ├── direct   (单次 LLM 调用，无工具)
                                              ├── react    (Thought/Action/Observation 循环)
                                              └── subagent (分解 -> 委托子代理 -> 合成)
                                              └── tools: calculator / search / doc
                                              └── trace:  全程结构化留痕
```

## 实验定位（重要，先读）

### 实验类型

**研究型实验项目**，不是生产系统。本仓库用"可复现的实验闭环"驱动设计决策：
设计 -> 实现 -> 评测 -> 发现缺陷 -> 修复 -> 重测，每一步都留痕（Issue/PR/trace/评测数据）。

### 目的

回答一个核心问题：**给定任务，选择哪种执行策略（direct / react / subagent）才能"对且便宜"？**

- 建立 `Task -> Policy -> Execution Strategy` 管线，量化各策略的适用域、正确率、成本；
- 采集结构化 trace 数据，为后续「学习式 Policy」（v5）铺路；
- 顺带验证运行时机制本身（ReAct 循环、subagent 委托）的健壮性。

### 执行策略定义

三种策略是"LLM 被调用的方式 / 循环深度"的三种形态（机制谱系详见 [RESEARCH.md](RESEARCH.md) §2）：

- **Direct（直接执行）**：**单次 LLM 调用、无工具、无循环**——模型仅凭自身知识直接产出答案。
  成本 = 1 次调用；能力上限 = 模型本身；典型失败 = 知识不足 / 心算错误。
  适用：常识问答、简单算术、总结/对比等不需要外部信息的任务。

- **ReAct（推理-行动循环）**：*ReAct: Synergizing Reasoning and Acting in Language Models*
  （Yao et al., 2022）。循环 `Thought -> Action(调用工具) -> Observation(回填结果) -> ... -> Final Answer`，
  **单一上下文**：每一步的工具观测都追加进同一条对话，上下文随步数膨胀。
  成本 = 2..N 次调用；典型失败 = 循环漂移、步数上限（max_steps）、上下文膨胀。
  适用：需要工具（计算/搜索/读文档）的单线程任务。

- **Subagent（子代理委托）**：planner 先把任务**分解**成独立子任务，每个子任务交给一个
  **隔离上下文**的全新 ReAct 循环执行（v2 起 worker 可并行），最后 synthesizer 合成最终答案。
  父上下文不被子任务污染，但总调用 = 分解 + 各子任务 + 合成，开销更大。
  典型失败 = 分解错误、合成丢失、worker 空输出。
  适用：可分解 / 可并行 / 长内容任务（如长文档分块、并行调研）。

### 方法论

- **评测 harness**（`eval/`）：对每个任务**强制**跑全部策略 × N 次采样（默认 N=5），
  从 trace 聚合 `llm_calls / tokens / latency / tool_calls / spawns`；
- **判定器**（`eval/runner.py`）：
  - 数学/chain 类：回答中**任一数字**匹配 expected（容忍千分位）；
  - search 类：关键词命中 **且** 对每个必需主题有语料引用（5-gram grounding，防止凭常识蒙对）；
  - spawns（子任务数）为独立机制指标，**不**计入正确性；
- **一致率**：policy 多数票 vs `optimal`（所有正确策略中 LLM 调用最少者，并列取 token 少），
  并配 **always-* 退化基线**作为参照系（否则"永远选 direct"就能拿 77.5%）；
- **变更流程**：先 Issue -> 再 PR（`Closes #n`）-> owner Review -> squash 合并。

### 进度

| 阶段 | 内容 | 实验/提交 |
|---|---|---|
| v1 | 管线骨架 + direct/react/subagent + calculator/search + trace | `90cdbb9` |
| Phase 1 | 评测 harness + N=5 多采样 + 并行执行 | [EXPERIMENT-001/002](eval/EXPERIMENT-001.md) |
| Phase 2 | 分类可靠性专项：prompt 变体 p0/p1/p2、结构化输出、chain 任务 | [EXPERIMENT-003](eval/EXPERIMENT-003.md) |
| Phase 3 | 判定器修复、ReAct 健壮性、subagent 泄漏、并行分类、退化基线 + 全量重测 | [EXPERIMENT-004](eval/EXPERIMENT-004.md) |
| v2 | 测量面扩展：longdoc 评测类（真·长文档任务）、并行 subagent、LLM-as-judge、分类稳定性（confidence）、LLM 层空输出重试 | [#34](https://github.com/justlearner010/adaptive-agent-runtime-lab/pull/34) |
| **v3（进行中）** | **「SubAgent 何时超过 ReAct」边界实证**：功效分析、多规模 longdoc、并行结构对照、超参敏感性、trace 过程证据 | 设计已定案 [v3-design.md](v3-design.md)，P2 基础设施进行中 |
| v4+ | 任务形态扩展（检索/选择性读取）、难度任务、多模型、学习式 Policy | Roadmap，未开始 |

**当前状态**：v3 实验设计中——先做成本-功效分析（结论：N=5 只能分辨 ±6-14pp，"多数票+McNemar"失效，改用每任务正确率+配对 t 检验），预算 ¥12~50 三档可选。

### 暂时结论（截至 EXPERIMENT-004 全量 + v2 冒烟，单模型 deepseek-v4-flash）

1. **direct 的适用域远宽于直觉**：除语料搜索外，所有类别 95~100% 正确且仅 1 次调用——
   **"默认 direct"是最优基线**（31/40 任务的 optimal 都是它）。
2. **react 只在需要外部信息时必要**：语料搜索类 95% vs direct 0%；其余任务用它纯属浪费。
3. **subagent 在旧评测集上最贵且无优势**（search 类 8.5 次调用/23.7s）——但旧评测集没有真正需要
   长文档分块的任务。**v2 已补上**：longdoc 任务类 + 并行 worker，冒烟验证机制可行
   （planner 能按页分解、worker 并行读取、合成正确）；**"subagent 在什么条件下开始超过 react"
   正是 v3 正在测的问题**（[v3-design.md](v3-design.md)）。
4. **LLM 分类成功时质量高**（p0 多数票一致率 92.5%，高出 always-direct 基线 15pp），
   但**部分任务分类不稳定**（如大数乘法 math-02 分类成功率仅 20%）；规则兜底接近无用（32.5%）。
5. **模型对结果影响显著**：deepseek-chat -> v4-flash 后整体正确率上升，结论需固定模型验证。
6. **推理模型需要 token headroom**：deepseek-v4-flash 在 max_tokens 偏紧时返回**空输出**，
   已修复（LLM 层空输出重试 + 各处 max_tokens 上调）——这是 v2 冒烟中最有价值的发现。

> 以上是**阶段性结论**，不是定论。每次模型/评测集/判定器变化都可能改写它们。

### 限制（诚实声明）

- **单模型、单轮任务**：无跨模型泛化性，无多轮/长上下文任务；
- **判定器是近似**：关键词 + 5-gram grounding 对轻度改写敏感；LLM-as-judge（v2）已实现但未大规模重放验证；
- **评测集小且同质**：48 任务 × 6 类（v2 新增 longdoc），每类仅 8 个；longdoc 尚未全量评测；
- **混杂因素**：EXPERIMENT-004 中模型变更与判定器修复叠加，无法单因素归因；
- **成本导向的 optimal**：未纳入答案质量与失败风险；
- 全量评测约 70 分钟+ / 数元 API 费用（4 workers，模型延迟主导）。

研究背景与机制对比见 [RESEARCH.md](RESEARCH.md)。

## 快速开始

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...            # 任意 OpenAI 兼容端点即可
python main.py "what is 23 * 47 + 12"
python main.py "compare react and subagent strategies"
python main.py "who is the president of the usa" --offline   # 无 key 也能跑（规则策略）
python main.py "..." --json             # 机器可读输出（含完整 trace）
```

也可以用项目根目录的 `.env` 文件（复制 `.env.example` 改好即可，已 gitignore）：
程序启动时自动加载，已存在的环境变量优先（`.env` 不覆盖手动 export 的值）。

配置：`OPENAI_API_KEY` / `OPENAI_BASE_URL`（默认 api.openai.com/v1）/ `OPENAI_MODEL`（默认 gpt-4o-mini）。
换 DeepSeek/Moonshot/本地 vLLM 只需改 base_url + model。

## 目录

```
main.py            CLI 入口
llm.py             OpenAI 兼容端点的最小封装（chat / chat_json，含空输出重试）
task_analyzer.py   Policy 层：LLM 分类（含 confidence）+ 规则 fallback（HybridPolicy）
router.py          Policy -> Executor 分发
executors/
  direct.py        单次调用
  react.py         ReAct 循环（JSON Action 协议）
  subagent.py      分解 -> 并行子代理 -> 合成（v2 起 worker 并行）
tools/
  calculator.py    AST 白名单安全计算器
  search.py        v1 本地语料占位（后续换真实搜索 API）
  doc.py           v2 长文档工具（确定性合成文档 + 分页读取）
eval/              评测 harness（runner / report / judge / tasks.json / 实验报告）
trace.py           结构化 trace（策略/LLM/工具/耗时）
v3-design.md       当前实验设计（v3）
AGENTS.md          实验设计决策上下文（协作者必读）
```

## 文档导航

| 文档 | 内容 | 读者 |
|---|---|---|
| [README.md](README.md) | 本文件：定位 / 快速开始 / 当前状态 | 所有人 |
| [RESEARCH.md](RESEARCH.md) | 背景研究：Agent Runtime 机制、执行策略谱系 | 想理解"为什么"的人 |
| [eval/EXPERIMENT-001..004](eval/EXPERIMENT-001.md) | 实验史：四轮评测与修复（中/英双语） | 想复现/引用结果的人 |
| [v3-design.md](v3-design.md) | 当前实验设计：SubAgent 何时超过 ReAct | 参与 v3 的人 |
| [AGENTS.md](AGENTS.md) | 实验设计决策上下文、贡献检查、决策日志 | 所有协作者（含 agent） |
| [CONTRIBUTING.md](CONTRIBUTING.md) | 协作流程：Issue -> PR -> Review | 想提改动的人 |

## 设计要点

- **Policy 层**：`LLMPolicy`（一次 LLM 调用输出 JSON 分类 + confidence）优先，`RulePolicy`（关键词启发式）兜底。
  分类失败或离线时自动降级，全程在 trace 中留痕。
- **ReAct 协议**：模型输出 `Action: {"tool": ..., "input": ...}` JSON，runtime 括号感知解析、执行、回填 Observation。
- **Subagent**：planner 拆分子任务 -> 每个子任务跑独立 ReAct 上下文（v2 起 **worker 并行**）-> synthesizer 合成。
  子任务上下文与主上下文隔离，不污染主 token 预算；空合成回退时剥离内部 prompt。
- **工具集**：calculator（安全算术）/ search（本地语料占位）/ doc（v2 长文档分页，供 longdoc 评测类）。
- **Trace 先行**：所有 trace 数据为后续「同任务集多策略对比评估」和「学习式 Policy」做准备；v3 起补充过程证据（父上下文 token、分解粒度、worker 遗漏）。

## 协作流程（重要）

从 v2 开始，**每新增/改版都必须**：

1. 先提 Issue（说明改动动机与方案）
2. 再提 PR（关联该 Issue）
3. 由仓库 owner 实际 Review 后合并

v1 为初始提交，直接落在 main。详见 [CONTRIBUTING.md](CONTRIBUTING.md)。
