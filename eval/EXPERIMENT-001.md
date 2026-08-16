# 实验报告 001：策略选择基准（Strategy Eval）

> [English](EXPERIMENT-001.en.md) | 中文

> 状态：v1 实验，待复核
> 日期：2026-08-13
> 模型：deepseek-chat（DeepSeek，OpenAI 兼容端点）
> 环境：Python 3.14 / openai 1.66.3 / 本地 `.env` 配置
> 代码版本：`feat/eval-harness`（PR #9）

---

## 1. 实验目的与研究问题

本 Lab 的核心管线是 `Task -> Policy -> Execution Strategy`：给定任务，先由 Policy 层决定策略（direct / react / subagent），再由对应执行器完成。阶段一的目标是回答三个问题：

1. **各执行策略分别在什么任务上"对且便宜"**（正确率、成本、延迟）
2. **Policy 层选得准不准**——与"最优策略"的一致率
3. **LLM 分类与规则启发式，谁的信号更可靠**

## 2. 系统与方法

### 2.1 被测系统

```
task -> task_analyzer (HybridPolicy) -> router -> executor
                                                  ├─ direct   单次 LLM 调用
                                                  ├─ react    Thought/Action/Observation 循环
                                                  └─ subagent planner 分解 -> 独立上下文子代理 -> synthesizer 合成
                                                  └─ tools: calculator(AST白名单) / search(本地语料)
                                                  └─ trace: 结构化事件（llm_call / tool_call / subagent / policy）
```

- **Policy 层**：`HybridPolicy` = `LLMPolicy`（LLM 输出 JSON 分类：strategy/complexity/tools/reasoning）优先，失败时降级 `RulePolicy`（关键词正则）。
- **强制策略**：`--force-strategy` 绕过 Policy 直接指定执行器，供评测使用；CLI 与 runner 共用同一 `run_pipeline()` 路径。

### 2.2 评测集（eval/tasks.json，24 任务 × 4 类）

| 类别 | n | 任务形态 | 判定方式 |
|---|---|---|---|
| math | 6 | 纯算术（含大数、函数） | 回答中提取数值 == expected（容差 1e-6，容忍千分位） |
| search | 6 | 查询本地语料 | 回答包含全部 must_contain 关键词 |
| direct | 6 | 常识/简单算术 | 关键词包含 |
| subagent | 6 | compare / summarize / report | 关键词包含；subagent 策略自身另要求 >=2 次 spawn（结构检查） |

评测值全部设计为**确定性**（精确算术、本地语料事实），不依赖模型世界观。

### 2.3 指标定义

对每个 (task, strategy) 组合，从 trace 事件聚合：

- `correct`：按类别判定器
- `llm_calls` / `tokens` / `latency_ms`：llm_call 事件计数 / usage 求和 / ms 求和
- `tool_calls` / `tool_failures`：tool_call 事件数 / ok=False 数
- `spawns`：subagent 分解出的子任务数（机制保真度）
- `optimal`：任务所有正确策略中 llm_calls 最少者（并列取 token 少）
- `一致率`：policy / rule 的选择 vs optimal 的相等比例；三策略全错的任务列为"评测集问题"不计入

### 2.4 执行方式

- 每任务 × 3 策略 = 72 次执行，另加每任务 1 次 HybridPolicy 分类 + 1 次 RulePolicy
- 全量约 100+ 次 LLM 调用（DeepSeek，费用约 ¥1~2）
- 首次全量超时（>30min），改为按类别分批执行（math / search / subagent / direct 四批）后合并

## 3. 过程与过程中发现的缺陷

实验过程中按 Issue→PR 流程修复了 5 个缺陷（3 个由本实验直接暴露）：

| # | 缺陷 | 发现方式 | 修复 |
|---|---|---|---|
| 1 | search 工具元组解包顺序错误（TypeError） | 前期手动验证 | PR #5 |
| 2 | chat_json 截断无兜底（subagent 崩溃） | 前期手动验证 | PR #7（纠正重试 + token 上限） |
| 3 | subagent planner 对 trivial 任务返回空 JSON 直接崩溃 | 首轮冒烟 | 降级为单子任务（自我委托） |
| 4 | 判定器把结构检查（>=2 spawns）错误套用在 direct/react 运行上 | 首轮全量结果分析 | 结构检查仅对 subagent 策略自身生效，spawns 降为独立指标 |
| 5 | synthesizer 空输出导致最终答案为空 | 首轮全量结果分析 | 回退到 worker 报告原文 |

另发现：全量运行超时（subagent 策略在部分任务上耗时长），增加按类别分批 + 进度输出。

## 4. 结果

### 4.1 正确率（按类别 × 策略）

| category | direct | react | subagent | 至少一策略解出 |
|---|---|---|---|---|
| direct (n=6) | 6/6 (100%) | 5/6 (83%) | 6/6 (100%) | 6/6 |
| math (n=6) | 6/6 (100%) | 6/6 (100%) | 5/6 (83%) | 6/6 |
| search (n=6) | 0/6 (0%) | 4/6 (67%) | 4/6 (67%) | 5/6 |
| subagent (n=6) | 6/6 (100%) | 5/6 (83%) | 4/6 (67%) | 6/6 |
| **合计** | **18/24** | **20/24** | **19/24** | **23/24** |

未解出：`search-01`（"search the corpus for what react is"，关键词 interleaves+reasoning 未被回答同时命中）。

### 4.2 成本（均值/任务，n=6）

| 类别 | direct | react | subagent |
|---|---|---|---|
| direct | 1 次 / 149 tok / 1.9s | 1 / 512 / 3.3s | 3 / 1076 / 6.8s / 1 spawn |
| math | 1 / 516 / 3.5s | 1 / 640 / 1.8s | 4 / 1354 / 5.8s / 1 spawn |
| search | 1 / 383 / 3.0s | 1 / 769 / 2.8s | 6 / 3877 / 16.9s / 1 spawn |
| subagent | 1 / 1391 / 12.4s | 1 / 950 / 6.1s | 6 / 5119 / 25.2s / 2 spawns |

subagent 策略在所有类别上都是最贵的（3~6 次调用，最高 25s）。

### 4.3 Policy 层质量（vs 最优策略，23 个可判定任务）

| 信号 | 一致率 | 细分 |
|---|---|---|
| **policy (Hybrid)** | **15/23 (65%)** | source=llm：14/16 (**87.5%**)；source=fallback：1/7 (**14%)** |
| **rule** | **6/23 (26%)** | - |

关键数字：LLM 分类**成功**时（16/23 任务，70%）几乎总是选对；**一旦失败降级到规则**，一致率掉到 14%。

## 5. 核心发现与讨论

1. **direct 被系统性低估**。math/direct/subagent 三类任务的 optimal 均为 direct（1 次调用且 100% 正确）。模型的心算、对比、总结能力足以覆盖这些任务，而 rule 策略看到 "calculate/compare/summarize" 就一律判 react/subagent——规则不知道"模型自己就会"。

2. **react 只在"需要外部信息"的任务上有意义**。search 类：direct 0%（模型没有语料知识）vs react 67%。这是工具存在意义的直接证据；反过来说，非外部信息任务用 react 纯属浪费（多一次调用，正确率不升）。

3. **subagent 是"贵且不一定好"**。最贵（最高 25s/6 次调用），且在最应该发挥的 subagent 类任务上反而 67% < direct 100%。原因：当前评测集的"对比/总结"任务**实际上不需要分解**——真正需要 subagent 的场景（长文档、并行调研、隔离上下文）本评测集未覆盖。

4. **策略层瓶颈是"分类输出的可靠性"，不是 LLM 的判断力**。LLM 分类成功时一致率 87.5%，几乎完美；但 7/23 任务分类失败（非法 JSON/截断/超时），降级到规则后一致率只剩 14%。chat_json 已有一轮纠正重试，仍不够。

5. **规则启发式接近无用（26% 一致率）**。作为 LLM 失败时的兜底，它的质量决定了整个 HybridPolicy 的下限——当前下限很低。

## 6. 限制（诚实声明）

1. **单次采样**。LLM 非确定性未处理：每个 (task, strategy) 只跑一次，一致率无置信区间。math-02 的 policy 选择在两次运行中分别是 react 和 direct，说明分类本身不稳定。
2. **关键词判定是近似**。假阳性（答非所问但碰巧含关键词）与假阴性（正确但措辞不同，如 search-01）都存在。
3. **评测集规模小且类别内同质**。每类仅 6 个任务；subagent 类任务设计有缺陷（对比/总结本不需要分解）；没有需要真正多步推理/长上下文的任务。
4. **"最优策略"定义是纯成本导向**（正确中取调用最少）。未考虑答案质量差异（同为 correct 但内容详略不同），也未考虑失败风险分布。
5. **延迟指标受网络波动影响**，latency_ms 只统计 LLM 侧耗时，跨批可比性一般。
6. **单一模型**（deepseek-chat），结论不具跨模型泛化性。
7. **评测环境与运行时共享进程内状态**（无），但 DeepSeek 服务端负载会影响结果复现。

## 7. 结论

- 三种策略各有适用域，但**边界与直觉不同**：direct 覆盖远比预想宽；subagent 的适用域比预想窄（且当前评测集无法证明其价值）；react 是"外部信息任务"的必需项。
- HybridPolicy 的 LLM 分类在成功时质量很高（87.5%），**当前主要矛盾是分类输出的可靠性**（70% 成功率），规则兜底把下限拖到 14%。
- eval harness 本身有效：72 次执行全部留下结构化 trace，缺陷发现→修复→重跑的闭环成立。

## 8. 下一步建议（为"结论还远远不够"列）

按优先级：

1. **多次采样**：每个 (task, strategy) 跑 N=5 次，报告均值与方差；policy 分类跑 N 次测稳定性。这是当前所有结论可信度的前提。
2. **重设计评测集**：加入真正需要 subagent 的任务（长文档分块、并行调研、需上下文隔离的任务）；增加多步推理与长链任务；每类扩到 10+。
3. **更强的判定器**：开放任务用 LLM-as-judge（对同一回答让独立模型判定正确性），替代关键词近似；数值任务保持确定性判定。
4. **分类可靠性专项**：把"分类失败率"作为一等指标（当前 30%）；提升 chat_json 健壮性（更宽的 token、schema 约束、多轮重试）；评估响应格式约束（structured output）是否可用。
5. **多模型对比**：deepseek-chat vs deepseek-reasoner（或另一个 OpenAI 兼容端点），验证结论的模型依赖性。
6. **最优策略定义细化**：引入质量维度（judge 评分）与失败风险（多次采样失败率），成本与质量权衡。
7. **策略层自身的对比实验**：LLM-only / rule-only / hybrid 三配置跑同一评测，量化各配置的性价比。

## 附：复现方法

```bash
cd ~/adaptive-agent-runtime-lab && source .venv/bin/activate
python -m eval.runner --category math      # 每类分批跑（避免全量超时）
python -m eval.runner --category search
python -m eval.runner --category subagent
python -m eval.runner --category direct
# 合并四批结果到 eval/results/FULL.json 后渲染报告
```
