"""Strategy evaluation harness (v4, extended in v2).

对每个任务强制跑三种执行策略，从 trace 提取指标，量化：
- 各策略在每类任务上的正确率
- 各策略的平均成本（LLM 调用次数 / token / 延迟）
- policy 层选择 vs 最优策略的一致率（LLM 分类 vs 规则分类）
- （v2）keyword 判定 vs LLM-as-judge 的误判率
- （v2）分类稳定性：llm 成功率 + 平均 confidence

## 用法

```bash
python -m eval.runner --limit 6              # 冒烟：6 个任务
python -m eval.runner                        # 全量：48 任务 x 3 策略 x 5 次
python -m eval.runner --category longdoc     # 只跑某类（math/search/direct/subagent/chain/longdoc）
python -m eval.runner --strategies react,direct
python -m eval.judge --input eval/results/<ts>.json   # v2：对已存答案做 LLM-as-judge 重放
```

结果 JSON 写入 `eval/results/<ts>.json`（已 gitignore），报告打印到终端。

## 指标定义

- `correct`：
  - math/chain：回答中**任一数字**匹配 expected（容忍千分位，不取"最后一个数字"）
  - search：`must_contain` 关键词 **且** 每个必需主题都有语料引用（5-gram grounding，防常识蒙对）
  - 其余（direct/subagent/longdoc）：`must_contain` 关键词
  - spawns（子任务数）是独立机制指标，**不**计入正确性
- `llm_calls` / `tokens` / `latency_ms`：从 trace 的 llm_call 事件聚合
- `tool_calls` / `tool_failures`：从 trace 的 tool_call 事件聚合
- `spawns`：subagent 分解出的子任务数（机制保真度指标）
- `optimal`：任务所有正确策略中 llm_calls 最少者（并列取 token 少）
- 一致率：policy 多数票 vs optimal，并配 always-* 退化基线作为参照系
- 三策略全错的任务列为"评测集问题"（任务设计或工具问题），不计入一致率

## longdoc 类（v2）

需要**真正分解**的任务族：`doc` 工具提供确定性长文档（~12KB，分页读取，每页上限 2500 字符），
任务要求读完整文档并报出全部事实（每个事实一个发明关键词，防模型靠常识蒙对）。
`SubagentExecutor` 的 worker 自 v2 起**并行执行**（ThreadPoolExecutor），
longdoc 是"并行 subagent"的第一个有测试意义的场景。

## 扩充任务集

编辑 `eval/tasks.json`，按类别追加：

```json
{"id": "math-07", "category": "math", "task": "...", "expected": "42"}
{"id": "search-07", "category": "search", "task": "...", "must_contain": ["term1", "term2"]}
{"id": "longdoc-09", "category": "longdoc", "task": "Use the doc tool to read document 'X' page by page and report every fact", "must_contain": ["kw1", "..."]}
```

注意：评测值必须是确定性的（本地语料事实 / 合成文档事实 / 精确算术），不要依赖模型世界观。
