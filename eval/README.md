"""Strategy evaluation harness (v4).

对每个任务强制跑三种执行策略，从 trace 提取指标，量化：
- 各策略在每类任务上的正确率
- 各策略的平均成本（LLM 调用次数 / token / 延迟）
- policy 层选择 vs 最优策略的一致率（LLM 分类 vs 规则分类）

## 用法

```bash
python -m eval.runner --limit 6              # 冒烟：6 个任务
python -m eval.runner                        # 全量：24 任务 x 3 策略
python -m eval.runner --category subagent    # 只跑某类
python -m eval.runner --strategies react,direct
```

结果 JSON 写入 `eval/results/<ts>.json`（已 gitignore），报告打印到终端。

## 指标定义

- `correct`：math 用数值相等（容忍千分位）；search/direct 用 `must_contain` 关键词；subagent 类任务按回答关键词判定，其中 **subagent 策略的运行**额外要求 >=2 次 spawn（结构检查，验证真的分解了），其他策略只看答案本身
- `llm_calls` / `tokens` / `latency_ms`：从 trace 的 llm_call 事件聚合
- `tool_calls` / `tool_failures`：从 trace 的 tool_call 事件聚合
- `spawns`：subagent 分解出的子任务数（机制保真度指标）
- `optimal`：任务所有正确策略中 llm_calls 最少者（并列取 token 少）
- 三策略全错的任务列为"评测集问题"（任务设计或工具问题），不计入一致率

## 扩充任务集

编辑 `eval/tasks.json`，按类别追加：

```json
{"id": "math-07", "category": "math", "task": "...", "expected": "42"}
{"id": "search-07", "category": "search", "task": "...", "must_contain": ["term1", "term2"]}
```

注意：评测值必须是确定性的（本地语料事实 / 精确算术），不要依赖模型世界观。
