# 实验报告 002：多采样评测（N=5）

> 状态：待复核
> 日期：2026-08-13
> 模型：deepseek-chat（DeepSeek）
> 对比：EXPERIMENT-001（单次采样，24 任务）→ 本实验（N=5 采样，32 任务，4 workers 并行）
> 代码版本：`feat/phase1-eval-rigor`（PR #11）

## 1. 相对 001 的方法变化

| 项 | 001 | 002 |
|---|---|---|
| 采样 | 每 (task,strategy) 1 次 | **5 次**，报告 mean±std |
| 任务集 | 24 | **32**（math/search/direct/subagent 各 8） |
| subagent 类任务 | compare/summarize（不需要分解） | 保留 + 新增多主题报告（subagent-07/08，3+ 主题） |
| policy 测量 | 单次分类 | **5 次分类**：选择分布 + llm 成功率；一致率用**多数票** |
| 执行 | 串行（全量曾超时 30min） | **4 workers 并行** + 429/5xx 指数退避重试 |

## 2. 结果

### 2.1 正确率（每格 = 正确采样数/总采样数，40 = 8 任务 × 5 次）

| category | direct | react | subagent |
|---|---|---|---|
| direct | 40/40 (100%) | 39/40 (97%) | 40/40 (100%) |
| math | 38/40 (95%) | 36/40 (90%) | 38/40 (95%) |
| search | 8/40 (20%) | 33/40 (82%) | 21/40 (52%) |
| subagent | 40/40 (100%) | 38/40 (95%) | 26/40 (65%) |

32/32 任务至少一策略解出（001 有 1 个未解出）。

### 2.2 成本（mean±std）

| 类别 | direct | react | subagent |
|---|---|---|---|
| direct | 1.0 次 / 160 tok / 1.0s | 1.3 / 511 / 1.9s | 3.4 / 1131 / 5.3s / 1.0 spawn |
| math | 1.0 / 613 / 4.2s | 1.9 / 799 / 3.0s | 4.2 / 1525 / 7.1s / 1.2 |
| search | 1.0 / 495 / 3.9s | 2.0 / 917 / 3.1s | 7.6 / 4786 / 19.4s / 2.0 |
| subagent | 1.0 / 1579 / 13.7s | 1.4 / 1015 / 5.4s | 5.3 / 4176 / 21.5s / 2.4 |

### 2.3 Policy 稳定性与一致率

- **一致率（多数票）**：policy 28/32 (**87.5%**)，rule 10/32 (31%)
- **分类稳定性**：选择分布可见明显波动（如 math-06: direct 2/5 vs react 3/5；subagent-07: direct 1/5 vs subagent 4/5）
- **llm 成功率（分类未降级比例）**：多数任务 100%，但 math-02 20%、subagent-07 20%、search-03/05 60%

## 3. 核心发现（对比 001）

1. **多数票是性价比最高的政策改进**：policy 一致率从 001 的单次 65% 提升到 **87.5%**——5 次分类取多数，无需改任何模型或 prompt。直接缓解了 001 发现的"分类可靠性"瓶颈。
2. **direct 的统治地位在多采样下更稳固**：除 search 外全部 95-100%，且 1 次调用。subagent 类任务（含多主题报告）100%——deepseek-chat 写对比/报告不需要分解。
3. **react 仍是 search 的唯一可靠策略**（82% vs direct 20%）；subagent 策略 52% 且 19.4s，性价比最差。
4. **subagent 策略自身依旧不稳**：自家类别上 65%（空合成/关键词漏配仍在发生），21.5s / 5.3 次调用。
5. **规则策略 31%**：多数票也没救回来，规则对带 "calculate/compare/summarize" 的任务系统性误判。

## 4. 局限（在 001 基础上的进展与遗留）

- **已解决**：单次采样 → N=5 有均值与方差；subagent 类任务重设计；全量超时 → 并行 + 重试。
- **遗留**：关键词判定仍是近似（search 20% 的 direct 误判中部分可能是措辞问题）；llm 分类成功率在部分任务上只有 20%（格式约束未解决）；评测集仍为单模型单轮任务；std 大（tokens/延迟波动真实存在）。

## 5. 下一步候选

- 分类成功率专项：structured output / response_format 约束，或更宽 max_tokens + 更多重试
- 多数票的成本核算：5 次分类的额外开销 vs 一致率收益（当前一致率收益远大于成本）
- 后续实验（plan-execute / 元认知层）沿用 N=5 基准

## 附：复现

```bash
python -m eval.runner --category math    --runs 5 --workers 4   # 每类 120 次执行
python -m eval.runner --category direct  --runs 5 --workers 4
python -m eval.runner --category search  --runs 5 --workers 4
python -m eval.runner --category subagent --runs 5 --workers 4
# 合并四类结果 -> FULL-N5.json 渲染报告
```
