# Experiment Report 002: Multi-Sampling Evaluation (N=5)

> Status: Pending review
> Date: 2026-08-13
> Model: deepseek-chat (DeepSeek)
> Compared to: EXPERIMENT-001 (single sampling, 24 tasks) → this experiment (N=5 sampling, 32 tasks, 4 parallel workers)
> Code version: `feat/phase1-eval-rigor` (PR #11)

## 1. Method Changes Relative to 001

| Item | 001 | 002 |
|---|---|---|
| Sampling | 1 per (task, strategy) | **5 times**, reporting mean±std |
| Task set | 24 | **32** (8 each for math/search/direct/subagent) |
| subagent-category tasks | compare/summarize (no decomposition needed) | Retained + added multi-topic reports (subagent-07/08, 3+ topics) |
| policy measurement | Single classification | **5 classifications**: choice distribution + llm success rate; agreement rate uses **majority vote** |
| Execution | Serial (full run once timed out at 30 min) | **4 parallel workers** + exponential backoff retries on 429/5xx |

## 2. Results

### 2.1 Accuracy (each cell = correct samples / total samples; 40 = 8 tasks × 5 runs)

| category | direct | react | subagent |
|---|---|---|---|
| direct | 40/40 (100%) | 39/40 (97%) | 40/40 (100%) |
| math | 38/40 (95%) | 36/40 (90%) | 38/40 (95%) |
| search | 8/40 (20%) | 33/40 (82%) | 21/40 (52%) |
| subagent | 40/40 (100%) | 38/40 (95%) | 26/40 (65%) |

32/32 tasks were solved by at least one strategy (001 had 1 unsolved).

### 2.2 Cost (mean±std)

| Category | direct | react | subagent |
|---|---|---|---|
| direct | 1.0 calls / 160 tok / 1.0s | 1.3 / 511 / 1.9s | 3.4 / 1131 / 5.3s / 1.0 spawn |
| math | 1.0 / 613 / 4.2s | 1.9 / 799 / 3.0s | 4.2 / 1525 / 7.1s / 1.2 |
| search | 1.0 / 495 / 3.9s | 2.0 / 917 / 3.1s | 7.6 / 4786 / 19.4s / 2.0 |
| subagent | 1.0 / 1579 / 13.7s | 1.4 / 1015 / 5.4s | 5.3 / 4176 / 21.5s / 2.4 |

### 2.3 Policy Stability and Agreement Rate

- **Agreement rate (majority vote)**: policy 28/32 (**87.5%**), rule 10/32 (31%)
- **Classification stability**: the choice distribution shows noticeable fluctuation (e.g., math-06: direct 2/5 vs react 3/5; subagent-07: direct 1/5 vs subagent 4/5)
- **llm success rate (proportion of classifications not degraded)**: 100% on most tasks, but math-02 20%, subagent-07 20%, search-03/05 60%

## 3. Key Findings (Compared to 001)

1. **Majority vote is the most cost-effective policy improvement**: policy agreement rate rose from 65% with single sampling in 001 to **87.5%** — taking the majority of 5 classifications requires no model or prompt changes. It directly alleviated the "classification reliability" bottleneck found in 001.
2. **direct's dominance is more robust under multi-sampling**: 95-100% everywhere except search, at 1 call. subagent-category tasks (including multi-topic reports) hit 100% — deepseek-chat does not need decomposition to write comparisons/reports.
3. **react remains the only reliable strategy for search** (82% vs direct 20%); the subagent strategy scores 52% at 19.4s, the worst cost-effectiveness.
4. **The subagent strategy itself remains unstable**: 65% on its own category (empty synthesis / missed keyword matches still occur), 21.5s / 5.3 calls.
5. **Rule strategy at 31%**: majority vote could not save it either; the rule systematically misclassifies tasks carrying "calculate/compare/summarize".

## 4. Limitations (Progress and Remaining Issues Relative to 001)

- **Resolved**: single sampling → N=5 provides mean and variance; subagent-category tasks redesigned; full-run timeout → parallelism + retries.
- **Remaining**: keyword-based judging is still approximate (some of the 20% direct misjudgments on search may be wording issues); llm classification success rate is only 20% on some tasks (format constraints unresolved); the eval set still consists of single-model, single-turn tasks; std is large (tokens/latency variance is real).

## 5. Candidate Next Steps

- Dedicated effort on classification success rate: structured output / response_format constraints, or wider max_tokens + more retries
- Cost accounting for majority vote: the extra overhead of 5 classifications vs the agreement rate gain (currently the agreement rate gain far outweighs the cost)
- Follow-up experiments (plan-execute / metacognitive layer) will keep the N=5 baseline

## Appendix: Reproduction

```bash
python -m eval.runner --category math    --runs 5 --workers 4   # 每类 120 次执行
python -m eval.runner --category direct  --runs 5 --workers 4
python -m eval.runner --category search  --runs 5 --workers 4
python -m eval.runner --category subagent --runs 5 --workers 4
# 合并四类结果 -> FULL-N5.json 渲染报告
```
