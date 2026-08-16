# Experiment Report 003: Strategy Reliability Deep-Dive (Definition Engineering + Structured Output + Chain-of-Thought Tasks)

> Status: Pending review
> Date: 2026-08-13
> Model: deepseek-chat (DeepSeek)
> Baseline: EXPERIMENT-002 (N=5, 32 tasks, p0 single variant)
> Code version: `feat/phase2-policy-reliability` (PR #12)

## 1. Method

- **Experiment A (definition engineering)**: comparison of three POLICY_PROMPT variants
  - `p0`: baseline (one-sentence definition)
  - `p1`: operationalized definition + per-strategy examples (intuition-based)
  - `p2`: data-grounded definition (reverse-engineered from 002's optimal-strategy labels: "default to direct; use react only for corpus queries and exact large-number arithmetic; subagent is barely necessary")
- **Experiment B (structured output)**: added `response_format=json_object` to `chat_json`, with automatic fallback when the endpoint does not support it
- **Experiment C (chain-of-thought tasks)**: added 8 deterministic tasks in the chain category (word problems / algebra / counting / large-number chains / powers / consecutive integers)
- Full run: 40 tasks × 3 strategies × 5 runs = 600 executions + 40 tasks × 3 variants × 5 runs = 600 classifications
- **Issue found and fixed along the way**: policy classification with `max_tokens=200` truncates the model's "thinking process" into an **empty response** (math-02 actually requires 537 tokens); fixed to 512 — this is the single biggest killer of classification reliability

## 2. Results

### 2.1 Accuracy (each cell = correct samples / total samples; 40 = 8 tasks × 5 runs)

| category | direct | react | subagent |
|---|---|---|---|
| direct | 40/40 (100%) | 36/40 (90%) | 40/40 (100%) |
| math | 40/40 (100%) | 38/40 (95%) | 38/40 (95%) |
| search | 8/40 (20%) | 33/40 (82%) | 24/40 (60%) |
| subagent | 40/40 (100%) | 37/40 (92%) | 18/40 (45%) |
| chain | 38/40 (95%) | 34/40 (85%) | 38/40 (95%) |

### 2.2 Prompt variant comparison (Experiments A/B, 40 tasks)

| variant | LLM success rate | Majority-vote agreement rate |
|---|---|---|
| p0 | **97%** | **33/40 (82.5%)** |
| p1 | 99% | 28/40 (70%) |
| p2 | 100% | 30/40 (75%) |

Compared with 002 (before the fix): p0 success rate 77% → 97% (max_tokens fix), agreement rate 29/40 → 33/40.

### 2.3 Cost (key rows)

| Category | direct | react | subagent |
|---|---|---|---|
| search | 1 call / 479 tok | 1.9 / 840 | 5.2 / 2832 / 13.7s |
| subagent | 1 / 1558 | 1.4 / 1074 | 4.7 / 3532 / 20.7s / 2.0 spawn |
| chain | 1 / 277 | 1.4 / 559 | 3.5 / 1325 / 6.1s |

## 3. Core Findings

1. **The #1 killer of classification reliability is token truncation, not format.** With `max_tokens=200`, the model "thinks" past the limit before outputting JSON and returns an empty string (math-02 actually requires 537 tokens). After the fix, p0 success rate rose from 77% to 97%. Structured output further guarantees format, but its contribution < that of the `max_tokens` fix.
2. **The "better" the definition, the higher the success rate — but the agreement rate actually drops.** p1/p2 success rates 99/100% vs p0 97%, but agreement rates 70/75% < p0 82.5%. Reason: more specific definitions ("use react for large-number arithmetic", "use subagent for reporting") push the model toward **empirically suboptimal** strategies — the data show that direct is optimal for these tasks. **Definition engineering should be reverse-engineered from data labels, not from intuition.**
3. **Applicability space map (final version)**:
   - direct: 95–100% on all categories except corpus search — including the newly added chain-of-thought tasks
   - react: **necessary only for corpus search** (82% vs direct 20%); on chain it is in fact the worst strategy (85%)
   - subagent: only 45% on its own category (this run), and the most expensive (20.7s) — **no use case could be found for it on this benchmark**
4. **Chain-of-thought tasks (chain) did not change the picture**: direct 95%, on par with subagent — the model's single-shot reasoning suffices, no decomposition needed; the model also computes large-number chains (chain-05) correctly in one pass (p0 even classifies it as direct).

## 4. Limitations

- The "agreement rate" in the variant comparison is computed by majority vote and still has noise within 5 votes (after the math-02 fix, p0 still has a 40% success rate on it)
- Structured output was validated only on the DeepSeek endpoint; the fallback path (400/422) is covered by unit tests and was not actually tested on other endpoints
- chain tasks are still not hard enough: direct at 95% shows that single-shot reasoning is sufficient; we need tasks that truly exceed single-shot capability (long-chain state tracking)
- The evaluation is still single-model (deepseek-chat)

## 5. Next-Step Candidates

1. **Take the p2 approach all the way**: automatically generate/validate definitions from optimal-strategy labels (a closed loop between definitions and data), and measure the agreement-rate ceiling
2. **Harder chain**: tasks that require multi-step state tracking (e.g., "the state after iterating a rule N steps") so that direct genuinely fails, and measure the react/subagent difference
3. Confidence: output confidence at classification time and combine it with majority vote (a step toward "uncertainty awareness" in the metacognitive layer)

## Appendix: Reproduction

```bash
python -m eval.runner --category <cat> --runs 5 --workers 4 --policy-variants p0,p1,p2
# Merge the results of the five categories and render the report (see eval/results/FULL-N5-PHASE2-FIXED.json)
```
