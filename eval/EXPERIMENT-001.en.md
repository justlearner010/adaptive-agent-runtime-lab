# Experiment Report 001: Strategy Selection Baseline (Strategy Eval)

> Status: v1 experiment, pending review
> Date: 2026-08-13
> Model: deepseek-chat (DeepSeek, OpenAI-compatible endpoint)
> Environment: Python 3.14 / openai 1.66.3 / local `.env` configuration
> Code version: `feat/eval-harness` (PR #9)

---

## 1. Experiment Purpose and Research Questions

The core pipeline of this Lab is `Task -> Policy -> Execution Strategy`: given a task, the Policy layer first decides the strategy (direct / react / subagent), and the corresponding executor then completes it. The goal of Phase 1 is to answer three questions:

1. **On which tasks is each execution strategy "correct and cheap"** (correctness, cost, latency)
2. **How accurate is the Policy layer's selection** — the agreement rate with the "optimal strategy"
3. **LLM classification vs. rule heuristics: whose signal is more reliable**

## 2. System and Method

### 2.1 System Under Test

```
task -> task_analyzer (HybridPolicy) -> router -> executor
                                                  ├─ direct    single LLM call
                                                  ├─ react    Thought/Action/Observation loop
                                                  └─ subagent planner decomposition -> independent-context subagents -> synthesizer synthesis
                                                  └─ tools: calculator(AST whitelist) / search(local corpus)
                                                  └─ trace: structured events (llm_call / tool_call / subagent / policy)
```

- **Policy layer**: `HybridPolicy` = `LLMPolicy` (LLM outputs a JSON classification: strategy/complexity/tools/reasoning) first, falling back to `RulePolicy` (keyword regex) on failure.
- **Forced strategy**: `--force-strategy` bypasses the Policy and directly specifies the executor, for evaluation use; the CLI and the runner share the same `run_pipeline()` path.

### 2.2 Evaluation Set (eval/tasks.json, 24 tasks × 4 categories)

| Category | n | Task form | Judging method |
|---|---|---|---|
| math | 6 | Pure arithmetic (incl. large numbers, functions) | numeric value extracted from answer == expected (tolerance 1e-6, thousands separators tolerated) |
| search | 6 | Query the local corpus | answer contains all must_contain keywords |
| direct | 6 | Common sense / simple arithmetic | keyword containment |
| subagent | 6 | compare / summarize / report | keyword containment; the subagent strategy itself additionally requires >=2 spawns (structural check) |

All evaluation values are designed to be **deterministic** (exact arithmetic, local-corpus facts), independent of the model's world knowledge.

### 2.3 Metric Definitions

For each (task, strategy) combination, aggregate from trace events:

- `correct`: judged per category
- `llm_calls` / `tokens` / `latency_ms`: llm_call event count / usage sum / ms sum
- `tool_calls` / `tool_failures`: tool_call event count / count with ok=False
- `spawns`: number of subtasks decomposed by subagent (mechanism fidelity)
- `optimal`: among all strategies that solve the task correctly, the one with the fewest llm_calls (ties broken by fewer tokens)
- `agreement rate`: proportion of policy / rule selections equal to optimal; tasks where all three strategies fail are listed as "evaluation-set issues" and excluded

### 2.4 Execution Procedure

- 3 strategies × each task = 72 executions, plus 1 HybridPolicy classification and 1 RulePolicy per task
- ~100+ LLM calls in total (DeepSeek, cost about ¥1~2)
- The first full run timed out (>30min); switched to per-category batched execution (four batches: math / search / subagent / direct), then merged

## 3. Process and Defects Discovered Along the Way

During the experiment, 5 defects were fixed via the Issue→PR workflow (3 of them directly exposed by this experiment):

| # | Defect | How found | Fix |
|---|---|---|---|
| 1 | Wrong tuple-unpacking order in the search tool (TypeError) | Early manual verification | PR #5 |
| 2 | chat_json truncation with no fallback (subagent crash) | Early manual verification | PR #7 (corrective retry + token cap) |
| 3 | subagent planner returned empty JSON for trivial tasks and crashed outright | First-round smoke test | Fall back to a single subtask (self-delegation) |
| 4 | The judge applied the structural check (>=2 spawns) incorrectly to direct/react runs | First-round full-result analysis | Structural check applies only to the subagent strategy itself; spawns demoted to a standalone metric |
| 5 | Empty synthesizer output produced an empty final answer | First-round full-result analysis | Fall back to the raw worker report text |

Also found: the full run timed out (the subagent strategy took long on some tasks); added per-category batching plus progress output.

## 4. Results

### 4.1 Correctness (by category × strategy)

| category | direct | react | subagent | solved by at least one strategy |
|---|---|---|---|---|
| direct (n=6) | 6/6 (100%) | 5/6 (83%) | 6/6 (100%) | 6/6 |
| math (n=6) | 6/6 (100%) | 6/6 (100%) | 5/6 (83%) | 6/6 |
| search (n=6) | 0/6 (0%) | 4/6 (67%) | 4/6 (67%) | 5/6 |
| subagent (n=6) | 6/6 (100%) | 5/6 (83%) | 4/6 (67%) | 6/6 |
| **Total** | **18/24** | **20/24** | **19/24** | **23/24** |

Not solved: `search-01` ("search the corpus for what react is", keywords interleaves+reasoning were not hit simultaneously in the answer).

### 4.2 Cost (mean per task, n=6)

| Category | direct | react | subagent |
|---|---|---|---|
| direct | 1 call / 149 tok / 1.9s | 1 / 512 / 3.3s | 3 / 1076 / 6.8s / 1 spawn |
| math | 1 / 516 / 3.5s | 1 / 640 / 1.8s | 4 / 1354 / 5.8s / 1 spawn |
| search | 1 / 383 / 3.0s | 1 / 769 / 2.8s | 6 / 3877 / 16.9s / 1 spawn |
| subagent | 1 / 1391 / 12.4s | 1 / 950 / 6.1s | 6 / 5119 / 25.2s / 2 spawns |

The subagent strategy is the most expensive in every category (3~6 calls, up to 25s).

### 4.3 Policy Layer Quality (vs. optimal strategy, 23 judgeable tasks)

| Signal | Agreement rate | Breakdown |
|---|---|---|
| **policy (Hybrid)** | **15/23 (65%)** | source=llm: 14/16 (**87.5%**); source=fallback: 1/7 (**14%)** |
| **rule** | **6/23 (26%)** | - |

Key numbers: when LLM classification **succeeds** (16/23 tasks, 70%) it is almost always right; **once it fails and falls back to rules**, the agreement rate drops to 14%.

## 5. Core Findings and Discussion

1. **direct is systematically underestimated**. The optimal strategy for the math/direct/subagent task categories is direct (1 call and 100% correct). The model's mental arithmetic, comparison, and summarization abilities suffice for these tasks, yet the rule strategy labels anything containing "calculate/compare/summarize" as react/subagent — the rules don't know that "the model can do it itself".

2. **react only matters for tasks that need external information**. In the search category: direct 0% (the model has no corpus knowledge) vs. react 67%. This is direct evidence that the tools have a reason to exist; conversely, using react on non-external-information tasks is pure waste (one extra call, no correctness gain).

3. **subagent is "expensive and not necessarily better"**. It is the most expensive (up to 25s/6 calls), and on the very subagent-category tasks where it should shine it scores 67% < direct 100%. Reason: the "compare/summarize" tasks in the current evaluation set **don't actually need decomposition** — the scenarios that genuinely need subagent (long documents, parallel research, isolated contexts) are not covered by this evaluation set.

4. **The bottleneck of the strategy layer is "reliability of the classification output", not the LLM's judgment**. When LLM classification succeeds, the agreement rate is 87.5%, nearly perfect; but classification failed on 7/23 tasks (invalid JSON / truncation / timeout), and after falling back to rules the agreement rate is only 14%. chat_json already has one round of corrective retry, still insufficient.

5. **The rule heuristics are close to useless (26% agreement rate)**. As the fallback when the LLM fails, its quality determines the floor of the entire HybridPolicy — and the current floor is very low.

## 6. Limitations (Honest Disclosure)

1. **Single sampling**. LLM non-determinism is not handled: each (task, strategy) runs only once, and the agreement rate has no confidence interval. The policy choice for math-02 was react in one run and direct in another, showing that the classification itself is unstable.
2. **Keyword judging is approximate**. Both false positives (answering off-topic but coincidentally containing keywords) and false negatives (correct but phrased differently, e.g. search-01) exist.
3. **The evaluation set is small and homogeneous within categories**. Only 6 tasks per category; the subagent-category tasks are flawed by design (compare/summarize don't need decomposition); there are no tasks requiring genuinely multi-step reasoning or long context.
4. **The "optimal strategy" definition is purely cost-driven** (fewest calls among correct ones). It doesn't account for answer-quality differences (both marked correct but with different levels of detail) or the distribution of failure risk.
5. **Latency metrics are affected by network fluctuation**; latency_ms only counts LLM-side time, so cross-batch comparability is mediocre.
6. **Single model** (deepseek-chat); the conclusions don't generalize across models.
7. **The evaluation environment shares in-process state with the runtime (none)**, but DeepSeek server-side load affects result reproducibility.

## 7. Conclusions

- Each of the three strategies has its own applicable domain, but **the boundaries differ from intuition**: direct covers far more than expected; subagent's applicable domain is narrower than expected (and the current evaluation set cannot demonstrate its value); react is a necessity for "external-information tasks".
- The HybridPolicy's LLM classification is high quality when it succeeds (87.5%), **and the current main contradiction is the reliability of the classification output** (70% success rate); the rule fallback drags the floor down to 14%.
- The eval harness itself works: all 72 executions left structured traces, and the defect-finding → fix → re-run closed loop holds.

## 8. Next-Step Recommendations (for the "conclusions are far from sufficient" list)

By priority:

1. **Multiple sampling**: run each (task, strategy) N=5 times and report mean and variance; run the policy classification N times to measure stability. This is the prerequisite for the credibility of all current conclusions.
2. **Redesign the evaluation set**: add tasks that genuinely need subagent (long-document chunking, parallel research, tasks requiring context isolation); add multi-step reasoning and long-chain tasks; expand each category to 10+.
3. **Stronger judge**: use LLM-as-judge for open tasks (an independent model judges correctness of the same answer), replacing keyword approximation; keep deterministic judging for numeric tasks.
4. **Dedicated work on classification reliability**: make "classification failure rate" a first-class metric (currently 30%); improve chat_json robustness (wider token limits, schema constraints, multi-round retry); evaluate whether response-format constraints (structured output) are usable.
5. **Multi-model comparison**: deepseek-chat vs. deepseek-reasoner (or another OpenAI-compatible endpoint), to verify the model dependence of the conclusions.
6. **Refine the optimal-strategy definition**: introduce a quality dimension (judge score) and failure risk (failure rate over multiple samples), weighing cost against quality.
7. **Comparative experiments on the strategy layer itself**: run LLM-only / rule-only / hybrid configurations on the same evaluation, quantifying the cost-effectiveness of each configuration.

## Appendix: Reproduction Procedure

```bash
cd ~/adaptive-agent-runtime-lab && source .venv/bin/activate
python -m eval.runner --category math      # run per category in batches (to avoid full-run timeout)
python -m eval.runner --category search
python -m eval.runner --category subagent
python -m eval.runner --category direct
# merge the four batches into eval/results/FULL.json, then render the report
```
