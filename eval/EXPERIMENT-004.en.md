# Experiment Report 004: Phase-3 Full Rerun After Checker Fix + Parallelization

> Status: Pending review
> Date: 2026-08-14
> Model: deepseek-v4-flash (DeepSeek, OpenAI-compatible endpoint; ⚠️ differs from the deepseek-chat used in 001–003, see §3)
> Comparison: EXPERIMENT-003 (deepseek-chat, old checker, serial classification)
> Code version: main @ 06599f1 (includes six fixes #23–#28)

## 1. Method Changes Relative to 003

| Item | 003 | 004 |
|---|---|---|
| Checker | last number / keyword / ≥2-spawns structure check | **any-number match / per-term corpus grounding / spawns decoupled from correctness** (#26) |
| Classification execution | serial (600 runs; the full set previously timed out) | **strategy + classification interleaved thread-pool submission, elapsed time taken as max** (#27) |
| Model | deepseek-chat | **deepseek-v4-flash** (.env change; a confound) |
| ReAct robustness | empty output → No answer produced; greedy regex swallowed transcriptions | empty-output retry + parenthesis-aware parsing (#23/#24) |
| Evaluation set | 40 tasks | 40 tasks (unchanged) |

## 2. Results

### 2.1 Accuracy (each cell = correct samples / total samples; 40 = 8 tasks × 5 runs)

| category | direct | react | subagent |
|---|---|---|---|
| chain | 40/40 (100%) | 40/40 (100%) | 40/40 (100%) |
| direct | 40/40 (100%) | 40/40 (100%) | 40/40 (100%) |
| math | 39/40 (97%) | 40/40 (100%) | 40/40 (100%) |
| search | 0/40 (0%) | 38/40 (95%) | 26/40 (65%) |
| subagent | 40/40 (100%) | 37/40 (92%) | 39/40 (97%) |

40/40 tasks were solved by at least one strategy.

### 2.2 Cost (mean, by task category × strategy)

| Category | direct | react | subagent |
|---|---|---|---|
| chain | 1.0 / 259 tok / 1.7s | 1.5 / 612 / 2.2s | 3.9 / 1661 / 7.7s / 1.1 spawn |
| math | 1.0 / 537 / 3.5s | 2.0 / 781 / 2.3s | 4.1 / 1479 / 6.2s / 1.0 |
| search | 1.0 / 480 / 3.8s | 2.4 / 1179 / 4.0s | 8.5 / 6509 / 23.7s / 1.6 |
| subagent | 1.0 / 1403 / 16.5s | 2.5 / 2406 / 10.5s | 7.0 / 6364 / 46.9s / 1.8 |

### 2.3 Policy Layer (40 tasks)

| variant | LLM success rate (mean) | Majority-vote agreement rate |
|---|---|---|
| p0 | 95% | **37/40 (92.5%)** |
| p1 | 99% | 33/40 (82.5%) |
| p2 | 100% | 35/40 (87.5%) |

**Degenerate baselines (added in #19)**: always-direct 31/40 (77.5%), always-react 9/40, always-subagent 0/40, rule 13/40 (32.5%).

- policy(p0) at 92.5% beats always-direct by **15pp** (only 5pp in 003)
- rule at 32.5%: still systematically misclassifies ("calculate/compare" triggers react/subagent)

### 2.4 Remaining Classification Instability

- math-02 (large-number multiplication): LLM success rate **20%**, choice distribution direct:1/react:4
- math-06: 40%, direct:2/react:3
- chain-05: 80%, direct:3/react:2
- All other tasks: 100%

## 3. Core Findings (vs. 003)

1. **The checker fix is real and effective**: search direct 20% → **0%** (false positives from common-sense answers matching keywords eliminated); chain at 100% across all three strategies (false negatives in the tail-end verification eliminated); subagent-category subagent strategy 45% → **97%** (false negatives from the spawns structure check eliminated). The 003 conclusion that "subagent is useless (45%)" was **partly an artifact of checker false negatives**; under correct judging, the subagent strategy is not bad — it is merely still the most expensive.
2. **The model change is the main confound**: deepseek-v4-flash is overall stronger than deepseek-chat (search react 82%→95%, subagent-category react held at 92%, and math direct also had one genuine arithmetic error, 97%). The accuracy gains in 004 cannot be attributed to "checker fix" or "model change" as a single factor.
3. **The value of the adaptive policy is quantified against baselines for the first time**: p0 92.5% vs. always-direct 77.5%, a 15pp edge (only 5pp in 003); the agreement-rate metric now has discriminative power (the degenerate-baseline table from #19 provides the reference frame).
4. **The "direct by default" conclusion is more robust under a stronger model**: 95–100% everywhere except search, and at 1 call each; large-number multiplication (math-02) direct 5/5, yet the classifier still leaned react (4/5) — classification bias is decoupled from model capability.
5. **The "expensive" verdict on the subagent strategy is rock solid**: 8.5 calls / 23.7s on the search category, 7.0 calls / 46.9s on the subagent category (std 52s). Accuracy is not bad, but cost-effectiveness still ranks last; the tasks that genuinely need it (long-document chunking, parallel research) are still not covered by the evaluation set.

## 4. Limitations (Honest Disclosure)

1. **Model change and checker fix are conflated**: single-factor attribution of the accuracy changes is impossible; a clean comparison would require rerunning the 003 baseline with the model held fixed.
2. **Single model** (deepseek-v4-flash); the findings do not generalize across models.
3. The keyword + grounding checker is still approximate (5-gram citation checking is sensitive to light paraphrasing and may miss correct citations).
4. The subagent applicability domain still lacks real-scenario tasks.
5. The full run took ~70 min (1200+ calls, 4 workers; v4-flash inference is slow); parallel classification eliminated the "additive" bottleneck, but total duration is still dominated by model latency.

## 5. Candidate Next Steps

1. **Fixed model, two-checker comparison**: run the old and new checkers each under the same model to attribute the "checker fix" and the "model change" separately
2. Add an **LLM-as-judge** checker to eval and compare its misjudgment rate against keyword/grounding
3. Add tasks that genuinely need subagent (long-document chunking / parallel research) to answer "does subagent actually have a real use case"
4. Classification-stability deep dive: the 20–40% success rates of math-02/math-06 (classification degradation persists after the #14 fix)

## Appendix: Reproduction

```bash
source .venv/bin/activate
python -m eval.runner --runs 5 --workers 4 --policy-variants p0,p1,p2
# results -> eval/results/<ts>.json (also saved as FULL-N5-PHASE3.json)
```
