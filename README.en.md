# Adaptive Agent Runtime Lab

> [English](README.en.md) | 中文

**What this is**: a **research-oriented experimental project** (not a production system). It answers one question through controlled experiments —
**"which execution strategy should an AI task use (answer directly / reason while using tools / delegate to subagents) to be both correct and cheap?"**
To that end it builds a minimal Agent runtime pipeline (`Task -> Policy -> Execution Strategy`) and measures it repeatedly with a self-made evaluation set,
using the measurements to drive design decisions.

**Who it's for**: researchers interested in "how an Agent runtime chooses execution strategies", and collaborators who want to join the experiments.

**How to read (5-minute route for newcomers)**:
1. [RESEARCH.md](RESEARCH.md) — background: the Agent Runtime mechanism and the taxonomy of the three execution strategies (10 minutes)
2. The "Experimental Positioning" section of this file — what we did, where we are, conclusions and limitations
3. [eval/EXPERIMENT-001..004](eval/EXPERIMENT-001.en.md) — experiment history (deep read as needed)
4. [v3-design.md](v3-design.md) — current experiment design (v3: when does SubAgent surpass ReAct)

Exploring how an agent runtime chooses different execution strategies (Execution Strategy) based on task type.

Core pipeline: **Task -> Policy -> Execution Strategy**

```
task ──▶ task_analyzer (Policy) ──▶ router ──▶ executor
                                              ├── direct   (single LLM call, no tools)
                                              ├── react    (Thought/Action/Observation loop)
                                              └── subagent (decompose -> delegate to subagents -> synthesize)
                                              └── tools: calculator / search / doc
                                              └── trace:  structured tracing throughout
```

## Experimental Positioning (Important — Read First)

### Experiment Type

**A research-oriented experimental project**, not a production system. This repository drives design decisions through a "reproducible experimental loop":
design -> implementation -> evaluation -> defect discovery -> fix -> retest, with every step leaving a trace (Issue/PR/trace/evaluation data).

### Purpose

Answer one core question: **given a task, which execution strategy (direct / react / subagent) is "correct and cheap"?**

- Build the `Task -> Policy -> Execution Strategy` pipeline and quantify each strategy's applicability domain, correctness, and cost;
- Collect structured trace data to pave the way for the future "learned Policy" (v5);
- Along the way, verify the robustness of the runtime mechanisms themselves (ReAct loop, subagent delegation).

### Execution Strategy Definitions

The three strategies are three variants of "how the LLM is invoked / loop depth" (see [RESEARCH.md](RESEARCH.md) §2 for the full mechanism taxonomy):

- **Direct (direct execution)**: **a single LLM call, no tools, no loop** — the model produces the answer directly from its own knowledge alone.
  Cost = 1 call; capability ceiling = the model itself; typical failures = insufficient knowledge / mental arithmetic errors.
  Suitable for: common-sense Q&A, simple arithmetic, summarization/comparison, and other tasks that need no external information.

- **ReAct (reasoning-acting loop)**: *ReAct: Synergizing Reasoning and Acting in Language Models*
  (Yao et al., 2022). The loop `Thought -> Action(call tool) -> Observation(fill in result) -> ... -> Final Answer`,
  in a **single context**: each step's tool observation is appended to the same conversation, so the context grows with each step.
  Cost = 2..N calls; typical failures = loop drift, step limit (max_steps), context growth.
  Suitable for: single-threaded tasks that need tools (calculation / search / document reading).

- **Subagent (sub-agent delegation)**: the planner first **decomposes** the task into independent subtasks, each handed to
  a fresh ReAct loop in an **isolated context** (workers can run in parallel since v2), and finally a synthesizer assembles the final answer.
  The parent context is not polluted by subtasks, but total calls = decomposition + each subtask + synthesis, which is more expensive.
  Typical failures = decomposition errors, information lost during synthesis, empty worker outputs.
  Suitable for: decomposable / parallelizable / long-content tasks (e.g., long-document chunking, parallel research).

### Methodology

- **Evaluation harness** (`eval/`): **forcibly** runs all strategies × N samples for every task (default N=5),
  aggregating `llm_calls / tokens / latency / tool_calls / spawns` from the trace;
- **Correctness checker** (`eval/runner.py`):
  - math/chain categories: **any digit** in the answer matches expected (thousands separators tolerated);
  - search category: keyword hits **and** corpus citations for every required topic (5-gram grounding, to prevent guessing correctly from general knowledge);
  - spawns (number of subtasks) is an independent mechanism metric and is **not** counted toward correctness;
- **Agreement rate**: policy majority vote vs. `optimal` (among all correct strategies, the one with the fewest LLM calls; ties broken by fewer tokens),
  alongside an **always-\* degenerate baseline** as a frame of reference (otherwise "always pick direct" would already score 77.5%);
- **Change process**: Issue first -> then PR (`Closes #n`) -> owner review -> squash merge.

### Progress

| Stage | Content | Experiment/Commit |
|---|---|---|
| v1 | Pipeline skeleton + direct/react/subagent + calculator/search + trace | `90cdbb9` |
| Phase 1 | Evaluation harness + N=5 multi-sampling + parallel execution | [EXPERIMENT-001](eval/EXPERIMENT-001.en.md) / [002](eval/EXPERIMENT-002.en.md) |
| Phase 2 | Classification reliability: prompt variants p0/p1/p2, structured output, chain tasks | [EXPERIMENT-003](eval/EXPERIMENT-003.en.md) |
| Phase 3 | Correctness-checker fixes, ReAct robustness, subagent leakage, parallel classification, degenerate baseline + full retest | [EXPERIMENT-004](eval/EXPERIMENT-004.en.md) |
| v2 | Expanded measurement surface: longdoc evaluation category (real long-document tasks), parallel subagent, LLM-as-judge, classification stability (confidence), LLM-layer empty-output retry | [#34](https://github.com/justlearner010/adaptive-agent-runtime-lab/pull/34) |
| **v3 (in progress)** | **"When does SubAgent surpass ReAct" boundary evidence**: power analysis, multi-scale longdoc, parallel-structure comparison, hyperparameter sensitivity, trace process evidence | Design finalized in [v3-design.md](v3-design.md); P2 infrastructure in progress |
| v4+ | Task-form expansion (retrieval / selective reading), difficulty tasks, multi-model, learned Policy | Roadmap, not started |

**Current status**: v3 is in experiment design — first a cost-power analysis (conclusion: N=5 can only resolve ±6-14pp, "majority vote + McNemar" no longer works; switching to per-task accuracy + paired t-test), with three budget tiers of ¥12-50 to choose from.

### Tentative Conclusions (as of the full EXPERIMENT-004 run + v2 smoke test, single model deepseek-v4-flash)

1. **direct's applicability domain is far wider than intuition suggests**: except for corpus search, all categories score 95~100% correct with only 1 call —
   **"direct by default" is the optimal baseline** (it is the `optimal` for 31/40 tasks).
2. **react is necessary only when external information is needed**: 95% on corpus-search tasks vs. direct's 0%; using it on other tasks is pure waste.
3. **subagent is the most expensive and offers no advantage on the old evaluation set** (8.5 calls / 23.7s on search tasks) — but the old evaluation set had no tasks that truly require
   long-document chunking. **v2 has filled this gap**: the longdoc task category + parallel workers, and the smoke test confirms the mechanism works
   (the planner can decompose by page, workers read in parallel, and the synthesis is correct); **"under what conditions subagent starts to surpass react"
   is exactly the question v3 is testing** ([v3-design.md](v3-design.md)).
4. **LLM classification is high quality when it succeeds** (p0 majority-vote agreement 92.5%, 15pp above the always-direct baseline),
   but **classification is unstable on some tasks** (e.g., large-number multiplication math-02 has only a 20% classification success rate); the rule-based fallback is nearly useless (32.5%).
5. **The model significantly affects results**: after deepseek-chat -> v4-flash, overall correctness rose; conclusions must be verified with a fixed model.
6. **Reasoning models need token headroom**: deepseek-v4-flash returns **empty outputs** when max_tokens is too tight;
   fixed (LLM-layer empty-output retry + raising max_tokens everywhere) — the most valuable finding from the v2 smoke test.

> The above are **interim conclusions**, not final verdicts. Any change in model / evaluation set / correctness checker may rewrite them.

### Limitations (Honest Disclosure)

- **Single model, single-turn tasks**: no cross-model generalization, no multi-turn / long-context tasks;
- **The correctness checker is approximate**: keywords + 5-gram grounding are sensitive to light rewording; LLM-as-judge (v2) is implemented but not yet validated via large-scale replay;
- **Small and homogeneous evaluation set**: 48 tasks × 6 categories (longdoc added in v2), only 8 per category; longdoc has not been fully evaluated;
- **Confounding factors**: in EXPERIMENT-004, the model change and correctness-checker fixes were combined, so single-factor attribution is impossible;
- **Cost-oriented optimal**: answer quality and failure risk are not factored in;
- A full evaluation run takes about 70 minutes+ / a few yuan of API cost (4 workers, model latency dominates).

See [RESEARCH.md](RESEARCH.md) for the research background and mechanism comparison.

## Quick Start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...            # any OpenAI-compatible endpoint works
python main.py "what is 23 * 47 + 12"
python main.py "compare react and subagent strategies"
python main.py "who is the president of the usa" --offline   # runs without a key (rule-based policy)
python main.py "..." --json             # machine-readable output (full trace included)
```

You can also use a `.env` file at the project root (copy `.env.example`, edit it, and you're done; it is already gitignored):
it is loaded automatically at startup, and already-set environment variables take precedence (`.env` does not override manually exported values).

Configuration: `OPENAI_API_KEY` / `OPENAI_BASE_URL` (defaults to api.openai.com/v1) / `OPENAI_MODEL` (defaults to gpt-4o-mini).
Switching to DeepSeek / Moonshot / a local vLLM only requires changing base_url + model.

## Directory Layout

```
main.py            CLI entry point
llm.py             minimal wrapper around OpenAI-compatible endpoints (chat / chat_json, with empty-output retry)
task_analyzer.py   Policy layer: LLM classification (with confidence) + rule fallback (HybridPolicy)
router.py          Policy -> Executor dispatch
executors/
  direct.py        single call
  react.py         ReAct loop (JSON Action protocol)
  subagent.py      decompose -> parallel subagents -> synthesize (workers run in parallel since v2)
tools/
  calculator.py    AST-whitelist safe calculator
  search.py        v1 local-corpus placeholder (real search API later)
  doc.py           v2 long-document tool (deterministic synthetic documents + paged reading)
eval/              evaluation harness (runner / report / judge / tasks.json / experiment reports)
trace.py           structured trace (strategy / LLM / tools / latency)
v3-design.md       current experiment design (v3)
AGENTS.md          experiment design decision context (required reading for collaborators)
```

## Documentation Navigation

| Document | Content | Audience |
|---|---|---|
| [README.md](README.md) | This file: positioning / quick start / current status | Everyone |
| [RESEARCH.md](RESEARCH.md) | Background research: Agent Runtime mechanisms, execution strategy taxonomy | Those who want to understand "why" |
| [eval/EXPERIMENT-001..004](eval/EXPERIMENT-001.en.md) | Experiment history: four rounds of evaluation and fixes (bilingual CN/EN) | Those who want to reproduce/cite results |
| [v3-design.md](v3-design.md) | Current experiment design: when does SubAgent surpass ReAct | Those involved in v3 |
| [AGENTS.md](AGENTS.md) | Experiment design decision context, contribution checks, decision log | All collaborators (including agents) |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Collaboration process: Issue -> PR -> Review | Those who want to propose changes |

## Design Highlights

- **Policy layer**: `LLMPolicy` (a single LLM call outputs a JSON classification + confidence) takes priority, with `RulePolicy` (keyword heuristics) as the fallback.
  It degrades automatically on classification failure or offline use, and everything is recorded in the trace.
- **ReAct protocol**: the model outputs an `Action: {"tool": ..., "input": ...}` JSON, which the runtime parses with bracket awareness, executes, and fills back into Observation.
- **Subagent**: planner splits subtasks -> each subtask runs in its own ReAct context (**workers in parallel since v2**) -> synthesizer assembles the result.
  Subtask contexts are isolated from the main context and do not pollute the main token budget; on empty synthesis it falls back with the internal prompt stripped out.
- **Toolset**: calculator (safe arithmetic) / search (local-corpus placeholder) / doc (v2 long-document paging, for the longdoc evaluation category).
- **Trace first**: all trace data prepares for the later "multi-strategy comparison on the same task set" and the "learned Policy"; starting from v3, process evidence is added (parent-context tokens, decomposition granularity, worker omissions).

## Collaboration Process (Important)

Starting from v2, **every addition / change must**:

1. File an Issue first (explaining the motivation and approach)
2. Then open a PR (linking that Issue)
3. Merge only after the repository owner has actually reviewed it

v1 was the initial commit and landed directly on main. See [CONTRIBUTING.md](CONTRIBUTING.md) for details.
