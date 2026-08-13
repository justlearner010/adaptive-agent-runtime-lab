# Adaptive Agent Runtime Lab

探索 Agent 运行时如何根据任务类型选择不同的执行策略（Execution Strategy）。

核心管线：**Task -> Policy -> Execution Strategy**

```
task ──▶ task_analyzer (Policy) ──▶ router ──▶ executor
                                              ├── direct   (单次 LLM 调用，无工具)
                                              ├── react    (Thought/Action/Observation 循环)
                                              └── subagent (分解 -> 委托子代理 -> 合成)
                                              └── tools: calculator / search
                                              └── trace:  全程结构化留痕
```

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

配置：`OPENAI_API_KEY` / `OPENAI_BASE_URL`（默认 api.openai.com/v1）/ `OPENAI_MODEL`（默认 gpt-4o-mini）。
换 DeepSeek/Moonshot/本地 vLLM 只需改 base_url + model。

## 目录

```
main.py            CLI 入口
llm.py             OpenAI 兼容端点的最小封装（chat / chat_json）
task_analyzer.py   Policy 层：LLM 分类 + 规则 fallback（HybridPolicy）
router.py          Policy -> Executor 分发
executors/
  direct.py        单次调用
  react.py         ReAct 循环（JSON Action 协议）
  subagent.py      分解 -> 独立上下文子代理 -> 合成
tools/
  calculator.py    AST 白名单安全计算器
  search.py        v1 本地语料占位（后续换真实搜索 API）
trace.py           结构化 trace（策略/LLM/工具/耗时）
```

## 设计要点

- **Policy 层**：`LLMPolicy`（一次 LLM 调用输出 JSON 分类）优先，`RulePolicy`（关键词启发式）兜底。
  分类失败或离线时自动降级，全程在 trace 中留痕。
- **ReAct 协议**：模型输出 `Action: {"tool": ..., "input": ...}` JSON，runtime 解析、执行、回填 Observation。
- **Subagent（v1 简化）**：planner 拆分子任务 -> 每个子任务跑独立 ReAct 上下文 -> synthesizer 合成。
  子任务上下文与主上下文隔离，不污染主 token 预算。
- **Trace 先行**：所有 trace 数据为后续「同任务集多策略对比评估」和「学习式 Policy」做准备。

研究背景与机制对比见 [RESEARCH.md](RESEARCH.md)。

## 协作流程（重要）

从 v2 开始，**每新增/改版都必须**：

1. 先提 Issue（说明改动动机与方案）
2. 再提 PR（关联该 Issue）
3. 由仓库 owner 实际 Review 后合并

v1 为初始提交，直接落在 main。详见 [CONTRIBUTING.md](CONTRIBUTING.md)。
