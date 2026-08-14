# 协作流程

本仓库的每次变更都要**可追溯、可评审、可验证**。从 v2 起严格执行本流程，
所有新增/改版必须走通下面的闭环：

```
提 Issue ──▶ 开 PR（关联 Issue）──▶ owner Review ──▶ 批准 ──▶ squash 合并
```

## 1. 提 Issue（必做，先于任何改动）

每次新增/改版前必须存在对应 Issue，说明**改什么、为什么改、怎么改**：

- [ ] 动机与目标（一句话说清背景）
- [ ] 方案要点与关键设计决策
- [ ] 必要时的复现/数据证据（失败 trace、实验数据、报错信息）
- [ ] 验证方式（测试、demo 命令、评测批次）

## 2. 开 PR（必做，且必须关联 Issue）

- [ ] 分支命名：`issue-<编号>-<短描述>`（如 `issue-12-search-api`）
- [ ] 标题：`<type>: <summary>`，type ∈ feat/fix/docs/refactor/test/chore
- [ ] 描述按 [PR 模板](../.github/pull_request_template.md) 填写，其中：
  - [ ] `Closes #<编号>` 指向对应 Issue（未关联的 PR 不予合并）
  - [ ] 动机、改动清单、影响面
  - [ ] 验证证据：`pytest` 结果、demo 输出
- [ ] 提交信息遵循 `<type>: <summary>`，关联 Issue 时附 `(closes #<编号>)`

## 3. Review（owner 执行）

Owner 评审清单：

- [ ] 是否关联 Issue，改动范围与 Issue 一致
- [ ] 是否符合架构约定：`Task -> Policy -> Execution Strategy` 分层、trace 先行
- [ ] 是否破坏现有测试；新逻辑是否有对应单测
- [ ] 行为变化是否在 trace/输出中留痕（可观测性）
- [ ] 有无回归风险（工具注册表、LLM 调用路径、eval 判定逻辑）

改动方按评审意见迭代，owner 批准后合并。

## 4. 合并规范

- 默认 **squash 合并**，保持 `main` 历史线性可读
- `main` 为唯一长期分支，始终保持可运行

## Definition of Done（合并前必须全部满足）

- [ ] 有对应 Issue，且 PR 已 `Closes` 关联
- [ ] `pytest` 全绿
- [ ] 描述中含验证证据（测试结果 / demo 输出）
- [ ] owner 已批准

## v1 说明

v1（Task -> Policy -> Execution Strategy 骨架 + direct/react/subagent + calculator/search + trace）
作为初始提交直接落在 main，从 v2 起严格执行上述流程。
