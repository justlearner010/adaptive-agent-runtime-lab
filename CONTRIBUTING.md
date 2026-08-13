# 协作流程

## 总原则

本仓库的每次变更都要**可追溯、可评审**：

- 变更动机 -> Issue
- 实现 -> PR（关联 Issue）
- 合并 -> 由 owner（justlearner010）Review 后合并

## 流程

1. **提 Issue**：写清楚要改什么、为什么改、方案要点（附关键设计决策）。
2. **开 PR**：标题关联 Issue（如 `Closes #12`），描述改动与验证方式（demo 输出 / 测试）。
3. **Review**：owner 在 PR 上评审；改动方按评审意见迭代；批准后合并。
4. **合并规范**：默认 squash 合并；提交信息遵循 `<type>: <summary>`（type: feat/fix/docs/refactor/test）。

## 分支约定

- `main` 为唯一长期分支，保持可运行。
- 功能分支命名：`issue-<编号>-<短描述>`（如 `issue-12-search-api`）。

## v1 说明

v1（Task -> Policy -> Execution Strategy 骨架 + direct/react/subagent + calculator/search + trace）
作为初始提交直接落在 main，从 v2 起严格执行上述流程。
