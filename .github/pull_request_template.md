## 关联 Issue

Closes #<issue-number>

<!-- 必填：本 PR 实现/修复哪个 Issue。未关联 Issue 的 PR 不予合并。 -->

## 动机

<!-- 为什么需要这个改动？一句话说明背景与目标。 -->

## 改动清单

<!-- 列出本次改动的核心内容，逐条勾选。 -->

- [ ] 改动 1
- [ ] 改动 2

## 验证

<!-- 必须提供可复核的验证证据，以下至少填写一项。 -->

- [ ] `pytest` 全绿（附关键输出）
- [ ] 运行 demo / 命令输出（粘贴关键片段）
- [ ] 涉及评测时：说明影响范围与重跑方式（如 `python -m eval.runner --category <cat>`）

## 影响面与风险

<!-- 影响哪些模块（task_analyzer / executors / tools / eval…）？有无行为变化或回归风险？ -->

## Checklist

- [ ] 已关联 Issue（`Closes #<编号>`）
- [ ] 分支命名符合规范：`issue-<编号>-<短描述>`
- [ ] 提交信息遵循 `<type>: <summary>`（type: feat/fix/docs/refactor/test/chore）
- [ ] 自测通过，无遗留 TODO / 调试代码
