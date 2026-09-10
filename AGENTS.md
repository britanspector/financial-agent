# Agent Working Notes

- 开始前先阅读 `README.md` 并检查 `git status`；保留用户已有改动，不做破坏性 Git 操作。
- 使用 `rg` 搜索、`apply_patch` 编辑；修改应聚焦当前任务，避免顺手重构。
- Tool 只依赖 Service/Provider 抽象，不直接调用数据库、HTTP 数据源或第三方 SDK。
- 配置和凭证只通过环境变量或被 Git 忽略的 `.env` 注入；不得写入源码、测试输出、日志或提交。
- 默认测试不得访问外网：`python -m pytest -q`。完整 synthetic 验收使用 `-m integration`，真实行情使用 `-m live`。
- 完成功能后运行相关测试、`git diff --check` 和敏感信息检查，并简洁更新 README 的阶段、能力与限制。
