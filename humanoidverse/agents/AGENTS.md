# agents 局部指南

这里维护 FB、TeCH（旧名 TLDR）算法实现和 preset。常规训练入口由上级 `humanoidverse/train.py` 组装。

## 配置事实

- `presets/__init__.py` 是 agent 注册表：`fb` 映射 FB，`tech` 和弃用 alias `tldr` 映射 TeCH。
- FB preset 默认每 1024 环境步更新、16 次 agent update、latent interval 100、trajectory buffer。
- TeCH preset 默认每 1024 环境步更新、128 次 agent update、latent interval 10。
- learner 的 per-GPU env/buffer 与 global env-step 预算由 CLI 语义决定，不要在 preset 中重新解释。

## 修改规则

- preset 只放算法默认值；机器人、数据和 XML 语义放在 RobotSpec/manifest。
- 任何 latent/action 维度变化都要同步 checkpoint metadata、export/inference 和 targeted unittest。
- `tldr` 仅用于读取旧命令和旧配置，公开文档使用 TeCH。
