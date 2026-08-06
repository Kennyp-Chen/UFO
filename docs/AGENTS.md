# docs 局部指南

文档面向实验者，继承根目录的训练、GPU、数据和 Git 规则。公开行为变更同时维护 `README.md` 与 `README_zh-CN.md`。

## 入口

- `TRAIN_INFERENCE.md`：常规训练与推理。
- `robot_config_training.md`：robot-aware 训练。
- `import_wizard.md`：新机器人和 RobotState 导入。
- `training_config_overview_zh.md`：当前完整配置矩阵和 Stage2 状态。
- `piplus_h0w_stage2_zh.md`、`tech_kick_stage2_zh.md`：专项 Stage2 操作。

命令示例必须只使用物理 GPU 0-3，临时输出使用 `/tmp`。文档中的绝对路径、旧 backend 或旧 checkpoint 只能作为历史记录，不能暗示可直接复现。发现命令与根目录规则冲突时，优先修正文档或明确标注冲突。
