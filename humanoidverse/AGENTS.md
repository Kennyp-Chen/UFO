# humanoidverse 局部指南

本目录是训练、环境、任务、推理和导出的 Python 包。遵循根目录 `AGENTS.md` 的 GPU、uv、数据和 Git 约束。

## 边界

- `train.py`：常规训练 CLI、robot/data precedence、smoke 和分布式启动。
- `training/`：Workspace、TrainConfig 和 robot-aware 初始化。
- `agents/`：FB/TeCH learner、preset、replay buffer 与 encoder。
- `config/`：Hydra 运行时组合；外部 robot/data 语义由根目录 `configs/` 传入。
- `envs/`、`tasks/`：MJLab 环境、终止、reward 和 curriculum。
- `utils/`：RobotSpec、motion data、schema 和通用运行工具。
- `speed_stage2.py`、`amp_stage2_piplus_22dof.py`、`tech_kick_stage2.py`：独立实验入口，不要将 Stage2 默认值偷偷写回常规训练。

## 约束

- action、observation、latent 和 joint order 必须由 RobotSpec/asset contract 传递；禁止按机器人名称在公共流程中硬编码。
- Stage2 checkpoint 必须先检查 task metadata，再加载 decoder/encoder；speed、AMP、kick 任务不能交叉播放。
- 训练输出、checkpoint、ONNX 和 replay buffer 写入明确的 `runs/` 或 `/tmp` 路径，不提交到 Git。
- 修改 CLI 或默认值时同步更新对应中文文档和测试。

## 验证

- Python 改动先运行对应 `uv run ruff check`，再运行最小 unittest。
- Stage2 合同优先验证 `tests.test_piplus_h0w_stage2_common`、`tests.test_speed_stage2`、`tests.test_amp_stage2_piplus_22dof`、`tests.test_tech_kick_stage2`。
