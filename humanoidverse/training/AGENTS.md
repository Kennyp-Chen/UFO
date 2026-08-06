# training 局部指南

本目录负责把 CLI、RobotSpec、manifest、Hydra 配置和 Workspace 接成一次训练。它是 robot-aware 语义的汇合点。

## 关键合同

- `TrainConfig` 管理 env、replay、checkpoint、eval、W&B 和 distributed 设置。
- `build_ufo_mjlab_config` 先解析 robot/data，再选择 agent preset，最后写入 Hydra robot/action overrides。
- CLI robot 与 manifest robot 同时存在时必须经过 compatibility assertion；不能静默选择一方。
- `--num-envs`、`--buffer-size` 按 GPU；`--num-env-steps` 为全局预算。

## 安全边界

- 常规入口使用根目录 `run_train.sh`，Stage2 独立入口不要绕过资产合同。
- 不把 G1/PiPlus 的关节顺序、动作缩放或 root pose 写成通用默认。
- smoke 只用于短验证，输出放 `/tmp`；正式运行才使用明确的 `runs/<name>`。

## 验证

- CLI 解析、robot precedence 和 smoke 预算发生变化时，补对应 unittest。
- 修改后至少运行受影响文件的 Ruff 和 targeted unittest；分布式训练不作为常规单测。
