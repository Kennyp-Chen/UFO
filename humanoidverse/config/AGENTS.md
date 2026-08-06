# Hydra 配置局部指南

本目录是 MJLab 运行时 Hydra 配置树，继承根目录规则。它消费外部 RobotTrainingSpec，不替代 `configs/robots/*.yaml`。

## 组合关系

- `base.yaml` 提供 defaults、seed、headless、sim type 和基础结构。
- `env/` 管理 task/legged motion 的 episode、termination、resample 和 reset。
- `simulator/mujoco.yaml` 当前为 200 FPS、control decimation 4、substeps 1。
- `domain_rand/`、`obs/`、`rewards/`、`callbacks/` 是可组合分组。
- `terrain/` 提供基础摩擦和 locomotion plane；`base_eval.yaml` 只服务 tracking evaluation 的日志/model 变量。
- `base/hydra.yaml`、`base/structure.yaml` 是 Hydra 目录/槽位组合；`base/fabric.yaml` 是保留的 Lightning Fabric DDP 组合，默认不走当前训练入口。
- `exp/bfm_zero/bfm_zero.yaml` 是完整 BFM Zero 实验组合。
- `robot/` 保存 G1、PiPlus BFM、PiPlus H0W 的 Hydra robot schema。

## 约束

- 改动 YAML 时明确记录 defaults 顺序和 override 来源，避免同名字段静默覆盖。
- 观测维度、action scale/clip、joint order 必须和 robot XML 及 checkpoint 一致。
- 不在此目录写入训练输出；运行时通过 `humanoidverse.train` 的 robot-aware overrides 注入。
- 新增分组要有最小组合/解析验证，优先沿用现有 Hydra 命名。
