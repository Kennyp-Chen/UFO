# Unitree 速度奖励 H0W 适配 Profile C

Profile C 是一个显式 opt-in 的速度奖励配置，名称为 `c_unitree_velocity_h0w_compat`。它把 Unitree Robotics 官方 G1 velocity task 的奖励项和 PPO 超参数迁移到 PiPlus H0W 22DoF 二阶段入口，同时保留 H0W 的机器人、动作、观测和终止语义。该 profile 不改变 Profile A 或 Profile B 的默认值，跨 profile 恢复训练也不会被静默接受。

## 1. 来源与实现

Unitree 侧来源固定为仓库 [`unitreerobotics/unitree_rl_mjlab`](https://github.com/unitreerobotics/unitree_rl_mjlab)，default branch 为 `main`，钉住 commit `1425b15f73bd4095f0df53709d7c389c3eb9e790`。依赖版本为 `mjlab==1.2.0` 和 `rsl-rl-lib==5.0.1`。

源代码位置如下，链接均指向上述固定 commit：

| 内容 | 文件 |
| --- | --- |
| base 配置 | [`src/tasks/velocity/velocity_env_cfg.py`](https://github.com/unitreerobotics/unitree_rl_mjlab/blob/1425b15f73bd4095f0df53709d7c389c3eb9e790/src/tasks/velocity/velocity_env_cfg.py) |
| G1 覆盖 | [`src/tasks/velocity/config/g1/env_cfgs.py`](https://github.com/unitreerobotics/unitree_rl_mjlab/blob/1425b15f73bd4095f0df53709d7c389c3eb9e790/src/tasks/velocity/config/g1/env_cfgs.py) |
| PPO 配置 | [`src/tasks/velocity/config/g1/rl_cfg.py`](https://github.com/unitreerobotics/unitree_rl_mjlab/blob/1425b15f73bd4095f0df53709d7c389c3eb9e790/src/tasks/velocity/config/g1/rl_cfg.py) |
| 奖励公式 | [`src/tasks/velocity/mdp/rewards.py`](https://github.com/unitreerobotics/unitree_rl_mjlab/blob/1425b15f73bd4095f0df53709d7c389c3eb9e790/src/tasks/velocity/mdp/rewards.py) |

UFO_HT 中的实现位置：

| 内容 | 文件 |
| --- | --- |
| H0W 奖励实现 | `humanoidverse/piplus_h0w_unitree_velocity.py` |
| PPO profile、`UNITREE_PPO_CONFIG` 和 `ppo_update` 扩展 | `humanoidverse/piplus_h0w_stage2.py` |
| Profile C 注册与 metadata | `humanoidverse/amp_stage2_piplus_22dof.py` |

## 2. 奖励对比

下表逐项记录 Unitree 原始 term 和权重、H0W Profile C 的对应 term 和权重，以及迁移分类。

### EXACT：公式与权重一致

| Unitree 原始 term / weight | H0W Profile C term / weight | 分类与公式 |
| --- | --- | --- |
| `track_linear_velocity` `1.0`，`std=sqrt(0.25)=0.5` | `tracking_lin_vel` `1.0` | EXACT。`exp(-(xy_err² + 2·z_err²)/0.25)`，z 误差权重为 xy 的 2 倍。 |
| `track_angular_velocity` `1.0`，`std=sqrt(0.5)` | `tracking_ang_vel` `1.0` | EXACT。`exp(-(yaw_err² + 0.05·‖torso_ω_xy‖²)/0.5)`。 |
| `body_orientation_l2` `-1.0`，G1 使用 `torso_link` | `orientation_l2` `-1.0`，H0W 使用 `torso` | EXACT。对投影重力的 xy 分量作平方惩罚。 |
| `body_ang_vel` `-0.05` | `ang_vel_xy_l2` `-0.05` | EXACT。惩罚 torso 世界坐标系角速度的 xy 平方。 |
| `joint_acc_l2` `-2.5e-7` | `dof_acc_l2` `-2.5e-7` | EXACT。 |
| `action_rate_l2` `-0.05` | `action_rate_l2` `-0.05` | EXACT。 |
| `foot_gait` `0.5`，period `0.6`，offset `[0.0,0.5]`，threshold `0.56`，command threshold `0.1` | `gait_phase` `0.5`，参数相同 | EXACT。 |
| `stand_still` `-1.0`，command threshold `0.1` | `stand_still` `-1.0` | EXACT。默认位形差平方，并由站立指令门控。 |

### ADAPTED：权重一致，公式或输入适配 H0W

| Unitree 原始 term / weight | H0W Profile C term / weight | 适配说明 |
| --- | --- | --- |
| `is_terminated` `-200.0` | `termination` `-200.0` | H0W 环境在奖励计算后分离 `terminated` 和 `truncated`，实现使用 `done` 标志，代码注释已说明这一点。 |
| `joint_pos_limits` `-10.0` | `dof_pos_limits` `-10.0` | 使用 `dof_pos_limits`。H0W 终止态捕获不含该字段时，该项为 0。 |
| `foot_clearance` `-1.0`，target height `0.10` | `feet_clearance` `-1.0` | H0W 没有 G1 脚部 site 传感器，因此改用 body position 的脚高度和 body velocity 的水平速度。摆动相门控公式为 `|foot_z−0.10|·‖v_xy‖`。 |
| `foot_slip` `-0.25` | `feet_slip` `-0.25` | 使用 `contact_forces` 判断接触脚，接触时惩罚脚部 xy 速度平方。 |
| `soft_landing` `-1e-3` | `soft_landing` `-1e-3` | 首次接触时惩罚接触力大小。 |

### OMITTED：H0W 无法表达，不做近似

以下 Unitree term 不在 H0W Profile C 中近似实现，并写入 metadata 的 `omitted_unitree_terms`：

| Unitree 原始 term / weight | 省略原因 |
| --- | --- |
| `angular_momentum` `-0.025` | H0W core 没有 `root_angmom` 传感器。 |
| `self_collisions` `-1.0`，G1 使用 self collision 传感器和 `force_threshold=10.0` | H0W 没有自碰撞传感器契约。 |
| `pose` / `variable_posture` `1.0` | G1 使用关节正则 std map，包括 `std_standing .*:0.05`，以及 walking、running 对 hip、knee、ankle、waist、shoulder、elbow、wrist 的正则分布。H0W 没有对应的 G1 姿态正则图。 |

## 3. PPO 对比

Profile C 的 `UNITREE_PPO_CONFIG` 对应 Unitree G1 `rl_cfg`。Profile A 和 Profile B 保持原有 PPO 默认值。

| Unitree G1 `rl_cfg` | Profile C | 说明 |
| --- | --- | --- |
| `value_loss_coef=1.0` | `value_coef=1.0` | 名称适配。 |
| `use_clipped_value_loss=True` | `clip_value_loss=True` | rsl_rl 语义为 `max((V−ret)², (V_old+clamp(V−V_old,−clip,+clip)−ret)²).mean()`。 |
| `clip_param=0.2` | `clip_ratio=0.2` | 名称适配。 |
| `entropy_coef=0.01` | `entropy_coef=0.01` | 一致。 |
| `num_learning_epochs=5` | CLI 默认 `--ppo-epochs 5` | 一致。 |
| `num_mini_batches=4` | `num_minibatches=4` | `minibatch_size = num_envs×rollout_steps//4`。 |
| `learning_rate=1.0e-3` | `learning_rate=1e-3` | 一致。 |
| `schedule="adaptive"` | `adaptive` | minibatch KL 大于 `2×desired_kl` 时 lr 除以 1.5，下限 `1e-5`；KL 大于 0 且小于 `desired_kl/2` 时 lr 乘以 1.5，上限 `1e-2`。 |
| `gamma=0.99` | `gamma=0.99` | 一致。 |
| `lam=0.95` | `gae_lambda=0.95` | 名称适配。 |
| `desired_kl=0.01` | `desired_kl=0.01` | 一致。 |
| `max_grad_norm=1.0` | `max_grad_norm=1.0` | 一致。 |
| `num_steps_per_env=24` | `rollout_steps=24` | 名称适配。 |

Profile A 和 Profile B 的默认值不变：`value_coef=0.5`、`entropy_coef=0.003`、`clip_value_loss=False`、`schedule=fixed`、`learning_rate=1e-4`、`rollout_steps=32`、`minibatch_size=1024`、`ppo_epochs=5`。显式传入 CLI 参数 `--learning-rate`、`--value-coef`、`--entropy-coef`、`--clip-value-loss`、`--desired-kl`、`--clip-ratio`、`--max-grad-norm`、`--rollout-steps`、`--minibatch-size`、`--discount`、`--gae-lambda` 时，会覆盖 profile 默认值。

## 4. 命令与终止语义

### Unitree 原始环境

Unitree base 配置的命令范围为：

| 命令 | 范围 |
| --- | --- |
| `lin_vel_x` | `(-1.0, 2.0)` |
| `lin_vel_y` | `(-1.0, 1.0)` |
| `ang_vel_z` | `(-1.0, 1.0)` |

它启用 `heading_command=True`，`heading_control_stiffness=0.5`，命令重采样时间范围为 `(3.0, 8.0)`，`rel_standing_envs=0.05`，episode 时长为 20 秒。终止条件为 `time_out + bad_orientation`，其中 bad orientation 阈值为 70 度。

Unitree 的 `commands_vel` curriculum 为：

| 阶段 | lin_vel_x | lin_vel_y | ang_vel_z |
| --- | --- | --- | --- |
| stage 0，step 0 | `(-0.5, 1.0)` | `(-0.5, 0.5)` | `(-1.0, 1.0)` |
| stage 1，`5000×24` 步后 | `(-1.0, 2.0)` | `(-1.0, 1.0)` | `(-1.0, 1.0)` |

### H0W Profile C

Profile C 的命令范围为 `(-0.5, -0.5, -1.0)` 到 `(1.0, 0.5, 1.0)`，恰好等于 Unitree curriculum 的 stage 0 范围。它使用 `stand_probability=0.05`，episode 时长为 20 秒。

H0W 不实现 heading command，直接使用 yaw 速度指令；H0W 没有地形 curriculum；H0W 使用环境自带的终止逻辑，包括 `time_out` 等，没有 Unitree 专用的 bad orientation 70 度项。这些都是既有 H0W 语义，Profile C 没有改动。

## 5. 用法

### 冒烟验证

以下命令已于 2026-08-06 在物理 GPU 2 上验证。临时输出写入 `/tmp`，命令只使用允许的物理 GPU 0 到 3。

```bash
CUDA_VISIBLE_DEVICES=2 MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=2 uv run python -m humanoidverse.amp_stage2_piplus_22dof \
  --reward-profile c_unitree_velocity_h0w_compat \
  --bfm-model <bfm-model.safetensors> --decoder-path <FBcprAuxModel.onnx> \
  --motion-dataset <piplus_h0w_lafan_10s-clipped.pkl> \
  --smoke --work-dir /tmp/ufo_smoke_profile_c
```

结果生成 `checkpoint_1.pt` 和 `config.json`。metrics 为 `effective_learning_rate=0.001`，即 adaptive schedule 的起始学习率，`reward_mean=0.037`，`locomotion_reward_mean=0.0378`，`termination_rate=0.0`。

### 正式训练

正式训练使用 Profile A 的其他参数，仅将 `--reward-profile a_mimiclite_speed_safety` 替换为 `--reward-profile c_unitree_velocity_h0w_compat`。切换 profile 后，PPO 默认值也随之切换为 `rollout_steps=24`、`learning_rate=1e-3`、`schedule=adaptive`、启用 clipped value loss、4 个 minibatch。

正式运行输出应写入 `runs/<name>`，不得把正式 checkpoint 写入 `/tmp`。训练命令仍须显式限制在物理 GPU 0 到 3 内。

### dry-run 与 metadata

加入 `--dry-run` 可检查生效配置。输出应包含 `ppo`、`effective_reward`、`omitted_unitree_terms` 和 `reward_source`。

checkpoint 与 `config.json` 的 metadata 记录：

- `reward_source`，包括钉住的 Unitree commit；
- `reward_source_versions`；
- `omitted_unitree_terms`；
- `ppo`；
- `effective_reward`。

跨 profile resume 会检查 metadata 中的 `reward_profile`，不匹配时拒绝恢复，避免把不同奖励或 PPO 合同的 checkpoint 混用。

## 6. 验证记录

- 单元测试通过：`tests/test_amp_stage2_piplus_22dof.py` 共 12 项，覆盖 Profile C 公式、PPO 合同和 A/B 回归；`tests/test_speed_stage2.py` 共 3 项通过。
- `uv run ruff check` 通过。
- Profile C 冒烟训练已生成 `/tmp/ufo_smoke_profile_c/checkpoint_1.pt` 和 `config.json`，metrics 与上节记录一致。
- 已有 Profile A 单卡运行 `runs/amp_stage2_piplus_h0w/profile_a_1gpu_4096env_20260806` 不受影响，Profile C 为 opt-in 配置。
