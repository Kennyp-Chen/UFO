# TeCH 二阶段踢球训练

## 方案

一阶段 TeCH 模型保持完整冻结。二阶段只训练一个 command encoder：

```text
机器人 actor 观测 + 8 帧足球任务历史
  -> command encoder
  -> 256 维 raw latent
  -> 一阶段 checkpoint 的 project_z()
  -> 冻结的 TeCH actor
  -> 22 维 PiPlus H0W 动作
```

足球任务历史的每帧为 11 维：

```text
[目标球速度 base_xyz, 左脚标记, 右脚标记,
 球相对位置 base_xyz, 球相对速度 base_xyz]
```

球状态和足球指令不修改冻结 actor 的输入，只进入新的 encoder。当前 PiPlus H0W checkpoint 使用 256 维 hypersphere latent，二阶段输出必须经过 checkpoint 自带的 `project_z()`，投影后范数为 16。

## 环境与奖励

环境在现有 MJLab PiPlus H0W 模型中加入半径 0.07 m、质量 0.20 kg 的自由足球，并记录左右脚与球的接触。课程从正前方静止球、任意脚开始，逐步增加球距离、方位、初速度、目标方向以及指定左右脚。

任务奖励包括：

- 指定脚接近球的 potential progress。
- 第一次正确触球奖励，错误脚和重复触球惩罚。
- 正确触球后目标球速度跟踪。
- 踢球后的默认姿态恢复约束。

现有 BFM Zero 环境安全项仍作为 `env_reward` 加入总奖励，包括动作变化、非期望接触、脚部姿态、打滑、关节位置/速度和力矩限制。躯干等非法接触或机器人倾倒会终止 episode。

## 运行

本地完整一阶段模型默认使用：

```text
runs/remote_tech_piplus_2h0w_8gpu_seed4728/checkpoint/model/model.safetensors
```

单卡 smoke：

```bash
MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 \
uv run python -m humanoidverse.tech_kick_stage2 \
  --device cuda:0 \
  --num-envs 2 \
  --disable-domain-randomization \
  --work-dir /tmp/ufo_tech_kick_stage2_smoke \
  --smoke
```

正式多卡训练：

```bash
MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
uv run python -m humanoidverse.tech_kick_stage2 \
  --gpu-ids all \
  --num-envs 1024 \
  --iterations 30000 \
  --work-dir runs/tech_kick_stage2_piplus_h0w
```

`--num-envs` 是每张 GPU 的环境数。训练会保存 `config.json` 和 `checkpoint_<iteration>.pt`；恢复训练使用 `--resume` 指向二阶段 checkpoint。

验证最早保存的 encoder，并录制带物理球的场景：

```bash
MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 \
uv run python -m humanoidverse.tech_kick_stage2_play \
  --model-folder /tmp/ufo_tech_kick_stage2_smoke \
  --checkpoint-selection first \
  --kick-foot left \
  --target-ball-velocity 1.5 0.0 0.0 \
  --output validation_videos/tech_kick_checkpoint_1.mp4
```

`--checkpoint-selection` 支持 `first` 和 `latest`，也可用 `--checkpoint` 显式指定文件。

重点监控 `contact_rate`、`correct_touch_rate`、`has_valid_kick_rate`、`ball_velocity_error_rate`、`termination_rate` 和各项 `reward/kick/*`。如果长期没有触球，优先检查随机 latent 是否覆盖抬腿/踢腿基础行为，再考虑用足球动作 latent 对 encoder 做监督初始化。
