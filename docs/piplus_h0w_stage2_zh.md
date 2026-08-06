# PiPlus H0W 22DoF 二阶段微调

实验历史、当前运行快照和 observation contract 决策统一维护在 [`docs/experiment_ledger.md`](experiment_ledger.md)。本页保留操作说明和历史上下文，不复制完整台账。

本迁移将 HT_BFM 的 PiPlus H0W 22DoF 二阶段算法接入 UFO 的 MJLab 运行时，并复用冻结的 H0W ONNX 解码器。

## 算法入口

| 算法 | 训练入口 | 回放入口 | 目标 |
| --- | --- | --- | --- |
| 速度条件 PPO | `humanoidverse.speed_stage2` | `humanoidverse.speed_stage2_play` | 三维速度命令到 256 维潜变量 |
| AMP 二阶段 | `humanoidverse.amp_stage2_piplus_22dof` | `humanoidverse.amp_stage2_piplus_22dof_play` | Profile A 纯 locomotion，或 Profile B AMP 加潜变量先验 |

两个入口均检查 H0W 的固定合同：MJCF 22 动作、ONNX `616 -> 22`、编码器输入 363 维、潜变量 256 维。检查点包含稳定的任务名，速度 PPO 与 AMP 回放和恢复训练会拒绝对方的检查点。

## 模型资产与路径

运行 H0W 二阶段时需要准备与 checkpoint 匹配的本地冻结 ONNX 解码器及其配套文件。模型目录被 `.gitignore` 排除，不会提交到 Git：

```text
model/piplus_h0w_bfm/decoder/bfmzero-piplus-h0w-isaac-20260629_214205/
├── config.json
├── config.yaml
├── exported/FBcprAuxModel.onnx
└── tracking_inference/zs_8.pkl
```

训练仍需要用户单独提供真实可读的 `model.safetensors`，默认位置是
`model/piplus_h0w_bfm/model.safetensors`。它只用于启动时校验 BFM 的 22DoF actor contract，不能用 ONNX 文件替代；当前用户目录之间的软链接不是可移植依赖。

播放时，ONNX 解码器是实际动作计算主体，不要求 `model.safetensors` 存在。旧 checkpoint metadata 中失效的 `decoder_path` 会回退到上述本地解码器目录；显式通过 CLI 指定的路径仍会严格校验。

## GPU 约束

本服务器只允许使用物理 GPU 0-3。单卡运行时显式设置可见卡：

```bash
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl uv run python -m humanoidverse.speed_stage2 --device cuda:0 ...
```

多卡仅允许以 `CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun ... --gpu-ids all` 启动。入口会拒绝 `LOCAL_RANK >= 4`。

## Profile A：纯 locomotion

Profile A 名称为 `a_mimiclite_speed_safety`。它仅使用 MimicLite 的速度跟踪、站立、姿态和动作平滑项；AMP 判别器和潜变量先验权重被强制为零。

```bash
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl uv run python -m humanoidverse.amp_stage2_piplus_22dof \
  --reward-profile a_mimiclite_speed_safety --device cuda:0 \
  --bfm-model /path/to/model.safetensors \
  --decoder-path model/piplus_h0w_bfm/decoder/bfmzero-piplus-h0w-isaac-20260629_214205/exported/FBcprAuxModel.onnx \
  --latent-reference model/piplus_h0w_bfm/decoder/bfmzero-piplus-h0w-isaac-20260629_214205/tracking_inference/zs_8.pkl \
  --robot-config configs/robots/piplus_h0w.yaml \
  --motion-dataset /path/to/piplus_h0w_lafan_10s-clipped.pkl
```

## Profile B：AMP 与潜变量先验

Profile B 名称为 `b_amp_23dof_reference`。它使用 194 维 AMP 特征：根部线速度、5 个关键刚体的根坐标相对位置，以及 8 帧 22DoF 关节历史。它保留来源算法的判别器、梯度惩罚、二次 AMP 奖励、潜变量近邻先验与足端接触奖励。

该 profile 必须提供专用 H0W walking/run-with-stand expert 数据。当前 `HT_BFM` 源目录不含默认的 `piplus_h0w_locomotion_run_with_stand.pkl`，因此会在启动前以明确错误退出；不会降级到 Profile A 或错误地把普通 locomotion 数据当作 AMP expert。

```bash
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl uv run python -m humanoidverse.amp_stage2_piplus_22dof \
  --reward-profile b_amp_23dof_reference --device cuda:0 \
  --expert-dataset /path/to/piplus_h0w_locomotion_run_with_stand.pkl \
  --bfm-model /path/to/model.safetensors \
  --decoder-path model/piplus_h0w_bfm/decoder/bfmzero-piplus-h0w-isaac-20260629_214205/exported/FBcprAuxModel.onnx \
  --robot-config configs/robots/piplus_h0w.yaml \
  --motion-dataset /path/to/piplus_h0w_lafan_10s-clipped.pkl
```

## 有界回放

训练输出目录中的最新 `checkpoint_<iteration>.pt` 会自动被选择。回放不打开交互窗口，适合服务器回归验证。

```bash
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl uv run python -m humanoidverse.speed_stage2_play \
  --model-folder runs/speed_stage2_piplus_h0w --device cuda:0 --max-steps 250

CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl uv run python -m humanoidverse.amp_stage2_piplus_22dof_play \
  --model-folder runs/amp_stage2_piplus_h0w --device cuda:0 --max-steps 250
```

## 实验记录

### 2026-08-06：speed_stage2 checkpoint_9800 续训（HT_BFM IsaacSim → UFO_HT MJLab）

- 背景：HT_BFM IsaacSim 训练已产出 `checkpoint_9800.pt`（`/root/autodl-tmp/chenyupeng/HT_BFM/logs/speed_stage2_piplus_22dof/full_4gpu_isaac_cpu_lvp_1024env_gpu_isolated/`）。原 GPU 0 上的 IsaacSim 单卡续训实验（work-dir `single_gpu_reward_v3_4096env_resume9800_20260806`，PID 723711）已按用户要求停止。
- 续训入口：UFO_HT 的 `humanoidverse.speed_stage2`（MJLab 接口），单卡 GPU 0，`CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1`。
- **训练配置与 HT_BFM 原实验完全对齐，避免混淆**（原配置来自 `single_gpu_reward_v3_4096env_resume9800_20260806/config.json` 与 IsaacSim 命令行）：
  - `--num-envs 4096 --iterations 19800 --rollout-steps 32 --ppo-epochs 4 --minibatch-size 8192 --learning-rate 5e-5 --save-every 100 --seed 4728`
  - reward 为 `velocity_command_only_plus_optional_environment_reward`（speed_stage2 内置 speed_tracking_reward，与 config.json 一致）
  - `--bfm-model` 使用 HT_BFM 的 `model.safetensors`（符号链接指向 hejunfu 的 remote_tech_piplus_2h0w_8gpu_seed4728 模型，与 IsaacSim 侧一致）；`--decoder-path` 使用 HT_BFM 导出的 `FBcprAuxModel.onnx`；`--robot-config` 使用 UFO_HT 的 `configs/robots/piplus_h0w.yaml`（contract 校验通过：22 动作 / 29 nq / 28 nv / 616->22 ONNX / encoder 363 / z 256）。
- checkpoint 兼容性：`checkpoint_task_is_compatible` 确认 task=`speed_stage2_piplus_22dof`、amp=False，结构含 policy/optimizer/iteration/metadata，可直接恢复。
- 启动命令与产物：见 `runs/speed_stage2_piplus_h0w_resume_9800_to_19800/`（`config.json`、`continuation_9800_to_19800.log`），迭代目标 19800（绝对迭代数，`range(start_iteration, args.iterations)`）。
- 实测速度：启动 150 s 完成 28 迭代（9800→9828），约 **5.4 s/iter**（GPU 解码器已生效，`onnxruntime-gpu` 1.23.2）；GPU 0 利用率 ~78-80%，显存 ~36.7-39 GiB。
- 同期 GPU 占用：GPU 1 为 AMP Profile A 单卡续训（PID 753371，~6.4 s/iter）；GPU 4-7 被其他用户任务占用；GPU 2/3 空闲。

### 2026-08-06：onnxruntime-gpu 安装记录

- UFO_HT 环境此前**未安装** onnxruntime / onnxruntime-gpu，`OnnxPiPlusH0WDecoder` 一直使用 CPU `ReferenceEvaluator` fallback。
- `pyproject.toml` 新增 `onnxruntime-gpu>=1.20.0`，`uv lock` 解析到 1.23.2，`uv sync` 完成。
- 验证：`ort.get_available_providers()` 返回 `['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']`；decoder 实例化后 `get_providers()` 为 `['CUDAExecutionProvider', 'CPUExecutionProvider']`，CPU fallback 未启用。
- 效果（AMP 单卡）：CPU 解码器 ~360 s/iter → GPU 解码器 ~6.4-6.7 s/iter，提速约 56×；进程 CPU 从 ~800% 降至 ~112%。
- 速度对比完整记录见 `docs/amp_stage2_4gpu_vs_1gpu_speed.md`。

## Profile C：Unitree 速度奖励 H0W 适配

Profile C 名称为 `c_unitree_velocity_h0w_compat`。它是显式 opt-in 的 Unitree 速度奖励迁移配置，使用固定的 Unitree Robotics `unitreerobotics/unitree_rl_mjlab` commit `1425b15f73bd4095f0df53709d7c389c3eb9e790`，对应依赖版本为 `mjlab==1.2.0` 和 `rsl-rl-lib==5.0.1`。Profile C 将 Unitree G1 velocity task 的奖励项和 PPO 默认值适配到 H0W 22DoF，不改变 Profile A 或 Profile B 的默认值。

奖励方面，线速度跟踪、角速度跟踪、躯干姿态、躯干 xy 角速度、关节加速度、动作变化率、步态相位和站立奖励保持 Unitree 的公式与权重。终止、关节位置限制、脚部抬脚、脚部滑移和软着陆奖励保留权重，但使用 H0W 可用的 `done`、`dof_pos_limits`、body position、body velocity 和 `contact_forces` 输入。H0W 无法表达的 `angular_momentum`、`self_collisions` 和 G1 姿态正则不做近似，并写入 `omitted_unitree_terms` metadata。完整逐项矩阵见 [`docs/unitree_velocity_profile_zh.md`](unitree_velocity_profile_zh.md)。

PPO 默认值切换为 `value_coef=1.0`、`clip_value_loss=True`、`clip_ratio=0.2`、`entropy_coef=0.01`、`learning_rate=1e-3`、`schedule=adaptive`、`rollout_steps=24`、`num_minibatches=4`、`ppo_epochs=5`、`gamma=0.99`、`gae_lambda=0.95`、`desired_kl=0.01` 和 `max_grad_norm=1.0`。CLI 显式传入对应 PPO 参数时覆盖 profile 默认值。命令范围为 `(-0.5, -0.5, -1.0)` 到 `(1.0, 0.5, 1.0)`，等于 Unitree curriculum stage 0；H0W 直接使用 yaw 指令，不实现 heading command，也没有地形 curriculum。H0W 终止逻辑仍使用环境自带语义，没有 Unitree 专用的 bad orientation 70 度项。

冒烟命令如下，临时输出写入 `/tmp`，物理 GPU 仅使用 GPU 2：

```bash
CUDA_VISIBLE_DEVICES=2 MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=2 uv run python -m humanoidverse.amp_stage2_piplus_22dof \
  --reward-profile c_unitree_velocity_h0w_compat \
  --bfm-model <bfm-model.safetensors> --decoder-path <FBcprAuxModel.onnx> \
  --motion-dataset <piplus_h0w_lafan_10s-clipped.pkl> \
  --smoke --work-dir /tmp/ufo_smoke_profile_c
```

该命令已于 2026-08-06 验证，生成 `checkpoint_1.pt` 和 `config.json`；metrics 包含 `effective_learning_rate=0.001`、`reward_mean=0.037`、`locomotion_reward_mean=0.0378` 和 `termination_rate=0.0`。加入 `--dry-run` 可查看 `ppo`、`effective_reward`、`omitted_unitree_terms` 和 `reward_source`。checkpoint 与 `config.json` metadata 记录固定来源、版本、PPO 和生效奖励配置，跨 profile resume 会因 `reward_profile` 不匹配而拒绝。
