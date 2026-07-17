# BFM-Zero 低显存训练配置参考

> 本文档分析 BFM-Zero（UFO 的前身）中的低显存训练方案，包含 29-DOF 和 23-DOF 两个版本，
> 以及 23-DOF 数据转换流程，供 UFO 适配低显存训练时参考。

---

## 目录

1. [背景](#1-背景)
2. [参数对比总表](#2-参数对比总表)
3. [各参数详解](#3-各参数详解)
4. [23-DOF 方案](#4-23-dof-方案)
5. [UFO 适配指南](#5-ufo-适配指南)
6. [附录：完整参数快照](#6-附录完整参数快照)

---

## 1. 背景

BFM-Zero 有 4 个训练入口，形成从**全量→低显存→极低显存**的梯度：

| 文件 | DOF | 后端 | 显存目标 | 显存节省 |
|------|:---:|:----:|:--------:|:--------:|
| `train.py` | 29 | Isaac Sim | 高显存（A100-80GB） | — |
| `train_low.py` | 29 | Isaac Sim | 中低显存（RTX 4090 ~24GB） | ≈60-70% |
| `train_low_23dof.py` | 23 | Isaac Sim | 低显存（RTX 4090） | ≈70-80% |

UFO 使用 MJLab 后端，参数通过 CLI 传递（非 hard-coded `TrainConfig`），但**优化思路完全一致**。

> ⚠️ BFM-Zero 使用 Isaac Sim 后端（`HumanoidVerseIsaacConfig`），UFO 使用 MJLab 后端（`HumanoidVerseMjlabConfig`），
> 因此配置不能直接复制粘贴。但所有网络架构参数（`z_dim`、`hidden_dim`、`hidden_layers` 等）
> 和 scaling 参数（`num_envs`、`buffer_size` 等）的概念与效果完全一致。

---

## 2. 参数对比总表

### 2.1 高影响参数（显存大头）

| 参数 | 基线 (train.py) | train_low.py (29-DOF) | train_low_23dof.py | 影响说明 |
|------|:---------------:|:--------------------:|:------------------:|:--------:|
| `online_parallel_envs` | 1024 | **512** (-50%) | 1024 (不变) | 并行环境→每步 batch → 显存/GPU |
| `buffer_size` | 5,120,000 | **2,560,000** (-50%) | **2,560,000** (-50%) | 回放缓冲→显存占用 |
| `compile` | `True` | **`False`** | **`False`** | torch.compile 显存+编译时间 |
| `cudagraphs` | `False` | **`False`** | **`False`** | CUDA Graph 显存 |

> **关键发现**：29-DOF 低显存版同时降低了 `online_parallel_envs` 和 `buffer_size`；但 23-DOF 版
> 因为 DOF 减少使得每个 env 的显存下降，所以保留了 1024 的 `online_parallel_envs`。

### 2.2 中等影响参数（网络架构）

| 参数 | 基线 (train.py) | train_low.py | train_low_23dof.py |
|------|:---------------:|:------------:|:------------------:|
| `z_dim` | 256 | **128** (-50%) | 256 (不变) |
| `hidden_dim` (f/actor/critic/aux_critic) | 2048 | **1024** (-50%) | 2048 (不变) |
| `hidden_dim` (backward) | 256 | **128** (-50%) | 256 (不变) |
| `hidden_dim` (discriminator) | 1024 | **512** (-50%) | 1024 (不变) |
| `hidden_layers` (f/actor/critic/aux_critic) | 6 | **4** (-33%) | **4** (-33%) |
| `hidden_layers` (discriminator) | 3 | 2 (-33%) | 2 (-33%) |

> **关键发现**：23-DOF 版保留了 29-DOF 全量网络大小（`z_dim=256`, `hidden_dim=2048`），
> 因为 DOF 减少本身降低了输入/输出层大小，且姿态空间更小使得训练更易收敛，不需要缩小网络容量。

### 2.3 低影响参数

| 参数 | 基线 (train.py) | train_low.py | train_low_23dof.py |
|------|:---------------:|:------------:|:------------------:|
| `batch_size` | 1024 | **504** | **512** |
| `inference_batch_size` | 500,000 | **256,000** | **256,000** |
| `seq_length` | 8 | **6** | 8 (不变) |
| `num_parallel` | 2 | 2 (不变) | 2 (不变) |

> `batch_size=504` 的设计原因：504 ÷ 6 (seq_length) = 84，可以整除；若使用 512 ÷ 6 ≈ 85.33 则不能整除。
> 23-DOF 版 `seq_length=8` 不变，所以 `batch_size=512`（512 ÷ 8 = 64，整除）。

### 2.4 训练调度参数

| 参数 | 基线 (train.py) | train_low.py | train_low_23dof.py |
|------|:---------------:|:------------:|:------------------:|
| `num_env_steps` | 384,000,000 | 384,000,000 | 384,000,000 |
| `num_agent_updates` | 16 | 16 | 16 |
| `checkpoint_every_steps` | 9,600,000 | 9,600,000 | **19,200,000** |
| `eval_every_steps` | 9,600,000 | 9,600,000 | **9,600,000** |
| `update_agent_every` | 1024 | 1024 | 1024 |
| `num_seed_steps` | 10240 | 10240 | 10240 |
| `log_every_updates` | 384,000 | 384,000 | 384,000 |

---

## 3. 各参数详解

### 3.1 `online_parallel_envs`: 并行环境数

```
基线: 1024 → train_low: 512
```

- 这是**显存最大头**。每个并行环境持有自己的仿真状态、观测张量、动作张量
- 减少 50% → 仿真相关显存（simulation state, rendering buffers, observation tensors）约减 50%
- 副作用：每步收集的 transition 减半，需要更多步数填满 buffer
- 在 `TrainConfig` 中通过 `capacity = buffer_size // online_parallel_envs` 计算 trajectory buffer 容量
- **UFO 对应 CLI 参数**：`--num-envs`

### 3.2 `buffer_size`: 回放缓冲大小

```
基线: 5120000 → low: 2560000
```

- replay buffer 存储在 GPU（`buffer_device='cuda'`），直接占用显存
- 每 512 envs 并行时，每小时产生约 512 × 3600 × FPS ≈ 92M 帧（29-DOF），5M buffer 约覆盖 3 分钟的训练
- 减少 50% → 缓冲显存减半，但采样多样性也降低
- **UFO 对应 CLI 参数**：`--buffer-size`

### 3.3 `z_dim`: 隐变量维度

```
基线: 256 → train_low: 128
```

- `z` 是 BFM-Zero/FB 的核心隐变量，参与 forward/backward/actor/critic/discriminator 所有计算
- 减少 50% 意味着：
  - 所有与 `z` 拼接的层输入维度降低
  - 模型参数量减少（特别是 backward 和 forward 的首层）
- 可能影响表现力（representation capacity），但 128 维在 29-DOF 上已验证可收敛

### 3.4 `hidden_dim` 与 `hidden_layers`: 网络宽度与深度

```
hidden_dim baseline: 2048 → low: 1024
hidden_layers baseline: 6 → low: 4
```

- 对 f（forward）、actor、critic、aux_critic 四个网络都减半宽度
- 参数量约减少：`(1024*1024)/(2048*2048) ≈ 25%` 每层 × 层数变化
- 加上层数减少（6→4），总参数量约为基线的 **15-20%**
- 1048→1024 在 Transformer 时代不算小，RL 任务通常不需要超大网络

### 3.5 `compile` 与 `cudagraphs`

```
compile: True → False
cudagraphs: False → False (不变)
```

- `torch.compile` 需要额外显存存储编译后的图和中间缓冲区
- 禁用 compile 可以减少约 1-2GB 显存，且避免 Triton kernel 编译时间
- `cudagraphs` 在低显存场景同样禁用

### 3.6 `inference_batch_size`

```
基线: 500000 → low: 256000
```

- 用于 backward inference（从 expert 数据或 rollout 数据中计算 z）
- 降低后减少了显存峰值，但增加了 inference 循环次数
- 对最终模型质量无影响（纯计算参数）

---

## 4. 23-DOF 方案

### 4.1 背景

23-DOF 方案通过**减少机器人自由度**来降低显存和计算需求。相比单纯缩小网络（train_low.py），
减少 DOF 的好处是**所有相关层都受益**（输入层、输出层、所有中间层与关节数相关的部分）。

### 4.2 移除的 6 个关节

| 索引（29-DOF） | 关节名称 | 所属部位 |
|:--------------:|:---------|:--------:|
| 13 | `waist_roll_joint` | 腰部滚转 |
| 14 | `waist_pitch_joint` | 腰部俯仰 |
| 20 | `left_wrist_pitch_joint` | 左手腕俯仰 |
| 21 | `left_wrist_yaw_joint` | 左手腕偏航 |
| 27 | `right_wrist_pitch_joint` | 右手腕俯仰 |
| 28 | `right_wrist_yaw_joint` | 右手腕偏航 |

### 4.3 保留的 23 个关节

```
左腿 (6):  hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll
右腿 (6):  hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll
腰部 (1):  waist_yaw
左臂 (5):  shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist_roll
右臂 (5):  shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist_roll
```

### 4.4 所需改动清单

要将一个 29-DOF 训练方案改为 23-DOF，需要以下所有改动：

**① 数据转换**
- 用 `convert_pkl_23dof.py` 将 29-DOF pickle 转成 23-DOF
  - `dof`: (frames, 29) → (frames, 23)
  - `pose_aa`: (frames, 30, 3) → (frames, 24, 3)
  - 其他字段不变
- 生成文件：`lafan_23dof.pkl` + `lafan_23dof_10s-clipped.pkl`

**② Hydra 机器人配置**
- `config/robot/g1/g1_23dof_hard_waist.yaml` — 23-DOF 的 Hydra 配置
- DOF 列表、PD 增益、contact/termination 语义全部按 23-DOF 调整

**③ MuJoCo XML**
- `data/robots/g1/g1_23dof.xml` — 移除了 6 个关节的 MuJoCo 模型
- 配套场景文件：`scene_23dof_freebase_*.xml`

**④ 训练配置**
- `lafan_tail_path` → 指向 23-DOF 数据
- `hydra_overrides: robot=g1/g1_23dof_hard_waist`
- 网络架构可选缩小（23-DOF 版保留了 2048 宽度，因为 DOF 减少已经降低了显存）

**⑤ 推理导出**
- `export_models_23dof.py` — 支持 23-DOF 的 tracking/goal/reward inference 导出
- 包含 `convert_29dof_to_23dof()` 函数，可在推理时动态转换 29→23 DOF

### 4.5 23-DOF 的权衡

| 优势 | 劣势 |
|:-----|:-----|
| 显存再降 10-15%（相比 train_low） | 无手腕俯仰/偏航，手部动作受限 |
| 动作空间从 29→23，探索更高效 | 无腰部滚转/俯仰，躯干灵活度降低 |
| 训练速度提升 ≈20% | 部分需要精细手腕控制的技能无法学习 |
| 数据需求量减少 | 与 29-DOF checkpoint 不兼容 |

---

## 5. UFO 适配指南

### 5.1 差异分析

UFO 和 BFM-Zero 在低显存场景的主要差异：

| 维度 | BFM-Zero | UFO |
|:-----|:---------|:---:|
| 训练后端 | Isaac Sim | MJLab |
| 参数传递 | Hard-coded `TrainConfig`（Python） | CLI args + `build_ufo_mjlab_config()` |
| 启动方式 | `python train_low.py` | `./run_train.sh ...` |
| 网络参数暴露 | 直接在 config 对象中设置 | 通过 agent preset 内部配置 |
| compile 控制 | `TrainConfig.compile` 字段 | `--disable-compile`（已添加） |

### 5.2 改造方案

在 UFO 中实现低显存模式有两种思路：

#### 方案 A：CLI 预设参数（推荐）

在 `run_train.sh` 或新增 `run_train_low.sh` 中提供低显存预设：

```bash
# run_train_low.sh 核心参数
./run_train.sh \
  --num-envs 512 \          # 基线 1024 → 512
  --buffer-size 2560000 \   # 基线 5120000 → 2560000
  --disable-compile \       # 禁用 torch.compile
  ...（其他参数同 normal）
```

**优点**：零代码改动，立即可用
**缺点**：不能调整 `z_dim`、`hidden_dim` 等网络架构参数（这些在 agent preset 内部）

#### 方案 B：新增 `--low-memory` 预设（推荐）

在 `train.py` 中新增预设标志，自动覆盖多组参数：

| CLI 新增 | 自动覆盖 |
|:---------|:---------|
| `--low-memory` | `--num-envs 512 --buffer-size 2560000 --disable-compile` |

在 `build_ufo_mjlab_config()` 或 preset 构建函数中，根据 low-memory 标志缩小网络：

```python
if low_memory:
    cfg.agent.model.archi.z_dim = 128
    cfg.agent.model.archi.f.hidden_dim = 1024
    cfg.agent.model.archi.f.hidden_layers = 4
    cfg.agent.model.archi.actor.hidden_dim = 1024
    cfg.agent.model.archi.actor.hidden_layers = 4
    cfg.agent.model.archi.critic.hidden_dim = 1024
    cfg.agent.model.archi.critic.hidden_layers = 4
```

**优点**：全面控制所有参数
**缺点**：需要改动 `train.py` 和 preset 代码

#### 方案 C：23-DOF 极低显存

在 UFO 中实现 23-DOF 需要：
1. 创建 `g1_23dof.xml`（从 29-DOF XML 移除 6 个关节）
2. 创建 `configs/robots/g1_23dof.yaml`（用户层 RobotTrainingSpec）
3. 创建 Hydra 配置（参考 BFM-Zero 的 `g1_23dof_hard_waist.yaml`）
4. 转换运动数据（参考 `convert_pkl_23dof.py`）
5. 在 `run_train.sh` 中指定 `--robot-config configs/robots/g1_23dof.yaml`

### 5.3 参数对照表（BFM-Zero → UFO）

| BFM-Zero 参数 | UFO CLI 参数 | 备注 |
|:-------------|:-------------|:-----|
| `online_parallel_envs` | `--num-envs` | 每 GPU 值 |
| `buffer_size` | `--buffer-size` | 每 GPU 值 |
| `compile` | `--disable-compile` | 已实现 |
| `z_dim` | 需代码改动 | 在 preset/model config 中 |
| `hidden_dim` | 需代码改动 | 在 preset/model config 中 |
| `hidden_layers` | 需代码改动 | 在 preset/model config 中 |
| `batch_size` | 需代码改动 | 在 preset train config 中 |
| `inference_batch_size` | 需代码改动 | 在 model config 中 |
| `seq_length` | 需代码改动 | 在 model config 中 |
| `num_seed_steps` | 需代码改动 | 在 preset 中 |
| `num_agent_updates` | 需代码改动 | 在 preset 中 |
| `lr_*` | 需代码改动 | 学习率通常无需降低 |
| `num_env_steps` | `--num-env-steps` | 全局值 |

### 5.4 显存估算参考

| 配置 | `online_parallel_envs` | 网络大小 | 预计显存 | 目标 GPU |
|:----|:---------------------:|:--------:|:--------:|:--------:|
| 全量 29-DOF | 1024 | 2048×6, z=256 | ≈40-48GB | A100-80GB |
| 低显存 29-DOF | 512 | 1024×4, z=128 | ≈16-20GB | RTX 4090-24GB |
| 23-DOF 标准 | 1024 | 2048×4, z=256 | ≈16-20GB | RTX 4090-24GB |
| 23-DOF 低显存 | 512 | 1024×4, z=128 | ≈10-14GB | RTX 4080-16GB |

---

## 6. 附录：完整参数快照

### 6.1 `train_low.py`（29-DOF 低显存）关键配置

```python
# === 训练规模 ===
online_parallel_envs = 512       # 基线 1024
num_env_steps = 384_000_000      # 不变
buffer_size = 2_560_000          # 基线 5_120_000

# === 网络架构 ===
z_dim = 128                      # 基线 256
f/actor/critic hidden_dim = 1024  # 基线 2048, layers=4 (基线 6)
backward hidden_dim = 128        # 基线 256
discriminator hidden_dim = 512   # 基线 1024, layers=2 (基线 3)

# === 训练参数 ===
batch_size = 504                 # 基线 1024 (504 ÷ 6 = 84 ← 整除)
inference_batch_size = 256_000   # 基线 500_000
seq_length = 6                   # 基线 8

# === 编译 ===
compile = False                  # 基线 True
cudagraphs = False               # 不变

# === 环境 ===
robot = g1_29dof_hard_waist
lafan_tail_path = lafan_29dof_10s-clipped.pkl
```

### 6.2 `train_low_23dof.py`（23-DOF 低显存）关键配置

```python
# === 训练规模 ===
online_parallel_envs = 1024      # 不变（DOF 减少已降显存）
buffer_size = 2_560_000          # 减半

# === 网络架构 ===
z_dim = 256                      # 不变
f/actor/critic hidden_dim = 2048 # 不变, layers=4 (基线 6)
backward hidden_dim = 256        # 不变
discriminator hidden_dim = 1024  # 不变, layers=2 (基线 3)

# === 训练参数 ===
batch_size = 512                 # 基线 1024 (512 ÷ 8 = 64 ← 整除)
inference_batch_size = 256_000   # 减半
seq_length = 8                   # 不变

# === 编译 ===
compile = False
cudagraphs = False

# === 环境 ===
robot = g1_23dof_hard_waist
lafan_tail_path = lafan_23dof_10s-clipped.pkl
```

### 6.3 23-DOF 关节索引映射

```python
# 29-DOF 关节顺序
DOF_NAMES_29 = [
    # 左腿 (0-5)
    'left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint',
    'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint',
    # 右腿 (6-11)
    'right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint',
    'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint',
    # 腰部 (12-14)  ← 仅保留 12: waist_yaw
    'waist_yaw_joint', 'waist_roll_joint ✗', 'waist_pitch_joint ✗',
    # 左臂 (15-21)  ← 保留 15-19, 移除 20-21
    'left_shoulder_pitch_joint', 'left_shoulder_roll_joint',
    'left_shoulder_yaw_joint', 'left_elbow_joint', 'left_wrist_roll_joint',
    'left_wrist_pitch_joint ✗', 'left_wrist_yaw_joint ✗',
    # 右臂 (22-28)  ← 保留 22-26, 移除 27-28
    'right_shoulder_pitch_joint', 'right_shoulder_roll_joint',
    'right_shoulder_yaw_joint', 'right_elbow_joint', 'right_wrist_roll_joint',
    'right_wrist_pitch_joint ✗', 'right_wrist_yaw_joint ✗',
]

# 移除的索引: [13, 14, 20, 21, 27, 28]
```

### 6.4 Pkl 数据字段结构

```
29-DOF:               23-DOF:
  dof: (T, 29)    →    dof: (T, 23)         ← 移除 6 列
  pose_aa: (T, 30, 3)  pose_aa: (T, 24, 3)  ← 移除 6 行
  root_trans_offset     (不变) (T, 3)
  root_rot              (不变) (T, 4)
  smpl_joints           (不变) (T, 24, 3)
  fps                   (不变) 30
  motion_name           (不变)
```

---

> **对应文件位置**：
> - BFM-Zero 仓库：`/home/hero/Projects/Robotics/RL/BFM-Zero/`
> - `train_low.py`: `humanoidverse/train_low.py`（29-DOF 低显存）
> - `train_low_23dof.py`: `humanoidverse/train_low_23dof.py`（23-DOF 低显存）
> - `23DOF_CONVERSION_SUMMARY.md`: `23DOF_CONVERSION_SUMMARY.md`
> - `convert_pkl_23dof.py`: `convert_pkl_23dof.py`（数据转换脚本）
> - 23-DOF Hydra 配置: `humanoidverse/config/robot/g1/g1_23dof_hard_waist.yaml`
> - 23-DOF MuJoCo XML: `humanoidverse/data/robots/g1/g1_23dof.xml`
