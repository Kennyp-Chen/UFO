# UFO 环境安装与训练指南

## 目录

1. [环境安装](#一环境安装)
2. [RTX 5090 (Blackwell) 特殊说明](#二rtx-5090-blackwell-特殊说明)
3. [训练框架概述](#三训练框架概述)
4. [训练命令详解](#四训练命令详解)
5. [训练后：推理与导出](#五训练后推理与导出)
6. [参数参考](#六参数参考)
7. [常见问题](#七常见问题)

---

## 一、环境安装

### 前置条件

- **Python 3.10**（严格绑定，`requires-python = "==3.10.*"`）
- **NVIDIA GPU + CUDA**（训练必须 GPU，推荐 8 卡）
- **系统中已安装 MuJoCo**（mjlab 依赖会自行处理）

### 1. 克隆仓库

```bash
git clone https://github.com/Roboparty/UFO.git
cd UFO
```

### 2. 安装 uv（Python 包管理器）

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.local/bin/env
```

或通过 pip 安装：

```bash
python -m pip install --user uv
export PATH="$HOME/.local/bin:$PATH"
```

### 3. 安装依赖

```bash
uv sync
```

这会创建 `.venv/` 虚拟环境，并按 `pyproject.toml` 安装所有依赖：

| 依赖 | 版本 | 用途 |
|------|------|------|
| torch | 2.7.x | 深度学习框架 |
| mujoco | 3.8.x | 物理引擎 |
| mjlab | 1.4.0 | MuJoCo 批处理仿真 |
| hydra-core | 1.2.x | 配置管理 |
| wandb | 0.26.x | 实验日志 |
| gymnasium | 0.29.x | RL 环境接口 |
| omegaconf | — | YAML 配置解析 |
| torchrunx | 0.3.x | 分布式启动 |

> **注意**：`pyproject.toml` 配置了 `[[tool.uv.index]] url = "https://pypi.nvidia.com/"` 用于 CUDA 相关包。如果无法访问，可配置镜像或通过环境变量覆盖。

### 4. 下载 G1 运动数据

大容量运动数据托管在 HuggingFace，不放在 Git 仓库中：

```bash
bash scripts/download_data.sh g1_lafan
```

下载完成后验证：

```bash
ls -lh humanoidverse/data/lafan_29dof_10s-clipped.pkl
ls -lh humanoidverse/data/lafan_29dof.pkl
```

| 文件 | 用途 |
|------|------|
| `lafan_29dof_10s-clipped.pkl` | 训练用——裁剪为 10 秒片段的 LaFAN 运动数据 |
| `lafan_29dof.pkl` | 推理用——完整运动序列 |

两个文件的 SHA256 校验由 `scripts/download_data.sh` 自动完成。

### 5. 配置 W&B（可选）

```bash
uv run wandb login
# 或设置环境变量
export WANDB_API_KEY=your_wandb_api_key
```

如果不使用 W&B，所有训练命令去掉 `--use-wandb` 即可。

---

## 二、RTX 5090 (Blackwell) 特殊说明

RTX 5090 基于 NVIDIA Blackwell 架构（计算能力 sm_120），当前 PyTorch 和 Triton 的正式版尚未完全支持。以下是在 5090 上运行 UFO 需要的特殊操作。

### 问题一：PyTorch 版本不支持 sm_120

**现象**：`RuntimeError: CUDA error: no kernel image is available for execution on the device`

**原因**：PyTorch 官方发布的 2.7.x 预编译包仅支持到 sm_90（Ada Lovelace / Hopper），不包含 Blackwell 的 CUDA kernel。

**解决**：安装 PyTorch nightly 版本（包含 sm_120 支持）。

```bash
uv pip install --pre --force-reinstall torch --index-url https://download.pytorch.org/whl/nightly/cu128
```

安装后验证：

```bash
python -c "import torch; print(torch.cuda.get_arch_list())"
```

输出应包含 `sm_120`。

### 问题二：pyproject.toml 版本约束冲突

**现象**：执行 `./run_train.sh` 时 uv 重新下载 torch。

**原因**：`pyproject.toml` 中写死了 `torch>=2.7.0,<2.8.0`，而 nightly 版本为 2.12.0.dev，不在约束范围内。`uv run` 每次会按约束重新解析依赖。

**解决（二选一）**：

**方案 A（临时，推荐）**：用 `.venv/bin/python` 直接运行，跳过 `uv run` 的依赖检查。

```bash
.venv/bin/python -m humanoidverse.train [参数...]
```

**方案 B（一劳永逸）**：放开 torch 版本约束。

编辑 `pyproject.toml`：

```toml
"torch>=2.7.0",   # 去掉 <2.8.0
```

然后重新锁依赖：

```bash
uv lock
```

之后 `./run_train.sh` 也可正常使用。

### 问题三：Triton 不支持 Blackwell，torch.compile 无法使用

**现象**：`torch._inductor.exc.InductorError: RuntimeError: CUDA driver error: device kernel image is invalid`

**原因**：UFO 代码中硬编码了多处 `torch.compile()` 调用（`trajectory.py`、`fb/agent.py` 等），Triton 3.7.0 尚未提供 Blackwell (sm_120) 的 kernel，编译时崩溃。

**解决**：在所有训练命令中加入 `--disable-compile` 参数（已在 `train.py` 中内置支持）。

```bash
.venv/bin/python -m humanoidverse.train \
  --agent fb \
  --data-manifest configs/data/example_mix.yaml \
  --gpu-ids single \
  --smoke \
  --disable-compile \
  --work-dir /tmp/ufo_smoke_g1
```

`--disable-compile` 会做三件事：
1. 设置 `TORCHDYNAMO_DISABLE=1` 环境变量
2. 设置 `torch._dynamo.config.disable = True`，让已编译的函数直接透传
3. 将 `torch.compile` 替换为恒等函数，后续所有新的编译调用都变成空操作

> **注意**：这会影响训练性能（eager mode 比 compiled mode 慢），但在 Triton 支持 Blackwell 之前这是必要的 workaround。RTX 5090 的纯算力优势可以在一定程度上弥补。后续 Triton 版本支持 sm_120 后即可去掉 `--disable-compile`。

### 总结：RTX 5090 训练完整命令

```bash
# 安装 nightly PyTorch（仅需一次）
uv pip install --pre --force-reinstall torch --index-url https://download.pytorch.org/whl/nightly/cu128

# 运行训练（用 .venv 直接执行，加 --disable-compile）
.venv/bin/python -m humanoidverse.train \
  --agent fb \
  --data-path humanoidverse/data/lafan_29dof_10s-clipped.pkl \
  --gpu-ids single \
  --num-envs 1024 \
  --num-env-steps 192000000 \
  --disable-compile \
  --work-dir runs/ufo_fb_g1_5090
```

---

## 三、训练框架概述

### 训练目标

UFO 的训练目标是通过**无监督强化学习**，让人形机器人学会跟踪给定的运动数据（motion tracking），从而掌握自然、多样、稳健的运动技能。

### 核心原理

```
┌─────────────────────────────────────────────────────────┐
│  训练循环                                                    │
│                                                             │
│  运动数据 (LaFAN)                                            │
│       │                                                      │
│       ▼                                                      │
│  MotionLib 采样 ─── 每 episode 从数据集中随机抽一段运动       │
│       │                                                      │
│       ▼                                                      │
│  MJLab 仿真环境 (MuJoCo 批处理)                                │
│  │  · 机器人目标状态 = 采样的运动帧                            │
│  │  · Agent 输出动作 → 物理仿真 → 新状态                      │
│  │  · 奖励 = 跟踪精度 + 辅助正则                              │
│       │                                                      │
│       ▼                                                      │
│  Agent 更新                                                   │
│  ├─ Forward 模型: 预测下一状态                                │
│  ├─ Backward 模型: 从轨迹推断潜在变量 z                       │
│  ├─ Actor: 基于 state + z 输出动作                            │
│  ├─ Critic: 评估状态价值                                      │
│  └─ 奖励: 跟踪奖励 + 辅助惩罚(避免不自然接触/姿态)             │
└─────────────────────────────────────────────────────────┘
```

**关键设计**：通过潜在变量 `z`（由 Backward 模型从运动轨迹中编码）引导 Actor 输出合适的动作，实现多样化的运动行为。

### 两个训练预设

| 预设 | CLI 名称 | 更新 z 间隔 | 特点 |
|------|----------|:----------:|------|
| **FB** | `--agent fb` | 每 100 步 | 默认预设，训练更稳定，适合通用运动跟踪 |
| **TeCH** | `--agent tech` | 每 10 步 | 更频繁更新潜在编码，对动态运动响应更快 |

> 历史说明：TeCH 在早期 UFO 版本中称为 TLDR。`--agent tldr` 保留为兼容别名，不推荐新使用。

### 训练参数语义

理解这些参数的**范围**是正确设置训练的关键：

| 参数 | 默认值 | 范围 | 说明 |
|------|--------|------|------|
| `--num-envs` | 1024 | **每 GPU** | 该 GPU 上并行仿真多少个环境 |
| `--num-env-steps` | 192M | **全局** | 所有 GPU 合计的总环境步数预算 |
| `--buffer-size` | 5.12M | **每 GPU** | 每个 GPU 上的经验回放缓冲区大小 |
| `--update-z-every-step` | 100(FB) / 10(TeCH) | — | 潜在编码 z 的更新间隔（步数） |

例如 8 GPU × 1024 envs = 8192 并行环境。

---

## 四、训练命令详解

### 场景 1：冒烟测试

**目标**：用极小规模快速验证环境安装、数据加载和训练循环是否正常。不产出有用策略，仅用于验证配置。

```bash
./run_train.sh \
  --agent fb \
  --data-manifest configs/data/example_mix.yaml \
  --gpu-ids single \
  --smoke \
  --work-dir /tmp/ufo_smoke_g1
```

`--smoke` 自动生效的设置：
- `num_envs` ≤ 16
- `num_env_steps` ≤ 2048
- 禁用 W&B
- 运行时间约 1-2 分钟

期望输出：训练日志正常打印，无报错退出。

---

### 场景 2：G1 通用运动跟踪（FB 预设）

**目标**：训练 Unitree G1 机器人掌握 LaFAN 数据集中的多样化运动能力，包括行走、跑步、跳跃等基本运动技能。这是 UFO 推荐的主力训练路径，经过最充分的测试。

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
./run_train.sh \
  --agent fb \
  --gpu-ids all \
  --num-envs 1024 \
  --num-env-steps 192000000 \
  --work-dir runs/ufo_fb_g1 \
  --data-path humanoidverse/data/lafan_29dof_10s-clipped.pkl \
  --update-z-every-step 100 \
  --buffer-size 5120000 \
  --use-wandb \
  --wandb-run-name ufo_fb_g1
```

| 参数 | 值 | 说明 |
|------|-----|------|
| 数据 | LaFAN 29 DOF | 通用运动数据集，10 秒裁剪片段 |
| 并行度 | 8 GPU × 1024 envs = 8192 | 高并行度加速训练 |
| 总步数 | 192,000,000 | 全局环境步数预算 |
| 保存间隔 | 每 3,200,000 步 | checkpoint 写入 `runs/ufo_fb_g1/` |
| 预计时间 | 8×A100 约 12-24 小时 | 取决于 GPU 型号和系统负载 |

---

### 场景 3：G1 通用运动跟踪（TeCH 预设）

**目标**：相比 FB，TeCH 以更高频率更新潜在编码 z（每 10 步 vs 每 100 步），对快速变化、动态性强的运动（如急停、转身、跳跃）响应更快。适合需要更敏捷运动控制的场景。

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
./run_train.sh \
  --agent tech \
  --gpu-ids all \
  --num-envs 1024 \
  --num-env-steps 192000000 \
  --work-dir runs/ufo_tech_g1 \
  --data-path humanoidverse/data/lafan_29dof_10s-clipped.pkl \
  --update-z-every-step 10 \
  --buffer-size 5120000 \
  --use-wandb \
  --wandb-run-name ufo_tech_g1
```

**FB 与 TeCH 的关键区别：**

| 对比项 | FB | TeCH |
|--------|:--:|:----:|
| `update_z_every_step` | 100 | 10 |
| `num_agent_updates` | 16 | 128 |
| 模型架构 | FBcprAux | GcrRlDistAux (TldrDistAux) |
| `--lr-scale` | 生效 | 忽略 |
| `--cartwheel-aux-safe` | 支持 | 不支持 |
| 适用场景 | 通用稳定训练 | 高频动态运动 |

---

### 场景 4：多数据源混合训练

**目标**：在基础运动分布中注入稀有高敏捷技能（如后空翻），通过加权数据源混合实现。适合需要"基础能力 + 特技"组合的场景。

准备数据 manifest 文件 `configs/data/lafan_cartwheel_mix.yaml`：

```yaml
datasets:
  - name: lafan
    format: ufo_pkl
    train_path: humanoidverse/data/lafan_29dof_10s-clipped.pkl
    weight: 0.95        # 95% 基础运动（行走/跑步/跳跃）

  - name: cartwheel
    format: ufo_pkl
    train_path: humanoidverse/data/soma_cartwheel_near10s.pkl
    weight: 0.05        # 5% 后空翻特技
```

启动训练：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
./run_train.sh \
  --agent fb \
  --gpu-ids all \
  --data-manifest configs/data/lafan_cartwheel_mix.yaml \
  --work-dir runs/ufo_fb_g1_cartwheel \
  --use-wandb \
  --wandb-run-name ufo_fb_g1_cartwheel
```

可选：加入 `--cartwheel-aux-safe` 启用安全辅助奖励集（仅 FB 支持）：

```bash
  --cartwheel-aux-safe
```

该选项会移除 locomotion 相关的 contact/foot-shape 惩罚，降低动作率惩罚系数，让后空翻训练更不容易因触发惩罚而失败。

**命令行多数据源方式（不写 manifest 文件）：**

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
./run_train.sh \
  --agent fb \
  --data-path \
    humanoidverse/data/lafan_29dof_10s-clipped.pkl \
    humanoidverse/data/soma_cartwheel_near10s.pkl \
  --data-mix-weights 0.95 0.05 \
  --work-dir runs/ufo_fb_g1_cartwheel_cli \
  --use-wandb
```

---

### 场景 5：单卡训练

**目标**：当只有单张 GPU 时的训练方案。可以完成完整训练流程，但时间显著延长。

```bash
CUDA_VISIBLE_DEVICES=0 \
./run_train.sh \
  --agent fb \
  --gpu-ids single \
  --num-envs 1024 \
  --num-env-steps 192000000 \
  --work-dir runs/ufo_fb_g1_single \
  --data-path humanoidverse/data/lafan_29dof_10s-clipped.pkl \
  --buffer-size 5120000 \
  --use-wandb \
  --wandb-run-name ufo_fb_g1_single
```

> `--num-envs` 和 `--buffer-size` 是每 GPU 值，所以单卡设置和 8 卡相同。单卡训练时间约为 8 卡的 6-8 倍。

---

### 场景 6：TeCH 冒烟测试

**目标**：快速验证 TeCH 预设的配置和训练循环是否正常工作。

```bash
./run_train.sh \
  --agent tech \
  --data-manifest configs/data/example_mix.yaml \
  --gpu-ids single \
  --smoke \
  --work-dir /tmp/ufo_smoke_tech_g1
```

---

### 场景 7：禁用领域随机化的训练

**目标**：当需要更确定性的训练环境进行调试或对比实验时，可以关闭领域随机化和观测噪声。

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
./run_train.sh \
  --agent fb \
  --gpu-ids all \
  --work-dir runs/ufo_fb_g1_nodr \
  --data-path humanoidverse/data/lafan_29dof_10s-clipped.pkl \
  --disable-dr \
  --disable-obs-noise \
  --use-wandb
```

领域随机化（Domain Randomization）默认包括：
- 物理参数随机化（摩擦、质量、阻尼等）
- 观测噪声
- 初始状态扰动

---

### 场景 8：自定义机器人训练

**目标**：将 UFO 适配到非 G1 的新机器人。此路径是实验性的。

```bash
# 1. 生成机器人配置草稿
uv run python -m humanoidverse.tools.robot_inspect \
  --xml /path/to/robot.xml \
  --name my_robot \
  --out configs/robots/my_robot.yaml \
  --hydra-out humanoidverse/config/robot/my_robot/my_robot_auto.yaml

# 2. 构建 RobotState 数据 manifest
uv run python -m humanoidverse.tools.data_build \
  --robot configs/robots/my_robot.yaml \
  --source "/path/to/motions/*.csv" \
  --format robot_state_csv \
  --name my_motion \
  --fps 50 \
  --clip-seconds 10 \
  --out configs/data/my_motion_auto_build.yaml \
  --rebuild-cache

# 3. 冒烟训练
./run_train.sh \
  --agent fb \
  --robot-config configs/robots/my_robot.yaml \
  --data-manifest configs/data/my_motion_auto_build.yaml \
  --gpu-ids single \
  --smoke \
  --work-dir /tmp/ufo_smoke_my_robot
```

> 详细说明见 [robot_config_training.md](robot_config_training.md) 和 [import_wizard.md](import_wizard.md)。

---

## 五、训练后：推理与导出

### Tracking Inference

训练完成后，用 tracking inference 评估策略效果，生成可视化视频：

```bash
CUDA_VISIBLE_DEVICES=0 \
uv run python -m humanoidverse.tracking_inference \
  --model-folder runs/ufo_fb_g1 \
  --data-path humanoidverse/data/lafan_29dof.pkl \
  --device cuda:0 \
  --headless \
  --save-mp4 \
  --motion-list 0 1 2 3 4 5 \
  --export-onnx true
```

| 参数 | 说明 |
|------|------|
| `--model-folder` | checkpoint 目录（`runs/ufo_fb_g1/` 下按时间戳的子目录） |
| `--data-path` | 推理用完整运动序列（**不要用训练用的 10s 裁剪文件**） |
| `--save-mp4` | 输出 MP4 视频 |
| `--motion-list` | 要推理的运动片段索引列表 |
| `--export-onnx` | 导出 ONNX 模型用于部署 |

输出位置：`<model-folder>/tracking_inference/`

### Goal Inference / Reward Inference

```bash
# Goal inference
CUDA_VISIBLE_DEVICES=0 \
uv run python -m humanoidverse.goal_inference \
  --model-folder runs/ufo_fb_g1 \
  --device cuda:0 --headless --save-mp4

# Reward inference
CUDA_VISIBLE_DEVICES=0 \
uv run python -m humanoidverse.reward_inference \
  --model-folder runs/ufo_fb_g1 \
  --device cuda:0 --headless --num-samples 150000 --save-mp4
```

### ONNX 导出说明

当 `--export-onnx true` 时，会导出：
- ONNX 策略文件（robot-config-aware）
- 元数据 JSON（记录机器人配置、关节列表、输入输出维度等）

导出的 ONNX 与当前 checkpoint 的机器人、动作维度、观测维度**绑定**，不能用于其他机器人。

---

## 六、参数参考

### 完整 CLI 参数表

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--agent` | `fb` | 训练预设：`fb` / `tech` / `tldr`（已弃用） |
| `--gpu-ids` | `single` | `single` / `all` / GPU ID 列表（如 `0,1,2,3`） |
| `--work-dir` | `runs/ufo` | 输出目录 |
| `--robot-config` | `configs/robots/g1_29dof.yaml` | 机器人 YAML 配置路径 |
| `--num-envs` | 1024 | **每 GPU** 并行环境数 |
| `--num-env-steps` | 192,000,000 | **全局**总环境步数 |
| `--checkpoint-every-steps` | 3,200,000 | checkpoint 保存间隔（全局步数） |
| `--data-path` | （见上） | 运动数据 pkl 文件路径，支持多个 |
| `--data-mix-weights` | — | 多数据源采样权重 |
| `--data-manifest` | — | YAML manifest 文件（不能与 `--data-path` 同时使用） |
| `--rebuild-motion-cache` | — | 重建 manifest 生成的数据缓存 |
| `--update-z-every-step` | 100(FB) / 10(TeCH) | 潜在编码 z 更新间隔 |
| `--buffer-size` | 5,120,000 | **每 GPU** 经验回放容量 |
| `--num-agent-updates` | 16(FB) / 128(TeCH) | 每次触发时的优化器更新次数 |
| `--disable-dr` | — | 关闭领域随机化 |
| `--disable-obs-noise` | — | 关闭观测噪声 |
| `--lr-scale` | 1.0 | FB 学习率缩放系数（TeCH 忽略） |
| `--clip-grad-norm` | 0.0 | 梯度裁剪阈值（0=禁用） |
| `--cartwheel-aux-safe` | — | 启用后空翻安全辅助奖励集（仅 FB） |
| `--seed` | 4728 | 随机种子 |
| `--use-wandb` | — | 启用 W&B 日志 |
| `--wandb-run-name` | — | W&B run 名称 |
| `--disable-eval-prioritization` | — | 禁用 tracking eval 和 expert 优先级采样 |
| `--smoke` | — | 冒烟模式（16 envs, 2048 steps, 无 W&B） |

---

## 七、常见问题

### 环境安装问题

| 问题 | 原因 | 解决 |
|------|------|------|
| `uv sync` 失败 | 网络问题或 Python 版本不对 | 确认 Python 3.10，检查网络代理 |
| `ModuleNotFoundError: mujoco` | MuJoCo 未安装 | 重新执行 `uv sync`，确认无报错 |
| 下载数据慢 | HuggingFace 网络问题 | 设置 `HF_ENDPOINT` 镜像或手动下载 |

### 训练问题

| 问题 | 原因 | 解决 |
|------|------|------|
| OOM（显存不足） | envs 或 buffer 太大 | 减少 `--num-envs` 或 `--buffer-size` |
| W&B 连接失败 | 未登录或网络不通 | 去掉 `--use-wandb` 不影响训练 |
| 训练速度过慢 | GPU 利用率低 | 确认 `--num-envs` 足够大（≥512/GPU） |
| 冒烟测试通过但正式训练失败 | 数据路径不对 | 确认 `--data-path` 指向完整的训练 pkl 文件 |

### 模型问题

| 问题 | 原因 | 解决 |
|------|------|------|
| checkpoint 加载失败 | 预设不匹配 | FB 和 TeCH 的 checkpoint **不兼容** |
| 策略表现差 | 训练不足或参数不当 | 检查训练日志中的 reward 曲线 |
| ONNX 导出失败 | 模型结构问题 | 确认 `--export-onnx true` 在 tracking inference 中 |

### 已知限制

- 🚫 不支持自动 motion retargeting（需外部工具预处理）
- 🚫 不同机器人之间不能复用 checkpoint
- 🚫 不支持跨机器人 shared-policy 训练
- ⚠️ 新机器人适配需要人工审核 contact bodies、PD gains、termination 语义
- ⚠️ 非 G1 的 goal inference 需要单独提供 goal JSON
- ⚠️ 非 G1 的 reward inference 仅支持 root/locomotion 任务
