# UFO: 仓库指南

## 仓库总览

UFO (**U**nsupervised **R**einforcement **L**earning **F**ramework for Humanoid **C**ontrol) 是 RoboParty Lab 的开源人形机器人强化学习框架。`main` 分支聚焦于 MJLab 训练、RobotState 数据导入、tracking/goal/reward inference 和 ONNX 导出。单元树 G1 是目前最完整、测试最充分的机器人路径。

### 与 BFMzero 的关系

UFO 是 BFMzero 的公开演化版本。BFMzero 遗留痕迹遍布代码库：

- `BFMZERO_MJLAB_CACHE_DIR` 环境变量作为 `UFO_CACHE_DIR` 的 fallback（`run_train.sh:43`, `train.py:19`），在分布式运行时也会透传给 torchrunx worker（`train.py:389`）
- Hydra 实验配置根路径为 `humanoidverse/config/exp/bfm_zero/bfm_zero.yaml`，引用 `project_name: BFMZero`
- 观测配置 `humanoidverse/config/obs/bfm_zero_obs.yaml`、奖励配置 `humanoidverse/config/rewards/reward_bfm_zero.yaml`
- `train.py` 中 `relative_config_path` 硬编码为 `"exp/bfm_zero/bfm_zero"`（第 203 行）
- `humanoidverse/agents/envs/humanoidverse_mjlab.py` 第 49 行使用 `HYDRA_CONFIG_REL_PATH = os.path.join("exp", "bfm_zero", "bfm_zero")`
- 示例配置 `configs/data/lafan_cartwheel_mix.yaml` 中路径硬编码为 `/data/xue/bfmzero/data/...`

> **注意**：BFMZero 名称不是 typo，是原始项目名。UFO 重构了公共接口但保留了内部 Hydra 配置结构。修改这些内部 BFMZero 引用需要确保 Hydra config 解析链完整。

## 核心架构

### 项目结构

```
UFO/
├── humanoidverse/          # 主 Python 包
│   ├── train.py            # 🔑 训练入口（build_ufo_mjlab_config → TrainConfig）
│   ├── train_mjlab.py      # 向后兼容的旧模块名（import 重定向）
│   ├── tracking_inference.py
│   ├── goal_inference.py
│   ├── reward_inference.py
│   ├── mjlab_inference_utils.py
│   ├── mjlab_reward_relabel.py
│   ├── agents/
│   │   ├── presets/        # FB / TeCH (原 TLDR) 训练预设
│   │   ├── envs/           # MJLab 环境桥（~1300 行，核心）
│   │   ├── buffers/        # trajectory + transition buffer
│   │   └── evaluations/    # tracking eval 配置
│   ├── config/             # 🔑 Hydra 配置树（base/exp/env/robot/rewards/obs...）
│   │   └── exp/bfm_zero/   # BFMZero 实验配置（UFO 沿用此路径）
│   ├── envs/               # env_utils、motion_observations
│   ├── tools/              # robot_inspect, data_build, data_inspect
│   ├── training/           # workspace（TrainConfig.build → workspace.train()）
│   └── utils/
│       ├── motion_data/    # 数据管道：adapter, clip, manifest, reader, converter, schema
│       ├── motion_lib/     # MotionLib 采样库
│       └── robot_spec/     # RobotSpec 解析验证
├── configs/                # 用户层面配置
│   ├── robots/g1_29dof.yaml     # 🔑 G1 机器人训练配置（RobotTrainingSpec）
│   └── data/                    # 数据 manifest YAML
├── docs/                   # 文档（import_wizard, robot_config_training, TRAIN_INFERENCE）
├── scripts/
│   ├── download_data.sh    # 从 HuggingFace 下载运动数据
│   └── smoke_release.sh    # 发布前冒烟测试
└── tests/                  # 10 个 unittest 测试文件
```

### 双层机器人配置

UFO 有两层机器人配置，**都要改才能适配新机器人**：

| 层级 | 位置 | 用途 | 手动编写 |
|------|------|------|----------|
| **用户层** | `configs/robots/<name>.yaml` | RobotTrainingSpec：训练入口读取的语义配置（base_body, feet, hands, control_joints, training.*） | 需要人工审核 |
| **Hydra 层** | `humanoidverse/config/robot/<group>/<name>.yaml` | 内部 Hydra 配置：DOF 列表、PD 增益、运动学参数、contact/termination 语义 | 大部分自动生成后需人工整理 |

`robot_inspect --hydra-out` 可同时生成两层草稿。生成配置会被标记 `metadata.review_status: draft`。

### 训练代理（agent）预设

| 预设 | CLI 名称 | 别名 | 特性 |
|------|---------|------|------|
| FB | `--agent fb` | - | 默认预设，update_z 间隔 100 步 |
| TeCH | `--agent tech` | `tldr`（已弃用） | 早期称 TLDR，update_z 间隔 10 步 |

- `--cartwheel-aux-safe` 仅适用于 FB
- `--lr-scale` 仅影响 FB，TeCH 忽略

## 关键命令

### 环境与数据
```bash
uv sync                                    # 安装依赖（Python 3.10 必须）
bash scripts/download_data.sh g1_lafan     # 下载 G1 LaFAN 运动数据（~200MB）
```

### 训练
```bash
# 冒烟测试
./run_train.sh --agent fb --data-manifest configs/data/example_mix.yaml --gpu-ids single --smoke

# 完整训练（8 GPU）
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./run_train.sh --agent fb --gpu-ids all \
  --num-envs 1024 --num-env-steps 192000000 --work-dir runs/ufo_fb_g1 \
  --data-path humanoidverse/data/lafan_29dof_10s-clipped.pkl \
  --update-z-every-step 100 --buffer-size 5120000 --use-wandb
```

### 推理
```bash
# Tracking inference + ONNX 导出
uv run python -m humanoidverse.tracking_inference \
  --model-folder runs/ufo_fb_g1 --device cuda:0 --headless \
  --save-mp4 --motion-list 0 --export-onnx true

# Goal / Reward inference 类似
```

### 工具
```bash
# 检查 MuJoCo XML 生成机器人配置草稿
uv run python -m humanoidverse.tools.robot_inspect --xml /path/to/robot.xml --name my_robot

# 检查/构建 RobotState 数据
uv run python -m humanoidverse.tools.data_inspect --robot configs/robots/my_robot.yaml --source "/*.csv"
uv run python -m humanoidverse.tools.data_build --robot configs/robots/my_robot.yaml --source "/*.csv" ...
```

### 测试
```bash
# 单个测试
uv run python tests/test_motion_data_adapter.py

# 完整冒烟测试
bash scripts/smoke_release.sh
```

## 重要约定与限制

### Python 版本
- **Python 3.10 必须**（`requires-python = "==3.10.*"`），torch 2.7.x 绑定
- 使用 `uv` 包管理，不要用 pip/poetry

### 缓存目录（`run_train.sh`）
- `UFO_CACHE_DIR` 自动推导：`--work-dir` 父目录如果是 `runs/` 则取同级 `cache/`
- 创建 `uv/`, `pycache/`, `tmp/`, `torchinductor/`, `triton/`, `cuda/`, `warp/` 子目录
- 通过 `BFMZERO_MJLAB_CACHE_DIR` 环境变量兼容旧工作流

### 分布式训练
- `--num-envs` 和 `--buffer-size` 是**每 GPU** 的值
- `--num-env-steps` 是**全局**样本预算
- 通过 `torchrunx` 启动多进程，NCCL backend
- 默认 `--gpu-ids single`（单卡），`all` 使用全部可见 GPU

### 运动数据
- 数据 manifest 支持多源加权混合（`configs/data/example_mix.yaml`）
- RobotState CSV/NPZ 格式：`root_pos` xyz + `root_quat` xyzw + DOF 位置
- 推理时**必须**使用 full motion sequences（不是 clipped training clips）
- 数据从 HuggingFace 下载（`xuewang/UFO-MotionData`），SHA256 校验

### 已知约束
- 🚫 不支持自动 motion retargeting
- 🚫 不同机器人之间不能复用 checkpoint
- 🚫 不支持跨机器人 shared-policy 训练
- ⚠️ 新机器人适配需要人工审核 contact bodies、PD gains、termination 语义
- ⚠️ 非 G1 的 goal inference 需要单独提供 goal JSON
- ⚠️ 非 G1 的 reward inference 仅支持 root/locomotion 任务

### Ruff 配置
- 行宽 140，忽略 E402（import 顺序）和 E731（lambda 赋值）
- 启用 I（import sorting）但仅作 lint 不自动 fix（CI 本地结果不一致）
- 运行：`uv run ruff check .` 或 `uv run ruff check --fix .`

### CI/CD
- 无 GitHub Actions（`.github/` 不存在）
- 发布前冒烟：`bash scripts/smoke_release.sh`（运行测试 + 短训练 + data_build）
- Git LFS 用于跟踪 `.pkl` 数据文件

### 其他
- `show_help.py` 打印 UFO 艺术字和快速帮助
- `index.html` 是空文件
- `AGENTS.md` 是首次创建，覆盖之前无指令文件的状态
