# UFO_HT 代理协作指南

## 项目定位

- 本仓库是面向人形机器人控制的 Python 3.10 科研项目，主要覆盖 MJLab 训练、RobotState 动作数据导入、tracking/goal/reward inference 和 ONNX 导出。
- `main`/当前训练分支负责训练、数据处理、推理与导出；Unitree G1 实机部署和遥操作属于 `deploy` 分支，不要把部署运行时逻辑混入当前分支。
- Unitree G1 是支持最完整的路线。新机器人接入仍属实验能力，必须提供匹配的 MuJoCo XML、可选 URDF、robot config，以及已经 retarget 到目标机器人的动作数据。
- checkpoint、动作维度、观测维度和机器人形态相互绑定。不要假设不同机器人之间可以直接复用 checkpoint 或 motion data。

## 目录导航

- `humanoidverse/agents/`：FB、TeCH 等算法及预设。
- `humanoidverse/envs/`、`humanoidverse/tasks/`：环境、任务、奖励和终止逻辑。
- `humanoidverse/training/`：训练初始化与 robot-aware 配置路径。
- `humanoidverse/tools/`：机器人检查、数据检查和数据构建工具。
- `humanoidverse/utils/motion_data/`：RobotState 适配、校验、裁剪和 manifest 处理。
- `humanoidverse/export/`：模型导出相关实现。
- `configs/robots/`：机器人配置；`configs/data/`：数据集 manifest。
- `tests/`：基于标准库 `unittest` 的测试。
- `scripts/`：数据下载及辅助脚本。
- `docs/`：训练、推理、机器人接入和专项流程说明。

## 本服务器 GPU 约束

- 本服务器训练只能使用前四张卡：物理 GPU 0、1、2、3。
- 严禁在训练、评估或多卡命令中使用 GPU 4、5、6、7；设置 `CUDA_VISIBLE_DEVICES` 时只能包含 `0`、`1`、`2`、`3`。
- 启动多卡训练前先用 `nvidia-smi` 确认前四张卡状态；优先通过 `CUDA_VISIBLE_DEVICES=0,1,2,3` 和项目参数选择卡集合，不要依赖未记录的全局 GPU 环境。

## 环境与依赖

- 使用 `uv` 和仓库根目录的 `pyproject.toml`、`uv.lock` 管理环境，不要混用裸 `pip` 修改 `.venv`。
- 安装或恢复环境：

  ```bash
  uv sync --locked
  ```

- Linux 上的 `torch` 和 `torchvision` 必须继续使用 `pytorch-cu128` 显式源；通用 PyPI 包使用项目配置的 USTC 默认镜像。不要仅为解决下载问题替换 PyTorch CUDA 构建。
- 修改依赖时同步更新 `pyproject.toml` 和 `uv.lock`，并说明新增依赖的必要性。普通代码改动不要无故重写锁文件。
- 大型 CUDA wheel 下载失败时，优先保留镜像配置并增加超时，例如：

  ```bash
  UV_HTTP_TIMEOUT=600 UV_CONCURRENT_DOWNLOADS=4 uv sync --locked
  ```

## 常用验证命令

按改动风险选择最小但充分的验证范围。提交完成声明前必须运行与改动对应的命令并检查退出码。

- Ruff 静态检查（优先检查本次改动文件）：

  ```bash
  uv run ruff check path/to/changed_file.py
  ```

  全仓 `uv run ruff check .` 可用于观察基线，但当前仓库存在既有 lint debt；不要为了通过全仓检查而批量改动无关代码。

- 全量单元测试：

  ```bash
  MUJOCO_GL=egl uv run python -m unittest discover -s tests -v
  ```

- 单文件测试示例：

  ```bash
  uv run python -m unittest tests.test_motion_data_adapter -v
  ```

- 依赖一致性检查：

  ```bash
  uv pip check --python .venv/bin/python
  ```

- GPU/CUDA 基础检查：

  ```bash
  uv run python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
  ```

- G1 smoke training 需要 GPU 和已下载的动作数据，不要把它当作每次改动都运行的快速测试：

  ```bash
  CUDA_VISIBLE_DEVICES=0 \
  ./run_train.sh \
    --agent fb \
    --data-manifest configs/data/example_mix.yaml \
    --gpu-ids single \
    --smoke \
    --work-dir /tmp/ufo_smoke_g1
  ```

## 实现约定

- 优先沿用现有模块边界、配置结构和 helper API；避免在功能修改中夹带无关重构。
- Python 代码使用类型标注，路径操作优先使用 `pathlib.Path`，复杂数据优先使用现有 schema/dataclass 和结构化解析器。
- Ruff 行宽为 140；遵循现有 `E402`、`E731` 例外和 import sorting 配置。
- 修复缺陷时先添加能复现问题的 `unittest`，确认失败原因正确后再实施最小修复。
- 新增 CLI 参数时同时覆盖解析、默认值、兼容 alias 和用户文档；公开行为变更应同步维护 `README.md` 与 `README_zh-CN.md`。
- robot-aware 路径应通过 `RobotSpec`、robot config 和 data manifest 传递语义，不要在公共流程中新增只适用于某一机器人名称或关节顺序的隐式假设。
- 修改训练默认值时明确区分 per-GPU 参数和全局参数；`--num-envs`、`--buffer-size` 是 per GPU，`--num-env-steps` 是全局样本预算。
- TeCH 是当前公开名称；`tldr` 只作为已弃用兼容 alias 保留。

## 数据、训练与生成物

- 大型 motion data、checkpoint、replay buffer、训练输出和渲染视频不应提交到 Git。
- 未经明确要求，不要自动下载数据、启动长时间训练、登录 W&B 或连接远端机器。
- 下载默认 G1 数据使用 `bash scripts/download_data.sh g1_lafan`；脚本会校验 SHA256。不要绕过校验或用不明文件覆盖现有数据。
- 训练统一从 `run_train.sh` 启动，以复用 `UFO_CACHE_DIR`、TorchInductor、Triton、CUDA 和 Warp 缓存设置。
- 快速验证输出写入 `/tmp`；正式运行输出写入明确的 `runs/<name>`，不要污染源码目录。
- 推理应使用完整动作序列而不是训练裁剪片段；ONNX 及 metadata 必须与对应 checkpoint 和机器人配置一起使用。

## Git 与协作安全

- 开始修改前运行 `git status --short`，把已有修改视为用户工作。不得回滚、覆盖或格式化无关文件。
- 当前工作区可能包含未提交的依赖源和锁文件修改；处理其他任务时必须保留它们。
- 不使用 `git reset --hard`、`git checkout --`、递归删除等破坏性命令，除非用户明确指定目标并授权。
- 不主动提交、推送、创建分支或 PR，除非用户明确要求。
- 每完成一项可独立验证的迁移或修复后，只暂存该项涉及的文件并创建一个原子提交；不得把已有未相关改动混入同一提交。
- 完成后报告改动文件、实际执行的验证命令、结果以及未执行测试的原因。

## 文档入口

- 首选中文总览：`README_zh-CN.md`；英文对应文档：`README.md`。
- 训练与推理：`docs/TRAIN_INFERENCE.md`。
- 训练配置全景与 Stage2 实验状态：`docs/training_config_overview_zh.md`。
- 新机器人数据导入：`docs/import_wizard.md`。
- robot-aware 训练：`docs/robot_config_training.md`。
- PiPlus 多卡训练和专项操作以 `docs/` 中对应说明为准，不要从文件名推断参数。

## 实验台账规则

- [`docs/experiment_ledger.md`](docs/experiment_ledger.md) 是所有实验进度的唯一权威台账。所有新实验、续训、checkpoint 创建、checkpoint 恢复、命令或配置变化、进程启动/停止、指标里程碑、失败/中断、硬件分配以及 observation/architecture 决策，都必须追加记录。
- 每条记录必须包含本地时间戳、状态、实验 ID、workdir、checkpoint lineage、完整命令或配置变化、硬件/GPU、验证结果和证据路径。历史记录只追加，不覆盖；更正必须以 dated amendment 追加。
- 不得只把实验进度留在聊天、shell history、运行目录或其他专题文档中；checkpoint、日志和训练输出仍留在运行目录，不提交到 Git。
- 修改运行中的 observation contract、decoder contract 或 critic architecture 前，必须先在台账记录 fresh-experiment 决策；不得让旧 checkpoint 静默跨合同恢复。

## 配置导航（自动整理）

- 常规入口：`run_train.sh` -> `humanoidverse.train`；FB 是默认 agent，`tech` 是 TeCH，`tldr` 仅保留兼容 alias。
- 配置分层：`humanoidverse/config/` 是 Hydra 运行时组合，`configs/robots/` 是完整 RobotTrainingSpec，`configs/data/` 是 motion manifest。
- 当前 Stage2 主线：H0W 22DoF frozen BFM decoder + command encoder；优先 `speed_stage2`，再对照 AMP Profile A；Profile B 需要专用 expert 数据，kick 是独立任务。
- Stage2 合同：decoder 616 -> 22，encoder 输入 363，latent 256 且投影范数 16；checkpoint、robot、motion data、backend 必须一起核对。
- Stage2 的详细默认值、已知风险和短回放指标见 `docs/training_config_overview_zh.md`；子目录 AGENTS 只补充各自边界，不重复本文件的全局规则。
