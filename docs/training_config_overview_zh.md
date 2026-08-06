# UFO_HT 训练配置总览

> 这份文档是当前实验分支的配置地图，重点记录 PiPlus H0W 的 Stage2 速度跟踪 command encoder。它描述“代码当前做什么”，不等同于一套已经验证收敛的超参数。大型 checkpoint、动作数据和运行输出不属于仓库配置。

## 1. 训练路径

标准入口只有一条：

```text
run_train.sh
  -> python -m humanoidverse.train
     -> 解析 agent / robot / data / GPU / budget 参数
     -> 读取 RobotTrainingSpec
     -> 组合 humanoidverse/config 的 Hydra MJLab 配置
     -> humanoidverse.training.workspace.Workspace
     -> FB 或 TeCH learner
```

`run_train.sh` 负责缓存目录、TorchInductor/Triton/Warp 和 CUDA 环境；不要直接绕过它启动常规训练。Stage2 是实验性独立入口，直接运行 `python -m humanoidverse.speed_stage2`、`amp_stage2_piplus_22dof` 或 `tech_kick_stage2`，但仍应遵守根目录的 GPU 和输出目录规则。

### 配置优先级

1. CLI 显式参数（`--robot-config`、`--data-manifest`/`--data-path`、预算和设备）。
2. 数据 manifest 中的 `robot_config`（若同时提供，必须与 CLI robot config 兼容）。
3. 默认机器人 `configs/robots/g1_29dof.yaml`、默认数据 `humanoidverse/data/lafan_29dof_10s-clipped.pkl`。
4. agent preset 再覆盖 learner/runtime 的默认值。
5. Stage2 的 task/profile 参数最后决定其自身 reward、decoder 和 checkpoint 合约，不能用常规 FB/TeCH preset 代替。

`--num-envs` 和 `--buffer-size` 是每张 GPU 的值；`--num-env-steps` 是全局样本预算。多卡只允许物理 GPU 0、1、2、3，并应显式设置 `CUDA_VISIBLE_DEVICES`。

常规 CLI 的实验开关也集中在 `humanoidverse/train.py`：`--data-path` 与 `--data-manifest` 互斥，多个 `--data-path` 可用 `--data-mix-weights` 指定比例；`--rebuild-motion-cache` 强制重建 manifest cache；`--update-z-every-step`、`--num-agent-updates` 用于更新密度消融；`--disable-dr`/`--disable-obs-noise` 控制随机化；FB 独有 `--lr-scale`、`--clip-grad-norm` 和 `--cartwheel-aux-safe`；`--init-model` 只加载模型权重、不恢复 optimizer/replay；`--enable-compile`、W&B 和 eval prioritization 是显式实验开关。`--smoke` 会把常规训练压到最多 16 env、2048 global env steps 且关闭 W&B。

## 2. 常规训练配置矩阵

| 层 | 位置 | 作用与当前默认 |
|---|---|---|
| 启动器 | `run_train.sh` | 设置缓存后调用 `humanoidverse.train`。 |
| CLI/默认 | `humanoidverse/train.py` | 默认 agent `fb`，每 GPU 1024 env，全局 192M env steps，checkpoint 间隔 3.2M，buffer 5.12M，默认 G1 29DoF。支持 `fb`、`tech`，`tldr` 是弃用兼容 alias。 |
| learner preset | `humanoidverse/agents/presets/fb.py` | FB：每 1024 环境步更新，16 次 agent update，trajectory buffer，latent interval 100。 |
| learner preset | `humanoidverse/agents/presets/tldr.py` | TeCH 兼容 preset：每 1024 环境步更新，128 次 agent update，latent interval 10。 |
| preset 注册 | `humanoidverse/agents/presets/__init__.py` | `fb -> build_fb_agent`；`tech/tldr -> build_tech_agent`。 |
| 运行时 | `humanoidverse/training/workspace.py` | 保存/恢复、W&B、评估、分布式和 replay buffer 的统一编排。内部 online env 默认 50，通常会被 CLI/preset 覆盖。 |
| 任务/环境 | `humanoidverse/envs/`、`humanoidverse/tasks/` | episode、motion reset/resample、termination、reward 和 curriculum。 |
| 推理/导出 | `humanoidverse/inference/`、`humanoidverse/export/` | tracking/goal/reward inference 与 ONNX backward encoder；模型必须绑定 robot/obs/action 维度。 |

### Hydra 配置树

`humanoidverse/config/base.yaml` 是组合根：seed=0、headless=true、num_envs=4096、MuJoCo、默认不启用 multi-GPU。主要分组如下：

- `config/env/`：`base_task`、`legged_base`、`legged_motions`，负责 episode 时长、动作/观测接口、motion 终止和重采样。
- `config/simulator/mujoco.yaml`：200 FPS、control decimation 4、substeps 1。
- `config/domain_rand/`：push、COM/link mass、friction、默认 DoF 随机化开启；PD/base mass/torque RFI/control delay 默认关闭。
- `config/obs/bfm_zero_obs.yaml`：base angular velocity、projected gravity、DoF pos/vel、actions、actor history 和局部 self observation；维度随 robot 变化。
- `config/rewards/reward_bfm_zero.yaml`：BFM Zero 的基础 reward scales（动作速率等惩罚也在此）。
- `config/callbacks/`：model save、autoresume、im_eval；默认保存/评估间隔为 500 个训练迭代量级。
- `config/exp/bfm_zero/bfm_zero.yaml`：完整实验组合，默认 G1 hard-waist、motion loader、plane terrain、lie-down 初始化和 BFM Zero 项目设置。
- `config/robot/`：Hydra robot schema 及 `g1`、`piplus_bfm`、`piplus_h0w` 形态配置；它不是 `configs/robots/*.yaml` 的替代品，而是由 robot-aware training 映射进去的运行时层。

当前仓库内的 Hydra 文件清单：

```text
humanoidverse/config/base.yaml
humanoidverse/config/base_eval.yaml
humanoidverse/config/base/{hydra,structure,fabric}.yaml
humanoidverse/config/env/{base_task,legged_base,legged_motions}.yaml
humanoidverse/config/simulator/mujoco.yaml
humanoidverse/config/domain_rand/domain_rand.yaml
humanoidverse/config/terrain/{terrain_base,terrain_locomotion_plane}.yaml
humanoidverse/config/obs/bfm_zero_obs.yaml
humanoidverse/config/rewards/reward_bfm_zero.yaml
humanoidverse/config/callbacks/{model_save,autoresume,im_eval}.yaml
humanoidverse/config/exp/bfm_zero/bfm_zero.yaml
humanoidverse/config/robot/robot_base.yaml
humanoidverse/config/robot/g1/g1_29dof_hard_waist.yaml
humanoidverse/config/robot/piplus/{piplus_bfm,piplus_h0w}.yaml
```

`base_eval.yaml` 只负责 tracking evaluation 的时间戳、model/eval 名称和日志目录；`base/hydra.yaml` 把 Hydra run/sweep 目录绑定到 `save_dir`；`base/structure.yaml` 声明 env/robot/terrain 组合槽位；`base/fabric.yaml` 是旧的 Lightning Fabric DDP 组合，默认不在当前 `humanoidverse.train` 路径启用。terrain 当前只有基础摩擦和 locomotion plane 组合。

## 3. Robot 与数据配置

### 外部 robot config

`configs/robots/*.yaml` 是用户可选的完整训练规格，不互相继承：

| 文件 | 形态 | 动作/关键约束 |
|---|---|---|
| `g1_29dof.yaml` | Unitree G1 | 29 DoF；默认主路线；绑定 `config/robot/g1/g1_29dof_hard_waist`。 |
| `piplus_bfm.yaml` | PiPlus BFM | 23 DoF；绑定 `config/robot/piplus/piplus_bfm`。 |
| `piplus_h0w.yaml` | PiPlus H0W | 22 DoF，显式 control joint order；绑定 `config/robot/piplus/piplus_h0w`；XML draft metadata 仍需在正式实验前复核。 |

robot config 同时承载 XML/URDF、关节语义、接触集合、actuator/control 参数和可选 Hydra overrides。checkpoint、motion data、观测和动作维度必须与同一 RobotSpec 配套。

### data manifest

`configs/data/` 描述数据来源和构建方式：

- `example_mix.yaml`：单个 LaFAN `ufo_pkl`，用于入口 smoke。
- `example_robot_state_auto_build.yaml`：G1 RobotState CSV，50 FPS、列映射和自动 clip 构建。
- `lafan_cartwheel_mix.yaml`：LaFAN 0.95 + cartwheel 0.05，并带推理路径。
- `piplus_lafan.yaml`：PiPlus BFM，环境变量 motion 目录、hash、23 joints、30 FPS。
- `piplus_h0w_lafan.yaml`：H0W，环境变量 motion 目录、hash、22 joints/27 bodies、30 FPS。
- `piplus_h0w_locomotion_recovery.yaml`：H0W RobotState PKL glob，覆盖 walk/run/sprint/fall-and-get-up，排除 dance/fight/jump，10 秒 clip、stride 自动构建。

实现入口是 `humanoidverse/utils/motion_data/manifest.py` 的 `ManifestMotionData`；mixed source 支持权重和 priority sampling。RobotState CSV/NPZ/PKL 适配必须通过 RobotSpec 校验，不能只按文件名推断关节顺序。

## 4. Stage2 总览

Stage2 的共同思想是：冻结一个已经训练好的动作 decoder（或完整 TeCH policy），只训练一个把 command 编码为 latent `z` 的小策略。它们共享部分命名，但不是同一任务、同一 checkpoint 或同一 reward。

### 共享 H0W BFM 合约

由 `humanoidverse/piplus_h0w_onnx_decoder.py` 和 `piplus_h0w_stage2.py` 固化：

| 项 | 当前值 |
|---|---:|
| robot/action dim | H0W / 22 |
| decoder actor input | 616 |
| command encoder input | 363 = command 3 + state 50 + last action 22 + actor history 288 |
| latent dim | 256 |
| latent projection | L2 norm 为 `sqrt(256)=16` |
| command scale | `(1.25, 5.0, 1.25)`，对应 vx、vy、yaw rate |
| decoder output | 22 actions，ONNX 输入 616、输出 22 |
| decoder backend | 优先 `onnxruntime`（可用 CUDA provider 时使用），否则 `onnx.reference.ReferenceEvaluator` |

当前 fallback 每步把 tensor 拷到 CPU NumPy；环境没有 onnxruntime-gpu 时会显著拖慢 rollout，不能把低 GPU 利用率误判为 encoder 没学到。

### Stage2 任务矩阵

| 入口 | task metadata | 冻结对象 | 学习对象 | 默认/特征 | 适用结论 |
|---|---|---|---|---|---|
| `humanoidverse/speed_stage2.py` | `speed_stage2_piplus_22dof` | H0W BFM ONNX decoder | stochastic command encoder + value head | env 64、rollout 32、PPO 5 epochs、lr 3e-4、gamma .98、stand/turn .15、resample 300、smoothing .15、20s episode、10k iterations | 当前最直接的速度跟踪实验；只启用 velocity reward，默认 reward weight=0。 |
| `humanoidverse/amp_stage2_piplus_22dof.py` Profile A | `amp_stage2_piplus_22dof` | H0W BFM ONNX decoder | command encoder | env 64、rollout 32、history 8、PPO 5、lr 1e-4、gamma .99、stand .05、turn .20、smoothing .02；函数会强制把 AMP/prior 权重置零 | 当前推荐的纯 locomotion/safety baseline；名字含 AMP 但 Profile A 实际不训练 discriminator。 |
| `humanoidverse/amp_stage2_piplus_22dof.py` Profile B | 同上 | decoder + AMP reference | command encoder + AMP discriminator/prior | AMP feature 194 = root lin vel 3 + 5 local key bodies 15 + 8x22 joint history 176；env reward 1、locomotion 1.1、AMP .06、prior .02 | 只有在专用 `piplus_h0w_locomotion_run_with_stand.pkl` 存在且验证后才可运行；缺失时应 hard-fail。 |
| `humanoidverse/tech_kick_stage2.py` | `tech_kick_stage2` | 完整 TeCH model | kick command encoder | env 64、rollout 24、PPO 5、30k iterations、lr 3e-4、3s episode；球半径 .07、质量 .20 | 球踢击任务，不是速度跟踪 baseline；不能与 speed/AMP checkpoint 互放。 |

Profile A 的直接平面速度奖励 `linvel_exp` 当前权重为 `3.7`，此前为 `3.1`；该值来源于 `humanoidverse/amp_stage2_piplus_22dof.py`。AMP 和 prior 权重仍为 `0`。Python 进程在启动时加载源代码，启动后再编辑源文件不会追溯改变已经运行的进程。

每个 playback 入口都会检查 task metadata，拒绝跨任务 checkpoint。speed playback 默认 command `(0.4, 0, 0)`、250 steps；AMP playback 同样受 task/profile 约束。路径 metadata 记录 task、robot、dataset、decoder、encoder 输入和 world size，但运行目录中的绝对路径未必能在当前机器复现。

Stage2 相关文件的职责边界：`piplus_h0w_onnx_decoder.py` 是 ONNX provider/维度校验，`piplus_h0w_stage2.py` 是 command encoder、GAE、PPO 和速度 reward 共用实现，`piplus_h0w_locomotion.py` 是 H0W reference-free MJLab 环境，`piplus_h0w_stage2_play.py` 是共享有界回放/渲染。`speed_stage2_play.py` 和 `amp_stage2_piplus_22dof_play.py` 是各自 playback；`speed_stage2_legacy_play.py` 只是旧 backend/路径兼容包装，不能当作新的训练入口。`tech_kick_stage2_play.py` 是 kick playback，`tech_kick_monitor.py` 只读运行日志并展示 dashboard。

## 5. 当前实验状态与风险

已有运行 metadata 显示 speed checkpoint 使用 H0W、decoder 616->22、latent 256、encoder 363、world size 4；部分路径仍指向外部 `HT_BFM/UFO-main`。这证明训练入口曾经跑通，但不证明跨后端行为等价。

现有 debug 记录的行为差异：同一 legacy speed checkpoint 在旧 Isaac 后端能产生约 `-0.46` 到 `-0.8` 的速度，在当前 UFO_HT MJLab 回放几乎静止（250 步根位移约 0.014 m，base vx 峰值约 0.172）。已排除 command 注入、渲染和 policy/decoder 未调用等表层原因；主要嫌疑是 backend/controller/action clipping、reset root height 和 upper-body 初始姿态合同不一致。UFO_HT 的 action clip/normalize 与旧 checkpoint 也不相同。

另一个性能风险是 ONNX CPU fallback：没有 `onnxruntime` 时每步发生 device/host copy，AMP 试跑约每迭代 70 秒；多卡 PPO 还会产生大量 all-reduce。先修复执行后端和数据通路，再比较 encoder 超参。

## 6. 推荐实验阶梯

1. **合同 smoke**：固定 H0W XML、22 joint order、616/363/256 维度和 decoder 输出，先运行 targeted Stage2 unittest。
2. **纯速度 baseline**：使用 `speed_stage2`，冻结同一 BFM decoder，只改变 command；记录 `vx/vy/yaw` 的 MAE、root displacement、termination 和 command-resample 对齐。
3. **等价性诊断**：对同一 checkpoint 做 zero-command、`vx=0.4`、`yaw=0.4` 三组短回放，逐层记录 latent norm、action clip、base velocity 和 reset 状态；不要先调 reward。
4. **Profile A 对照**：只有 speed baseline 合同稳定后再比较 AMP Profile A 的 smoothing/stand/turn 设置。
5. **Profile B/长期训练**：准备并 hash 专用 locomotion expert，确认 ONNX CUDA provider 或批量 reference 执行，再扩大 rollout/多卡预算。

建议把以下指标写入每次 checkpoint metadata 或独立 CSV：`command/vx`、`command/vy`、`command/yaw_rate`、`base/vx`、`base/vy`、`base/yaw_rate`、三轴 MAE、latent L2 norm、action saturation ratio、episode termination reason、decoder provider、world size。

### 受约束的启动示例

常规 G1 smoke：

```bash
CUDA_VISIBLE_DEVICES=0 ./run_train.sh \
  --agent fb --data-manifest configs/data/example_mix.yaml \
  --gpu-ids single --smoke --work-dir /tmp/ufo_smoke_g1
```

H0W speed Stage2（资产路径需由实验者显式提供）：

```bash
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl uv run python -m humanoidverse.speed_stage2 \
  --device cuda:0 --smoke --work-dir /tmp/ufo_speed_stage2_smoke \
  --bfm-model <frozen-bfm-checkpoint> \
  --decoder-path <h0w-decoder.onnx> \
  --robot-config configs/robots/piplus_h0w.yaml \
  --motion-dataset <h0w-motion-data>
```

Stage2 targeted tests：

```bash
MUJOCO_GL=egl uv run python -m unittest \
  tests.test_piplus_h0w_stage2_common \
  tests.test_speed_stage2 \
  tests.test_amp_stage2_piplus_22dof \
  tests.test_tech_kick_stage2 -v
```

## 7. 已知文档/环境不一致

- `docs/tech_kick_stage2_zh.md` 中仍有使用 GPU 4-7 的旧多卡命令；本服务器规则优先，实际命令必须改为 0-3 范围。
- Profile B 的 expert 数据不是仓库内置资产，缺失时不能退回随机或 LaFAN 数据冒充 AMP reference。
- 旧运行 metadata 的绝对路径和旧 backend checkpoint 不保证可直接在当前 MJLab 回放；复现实验必须重新填写 robot/data/decoder 路径并检查合同。
- `onnxruntime-gpu` 属于环境能力而非代码默认；安装依赖前应同步评估 `pyproject.toml`/`uv.lock`，本总览不修改依赖。

相关入口：[`TRAIN_INFERENCE.md`](TRAIN_INFERENCE.md)、[`robot_config_training.md`](robot_config_training.md)、[`piplus_h0w_stage2_zh.md`](piplus_h0w_stage2_zh.md)、[`tech_kick_stage2_zh.md`](tech_kick_stage2_zh.md)。
