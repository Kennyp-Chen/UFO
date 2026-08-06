# AMP Stage2 Profile A：4-GPU vs 1-GPU 更新速度对比

- 日期：2026-08-06
- Agent：`humanoidverse.amp_stage2_piplus_22dof`（reward profile `a_mimiclite_speed_safety`）
- 目标：对比同等总规模（4096 envs）下 4-GPU 与单卡续训的迭代速度，验证单卡是否更优。
- 结论先行：**单卡 4096 envs 比 4 卡 × 1024 envs 慢约 4 倍（360 s/iter vs 90 s/iter）**，当前 CPU 解码器路径下单卡没有优势；继续沿用 4-GPU 配置，真正的提速方向是安装 `onnxruntime-gpu` 用 GPU 解码器。

## 环境状态（测量期间）

- GPU 4-7 被其他用户（hejunfu）的 tech 训练任务占满（util 82-92%，每卡 ~42 GiB），全程存在 CPU 竞争。
- IsaacSim speed_stage2 训练（PID 924034）在两次测量之间结束：4-GPU 测量时 GPU 0-3 仍有其 2.3-4.0 GiB 占用；单卡测量时 GPU 0/2/3 已释放至 ~4 MiB。
- 单卡任务运行于 **GPU 1**（测量时显存占用最低：2330 MiB，GPU 0 为 3970 MiB）。
- 解码器：UFO_HT 环境无 `onnxruntime`/`onnxruntime-gpu`，`OnnxPiPlusH0WDecoder` 回退到 CPU `ReferenceEvaluator`，每步动作推理发生在 CPU（GPU→CPU→GPU 拷贝）。

## 4-GPU 测量（profile_a_4gpu_4096env_20260805）

- 命令（摘要）：`torchrun --standalone --nproc_per_node=4 -m humanoidverse.amp_stage2_piplus_22dof ... --num-envs 1024 --rollout-steps 32 --minibatch-size 1024 --ppo-epochs 5 ... --iterations 20000 --resume checkpoint_10000.pt`
- 每迭代：每卡 1024 envs × 32 rollout = 32768 样本，160 个 PPO 更新/卡（4 卡并行），全局样本 131072。
- 采样窗口 180.4 s：iteration 10184 → 10186，**90.2 s/iter**（0.0111 iter/s）。
- GPU 0-3 利用率采样时 0-2%（CPU 解码器为瓶颈；GPU 4-7 竞争加剧 CPU 压力）。

## 单卡测量（profile_a_1gpu_4096env_20260806）

- 命令（摘要）：`CUDA_VISIBLE_DEVICES=1 torchrun --standalone --nproc_per_node=1 -m humanoidverse.amp_stage2_piplus_22dof ... --gpu-ids single --num-envs 4096 --rollout-steps 32 --minibatch-size 1024 --ppo-epochs 5 ... --iterations 20000 --resume checkpoint_10100.pt`
- 每迭代：4096 envs × 32 rollout = 131072 样本（与 4-GPU 全局一致），640 个 PPO 更新（单卡串行；4-GPU 时为 4 卡并行 160×4）。
- 采样窗口 360.4 s：iteration 10102 → 10103，**360.4 s/iter**（0.00277 iter/s）；此前窗口 10101 → 10102 同样约 360 s。
- GPU 1 显存 ~36.4 GiB，进程 CPU ~800%（CPU 解码器 + 640 次串行 PPO 更新）。

## 对比汇总

| 指标 | 4-GPU（4×1024 envs） | 1-GPU（1×4096 envs） | 比值 |
|---|---|---|---|
| 每迭代样本数 | 131072 | 131072 | 1:1 |
| PPO 更新/迭代 | 160×4 并行 | 640 串行 | 4:1 串行化 |
| 实测 s/iter（CPU 解码器） | 90.2 | ~360 | 4.0× |
| 实测 iter/s（CPU 解码器） | 0.0111 | 0.00277 | 4.0× |
| 实测 s/iter（GPU 解码器） | — | ~6.4 | — |
| 单卡显存占用 | ~10.5 GiB/卡 | ~36.4 GiB | — |

## 2026-08-06 更新：安装 onnxruntime-gpu 后的实测

- 确认此前 **onnxruntime / onnxruntime-gpu 均未安装**，`OnnxPiPlusH0WDecoder` 一直走 CPU `ReferenceEvaluator` fallback（`piplus_h0w_onnx_decoder.py:65-74`）。
- 已在 `pyproject.toml` 增加 `onnxruntime-gpu>=1.20.0`（锁定 1.23.2），`uv lock` + `uv sync` 完成，`.venv` 内 `CUDAExecutionProvider` 可用。
- 快速验证：`OnnxPiPlusH0WDecoder` 实例化后 `get_providers()` 返回 `['CUDAExecutionProvider', 'CPUExecutionProvider']`，CPU fallback 未启用。
- 重启单卡任务（GPU 1，PID 753371，同样命令）后实测：
  - 180 s 窗口：iteration 10117 → 10145 = 28 迭代 → **6.43 s/iter**
  - 120 s 窗口：iteration 10146 → 10164 = 18 迭代 → **6.67 s/iter**
  - 相比 CPU 解码器单卡（~360 s/iter）提速 **~56×**；进程 CPU 从 ~800% 降至 ~112%，GPU 1 利用率升至 61-78%。
- 该任务已停止后再次重启（PID 753371 为当前活动进程，见下节），继续从 checkpoint_10100 训练至 20000。

## 分析

- 单卡将 4 卡并行的工作（仿真 4096 envs、640 个 PPO 更新、每步 CPU 解码 4096 次）全部串行化，理论上限就是 4 卡并行速度的 1/4，实测恰好接近该上限。
- 因此**瓶颈不是多卡通信**（per-param all_reduce 在本规模下影响有限），而是**单卡串行总量**与 **CPU 解码器**。
- 单卡真正可能变快的场景：4096 envs 全部跑在 GPU 解码器 + 减少 PPO 更新次数（如 `--minibatch-size 4096` 使每迭代仅 160 次更新），或 1024 envs 小规模 smoke/调试。

## 建议

1. **正式训练采用单卡 + GPU 解码器**：单卡 4096 envs 已实测 ~6.4-6.7 s/iter，与 4-GPU CPU 解码器时代（90 s/iter）相比快 ~14×，且只占用 1 张 GPU（当前 GPU 1），其余卡可跑其他任务。
2. 若需进一步提速，可调大 `--minibatch-size` 减少每迭代 PPO 更新次数（640→160），或恢复 4-GPU 并行（此时多卡通信收益才体现出来）。
3. per-param `all_reduce`（`humanoidverse/distributed.py`）优化优先级仍然较低。
4. 单卡任务继续训练至目标 20000 迭代，checkpoint 每 100 迭代保存于
   `runs/amp_stage2_piplus_h0w/profile_a_1gpu_4096env_20260806/`。

## 日志与产物

- 4-GPU 日志：`runs/amp_stage2_piplus_h0w/profile_a_4gpu_4096env_20260805/continuation_10000_to_20000.log`（停止于 ~10186）
- 单卡日志：`runs/amp_stage2_piplus_h0w/profile_a_1gpu_4096env_20260806/continuation_10100_to_20000.log`
- 恢复起点：`profile_a_4gpu_4096env_20260805/checkpoint_10100.pt`

## 2026-08-06：单卡活动运行的 Profile A 权重启动凭据

本节记录当前活动的 1-GPU 续训运行，不把它与已停止的 4-GPU 运行混为一谈。当前工作目录为 `/root/autodl-tmp/chenyupeng/UFO_HT`，worker PID 为 `753371`，进程启动时间为 `2026-08-06 19:11:51 +0800`。worker 命令使用 AMP Profile A，work-dir 为 `runs/amp_stage2_piplus_h0w/profile_a_1gpu_4096env_20260806`，并从 `checkpoint_10100.pt` 恢复。

启动顺序证据如下：

- 源文件 `humanoidverse/amp_stage2_piplus_22dof.py` 的 mtime 为 `2026-08-06 19:11:49.089 +0800`。
- worker 于 `2026-08-06 19:11:51 +0800` 启动，晚于该源文件 mtime。
- 随后写入的 `.pyc` 时间为 `2026-08-06 19:12:12.837 +0800`，其中包含浮点常量 `3.7`。
- 运行配置写入时间为 `2026-08-06 19:12:14.049 +0800`。

因此，对这次运行的结论是：`linvel_exp=3.7` **loaded at startup**。启动后再编辑源文件不会追溯改变已经启动的 Python worker。运行 metadata 没有序列化单独的 reward-term map，所以该结论来自启动时的源文件、worker 时间边界和 `.pyc` 常量证据，而不是来自 metadata 中的逐项 reward 映射。

下一次对比所需的指标仍属于待补充的 instrumentation，不是当前日志已经记录的数值：

- 各启用 locomotion reward term 的逐项均值，至少包括 `linvel_exp`、`linvel_projection`、`angvel_z_exp`、`body_upright` 和动作平滑惩罚；Profile A 的 AMP/prior 应持续记录为零。
- `vx`、`vy`、`yaw_rate` 的 MAE。下一轮对比的候选协议是 `|vx 误差| <= 0.15 m/s`、`|vy 误差| <= 0.15 m/s`、`|yaw_rate 误差| <= 0.20 rad/s`，并以三个条件同时满足作为 combined velocity success rate 的阈值；这不是当前仓库已经固化的成功标准。
- per-axis velocity success rate 和 combined velocity success rate，按候选阈值分别统计。
- `termination_rate`、`body_upright`，以及 action-rate penalties。

这些指标用于后续速度和质量对比；目前不要据此宣称任何 speed 或 quality improvement。
