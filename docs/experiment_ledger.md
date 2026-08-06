# UFO_HT 实验台账

本文件是 UFO_HT 实验进度的权威、追加式记录。每次实验进度事件都必须追加一条带时间戳的记录，不得覆写历史结果。记录命令、配置变化、workdir、checkpoint lineage、硬件和验证来源。checkpoint、日志、运行输出等生成物只保留在对应运行目录，不提交到 Git。

## 记录规则

| 字段 | 要求 |
| --- | --- |
| 时间戳 | 使用本地时区并精确到秒 |
| 状态 | planned、active、completed、blocked、stopped 或 snapshot |
| 实验 ID | 稳定且可搜索，例如 `profile_c_formal` |
| 路径 | 记录 workdir、checkpoint 和日志路径，必要时注明外部来源 |
| lineage | 说明 resume checkpoint、目标迭代和是否允许恢复 |
| 变化 | 记录命令行、配置、backend、reward、observation 或 architecture 变化 |
| 证据 | 记录测试、dry-run、smoke、日志或 metadata 来源；snapshot 指标不是最终结果 |

后续更新必须追加在文件末尾。更正历史信息时追加 amendment，并保留原记录和原时间戳。

## Repository contract baseline

H0W Stage2 的冻结合同如下：action 22；state 50 = relative dof_pos 22 + dof_vel 22 + projected_gravity 3 + base_ang_vel 3；history 288 = 4 frames of base_ang_vel/projected_gravity/dof_pos/dof_vel/actions；command encoder 363 = command 3 + state 50 + last_action 22 + history 288；decoder 616 -> 22 = state 50 + last_action 22 + history 288 + latent 256。latent projection norm 16，velocity scale 0.05，joint pos scale 1.0。relative joint position 是 `dof_pos - (default_dof_pos + default_dof_pos_offset)`，default offset randomization 开启，范围为 `[-0.02, 0.02]`。

证据路径：`humanoidverse/agents/envs/humanoidverse_mjlab.py:_raw_actor_obs/get_observation`、`humanoidverse/piplus_h0w_stage2.py:flatten_encoder_observation/_encoder_input_scale/CommandEncoderPolicy`、`humanoidverse/piplus_h0w_onnx_decoder.py:build_actor_observation/project_z`、`humanoidverse/config/obs/bfm_zero_obs.yaml`、`humanoidverse/config/domain_rand/domain_rand.yaml`、`humanoidverse/piplus_h0w_unitree_velocity.py:UnitreeVelocityLocomotionRewardState`、`humanoidverse/amp_stage2_piplus_22dof.py` 的 profile registration/metadata，以及 `tests/test_piplus_h0w_stage2_common.py` 和 `tests/test_amp_stage2_piplus_22dof.py`。

## Experiment index

| Experiment ID | Status | Workdir or source | Checkpoint lineage / result |
| --- | --- | --- | --- |
| `h0w_speed_continuation_9800_19800` | historical | `runs/speed_stage2_piplus_h0w_resume_9800_to_19800/` | checkpoint 9800 to target 19800 |
| `profile_a_4gpu_benchmark` | stopped | `runs/amp_stage2_piplus_h0w/profile_a_4gpu_4096env_20260805` | stopped around iteration 10186; CPU decoder era |
| `profile_a_1gpu_continuation` | active | `runs/amp_stage2_piplus_h0w/profile_a_1gpu_4096env_20260806` | PID 753371; resumed checkpoint_10100, target 20000; snapshot around 12155 |
| `profile_c_smoke` | completed | `/tmp/ufo_smoke_profile_c` | checkpoint_1/config.json |
| `profile_c_formal` | active snapshot | `runs/amp_stage2_piplus_h0w/profile_c_1gpu_4096env_20260806` | PID 885173; checkpoint_600.pt exists |
| `onnxruntime_gpu_decoder` | completed | `docs/amp_stage2_4gpu_vs_1gpu_speed.md` | locked 1.23.2, CUDA provider, about 56x decoder speedup |
| `profile_b_amp_reference` | blocked/planned | no run workdir | dedicated `piplus_h0w_locomotion_run_with_stand.pkl` absent; never substitute generic motion data |

## Detailed records

### 2026-08-06, `h0w_speed_continuation_9800_19800`, historical

HT_BFM IsaacSim produced `checkpoint_9800.pt`; UFO_HT continued it with `humanoidverse.speed_stage2` and MJLab to absolute target iteration 19800. The recorded workdir is `runs/speed_stage2_piplus_h0w_resume_9800_to_19800/`. The documented configuration was 4096 envs, rollout 32, PPO epochs 4, minibatch 8192, learning rate `5e-5`, save every 100, seed 4728. GPU decoder execution used `onnxruntime-gpu` 1.23.2. See [`docs/piplus_h0w_stage2_zh.md`](piplus_h0w_stage2_zh.md).

### 2026-08-05, `profile_a_4gpu_benchmark`, stopped

Workdir `runs/amp_stage2_piplus_h0w/profile_a_4gpu_4096env_20260805`; the four GPU benchmark/continuation stopped around iteration 10186. This is the CPU decoder era and must not be compared with later GPU decoder timings without that provenance.

### 2026-08-06, `profile_a_1gpu_continuation`, active

PID 753371, workdir `runs/amp_stage2_piplus_h0w/profile_a_1gpu_4096env_20260806`, resumed checkpoint_10100, target 20000. The inspection snapshot log was around iteration 12155. This is a status snapshot, not a final result.

### 2026-08-06, `profile_c_smoke`, completed

The opt-in Unitree velocity reward smoke run used `/tmp/ufo_smoke_profile_c`, generating `checkpoint_1.pt` and `config.json`. Recorded metrics were `effective_learning_rate=0.001`, `reward_mean=0.037`, `locomotion_reward_mean=0.0378`, and `termination_rate=0.0`. Metadata confirms the pinned source, omitted terms, PPO settings, effective reward, and profile identity.

### 2026-08-06 22:53:52 +0800, `profile_c_formal`, snapshot

This is a read-only live snapshot. Future updates must append a new timestamped entry. PID 885173 was launched with physical GPU 2 and therefore uses logical `cuda:0`; workdir is `runs/amp_stage2_piplus_h0w/profile_c_1gpu_4096env_20260806`. The command used 4096 envs, target 20000 iterations, history 8, seed 4728, command stand 0.05, turn 0.20, smoothing 0.02, episode 20 s, save every 100, and no resume. `checkpoint_600.pt` exists and the log reached approximately iteration 675. Snapshot metrics were reward_mean 0.01897, locomotion_reward_mean 0.01916, approx_kl 0.01274, effective learning rate `1e-5`, termination rate 0.000865, and value loss 0.1576. Metadata confirms Profile C Unitree PPO and reward source/omitted terms. These are not final metrics.

Launch command and immutable assets:

```bash
CUDA_VISIBLE_DEVICES=2 MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=2 \
UFO_CACHE_DIR=/root/autodl-tmp/chenyupeng/UFO_HT/cache \
uv run --no-sync python -u -m humanoidverse.amp_stage2_piplus_22dof \
  --reward-profile c_unitree_velocity_h0w_compat \
  --gpu-ids single --device cuda:0 \
  --bfm-model /root/autodl-tmp/hejunfu/projects/UFO-main/checkpoints/remote_tech_piplus_2h0w_8gpu_seed4728/model/model.safetensors \
  --decoder-path /root/autodl-tmp/chenyupeng/HT_BFM/model/piplus_h0w_bfm/decoder/bfmzero-piplus-h0w-isaac-20260629_214205/exported/FBcprAuxModel.onnx \
  --robot-config /root/autodl-tmp/chenyupeng/UFO_HT/configs/robots/piplus_h0w.yaml \
  --motion-dataset /root/autodl-tmp/chenyupeng/HT_BFM/humanoidverse/data/piplus_h0w_lafan/piplus_h0w_lafan_10s-clipped.pkl \
  --work-dir /root/autodl-tmp/chenyupeng/UFO_HT/runs/amp_stage2_piplus_h0w/profile_c_1gpu_4096env_20260806 \
  --iterations 20000 --num-envs 4096 --history-length 8 \
  --command-stand-prob 0.05 --command-turn-prob 0.20 --command-smoothing 0.02 \
  --max-episode-length-s 20.0 --save-every 100 --seed 4728
```

The corresponding metadata is `runs/amp_stage2_piplus_h0w/profile_c_1gpu_4096env_20260806/config.json`; it records the H0W 22DoF/616->22/363/256 contract, Unitree source commit, PPO profile, effective reward map, and omitted terms. The external BFM, decoder, and motion paths above are machine-specific provenance, not portable repository defaults.

## Pinned Unitree observation semantics

The pinned source is `unitreerobotics/unitree_rl_mjlab` commit `1425b15f73bd4095f0df53709d7c389c3eb9e790`, with `mjlab==1.2.0` and `rsl-rl-lib==5.0.1`. Unitree actor terms include base_ang_vel, projected_gravity, command, phase, `joint_pos` via `mdp.joint_pos_rel`, `joint_vel` via `mdp.joint_vel_rel`, actions, and rough-terrain height_scan. Critic includes actor terms plus base_lin_vel, clean height_scan, foot_height, foot_air_time, foot_contact, and foot_contact_forces. Flat G1 removes height_scan and terrain curriculum. No exact flattened dimension is asserted without direct verification.

The environment exposes `privileged_state`, but standalone H0W Stage2 does not feed it into PPO value prediction. `CommandEncoderPolicy` has one trunk/value head and value uses the same 363 actor features, so a Unitree-style asymmetric critic is not currently implemented. H0W keeps relative actor joint positions and must not claim Unitree uses absolute joint positions.

## Observation decision and future ablation protocol

### 2026-08-06 22:53:52 +0800, decision

Keep the H0W actor, CommandEncoder, history, and decoder observation unchanged for current Profile C. Do not change relative actor joint positions to absolute. Relative positions are default-pose centered and consistent with H0W default pose, randomized offset, action/decoder contract, and reset/history semantics; pinned Unitree also uses relative joint position terms. Do not force Unitree terrain or foot critic terms into H0W when sensors/body contracts differ. A future asymmetric critic should prefer H0W available clean privileged features and explicitly version its critic contract. Privileged critic features may improve value estimation/sample efficiency, but are not required for actor correctness; current run is stable enough that no mid-run change is justified.

Future ablation protocol:

1. Baseline/control keeps current Profile C actor and 363/616 contracts. Each arm gets a fresh workdir and metadata containing exact command, config, hardware, seed, source commit, decoder checksum, and checkpoint lineage.
2. Candidate A adds absolute joint position only to a newly designed critic. Candidate B adds only H0W-available clean privileged features. Do not copy Unitree terrain or foot terms without matching sensors and body semantics.
3. Add TDD-first tests in `tests/test_piplus_h0w_stage2_common.py` and `tests/test_amp_stage2_piplus_22dof.py` for actor invariance, critic dimensions, metadata versioning, and incompatible resume rejection.
4. Compare identical seeds and command distributions. Report reward_mean, locomotion_reward_mean, termination rate, approximate KL, effective learning rate, value loss, episode length, throughput, and short replay stability.
5. No current checkpoint can be resumed after changing actor observation semantics or critic architecture. Such a change starts a fresh experiment and checkpoint lineage.

## Append-only updates

### 2026-08-06 22:53:52 +0800, ledger initialization

This initial entry consolidates the repository baseline, historical/current records, Profile C snapshot, and observation decision. Amendments and future progress events append below this line.

### 2026-08-06 23:03:54 +0800, `profile_c_formal`, progress snapshot

PID 885173 remained alive on the user-assigned physical GPU 2. The log reached iteration 802 and `checkpoint_800.pt` exists under the formal workdir. Snapshot metrics at this timestamp were reward_mean 0.01770, locomotion_reward_mean 0.01791, approx_kl 0.01219, effective learning rate `1e-5`, termination rate 0.001078, and value loss 0.2046. This is an append-only progress update, not a final evaluation; future checkpoint or process milestones must append another timestamped record.

### 2026-08-06 23:15:39 +0800, `profile_c_formal`, observation decision amendment and progress snapshot

The pinned Unitree flat configuration was checked against the exact commit before deciding whether H0W should copy its foot observations. In the flat task, Unitree removes `terrain_scan` from both actor and critic, so there is no `height_scan` observation. Unitree still defines critic-only foot terms and keeps `joint_pos`/`joint_vel` relative through `mdp.joint_pos_rel`/`mdp.joint_vel_rel`. This corrects any earlier interpretation that Unitree's joint observations were absolute.

The current H0W robot can expose the following flat-style critic information without changing the 363-dimensional CommandEncoder or 616-dimensional frozen decoder: clean `base_lin_vel` (already available as `core.base_lin_vel`), two foot heights from `core.body_pos[:, feet_indices, 2]`, two foot horizontal velocities from `core.body_vel[:, feet_indices, :2]`, two contact flags and two force vectors from `core.contact_forces[:, feet_indices]`. The current contact sensor is a body-wide `body_contact` sensor with `found` and `force` fields, mapped back into the H0W body array; it is not a Unitree foot-only `feet_ground_contact` sensor. `foot_height` is therefore directly implementable with body-position semantics, while `foot_contact`/`foot_contact_forces` are already available with a broader sensor contract. `foot_air_time` requires a new per-foot temporal counter; it is not currently emitted as an observation. A terrain scan is not needed for the Unitree flat comparison and is not present in H0W's current actor/critic observation path.

Decision: do not add these features to the active Profile C run. If a future asymmetric critic is tested, use H0W-native clean foot/base features first, keep the actor and decoder contract unchanged, explicitly version the critic input, and compare against a no-privileged-critic control. Do not copy rough-terrain `height_scan` or claim exact Unitree sensor equivalence without matching the sensor/body semantics.

The live Profile C process PID 885173 remained active on physical GPU 2. The log reached iteration 992 and `checkpoint_900.pt` exists. Snapshot metrics were reward_mean 0.01875, locomotion_reward_mean 0.01897, approx_kl 0.01104, effective learning rate `2.25e-5`, termination rate 0.000936, and value loss 0.1536. These are progress snapshots, not final evaluation results.
