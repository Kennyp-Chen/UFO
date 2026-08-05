# PiPlus H0W 22DoF Stage2 Migration Design

## Goal

Move the PiPlus H0W 22DoF second-stage fine-tuning algorithms from the sibling
`HT_BFM` repository into UFO's MJLab runtime without changing the existing
TeCH-kick path.

## Scope

The migration exposes two independent command-encoder training entries:

- `humanoidverse.speed_stage2`: frozen 22DoF ONNX BFM decoder plus PPO velocity
  tracking. It has no AMP discriminator or expert-style reward.
- `humanoidverse.amp_stage2_piplus_22dof`: frozen decoder plus PPO, with two
  named reward profiles. Profile A is the pure MimicLite locomotion baseline;
  Profile B adds the 194D AMP discriminator and latent prior when a dedicated
  walking/run-with-stand expert dataset is supplied.

Both entries write distinct `task` metadata and have dedicated playback
commands. Playback must reject a checkpoint from the other task.

## Architecture

Shared components live in small H0W-specific helpers: an ONNX decoder adapter,
PPO command encoder/GAE implementation, and an MJLab locomotion environment
builder. The environment uses the tracked `configs/robots/piplus_h0w.yaml`
robot contract and enables `locomotion_mode`, so reset is not tied to a motion
reference. The frozen decoder receives the existing actor observation contract
`state(50) + last_action(22) + history_actor(288) + projected_z(256)`.

The decoder adapter prefers `onnxruntime` when installed and otherwise uses
the already-required `onnx.reference` evaluator. This keeps the migration
usable in the locked UFO environment without altering the existing dependency
lock. The reference evaluator is a correctness fallback, not a high-throughput
training backend.

## Contracts

- Robot action dimension: 22; H0W decoder input/output: 616/22; latent: 256D,
  normalized to L2 norm 16 before decoder inference.
- PPO encoder input: command(3) + actor observations, therefore 363 values for
  the tracked H0W configuration.
- AMP online/expert feature: local root velocity(3), five local key bodies(15),
  and eight frames of joint positions(176), totaling 194 values.
- Profile A never constructs an AMP discriminator, even when stale AMP command
  line flags are supplied. Profile B requires dedicated filtered expert data
  and fails before rollout when it is absent.
- DDP uses the project's `humanoidverse.distributed` helpers. GPU visibility is
  constrained by `AGENTS.md` to physical devices 0-3.

## Verification

Unit tests cover decoder input validation, PPO reward/GAE boundaries, numeric
checkpoint selection, the 194D AMP feature contract, reward-profile isolation,
and task-specific playback compatibility. Runtime verification runs one
iteration/short rollout for speed PPO and Profile A pure locomotion on GPU 0,
using the source repository's local decoder and motion assets only as explicit
test inputs. Profile B is dry-run tested for its missing-data guard because the
dedicated walking/run-with-stand artifact is absent in both repositories.
