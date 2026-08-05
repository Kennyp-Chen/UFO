# PiPlus H0W Stage2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the PiPlus H0W 22DoF speed PPO and AMP stage-2 algorithms to UFO's MJLab training runtime.

**Architecture:** Keep the new algorithm modules separate from TeCH-kick. A shared decoder/PPO/environment layer owns only the frozen H0W contract; the speed and AMP entrypoints own their distinct rewards, checkpoints, and playback validation.

**Tech Stack:** Python 3.10, PyTorch, MuJoCo/MJLab, ONNX reference evaluator, `unittest`.

## Global Constraints

- Training, evaluation, and DDP must only use physical GPUs 0-3.
- Do not commit checkpoints, motion data, or generated run directories.
- Preserve the existing uncommitted `pyproject.toml` and `uv.lock` changes.
- Each completed, independently tested migration unit is an atomic commit.

---

### Task 1: Shared H0W Contract

**Files:**
- Create: `humanoidverse/piplus_h0w_onnx_decoder.py`
- Create: `humanoidverse/piplus_h0w_stage2.py`
- Test: `tests/test_piplus_h0w_stage2.py`

- [ ] Write tests for the 22DoF/616D decoder and PPO GAE contracts.
- [ ] Verify tests fail because the modules do not yet exist.
- [ ] Implement the minimal frozen decoder adapter, command encoder, GAE, PPO update, and checkpoint helpers.
- [ ] Run the new unit test and commit the tested files.

### Task 2: Speed PPO Locomotion

**Files:**
- Create: `humanoidverse/piplus_h0w_locomotion.py`
- Create: `humanoidverse/speed_stage2.py`
- Create: `humanoidverse/speed_stage2_play.py`
- Modify: `tests/test_piplus_h0w_stage2.py`

- [ ] Write failing tests for speed reward and speed checkpoint task isolation.
- [ ] Implement the MJLab locomotion builder, speed PPO entrypoint, and playback entrypoint.
- [ ] Run targeted tests, then a one-iteration GPU smoke using explicit local assets.
- [ ] Commit the tested migration unit.

### Task 3: AMP Profiles

**Files:**
- Create: `humanoidverse/amp_stage2_piplus_22dof.py`
- Create: `humanoidverse/amp_stage2_piplus_22dof_play.py`
- Modify: `tests/test_piplus_h0w_stage2.py`

- [ ] Write failing tests for the 194D feature contract, Profile A isolation, Profile B data guard, and playback task isolation.
- [ ] Implement expert feature construction, discriminator, MimicLite locomotion rewards, AMP profile training, and playback.
- [ ] Run targeted tests, Profile A GPU smoke, and Profile B missing-data dry run.
- [ ] Commit the tested migration unit.

### Task 4: Documentation And Final Verification

**Files:**
- Create: `docs/piplus_h0w_stage2_zh.md`

- [ ] Document asset arguments, profile semantics, checkpoints, playback, and GPU 0-3 commands.
- [ ] Run relevant Ruff checks, complete unit suite, speed PPO smoke, and pure locomotion smoke.
- [ ] Commit documentation and report exact verification evidence.
