"""Play a 22DoF PiPlus H0W command-encoder Stage2 checkpoint with MP4 capture.

This player reuses the HT_BFM legacy backend (HumanoidVerseIsaacConfig +
LeggedRobotBase semantics) because the Stage2 checkpoints are trained under that
contract: ``normalize_action_to=32``, legacy reset pose and actuator path. The
UFO_HT MJLab playback backend is not behaviorally compatible with these
checkpoints (see ``speed_stage2_legacy_play.py`` for the speed task).

The module runs the legacy playback logic in-process: it prepends the HT_BFM
repository to ``sys.path`` and drops any already-loaded ``humanoidverse``
package so that all imports resolve against the HT_BFM tree, then renders qpos
with the same offscreen MuJoCo renderer used by the legacy ``speed_stage2_play``.

Run headless with EGL on an idle GPU, e.g.::

    CUDA_VISIBLE_DEVICES=3 MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=3 \
    uv run python -m humanoidverse.command_encoder_play_legacy \
      --checkpoint runs/amp_stage2_piplus_h0w/profile_a_1gpu_4096env_20260806/checkpoint_10300.pt \
      --output /tmp/command_encoder.mp4
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Final

PROJECT_ROOT: Final = Path(__file__).resolve().parents[1]
DEFAULT_LEGACY_REPO: Final = Path(os.environ.get("UFO_HT_LEGACY_REPO", PROJECT_ROOT.parent / "HT_BFM"))
DEFAULT_DECODER_PATH: Final = (
    PROJECT_ROOT / "model/piplus_h0w_bfm/decoder"
    / "bfmzero-piplus-h0w-isaac-20260629_214205/exported/FBcprAuxModel.onnx"
)
DEFAULT_BFM_MODEL: Final = PROJECT_ROOT / "model/piplus_h0w_bfm/model.safetensors"
DEFAULT_ROBOT_CONFIG: Final = DEFAULT_LEGACY_REPO / "humanoidverse/config/robot/piplus/PiPlus_S_12L8A0G2H0W.yaml"
DEFAULT_EXPERT_DATASET: Final = DEFAULT_LEGACY_REPO / "humanoidverse/data/piplus_h0w_lafan/piplus_h0w_lafan_10s-clipped.pkl"
DEFAULT_DECODER_FACTORY: Final = "humanoidverse.piplus_h0w_onnx_decoder:load_decoder"

AMP_STAGE2_TASK: Final = "amp_stage2_piplus_22dof"
COMMAND_LOW: Final = (-0.5, -0.5, -1.0)
COMMAND_HIGH: Final = (1.0, 0.5, 1.0)


def _resolved_file(path: Path, name: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{name} does not exist: {resolved}")
    return resolved


def _resolved_directory(path: Path, name: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"{name} does not exist: {resolved}")
    return resolved


def _legacy_asset_default(repo: Path, relative_path: str) -> Path:
    return repo / relative_path


def install_legacy_imports(htbfm_repo: Path) -> None:
    """Prepend the HT_BFM repo to sys.path and drop cached humanoidverse modules.

    Stage2 playback imports the HT_BFM tree (``amp_stage2``, ``speed_stage2``,
    env builders). When this module is launched with ``python -m`` from the
    UFO_HT checkout, the ``humanoidverse`` package may already be cached for the
    local tree; removing it forces every subsequent ``humanoidverse.*`` import
    to resolve against the HT_BFM repository at ``sys.path[0]``.
    """
    repo = _resolved_directory(htbfm_repo, "HT_BFM repository")
    sys.path.insert(0, str(repo))
    for name in [name for name in tuple(sys.modules) if name == "humanoidverse" or name.startswith("humanoidverse.")]:
        del sys.modules[name]


def checkpoint_task_is_compatible(metadata: dict[str, object]) -> bool:
    return metadata.get("task") in {None, AMP_STAGE2_TASK}


def _validate_command(command: tuple[float, float, float]) -> None:
    values = tuple(float(value) for value in command)
    if len(values) != 3 or any(value < low or value > high for value, low, high in zip(values, COMMAND_LOW, COMMAND_HIGH)):
        raise ValueError(f"fixed command must be within {COMMAND_LOW} and {COMMAND_HIGH}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--htbfm-repo", type=Path, default=DEFAULT_LEGACY_REPO)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--bfm-model", type=Path, default=None)
    parser.add_argument("--decoder-path", type=Path, default=DEFAULT_DECODER_PATH)
    parser.add_argument("--decoder-factory", default=DEFAULT_DECODER_FACTORY)
    parser.add_argument("--robot-config", type=Path, default=None)
    parser.add_argument("--expert-dataset", type=Path, default=None)
    parser.add_argument("--device", default="cpu", help="Policy/decoder device; cpu matches the legacy playback path.")
    parser.add_argument("--fixed-command", type=float, nargs=3, default=(0.4, 0.0, 0.0), metavar=("VX", "VY", "WZ"))
    parser.add_argument("--command-smoothing", type=float, default=0.15)
    parser.add_argument("--max-steps", type=int, default=250)
    parser.add_argument("--max-episode-length-s", type=float, default=20.0)
    parser.add_argument("--output", type=Path, required=True, help="MP4 output path.")
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--render-size", type=int, default=720)
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    _validate_command(args.fixed_command)
    if args.max_steps <= 0:
        raise ValueError("--max-steps must be positive")
    if args.fps <= 0:
        raise ValueError("--fps must be positive")
    if args.render_size <= 0:
        raise ValueError("--render-size must be positive")
    if not 0.0 < args.command_smoothing <= 1.0:
        raise ValueError("--command-smoothing must be in (0, 1]")
    checkpoint_path = _resolved_file(args.checkpoint, "checkpoint")
    bfm_model = _resolved_file(args.bfm_model, "BFM model") if args.bfm_model is not None else None
    decoder_path = _resolved_file(args.decoder_path, "decoder")
    robot_config = _resolved_file(
        args.robot_config
        if args.robot_config is not None
        else _legacy_asset_default(args.htbfm_repo, "humanoidverse/config/robot/piplus/PiPlus_S_12L8A0G2H0W.yaml"),
        "robot config",
    )
    expert_dataset = _resolved_file(
        args.expert_dataset
        if args.expert_dataset is not None
        else _legacy_asset_default(args.htbfm_repo, "humanoidverse/data/piplus_h0w_lafan/piplus_h0w_lafan_10s-clipped.pkl"),
        "expert dataset",
    )
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    install_legacy_imports(args.htbfm_repo)
    # All imports below resolve against the HT_BFM repository.
    import mediapy as media
    import torch

    from humanoidverse import amp_stage2 as amp
    from humanoidverse.amp_stage2_piplus_22dof import _load_h0w_robot_contract, build_h0w_amp_env
    from humanoidverse.amp_stage2_piplus_22dof_play import _policy_shape
    from humanoidverse.speed_stage2 import _decoder_z_dim, _load_decoder, _to_torch_obs
    from humanoidverse.speed_stage2_play import MujocoQposRenderer

    device = torch.device(args.device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    metadata = checkpoint.get("metadata", {})
    if not checkpoint_task_is_compatible(metadata):
        raise ValueError(f"Checkpoint task {metadata.get('task')!r} is not {AMP_STAGE2_TASK}")
    policy_state = checkpoint.get("policy")
    if not isinstance(policy_state, dict):
        raise TypeError("Checkpoint policy state must be a mapping")
    input_dim, hidden_dim, z_dim = _policy_shape(policy_state)

    env = build_h0w_amp_env(
        device=str(device),
        expert_dataset=str(expert_dataset),
        robot_config=str(robot_config),
        num_envs=1,
        seed=args.seed,
        max_episode_length_s=args.max_episode_length_s,
        simulator="mujoco",
        disable_domain_randomization=True,
    )
    renderer = None
    writer = None
    try:
        observation, _ = env.reset(to_numpy=False, reset_to_default_pose=True)
        observation_t = _to_torch_obs(observation, device)
        commands = torch.zeros(1, 3, device=device)
        expected_input_dim = int(amp.flatten_encoder_observation(observation_t, commands).shape[-1])
        if input_dim != expected_input_dim:
            raise ValueError(f"Checkpoint expects encoder input {input_dim}, environment provides {expected_input_dim}")
        policy = amp.CommandEncoderPolicy(input_dim, z_dim, hidden_dim=hidden_dim).to(device)
        policy.load_state_dict(policy_state)
        policy.eval()
        decoder = _load_decoder(bfm_model, decoder_path, args.decoder_factory, device)
        if _decoder_z_dim(decoder) != z_dim:
            raise ValueError(f"Checkpoint z_dim={z_dim} does not match decoder z_dim={_decoder_z_dim(decoder)}")
        target = torch.tensor(args.fixed_command, dtype=torch.float32, device=device).unsqueeze(0)
        low = torch.tensor(COMMAND_LOW, device=device)
        high = torch.tensor(COMMAND_HIGH, device=device)
        if torch.any(target < low) or torch.any(target > high):
            raise ValueError(f"--fixed-command must be within {COMMAND_LOW} and {COMMAND_HIGH}")
        qpos, _ = env._get_qpos_qvel(to_numpy=True)
        robot_contract = _load_h0w_robot_contract(robot_config)
        renderer = MujocoQposRenderer(
            robot_contract.xml_path,
            render_size=args.render_size,
            expected_qpos_size=int(qpos.shape[-1]),
        )
        writer = media.VideoWriter(output, shape=(args.render_size, args.render_size), fps=args.fps)
        writer.__enter__()
        print(
            f"[INFO] checkpoint={checkpoint_path} iteration={checkpoint.get('iteration', 'unknown')} "
            f"command={args.fixed_command} device={device}",
            flush=True,
        )
        commands.zero_()
        step = 0
        while step < args.max_steps:
            commands.add_(args.command_smoothing * (target - commands))
            with torch.inference_mode():
                obs_t = _to_torch_obs(observation, device)
                features = amp.flatten_encoder_observation(obs_t, commands)
                raw_z = policy.deterministic_z(features)
                action = decoder.act(obs_t, decoder.project_z(raw_z))
            observation, _reward, terminated, truncated, _info = env.step(action, to_numpy=False)
            qpos, _ = env._get_qpos_qvel(to_numpy=True)
            frame = renderer.render_qpos(qpos[0])
            writer.add_image(frame)
            step += 1
            if step == 1 or step % 50 == 0:
                velocity = env._env.base_lin_vel[0].detach().cpu().numpy()
                yaw_velocity = float(env._env.base_ang_vel[0, 2].detach().cpu())
                print(
                    f"[INFO] step={step} velocity="
                    f"{[round(float(velocity[0]), 3), round(float(velocity[1]), 3), round(yaw_velocity, 3)]} "
                    f"terminated={bool(terminated.any())} truncated={bool(truncated.any())}",
                    flush=True,
                )
    finally:
        if writer is not None:
            writer.close()
        if renderer is not None:
            renderer.close()
        env.close()
    print(f"[INFO] Saved MP4: {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
