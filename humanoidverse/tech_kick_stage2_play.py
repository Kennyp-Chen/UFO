"""Validate a TeCH Kick Stage2 encoder and record the physical ball scene."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import mediapy as media
import numpy as np
import torch

from humanoidverse.agents.envs.tech_kick_mjlab import DEFAULT_BALL_MASS, DEFAULT_BALL_RADIUS, build_tech_kick_env
from humanoidverse.agents.load_utils import load_model_from_checkpoint_dir
from humanoidverse.tasks.kick import KickCommandState, KickObservationHistory, flatten_kick_encoder_observation
from humanoidverse.tech_kick_stage2 import CommandEncoderPolicy, _to_torch_obs, freeze_tech, tech_action
from humanoidverse.utils.torch_utils import my_quat_rotate

FOOT_ENCODINGS = {
    "left": (1.0, 0.0),
    "right": (0.0, 1.0),
    "either": (1.0, 1.0),
}


def stage2_checkpoints(model_folder: Path) -> list[Path]:
    checkpoints: list[tuple[int, Path]] = []
    for path in model_folder.glob("checkpoint_*.pt"):
        match = re.fullmatch(r"checkpoint_(\d+)\.pt", path.name)
        if match:
            checkpoints.append((int(match.group(1)), path))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint_<iteration>.pt files in {model_folder}")
    return [path for _, path in sorted(checkpoints)]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-folder", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--checkpoint-selection", choices=("first", "latest"), default="latest")
    parser.add_argument("--tech-checkpoint", type=Path, default=None)
    parser.add_argument("--robot-config", type=Path, default=None)
    parser.add_argument("--expert-dataset", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--target-ball-velocity", type=float, nargs=3, default=(1.5, 0.0, 0.0), metavar=("VX", "VY", "VZ"))
    parser.add_argument("--kick-foot", choices=tuple(FOOT_ENCODINGS), default="left")
    parser.add_argument("--ball-distance", type=float, default=0.30)
    parser.add_argument("--ball-angle", type=float, default=0.0)
    parser.add_argument("--max-steps", type=int, default=250)
    parser.add_argument("--max-episode-length-s", type=float, default=5.0)
    parser.add_argument("--render-size", type=int, default=720)
    parser.add_argument("--camera-distance", type=float, default=2.5)
    parser.add_argument("--camera-azimuth", type=float, default=135.0)
    parser.add_argument("--camera-elevation", type=float, default=-18.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--log-every-steps", type=int, default=50)
    return parser.parse_args()


def _metadata_path(metadata: dict[str, object], key: str, override: Path | None) -> Path:
    path = override if override is not None else Path(str(metadata[key]))
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Missing {key}: {resolved}")
    return resolved


def _resolve(args: argparse.Namespace) -> tuple[Path, Path, Path, Path, dict[str, object]]:
    model_folder = args.model_folder.expanduser().resolve()
    metadata_path = model_folder / "config.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing Stage2 metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("task") != "tech_kick_stage2":
        raise ValueError(f"Expected tech_kick_stage2 metadata, got {metadata.get('task')!r}")
    if args.checkpoint is not None:
        checkpoint = args.checkpoint.expanduser().resolve()
    else:
        checkpoints = stage2_checkpoints(model_folder)
        checkpoint = checkpoints[0] if args.checkpoint_selection == "first" else checkpoints[-1]
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing Stage2 checkpoint: {checkpoint}")
    return (
        checkpoint,
        _metadata_path(metadata, "tech_checkpoint", args.tech_checkpoint),
        _metadata_path(metadata, "robot_config", args.robot_config),
        _metadata_path(metadata, "expert_dataset", args.expert_dataset),
        metadata,
    )


def play(args: argparse.Namespace) -> Path:
    if args.max_steps <= 0 or args.render_size <= 0 or args.ball_distance <= 0.0:
        raise ValueError("max-steps, render-size, and ball-distance must be positive")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    elif device.type != "cpu":
        raise ValueError(f"Unsupported device: {device}")

    checkpoint_path, tech_checkpoint, robot_config, expert_dataset, metadata = _resolve(args)
    model_folder = args.model_folder.expanduser().resolve()
    output = (
        args.output.expanduser().resolve()
        if args.output is not None
        else model_folder / "validation" / f"{checkpoint_path.stem}_{args.kick_foot}.mp4"
    )
    output.parent.mkdir(parents=True, exist_ok=True)

    env, _ = build_tech_kick_env(
        device=str(device),
        robot_config=str(robot_config),
        expert_dataset=str(expert_dataset),
        num_envs=1,
        seed=args.seed,
        max_episode_length_s=args.max_episode_length_s,
        ball_radius=float(metadata.get("ball_radius", DEFAULT_BALL_RADIUS)),
        ball_mass=float(metadata.get("ball_mass", DEFAULT_BALL_MASS)),
        disable_domain_randomization=True,
        render_mode="rgb_array",
        render_size=args.render_size,
        camera_distance=args.camera_distance,
        camera_azimuth=args.camera_azimuth,
        camera_elevation=args.camera_elevation,
    )
    env._env.set_ball_reset_distribution(
        radius_range=(args.ball_distance, args.ball_distance),
        angle_range=(args.ball_angle, args.ball_angle),
        speed_range=(0.0, 0.0),
    )
    model = load_model_from_checkpoint_dir(str(tech_checkpoint), device=device.type, strict=False)
    freeze_tech(model)
    observation, _ = env.reset(to_numpy=False)
    observation_t = _to_torch_obs(observation, device)

    command_state = KickCommandState(1, device)
    target_velocity_b = torch.tensor(args.target_ball_velocity, device=device, dtype=torch.float32).unsqueeze(0)
    command_state.target_velocity_w[:] = my_quat_rotate(env._env.base_quat, target_velocity_b)
    command_state.foot_encoding[:] = torch.tensor(FOOT_ENCODINGS[args.kick_foot], device=device)
    command = command_state.command(env._env.base_quat)
    history = KickObservationHistory(1, int(metadata.get("kick_history_length", 8)), device)
    env_ids = torch.arange(1, device=device)
    history.reset(env_ids, command, env._env.ball_position_b, env._env.ball_linear_velocity_b)

    encoder_input = flatten_kick_encoder_observation(observation_t, history)
    encoder_arch = model.cfg.archi.goal_encoder
    policy = CommandEncoderPolicy(
        encoder_input.shape[-1],
        int(metadata["z_dim"]),
        hidden_dim=int(encoder_arch.hidden_dim),
        hidden_layers=int(encoder_arch.hidden_layers),
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    policy.load_state_dict(checkpoint["policy"])
    policy.eval()

    frames = [np.asarray(env.render())]
    fps = 1.0 / float(env._env.dt)
    contact_steps = 0
    peak_ball_speed = 0.0
    resets = 0
    try:
        for step in range(args.max_steps):
            with torch.inference_mode():
                observation_t = _to_torch_obs(observation, device)
                encoder_input = flatten_kick_encoder_observation(observation_t, history)
                raw_z = policy.deterministic_z(encoder_input)
                action, _ = tech_action(model, observation_t, raw_z)
            observation, _reward, terminated, truncated, _info = env.step(action, to_numpy=False)
            done = terminated | truncated
            command = command_state.command(env._env.base_quat)
            history.append(command, env._env.ball_position_b, env._env.ball_linear_velocity_b)
            if torch.any(done):
                resets += int(done.sum())
                command_state.target_velocity_w[:] = my_quat_rotate(env._env.base_quat, target_velocity_b)
                command = command_state.command(env._env.base_quat)
                history.reset(env_ids, command, env._env.ball_position_b, env._env.ball_linear_velocity_b)

            contact = bool(torch.any(env._env.foot_ball_contact_force > 0.1))
            contact_steps += int(contact)
            ball_speed = float(torch.linalg.vector_norm(env._env.ball_lin_vel_w[0]).detach().cpu())
            peak_ball_speed = max(peak_ball_speed, ball_speed)
            frames.append(np.asarray(env.render()))
            if step == 0 or (args.log_every_steps > 0 and (step + 1) % args.log_every_steps == 0):
                ball_position = env._env.ball_position_b[0].detach().cpu().numpy().round(3).tolist()
                print(
                    f"[INFO] step={step + 1} ball_position_b={ball_position} ball_speed={ball_speed:.3f} "
                    f"contact={contact} reset={bool(done.any())}",
                    flush=True,
                )
    finally:
        env.close()

    media.write_video(str(output), frames, fps=fps)
    print(
        f"[INFO] Saved validation video: {output} frames={len(frames)} "
        f"contact_steps={contact_steps} peak_ball_speed={peak_ball_speed:.3f} resets={resets}",
        flush=True,
    )
    return output


def main() -> None:
    play(_parse_args())


if __name__ == "__main__":
    main()
