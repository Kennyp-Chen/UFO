"""Shared bounded playback for PiPlus H0W stage-2 checkpoints."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import mediapy as media
import mujoco
import numpy as np
import torch

from humanoidverse.piplus_h0w_locomotion import build_h0w_locomotion_env
from humanoidverse.piplus_h0w_onnx_decoder import load_decoder
from humanoidverse.piplus_h0w_stage2 import CommandEncoderPolicy, flatten_encoder_observation, latest_checkpoint, to_torch_observation


def load_playback_checkpoint(
    *,
    model_folder: Path,
    checkpoint: Path | None,
    expected_task: str,
    device: torch.device,
) -> tuple[Path, dict[str, Any], Mapping[str, torch.Tensor]]:
    """Load a task-scoped checkpoint, defaulting to its numeric latest version."""
    checkpoint_path = checkpoint.expanduser().resolve() if checkpoint is not None else latest_checkpoint(model_folder)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"PiPlus H0W stage-2 checkpoint does not exist: {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"PiPlus H0W checkpoint must be a mapping: {checkpoint_path}")
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, dict) or metadata.get("task") not in {None, expected_task}:
        actual = metadata.get("task") if isinstance(metadata, dict) else None
        raise ValueError(f"Checkpoint task {actual!r} cannot be played as {expected_task!r}")
    metadata = {**metadata, "iteration": int(payload.get("iteration", 0))}
    policy_state = payload.get("policy")
    if not isinstance(policy_state, Mapping):
        raise ValueError("PiPlus H0W checkpoint does not contain a command encoder policy")
    return checkpoint_path, metadata, policy_state


def resolve_playback_asset(
    override: Path | None,
    metadata: Mapping[str, Any],
    key: str,
    *,
    fallback: Path | None = None,
) -> Path | None:
    if override is None and key == "bfm_model" and not metadata.get(key):
        return None
    candidate = override if override is not None else Path(str(metadata.get(key, "")))
    resolved = candidate.expanduser().resolve()
    if not resolved.is_file():
        if override is None and key == "bfm_model":
            return None
        if override is None and fallback is not None:
            fallback_path = fallback.expanduser().resolve()
            if fallback_path.is_file():
                return fallback_path
        raise FileNotFoundError(f"PiPlus H0W playback asset {key!r} does not exist: {resolved}")
    return resolved


def resolve_required_playback_asset(
    override: Path | None,
    metadata: Mapping[str, Any],
    key: str,
    *,
    fallback: Path | None = None,
) -> Path:
    resolved = resolve_playback_asset(override, metadata, key, fallback=fallback)
    if resolved is None:
        raise FileNotFoundError(f"PiPlus H0W playback asset {key!r} is required")
    return resolved


def _add_floor_if_missing(spec: mujoco.MjSpec, floor_name: str) -> None:
    """Add a ground plane only when the robot XML does not already define one.

    The 22DoF H0W XML ships a built-in infinite ``floor`` plane. Adding a second
    coplanar plane produces z-fighting flicker in renders, so skip it in that case.
    """
    has_plane = any(g.type == mujoco.mjtGeom.mjGEOM_PLANE for g in spec.worldbody.geoms)
    if not has_plane:
        spec.worldbody.add_geom(
            name=floor_name,
            type=mujoco.mjtGeom.mjGEOM_PLANE,
            pos=[0.0, 0.0, 0.0],
            size=[20.0, 20.0, 0.02],
            rgba=[0.45, 0.47, 0.50, 1.0],
            contype=0,
            conaffinity=0,
        )


class MujocoQposRenderer:
    """Render policy qpos with the target robot's MuJoCo XML (offscreen, for MP4 capture).

    Mirrors the known-good recording path used by ``speed_stage2_play``: a standalone
    ``mujoco.Renderer`` driven by qpos from the environment, which avoids the noisy and
    unstable MJLab ``rgb_array`` render path.
    """

    def __init__(
        self,
        xml_path: Path,
        render_size: int = 720,
        *,
        camera_distance: float = 3.0,
        camera_azimuth: float = 135.0,
        camera_elevation: float = -18.0,
        expected_qpos_size: int | None = None,
    ) -> None:
        spec = mujoco.MjSpec.from_file(str(xml_path))
        _add_floor_if_missing(spec, "piplus_h0w_stage2_render_floor")
        spec.worldbody.add_light(
            name="piplus_h0w_stage2_render_light",
            pos=[0.0, -3.0, 4.0],
            dir=[0.2, 0.5, -1.0],
            diffuse=[0.8, 0.8, 0.8],
            ambient=[0.35, 0.35, 0.35],
        )
        self.model = spec.compile()
        if expected_qpos_size is not None and self.model.nq != int(expected_qpos_size):
            raise ValueError(f"Expected renderer nq={expected_qpos_size}, got nq={self.model.nq}")
        self.model.vis.global_.offwidth = max(int(self.model.vis.global_.offwidth), int(render_size))
        self.model.vis.global_.offheight = max(int(self.model.vis.global_.offheight), int(render_size))
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height=render_size, width=render_size)
        self.camera = mujoco.MjvCamera()
        self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.camera.distance = float(camera_distance)
        self.camera.azimuth = float(camera_azimuth)
        self.camera.elevation = float(camera_elevation)

    def render_qpos(self, qpos: np.ndarray) -> np.ndarray:
        qpos = np.asarray(qpos, dtype=np.float64).reshape(-1)
        if qpos.size != self.model.nq:
            raise ValueError(f"Expected qpos size {self.model.nq}, got {qpos.size}")
        self.data.qpos[:] = qpos
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self.camera.lookat[:] = [float(qpos[0]), float(qpos[1]), max(float(qpos[2]), 0.75)]
        self.renderer.update_scene(self.data, camera=self.camera)
        return np.ascontiguousarray(self.renderer.render()).astype(np.uint8, copy=False)

    def close(self) -> None:
        self.renderer.close()


def policy_from_state(policy_state: Mapping[str, torch.Tensor], device: torch.device) -> CommandEncoderPolicy:
    """Recreate this repository's one-layer command encoder from its state dict."""
    try:
        input_dim = int(policy_state["trunk.0.weight"].shape[1])
        hidden_dim = int(policy_state["trunk.0.weight"].shape[0])
        z_dim = int(policy_state["latent_mean.weight"].shape[0])
    except (AttributeError, KeyError) as exc:
        raise ValueError("Checkpoint does not contain a compatible PiPlus H0W command encoder") from exc
    policy = CommandEncoderPolicy(input_dim, z_dim=z_dim, hidden_dim=hidden_dim).to(device)
    policy.load_state_dict(policy_state)
    policy.eval()
    return policy


def playback(
    *,
    checkpoint_path: Path,
    metadata: Mapping[str, Any],
    policy_state: Mapping[str, torch.Tensor],
    bfm_model: Path | None,
    decoder_path: Path,
    robot_config: Path,
    motion_dataset: Path,
    device: torch.device,
    fixed_command: tuple[float, float, float],
    command_smoothing: float,
    max_steps: int,
    max_episode_length_s: float,
    seed: int,
    video_path: Path | None = None,
    render_every: int = 2,
    render_size: int = 720,
) -> dict[str, float | int | str]:
    """Run deterministic command-to-decoder playback without an interactive UI."""
    if max_steps <= 0:
        raise ValueError("--max-steps must be positive")
    if not 0.0 < command_smoothing <= 1.0:
        raise ValueError("--command-smoothing must be in (0, 1]")
    if render_every <= 0:
        raise ValueError("--render-every must be positive")
    if render_size <= 0:
        raise ValueError("--render-size must be positive")
    command = torch.tensor(fixed_command, device=device, dtype=torch.float32).unsqueeze(0)
    lower = torch.tensor((-0.8, -0.5, -0.8), device=device)
    upper = torch.tensor((0.8, 0.5, 0.8), device=device)
    if torch.any(command < lower) or torch.any(command > upper):
        raise ValueError("--fixed-command must be within [-0.8, -0.5, -0.8] and [0.8, 0.5, 0.8]")

    environment, robot_training = build_h0w_locomotion_env(
        device=str(device),
        robot_config=robot_config,
        motion_dataset=motion_dataset,
        num_envs=1,
        seed=seed,
        max_episode_length_s=max_episode_length_s,
        disable_domain_randomization=True,
    )
    resolved_video_path = video_path.expanduser().resolve() if video_path is not None else None
    writer: media.VideoWriter | None = None
    video_renderer: MujocoQposRenderer | None = None
    rendered_frames = 0
    try:
        observation, _ = environment.reset(to_numpy=False)
        observation_t = to_torch_observation(observation, device)
        policy = policy_from_state(policy_state, device)
        expected_input = flatten_encoder_observation(observation_t, command).shape[-1]
        if policy.input_dim != expected_input:
            raise ValueError(f"Checkpoint expects encoder input dim {policy.input_dim}, environment provides {expected_input}")
        decoder = load_decoder(bfm_model, decoder_path, device)
        if policy.z_dim != decoder.z_dim:
            raise ValueError(f"Checkpoint z_dim={policy.z_dim} does not match decoder z_dim={decoder.z_dim}")
        if resolved_video_path is not None:
            resolved_video_path.parent.mkdir(parents=True, exist_ok=True)
            qpos, _ = environment._get_qpos_qvel(to_numpy=True)
            video_renderer = MujocoQposRenderer(
                robot_training.robot.xml_path,
                render_size=render_size,
                expected_qpos_size=int(qpos.shape[-1]),
            )
            writer = media.VideoWriter(
                str(resolved_video_path),
                shape=(render_size, render_size),
                fps=1.0 / (environment._env.dt * render_every),
                ffmpeg_args=("-preset", "ultrafast"),
            )
            writer.__enter__()

        commands = torch.zeros_like(command)
        rewards: list[torch.Tensor] = []
        terminations = 0
        for _step in range(max_steps):
            commands.add_(command_smoothing * (command - commands))
            with torch.inference_mode():
                observation_t = to_torch_observation(observation, device)
                raw_z = policy.deterministic_z(flatten_encoder_observation(observation_t, commands))
                action = decoder.act(observation_t, decoder.project_z(raw_z))
            observation, reward, terminated, truncated, _info = environment.step(action.to(environment._env.device), to_numpy=False)
            rewards.append(reward.to(device))
            terminations += int((terminated | truncated).sum())
            if video_renderer is not None and _step % render_every == 0:
                qpos, _ = environment._get_qpos_qvel(to_numpy=True)
                frame = video_renderer.render_qpos(qpos[0])
                writer.add_image(frame)
                rendered_frames += 1
        result: dict[str, float | int | str] = {
            "checkpoint": str(checkpoint_path),
            "iteration": int(metadata.get("iteration", 0)),
            "steps": max_steps,
            "reward_mean": float(torch.stack(rewards).mean()),
            "terminations": terminations,
        }
        if resolved_video_path is not None:
            result["video_path"] = str(resolved_video_path)
            result["video_fps"] = 1.0 / (environment._env.dt * render_every)
            result["rendered_frames"] = rendered_frames
        return result
    finally:
        if writer is not None:
            writer.close()
        if video_renderer is not None:
            video_renderer.close()
        environment.close()


def print_playback_result(result: Mapping[str, float | int | str]) -> None:
    print(json.dumps(dict(result), sort_keys=True), flush=True)
