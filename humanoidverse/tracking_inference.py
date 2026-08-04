"""Tracking inference and video export for UFO policies.

Policy rollout is rendered from the training environment, while the reference
motion is rendered from the configured robot MJCF with pure MuJoCo qpos playback.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import joblib
import mediapy as media
import numpy as np
import torch
from torch.utils._pytree import tree_map

from humanoidverse.agents.load_utils import load_model_from_checkpoint_dir
from humanoidverse.export.backward_encoder import (
    UnsupportedBackwardEncoderExport,
    export_backward_encoder_from_model,
)
from humanoidverse.mjlab_inference_utils import (
    MujocoQposRenderer,
    add_bool_arg,
    checkpoint_load_device,
    load_mjlab_env_cfg,
    render_policy_frame,
)
from humanoidverse.utils.helpers import export_meta_policy_as_onnx, get_backward_observation
from humanoidverse.utils.motion_data import prepare_manifest_dataset_path, prepare_manifest_robot_config_path
from humanoidverse.utils.robot_spec import assert_robot_configs_compatible, load_robot_training_spec, resolve_robot_config_path

DEFAULT_ROBOT_CONFIG = "configs/robots/g1_29dof.yaml"


def _format_duration(seconds: float) -> str:
    total_seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _format_rollout_progress(
    *,
    motion_id: int,
    completed_steps: int,
    total_steps: int,
    rendered_frames: int,
    elapsed_s: float,
) -> str:
    progress = completed_steps / max(total_steps, 1)
    steps_per_second = completed_steps / max(elapsed_s, 1e-9)
    eta_s = max(total_steps - completed_steps, 0) / max(steps_per_second, 1e-9)
    return (
        f"[INFO] motion_id={motion_id} progress={completed_steps}/{total_steps} "
        f"({progress:.1%}) rendered_frames={rendered_frames} "
        f"elapsed={_format_duration(elapsed_s)} rate={steps_per_second:.1f} steps/s "
        f"eta={_format_duration(eta_s)}"
    )


def _resize_nearest(frame: np.ndarray, height: int, width: int) -> np.ndarray:
    if frame.shape[:2] == (height, width):
        return frame
    y_idx = np.linspace(0, frame.shape[0] - 1, height).astype(np.int64)
    x_idx = np.linspace(0, frame.shape[1] - 1, width).astype(np.int64)
    return frame[y_idx[:, None], x_idx[None, :]]


def _collect_torque_metrics(core_env: Any, joint_names: list[str], step: int) -> dict[str, Any]:
    """Capture post-step actuator torques and configured hard/soft limits."""
    torques = torch.as_tensor(core_env.torques[0]).detach().cpu().numpy().astype(np.float64)
    limits = torch.as_tensor(core_env.torque_limits).detach().cpu().numpy().astype(np.float64)
    if torques.shape != limits.shape or torques.shape[0] != len(joint_names):
        raise ValueError(
            "Torque telemetry shape does not match the control joints: "
            f"torques={torques.shape}, limits={limits.shape}, joints={len(joint_names)}"
        )
    soft_factor = float(core_env.config.rewards.reward_limit.soft_torque_limit)
    soft_limits = limits * soft_factor
    abs_torques = np.abs(torques)
    ratios = abs_torques / np.maximum(limits, 1.0e-8)
    soft_ratios = abs_torques / np.maximum(soft_limits, 1.0e-8)
    return {
        "step": int(step),
        "time_s": float((step + 1) * core_env.dt),
        "torques": torques,
        "limits": limits,
        "ratios": ratios,
        "soft_ratios": soft_ratios,
        "hard_saturation": abs_torques >= limits * (1.0 - 1.0e-4),
        "soft_limit": abs_torques >= soft_limits,
    }


def _write_torque_metrics(
    output_dir: Path,
    motion_id: int,
    joint_names: list[str],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Write per-step torque telemetry and a per-joint saturation summary."""
    csv_path = output_dir / f"torque_stats_{motion_id}.csv"
    fieldnames = ["step", "time_s", "max_abs_torque", "max_ratio", "hard_saturation_count", "soft_limit_count"]
    fieldnames += [f"torque_{name}" for name in joint_names]
    fieldnames += [f"ratio_{name}" for name in joint_names]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            record: dict[str, Any] = {
                "step": row["step"],
                "time_s": f"{row['time_s']:.6f}",
                "max_abs_torque": f"{float(np.max(np.abs(row['torques']))):.6f}",
                "max_ratio": f"{float(np.max(row['ratios'])):.6f}",
                "hard_saturation_count": int(np.sum(row["hard_saturation"])),
                "soft_limit_count": int(np.sum(row["soft_limit"])),
            }
            record.update({f"torque_{name}": f"{float(value):.6f}" for name, value in zip(joint_names, row["torques"])})
            record.update({f"ratio_{name}": f"{float(value):.6f}" for name, value in zip(joint_names, row["ratios"])})
            writer.writerow(record)

    if not rows:
        summary = {"motion_id": int(motion_id), "steps": 0, "joints": {}}
    else:
        torques = np.stack([row["torques"] for row in rows])
        ratios = np.stack([row["ratios"] for row in rows])
        hard = np.stack([row["hard_saturation"] for row in rows])
        soft = np.stack([row["soft_limit"] for row in rows])
        summary = {
            "motion_id": int(motion_id),
            "steps": len(rows),
            "dt_s": float(rows[0]["time_s"]),
            "joints": {
                name: {
                    "limit": float(rows[0]["limits"][idx]),
                    "max_abs_torque": float(np.max(np.abs(torques[:, idx]))),
                    "max_ratio": float(np.max(ratios[:, idx])),
                    "hard_saturation_steps": int(np.sum(hard[:, idx])),
                    "soft_limit_steps": int(np.sum(soft[:, idx])),
                }
                for idx, name in enumerate(joint_names)
            },
            "max_ratio": float(np.max(ratios)),
            "hard_saturation_steps": int(np.sum(np.any(hard, axis=1))),
            "soft_limit_steps": int(np.sum(np.any(soft, axis=1))),
        }
    summary_path = output_dir / f"torque_summary_{motion_id}.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return {"csv": csv_path, "summary": summary_path, "data": summary}


def _control_to_qpos_order_indices(robot_training: Any) -> tuple[np.ndarray, list[str]]:
    control_joint_names = list(robot_training.robot.control_joint_names)
    qpos_joint_names = sorted(control_joint_names, key=lambda joint: robot_training.robot.joint_qpos_addr[joint])
    qpos_addrs = [int(robot_training.robot.joint_qpos_addr[joint]) for joint in qpos_joint_names]
    if len(set(qpos_addrs)) != len(qpos_addrs):
        raise ValueError(f"Duplicate MuJoCo qpos addresses for control joints: {list(zip(qpos_joint_names, qpos_addrs))}")
    index_by_control_joint = {joint: idx for idx, joint in enumerate(control_joint_names)}
    return np.asarray([index_by_control_joint[joint] for joint in qpos_joint_names], dtype=np.int64), qpos_joint_names


def _expert_qpos_from_obs(
    obs_dict: dict[str, torch.Tensor],
    *,
    num_dof: int,
    dof_qpos_order_indices: np.ndarray,
) -> np.ndarray:
    root_pos = obs_dict["ref_body_pos"][:, 0].detach().cpu().numpy()
    root_quat_wxyz = np.roll(obs_dict["ref_body_rots"][:, 0].detach().cpu().numpy(), 1, axis=-1)
    # MotionLib stores dof_pos in policy/control-joint order. MuJoCo qpos playback
    # expects hinge joints sorted by qpos address, which can differ for robots such
    # as X2 where actuator order places head joints before arm joints.
    dof_pos = obs_dict["dof_pos"].detach().cpu().numpy()[:, dof_qpos_order_indices]
    qpos = np.concatenate([root_pos, root_quat_wxyz, dof_pos], axis=-1)
    expected = 7 + int(num_dof)
    if qpos.shape[-1] != expected:
        raise ValueError(f"Expected expert qpos shape (*, {expected}), got {qpos.shape}")
    return qpos


def _target_states_from_obs(obs_dict: dict[str, torch.Tensor], device: str, *, num_dof: int) -> dict[str, torch.Tensor]:
    root_state_xyzw = torch.cat(
        [
            obs_dict["ref_body_pos"][0, 0],
            obs_dict["ref_body_rots"][0, 0],
            obs_dict["ref_body_vels"][0, 0],
            obs_dict["ref_body_angular_vels"][0, 0],
        ],
        dim=-1,
    ).to(device=device, dtype=torch.float32)
    dof_state = torch.zeros((int(num_dof), 2), device=device, dtype=torch.float32)
    dof_state[:, 0] = obs_dict["dof_pos"][0].to(device=device, dtype=torch.float32)
    dof_state[:, 1] = obs_dict["ref_dof_vel"][0].to(device=device, dtype=torch.float32)
    return {"root_states": root_state_xyzw.unsqueeze(0), "dof_states": dof_state.unsqueeze(0)}


@torch.no_grad()
def _tracking_z(model: torch.nn.Module, obs: Any) -> torch.Tensor:
    if hasattr(model, "tracking_inference"):
        return model.tracking_inference(obs)

    z = model.backward_map(obs)
    seq_length = int(getattr(model.cfg, "seq_length", 1))
    for step in range(z.shape[0]):
        end_idx = min(step + seq_length, z.shape[0])
        z[step] = z[step:end_idx].mean(dim=0)
    return model.project_z(z)


def _export_policy_model(model: torch.nn.Module, output_dir: Path, robot_training: Any) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    model_name = model.__class__.__name__
    output_name = f"{model_name}.onnx"
    control_joint_names = list(robot_training.robot.control_joint_names)
    num_dof = len(control_joint_names)
    export_metadata = export_meta_policy_as_onnx(
        model,
        output_dir,
        output_name,
        z_dim=model.cfg.archi.z_dim,
    )
    if int(export_metadata["output_action_dim"]) != num_dof:
        raise ValueError(
            "Policy action dim does not match robot control joint count: "
            f"output_action_dim={export_metadata['output_action_dim']}, num_dof={num_dof}"
        )
    export_metadata.update(
        {
            "robot_name": robot_training.robot.name,
            "robot_config_path": str(Path(robot_training.config_path).expanduser().resolve()),
            "xml_path": str(Path(robot_training.robot.xml_path).expanduser().resolve()),
            "num_dof": num_dof,
            "control_joint_names": control_joint_names,
        }
    )
    metadata_path = output_dir / f"{model_name}.meta.json"
    metadata_path.write_text(json.dumps(export_metadata, indent=2, sort_keys=True) + "\n")
    print(f"[INFO] Exported model to {output_dir / output_name}")
    print(f"[INFO] Wrote policy ONNX metadata to {metadata_path}")
    return export_metadata


def _export_tracking_onnx(model: torch.nn.Module, output_dir: Path, robot_training: Any) -> None:
    _export_policy_model(model, output_dir, robot_training)
    try:
        export_backward_encoder_from_model(model, output_dir / "backward_encoder.onnx")
    except UnsupportedBackwardEncoderExport as exc:
        print(f"[INFO] Skip backward encoder ONNX export: {exc}")


def _resolve_tracking_robot_config(
    cli_robot_config: str | Path | None,
    manifest_robot_config: str | Path | None,
) -> Path:
    if cli_robot_config is not None and manifest_robot_config is not None:
        return assert_robot_configs_compatible(cli_robot_config, manifest_robot_config)
    if cli_robot_config is not None:
        return resolve_robot_config_path(cli_robot_config)
    if manifest_robot_config is not None:
        return resolve_robot_config_path(manifest_robot_config)
    return resolve_robot_config_path(DEFAULT_ROBOT_CONFIG)


def run_tracking_inference(
    *,
    model_folder: Path,
    data_path: Path | None,
    robot_config: Path | None,
    headless: bool,
    device: str,
    save_mp4: bool,
    disable_dr: bool,
    disable_obs_noise: bool,
    motion_list: list[int],
    render_size: int,
    camera_distance: float,
    camera_azimuth: float,
    camera_elevation: float,
    fps: int,
    render_every: int,
    max_steps: int | None,
    log_every_steps: int,
    max_episode_length_s: float,
    export_onnx: bool,
    log_torque: bool = True,
) -> None:
    if render_every <= 0:
        raise ValueError("render_every must be positive")

    model_folder = model_folder.expanduser().resolve()
    checkpoint_dir = model_folder / "checkpoint"
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Missing checkpoint directory: {checkpoint_dir}")

    robot_config = _resolve_tracking_robot_config(robot_config, None)
    robot_training = load_robot_training_spec(robot_config)
    robot_xml = Path(robot_training.robot.xml_path).expanduser().resolve()
    if not robot_xml.exists():
        raise FileNotFoundError(f"Missing robot XML: {robot_xml}")
    control_joint_names = list(robot_training.robot.control_joint_names)
    num_dof = len(control_joint_names)
    dof_qpos_order_indices, qpos_joint_names = _control_to_qpos_order_indices(robot_training)

    model_load_device = checkpoint_load_device(device)
    model = load_model_from_checkpoint_dir(checkpoint_dir, device=model_load_device)
    model.to(device)
    model.eval()

    if export_onnx:
        _export_tracking_onnx(model, model_folder / "exported", robot_training)

    env_cfg, use_root_height_obs = load_mjlab_env_cfg(
        model_folder,
        data_path=data_path,
        robot_config=robot_config,
        device=device,
        headless=headless,
        disable_dr=disable_dr,
        disable_obs_noise=disable_obs_noise,
        max_episode_length_s=max_episode_length_s,
    )
    wrapped_env, _ = env_cfg.build(num_envs=1)
    env = wrapped_env._env

    output_dir = model_folder / "tracking_inference"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] UFO tracking inference model_folder={model_folder}")
    print(f"[INFO] Rollout XML={env_cfg.mjcf_path}")
    print(f"[INFO] Motion data={env_cfg.lafan_tail_path}")
    print(f"[INFO] Expert renderer XML={robot_xml}")
    if qpos_joint_names != control_joint_names:
        print(f"[INFO] Expert qpos joint order differs from control order: {qpos_joint_names}")
    print(f"[INFO] device={device} disable_dr={disable_dr} disable_obs_noise={disable_obs_noise} save_mp4={save_mp4}")

    env._motion_lib.load_all_motions()
    env.is_evaluating = True
    expert_renderer = (
        MujocoQposRenderer(
            robot_xml,
            render_size=render_size,
            camera_distance=camera_distance,
            camera_azimuth=camera_azimuth,
            camera_elevation=camera_elevation,
            expected_qpos_size=7 + num_dof,
        )
        if save_mp4
        else None
    )
    try:
        for motion_id in motion_list:
            backward_obs, obs_dict = get_backward_observation(env, motion_id, use_root_height_obs=use_root_height_obs)
            z = _tracking_z(model, tree_map(lambda x: x[1:].to(device) if hasattr(x, "to") else x, backward_obs))
            joblib.dump(z.detach().cpu().numpy(), output_dir / f"zs_{motion_id}.pkl")
            print(f"[INFO] Saved z embedding: {output_dir / f'zs_{motion_id}.pkl'}")

            target_states = _target_states_from_obs(obs_dict, device=device, num_dof=num_dof)
            observation, _ = wrapped_env.reset(to_numpy=False, target_states=target_states)
            episode_len = int(z.shape[0])
            if max_steps is not None:
                episode_len = min(episode_len, int(max_steps))
            expert_qpos = _expert_qpos_from_obs(
                obs_dict,
                num_dof=num_dof,
                dof_qpos_order_indices=dof_qpos_order_indices,
            )
            video_path = output_dir / f"tracking_{motion_id}.mp4"
            video_writer: media.VideoWriter | None = None
            rendered_frames = 0
            torque_rows: list[dict[str, Any]] = []
            use_env_render = True
            rollout_started_at = time.monotonic()

            print(f"[INFO] Running policy rollout for motion_id={motion_id}, steps={episode_len}", flush=True)
            try:
                for step in range(episode_len):
                    action = model.act(observation, z[step].unsqueeze(0), mean=True)
                    observation, _reward, terminated, truncated, _info = wrapped_env.step(action, to_numpy=False)
                    if log_torque:
                        torque_rows.append(_collect_torque_metrics(env, control_joint_names, step))

                    if save_mp4 and (step % render_every == 0 or step + 1 == episode_len):
                        policy_frame, use_env_render = render_policy_frame(
                            wrapped_env,
                            expert_renderer,
                            use_env_render=use_env_render,
                        )
                        expert_frame = expert_renderer.render_qpos(expert_qpos[min(step + 1, len(expert_qpos) - 1)])
                        expert_frame = _resize_nearest(expert_frame, policy_frame.shape[0], policy_frame.shape[1])
                        frame = np.concatenate([expert_frame, policy_frame], axis=1)
                        if video_writer is None:
                            video_writer = media.VideoWriter(
                                str(video_path),
                                shape=frame.shape[:2],
                                fps=fps / render_every,
                                ffmpeg_args=("-preset", "ultrafast"),
                            )
                            video_writer.__enter__()
                        video_writer.add_image(frame)
                        rendered_frames += 1

                    if step == 0 or (step + 1) == episode_len or (log_every_steps > 0 and (step + 1) % log_every_steps == 0):
                        print(
                            _format_rollout_progress(
                                motion_id=motion_id,
                                completed_steps=step + 1,
                                total_steps=episode_len,
                                rendered_frames=rendered_frames,
                                elapsed_s=time.monotonic() - rollout_started_at,
                            ),
                            flush=True,
                        )

                    if bool(torch.as_tensor(terminated).any()) or bool(torch.as_tensor(truncated).any()):
                        print(f"[INFO] Episode ended at step={step}; stopping rollout for motion_id={motion_id}")
                        break
            finally:
                if video_writer is not None:
                    video_writer.close()

            if save_mp4:
                if rendered_frames == 0:
                    raise RuntimeError(f"No frames were rendered for motion_id={motion_id}")
                print(f"[INFO] Saved side-by-side video: {video_path}")
            if log_torque:
                torque_paths = _write_torque_metrics(output_dir, motion_id, control_joint_names, torque_rows)
                torque_summary = torque_paths["data"]
                print(
                    f"[INFO] Torque telemetry motion_id={motion_id}: "
                    f"max_ratio={torque_summary['max_ratio']:.3f}, "
                    f"hard_saturation_steps={torque_summary['hard_saturation_steps']}, "
                    f"soft_limit_steps={torque_summary['soft_limit_steps']} "
                    f"csv={torque_paths['csv']} summary={torque_paths['summary']}",
                    flush=True,
                )
    finally:
        if expert_renderer is not None:
            expert_renderer.close()
        wrapped_env.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="UFO tracking inference with MuJoCo expert rendering.")
    parser.add_argument("--model-folder", type=Path, required=True)
    parser.add_argument("--data-path", type=Path, default=None)
    parser.add_argument("--robot-config", type=Path, default=None, help="Robot YAML for rollout and expert rendering.")
    parser.add_argument("--data-manifest", type=Path, default=None, help="Motion data manifest. Use with --dataset.")
    parser.add_argument("--dataset", default=None, help="Dataset name inside --data-manifest for tracking inference.")
    parser.add_argument("--rebuild-motion-cache", action="store_true", help="Rebuild manifest-generated motion pkl cache.")
    add_bool_arg(parser, "--headless", True, "Run MuJoCo in headless mode.")
    parser.add_argument("--device", default="cuda:0")
    add_bool_arg(parser, "--save-mp4", False, "Save side-by-side expert/policy MP4.")
    add_bool_arg(parser, "--disable-dr", False, "Disable domain randomization.")
    add_bool_arg(parser, "--disable-obs-noise", False, "Disable observation noise.")
    parser.add_argument("--motion-list", type=int, nargs="+", default=[20])
    parser.add_argument("--render-size", type=int, default=480)
    parser.add_argument("--camera-distance", type=float, default=3.0)
    parser.add_argument("--camera-azimuth", type=float, default=135.0)
    parser.add_argument("--camera-elevation", type=float, default=-18.0)
    parser.add_argument("--fps", type=int, default=50)
    parser.add_argument(
        "--render-every",
        type=int,
        default=1,
        help="Render every N policy steps while keeping the video duration unchanged.",
    )
    parser.add_argument("--max-steps", type=int, default=None, help="Optional cap on rollout/video frames for quick previews.")
    parser.add_argument("--log-every-steps", type=int, default=100, help="Print rollout/render progress every N steps; 0 disables periodic logs.")
    parser.add_argument("--max-episode-length-s", type=float, default=10000.0)
    add_bool_arg(parser, "--export-onnx", True, "Export ONNX next to the checkpoint before inference.")
    add_bool_arg(parser, "--log-torque", True, "Record per-step actuator torque and saturation telemetry.")
    args = parser.parse_args()
    manifest_robot_config = None
    if args.data_manifest is not None:
        if args.data_path is not None:
            parser.error("--data-manifest and --data-path cannot be used together")
        if args.dataset is None:
            parser.error("--dataset is required when --data-manifest is provided")
        manifest_robot_config = prepare_manifest_robot_config_path(args.data_manifest)
        args.data_path = Path(
            prepare_manifest_dataset_path(
                args.data_manifest,
                args.dataset,
                split="inference",
                rebuild_cache=bool(args.rebuild_motion_cache),
            )
        )
    args.robot_config = _resolve_tracking_robot_config(args.robot_config, manifest_robot_config)
    return args


def main() -> None:
    args = parse_args()
    run_tracking_inference(
        model_folder=args.model_folder,
        data_path=args.data_path,
        robot_config=args.robot_config,
        headless=args.headless,
        device=args.device,
        save_mp4=args.save_mp4,
        disable_dr=args.disable_dr,
        disable_obs_noise=args.disable_obs_noise,
        motion_list=args.motion_list,
        render_size=args.render_size,
        camera_distance=args.camera_distance,
        camera_azimuth=args.camera_azimuth,
        camera_elevation=args.camera_elevation,
        fps=args.fps,
        render_every=args.render_every,
        max_steps=args.max_steps,
        log_every_steps=args.log_every_steps,
        max_episode_length_s=args.max_episode_length_s,
        export_onnx=args.export_onnx,
        log_torque=args.log_torque,
    )


if __name__ == "__main__":
    main()
