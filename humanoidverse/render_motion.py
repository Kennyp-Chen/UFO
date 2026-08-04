"""Render a UFO expert motion to MP4 with a selected robot model."""

from __future__ import annotations

import argparse
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import mediapy as media
import mujoco
import numpy as np

from humanoidverse.mjlab_inference_utils import MujocoQposRenderer
from humanoidverse.utils.motion_data import prepare_manifest_dataset_path, prepare_manifest_robot_config_path
from humanoidverse.utils.motion_data.adapters import load_ufo_pkl
from humanoidverse.utils.robot_spec import RobotSpec, assert_robot_configs_compatible, load_robot_spec, resolve_robot_config_path


def _scalar_fps(motion: Mapping[str, Any], motion_key: str) -> float:
    fps_array = np.asarray(motion["fps"])
    if fps_array.size != 1:
        raise ValueError(f"Motion {motion_key!r} fps must be scalar, got shape={fps_array.shape}")
    fps = float(fps_array.reshape(-1)[0])
    if not np.isfinite(fps) or fps <= 0.0:
        raise ValueError(f"Motion {motion_key!r} fps must be positive, got {fps}")
    return fps


def _motion_field(motion: Mapping[str, Any], names: Sequence[str], *, motion_key: str) -> np.ndarray:
    for name in names:
        if name in motion:
            return np.asarray(motion[name])
    raise ValueError(f"Motion {motion_key!r} is missing one of fields {list(names)}")


def _motion_control_dof(motion: Mapping[str, Any], robot: RobotSpec, *, motion_key: str) -> np.ndarray:
    dof = _motion_field(motion, ("dof_pos", "dof"), motion_key=motion_key)
    if dof.ndim != 2:
        raise ValueError(f"Motion {motion_key!r} dof must have shape [T, D], got {dof.shape}")

    control_joints = list(robot.control_joint_names)
    motion_joint_names = motion.get("joint_names")
    if motion_joint_names is None:
        if dof.shape[1] != len(control_joints):
            raise ValueError(
                f"Motion {motion_key!r} has no joint_names and dof width {dof.shape[1]} does not match "
                f"robot control joint count {len(control_joints)}"
            )
        print(f"[WARN] Motion {motion_key!r} has no joint_names; assuming robot control-joint order.")
        return dof

    motion_joint_names = [str(name) for name in motion_joint_names]
    if len(motion_joint_names) != dof.shape[1]:
        raise ValueError(
            f"Motion {motion_key!r} joint_names count {len(motion_joint_names)} does not match dof width {dof.shape[1]}"
        )
    if len(set(motion_joint_names)) != len(motion_joint_names):
        raise ValueError(f"Motion {motion_key!r} contains duplicate joint names")
    missing = [name for name in control_joints if name not in motion_joint_names]
    extra = [name for name in motion_joint_names if name not in control_joints]
    if missing or extra:
        raise ValueError(
            f"Motion {motion_key!r} joint set does not match robot {robot.name}: missing={missing}, extra={extra}"
        )
    motion_index = {name: index for index, name in enumerate(motion_joint_names)}
    return dof[:, [motion_index[name] for name in control_joints]]


def motion_to_qpos(motion: Mapping[str, Any], robot: RobotSpec, model: mujoco.MjModel, *, motion_key: str) -> np.ndarray:
    """Convert one UFO motion record to MuJoCo qpos order."""

    if robot.free_joint is None:
        raise ValueError(f"Robot {robot.name} does not have a free root joint")
    if model.nq != robot.nq:
        raise ValueError(f"Robot config nq={robot.nq} does not match renderer model nq={model.nq}")
    if robot.dof_unit != "rad":
        raise ValueError(f"Rendering currently requires dof_unit=rad, got {robot.dof_unit!r}")

    root_pos = _motion_field(motion, ("root_trans_offset",), motion_key=motion_key)
    root_quat = _motion_field(motion, ("root_rot", "root_quat"), motion_key=motion_key)
    control_dof = _motion_control_dof(motion, robot, motion_key=motion_key)
    if root_pos.ndim != 2 or root_pos.shape[1] != 3:
        raise ValueError(f"Motion {motion_key!r} root_trans_offset must have shape [T, 3], got {root_pos.shape}")
    frame_count = root_pos.shape[0]
    if root_quat.shape != (frame_count, 4):
        raise ValueError(f"Motion {motion_key!r} root quaternion must have shape [{frame_count}, 4], got {root_quat.shape}")
    if control_dof.shape != (frame_count, len(robot.control_joint_names)):
        raise ValueError(
            f"Motion {motion_key!r} dof must have shape [{frame_count}, {len(robot.control_joint_names)}], "
            f"got {control_dof.shape}"
        )
    for name, array in (("root_trans_offset", root_pos), ("root quaternion", root_quat), ("dof", control_dof)):
        if not np.all(np.isfinite(array)):
            raise ValueError(f"Motion {motion_key!r} {name} contains NaN or infinity")

    quat_norm = np.linalg.norm(root_quat, axis=1, keepdims=True)
    if np.any(quat_norm < 1e-8):
        raise ValueError(f"Motion {motion_key!r} contains a zero-length root quaternion")
    root_quat = root_quat / quat_norm
    root_quat_wxyz = np.roll(root_quat, 1, axis=-1) if robot.root_quat_order == "xyzw" else root_quat

    qpos = np.repeat(model.qpos0[None, :], frame_count, axis=0)
    root_addr = robot.joint_qpos_addr[robot.free_joint]
    qpos[:, root_addr : root_addr + 3] = root_pos
    qpos[:, root_addr + 3 : root_addr + 7] = root_quat_wxyz
    for control_index, joint_name in enumerate(robot.control_joint_names):
        if robot.joint_types[joint_name] not in {"hinge", "slide"}:
            raise ValueError(
                f"Control joint {joint_name!r} has unsupported type {robot.joint_types[joint_name]!r}; "
                "only hinge and slide joints map to one qpos value"
            )
        qpos[:, robot.joint_qpos_addr[joint_name]] = control_dof[:, control_index]
    return qpos


def select_motion(
    motions: Mapping[str, Mapping[str, Any]], *, motion_key: str | None, motion_index: int | None
) -> tuple[str, Mapping[str, Any]]:
    keys = list(motions)
    if motion_key is not None:
        if motion_key not in motions:
            raise ValueError(f"Motion key {motion_key!r} not found. Use --list-motions to inspect available keys.")
        return motion_key, motions[motion_key]
    selected_index = 0 if motion_index is None else int(motion_index)
    if selected_index < 0 or selected_index >= len(keys):
        raise ValueError(f"Motion index {selected_index} is out of range [0, {len(keys) - 1}]")
    selected_key = keys[selected_index]
    return selected_key, motions[selected_key]


def frame_indices(frame_count: int, *, start_frame: int, max_frames: int | None, stride: int) -> np.ndarray:
    if stride <= 0:
        raise ValueError(f"--stride must be positive, got {stride}")
    if start_frame < 0 or start_frame >= frame_count:
        raise ValueError(f"--start-frame must be in [0, {frame_count - 1}], got {start_frame}")
    indices = np.arange(start_frame, frame_count, stride, dtype=np.int64)
    if max_frames is not None:
        if max_frames <= 0:
            raise ValueError(f"--max-frames must be positive, got {max_frames}")
        indices = indices[:max_frames]
    return indices


def _safe_filename(value: str) -> str:
    result = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return result or "motion"


def _list_motions(motions: Mapping[str, Mapping[str, Any]], *, limit: int | None) -> None:
    print("index\tframes\tfps\tduration_s\tmotion_key")
    for index, (key, motion) in enumerate(motions.items()):
        if limit is not None and index >= limit:
            print(f"... {len(motions) - limit} more motions; use --list-limit 0 to show all")
            break
        frames = int(np.asarray(motion["root_trans_offset"]).shape[0])
        fps = _scalar_fps(motion, str(key))
        print(f"{index}\t{frames}\t{fps:g}\t{frames / fps:.3f}\t{key}")


def _resolve_inputs(args: argparse.Namespace, parser: argparse.ArgumentParser) -> tuple[Path, Path]:
    if args.data_manifest is not None:
        if not args.dataset:
            parser.error("--dataset is required with --data-manifest")
        manifest_robot = prepare_manifest_robot_config_path(args.data_manifest)
        if args.robot_config is not None and manifest_robot is not None:
            robot_config = assert_robot_configs_compatible(args.robot_config, manifest_robot)
        elif args.robot_config is not None:
            robot_config = resolve_robot_config_path(args.robot_config)
        elif manifest_robot is not None:
            robot_config = resolve_robot_config_path(manifest_robot)
        else:
            parser.error("Manifest has no robot_config; pass --robot-config")
        data_path = Path(
            prepare_manifest_dataset_path(
                args.data_manifest,
                args.dataset,
                split=args.split,
                rebuild_cache=bool(args.rebuild_motion_cache),
            )
        )
        return robot_config, data_path

    if args.robot_config is None:
        parser.error("--robot-config is required with --data-path")
    return resolve_robot_config_path(args.robot_config), args.data_path.expanduser().resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render one expert trajectory as an MP4 with a selected robot MJCF.")
    parser.add_argument("--robot-config", type=Path, default=None, help="Robot YAML; inferred from the manifest when omitted.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--data-path", type=Path, help="UFO motion pkl path.")
    source.add_argument("--data-manifest", type=Path, help="Motion data manifest.")
    parser.add_argument("--dataset", help="Dataset name in --data-manifest.")
    parser.add_argument("--split", choices=("train", "inference"), default="inference")
    parser.add_argument("--rebuild-motion-cache", action="store_true")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--motion-key", help="Exact motion dictionary key.")
    selection.add_argument("--motion-index", type=int, help="Zero-based motion index; defaults to 0.")
    parser.add_argument("--list-motions", action="store_true", help="List motions and exit without rendering.")
    parser.add_argument("--list-limit", type=int, default=100, help="Maximum listed motions; 0 lists all.")
    parser.add_argument("--output", type=Path, default=None, help="Output MP4; defaults to renders/<robot>_<motion>.mp4.")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=None, help="Maximum number of encoded frames after applying stride.")
    parser.add_argument("--stride", type=int, default=1, help="Render every Nth source frame.")
    parser.add_argument("--fps", type=float, default=None, help="Output FPS; defaults to source FPS divided by stride.")
    parser.add_argument("--render-size", type=int, default=480, help="Square output size in pixels; must be positive and even.")
    parser.add_argument("--camera-distance", type=float, default=2.2)
    parser.add_argument("--camera-azimuth", type=float, default=135.0)
    parser.add_argument("--camera-elevation", type=float, default=-18.0)
    parser.add_argument("--log-every-frames", type=int, default=100, help="Progress interval; 0 disables periodic logs.")
    return parser


def run(args: argparse.Namespace, parser: argparse.ArgumentParser) -> Path | None:
    robot_config, data_path = _resolve_inputs(args, parser)
    robot = load_robot_spec(robot_config)
    motions = load_ufo_pkl(data_path, source_name=f"render:{data_path.name}")
    if args.list_motions:
        if args.list_limit < 0:
            raise ValueError(f"--list-limit must be non-negative, got {args.list_limit}")
        _list_motions(motions, limit=None if args.list_limit == 0 else args.list_limit)
        return None

    motion_key, motion = select_motion(motions, motion_key=args.motion_key, motion_index=args.motion_index)
    source_fps = _scalar_fps(motion, motion_key)
    model = mujoco.MjModel.from_xml_path(str(robot.xml_path))
    qpos = motion_to_qpos(motion, robot, model, motion_key=motion_key)
    indices = frame_indices(qpos.shape[0], start_frame=args.start_frame, max_frames=args.max_frames, stride=args.stride)
    output_fps = source_fps / args.stride if args.fps is None else float(args.fps)
    if not np.isfinite(output_fps) or output_fps <= 0.0:
        raise ValueError(f"Output FPS must be positive, got {output_fps}")
    if args.render_size <= 0 or args.render_size % 2 != 0:
        raise ValueError(f"--render-size must be positive and even, got {args.render_size}")

    output = args.output
    if output is None:
        output = Path("renders") / f"{_safe_filename(robot.name)}_{_safe_filename(motion_key)}.mp4"
    output = output.expanduser().resolve()
    if output.suffix.lower() != ".mp4":
        raise ValueError(f"Output path must end in .mp4, got {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"[INFO] Rendering robot={robot.name} motion={motion_key!r} source={data_path} "
        f"source_frames={qpos.shape[0]} output_frames={len(indices)} output_fps={output_fps:g}"
    )
    renderer = MujocoQposRenderer(
        Path(robot.xml_path),
        render_size=args.render_size,
        camera_distance=args.camera_distance,
        camera_azimuth=args.camera_azimuth,
        camera_elevation=args.camera_elevation,
        expected_qpos_size=robot.nq,
    )
    try:
        with media.VideoWriter(output, shape=(args.render_size, args.render_size), fps=output_fps, codec="h264", crf=18) as writer:
            for output_index, source_index in enumerate(indices, start=1):
                writer.add_image(renderer.render_qpos(qpos[source_index]))
                if output_index == 1 or output_index == len(indices) or (
                    args.log_every_frames > 0 and output_index % args.log_every_frames == 0
                ):
                    print(f"[INFO] Rendered {output_index}/{len(indices)} frames", flush=True)
    finally:
        renderer.close()

    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError(f"MP4 writer did not create a non-empty file: {output}")
    print(f"[INFO] Saved MP4: {output} ({output.stat().st_size} bytes)")
    return output


def main() -> None:
    parser = build_parser()
    run(parser.parse_args(), parser)


if __name__ == "__main__":
    main()
