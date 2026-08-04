"""Validate a robot-config-aware BFM training setup before launching GPUs."""

from __future__ import annotations

import argparse
import hashlib
import xml.etree.ElementTree as ET
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from omegaconf import OmegaConf

from humanoidverse.utils.motion_data import prepare_motion_manifest
from humanoidverse.utils.robot_spec import (
    assert_robot_configs_compatible,
    load_robot_training_spec,
    resolve_robot_config_path,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_finite(value: Any, *, field: str, source: str, motion_name: str) -> np.ndarray:
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"{source}:{motion_name} field {field} must be numeric, got dtype={array.dtype}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{source}:{motion_name} field {field} contains NaN or infinity")
    return array


def validate_motion_file(
    path: str | Path,
    *,
    control_joints: Sequence[str],
    body_count: int,
    require_joint_names: bool = True,
) -> dict[str, Any]:
    source_path = Path(path).expanduser().resolve()
    data = joblib.load(source_path)
    if not isinstance(data, Mapping) or not data:
        raise ValueError(f"Motion file must contain a non-empty mapping: {source_path}")

    expected_joints = list(control_joints)
    fps_counts: dict[float, int] = {}
    frame_count = 0
    duration_seconds = 0.0
    for raw_name, raw_motion in data.items():
        motion_name = str(raw_name)
        if not isinstance(raw_motion, Mapping):
            raise ValueError(f"{source_path}:{motion_name} must be a mapping")
        for field in ("root_trans_offset", "pose_aa", "fps"):
            if field not in raw_motion:
                raise ValueError(f"{source_path}:{motion_name} is missing required field {field}")
        root_rot_field = "root_rot" if "root_rot" in raw_motion else "root_quat"
        dof_field = "dof_pos" if "dof_pos" in raw_motion else "dof"
        if root_rot_field not in raw_motion:
            raise ValueError(f"{source_path}:{motion_name} is missing root_rot/root_quat")
        if dof_field not in raw_motion:
            raise ValueError(f"{source_path}:{motion_name} is missing dof_pos/dof")

        root_pos = _require_finite(raw_motion["root_trans_offset"], field="root_trans_offset", source=str(source_path), motion_name=motion_name)
        root_rot = _require_finite(raw_motion[root_rot_field], field=root_rot_field, source=str(source_path), motion_name=motion_name)
        pose_aa = _require_finite(raw_motion["pose_aa"], field="pose_aa", source=str(source_path), motion_name=motion_name)
        dof = _require_finite(raw_motion[dof_field], field=dof_field, source=str(source_path), motion_name=motion_name)

        if root_pos.ndim != 2 or root_pos.shape[1] != 3:
            raise ValueError(f"{source_path}:{motion_name} root_trans_offset must be [T, 3], got {root_pos.shape}")
        frames = int(root_pos.shape[0])
        if root_rot.shape != (frames, 4):
            raise ValueError(f"{source_path}:{motion_name} root_rot must be [T, 4], got {root_rot.shape}")
        if pose_aa.shape != (frames, body_count, 3):
            raise ValueError(
                f"{source_path}:{motion_name} pose_aa must be [T, {body_count}, 3], got {pose_aa.shape}"
            )
        if dof.shape != (frames, len(expected_joints)):
            raise ValueError(
                f"{source_path}:{motion_name} dof must be [T, {len(expected_joints)}], got {dof.shape}"
            )

        joint_names = raw_motion.get("joint_names")
        if joint_names is None:
            if require_joint_names:
                raise ValueError(f"{source_path}:{motion_name} is missing joint_names")
        elif list(joint_names) != expected_joints:
            raise ValueError(f"{source_path}:{motion_name} joint_names do not match robot control_joints order")

        fps_array = np.asarray(raw_motion["fps"])
        if fps_array.size != 1:
            raise ValueError(f"{source_path}:{motion_name} fps must be scalar, got {fps_array.shape}")
        fps = float(fps_array.reshape(-1)[0])
        if not np.isfinite(fps) or fps <= 0.0:
            raise ValueError(f"{source_path}:{motion_name} fps must be positive, got {fps}")

        quat_norm = np.linalg.norm(root_rot, axis=1)
        if np.max(np.abs(quat_norm - 1.0)) > 5e-3:
            raise ValueError(f"{source_path}:{motion_name} root_rot contains non-unit quaternions")

        fps_counts[fps] = fps_counts.get(fps, 0) + 1
        frame_count += frames
        duration_seconds += frames / fps

    return {
        "path": str(source_path),
        "motions": len(data),
        "frames": frame_count,
        "duration_seconds": duration_seconds,
        "fps": fps_counts,
    }


def _manifest_datasets(path: Path) -> list[dict[str, Any]]:
    raw = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(raw, dict) or not isinstance(raw.get("datasets"), list):
        raise ValueError(f"Invalid motion manifest: {path}")
    return [dict(item) for item in raw["datasets"]]


def _validate_hash(path: Path, expected: str | None, *, label: str) -> None:
    if not expected:
        return
    actual = _sha256(path)
    if actual.lower() != str(expected).lower():
        raise ValueError(f"{label} SHA-256 mismatch: expected={expected}, actual={actual}, path={path}")


def _validate_motionlib_mjcf_layout(xml_path: str | Path, control_joints: Sequence[str]) -> None:
    source_path = Path(xml_path).expanduser().resolve()
    xml_root = ET.parse(source_path).getroot()
    root_body = xml_root.find("worldbody/body")
    if root_body is None:
        raise ValueError(f"MJCF has no root body: {source_path}")
    if root_body.find("freejoint") is None:
        free_joint = root_body.find("joint[@type='free']")
        if free_joint is not None:
            raise ValueError(
                f"MJCF root uses <joint type='free'>, which MotionLib counts as an actuated body; "
                f"use <freejoint name='{free_joint.get('name', 'root')}'/> in the training MJCF"
            )
        raise ValueError(f"MJCF root body has no freejoint: {source_path}")

    motionlib_joints: list[str] = []
    for body in root_body.iter("body"):
        body_joints = body.findall("joint")
        if len(body_joints) > 1:
            raise ValueError(f"MotionLib requires at most one actuated joint per body, got body={body.get('name')}")
        if body_joints:
            motionlib_joints.append(str(body_joints[0].get("name")))
    if motionlib_joints != list(control_joints):
        raise ValueError(
            "MotionLib MJCF body-to-joint order does not match robot control_joints: "
            f"motionlib={motionlib_joints}, control={list(control_joints)}"
        )


def validate_setup(
    *,
    data_manifest: str | Path,
    robot_config: str | Path | None = None,
    require_joint_names: bool = True,
    verify_hashes: bool = True,
) -> list[dict[str, Any]]:
    manifest_path = Path(data_manifest).expanduser().resolve()
    prepared = prepare_motion_manifest(manifest_path)
    selected_robot = prepared.robot_config_path
    if robot_config is not None and selected_robot is not None:
        selected_robot = str(assert_robot_configs_compatible(robot_config, selected_robot))
    elif robot_config is not None:
        selected_robot = str(resolve_robot_config_path(robot_config))
    if selected_robot is None:
        raise ValueError("A robot config is required through --robot-config or manifest robot_config")

    training = load_robot_training_spec(selected_robot)
    _validate_motionlib_mjcf_layout(training.robot.xml_path, training.robot.control_joint_names)
    datasets = _manifest_datasets(manifest_path)
    if len(datasets) != len(prepared.train_data_paths):
        raise ValueError("Prepared manifest dataset count does not match configured datasets")

    results: list[dict[str, Any]] = []
    for dataset, train_path_text in zip(datasets, prepared.train_data_paths):
        dataset_name = str(dataset["name"])
        train_path = Path(train_path_text).resolve()
        if verify_hashes:
            _validate_hash(train_path, dataset.get("train_sha256"), label=f"{dataset_name}:train")
        train_result = validate_motion_file(
            train_path,
            control_joints=training.robot.control_joint_names,
            body_count=len(training.robot.body_names),
            require_joint_names=require_joint_names,
        )
        train_result.update(dataset=dataset_name, split="train")
        results.append(train_result)

        inference_path_text = prepared.inference_paths.get(dataset_name)
        if inference_path_text:
            inference_path = Path(inference_path_text).expanduser().resolve()
            if verify_hashes:
                _validate_hash(inference_path, dataset.get("inference_sha256"), label=f"{dataset_name}:inference")
            inference_result = validate_motion_file(
                inference_path,
                control_joints=training.robot.control_joint_names,
                body_count=len(training.robot.body_names),
                require_joint_names=require_joint_names,
            )
            inference_result.update(dataset=dataset_name, split="inference")
            results.append(inference_result)

    print(
        f"[OK] robot={training.robot.name} nq={training.robot.nq} nv={training.robot.nv} "
        f"actions={len(training.robot.control_joint_names)} bodies={len(training.robot.body_names)}"
    )
    for result in results:
        print(
            f"[OK] dataset={result['dataset']} split={result['split']} motions={result['motions']} "
            f"frames={result['frames']} duration_s={result['duration_seconds']:.1f} fps={result['fps']} "
            f"path={result['path']}"
        )
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a robot-aware BFM setup and its expert trajectories.")
    parser.add_argument("--data-manifest", required=True, type=Path)
    parser.add_argument("--robot-config", type=Path, default=None)
    parser.add_argument("--allow-missing-joint-names", action="store_true")
    parser.add_argument("--skip-hashes", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_setup(
        data_manifest=args.data_manifest,
        robot_config=args.robot_config,
        require_joint_names=not args.allow_missing_joint_names,
        verify_hashes=not args.skip_hashes,
    )


if __name__ == "__main__":
    main()
