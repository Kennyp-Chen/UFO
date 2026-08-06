"""PiPlus H0W 22DoF AMP stage-2 fine tuning on UFO's MJLab runtime."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import joblib
import mujoco
import numpy as np
import torch
from torch import nn
from torch.nn import functional as functional

from humanoidverse.distributed import average_gradients, average_metrics, barrier, broadcast_module_state, is_distributed
from humanoidverse.piplus_h0w_locomotion import build_h0w_locomotion_env, h0w_robot_training_spec
from humanoidverse.piplus_h0w_onnx_decoder import H0W_ACTION_DIM, load_decoder
from humanoidverse.piplus_h0w_stage2 import (
    UNITREE_PPO_PROFILE,
    CommandEncoderPolicy,
    PpoProfile,
    PpoRollout,
    compute_gae,
    flatten_encoder_observation,
    ppo_update,
    sample_commands,
    to_torch_observation,
)
from humanoidverse.piplus_h0w_unitree_velocity import (
    UNITREE_OMITTED_TERMS,
    UNITREE_REWARD_SOURCE,
    UNITREE_REWARD_SOURCE_VERSIONS,
    UNITREE_VELOCITY_WEIGHTS,
    UnitreeVelocityLocomotionRewardState,
)
from humanoidverse.speed_stage2 import (
    DEFAULT_BFM_MODEL,
    DEFAULT_DECODER_PATH,
    DEFAULT_MOTION_DATASET,
    DEFAULT_ROBOT_CONFIG,
    validate_h0w_assets,
)
from humanoidverse.utils.torch_utils import quat_rotate_inverse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
AMP_STAGE2_TASK = "amp_stage2_piplus_22dof"
H0W_KEY_BODIES = (
    "l_ankle_roll_link",
    "r_ankle_roll_link",
    "l_elbow_link",
    "r_elbow_link",
    "head_pitch_link",
)
H0W_AMP_FEATURE_DIM = 194
FEET_CLEARANCE_TARGET = 0.13
FEET_CLEARANCE_SIGMA = 0.05
DEFAULT_LATENT_REFERENCE = (
    PROJECT_ROOT / "model/piplus_h0w_bfm/decoder/bfmzero-piplus-h0w-isaac-20260629_214205/tracking_inference/zs_8.pkl"
)
DEFAULT_H0W_AMP_WALKING_EXPERT_DATASET = (
    PROJECT_ROOT / "humanoidverse/data/piplus_h0w_lafan/piplus_h0w_locomotion_run_with_stand.pkl"
)
PROFILE_A_MIMICLITE_SPEED_SAFETY = "a_mimiclite_speed_safety"
PROFILE_B_AMP_23DOF_REFERENCE = "b_amp_23dof_reference"
PROFILE_C_UNITREE_VELOCITY_H0W_COMPAT = "c_unitree_velocity_h0w_compat"

MIMICLITE_LOCOMOTION_WEIGHTS = {
    "linvel_exp": 3.7,
    "linvel_projection": 1.5,
    "angvel_z_exp": 3.4,
    "backward_velocity_progress": 0.9,
    "turn_rate_progress": 0.65,
    "single_foot_contact": 0.85,
    "angvel_xy_l2": 0.035,
    "body_upright": 1.1,
    "stand_still": 0.8,
    "feet_air_time": 2.5,
    "feet_clearance": 2.0,
    "energy_l1": 2.0e-4,
    "joint_acc_l2": 1.0e-7,
    "action_rate_l2": 0.005,
    "action_rate2_l2": 0.005,
    "joint_vel_l2": 1.0e-3,
    "joint_deviation_l2": 0.11,
}


@dataclass(frozen=True)
class AmpRewardProfile:
    name: str
    description: str
    default_expert_dataset: Path
    env_reward_weight: float
    locomotion_reward_weight: float
    amp_weight: float
    latent_prior_weight: float
    uses_amp: bool
    uses_latent_prior: bool
    locomotion_weights: Mapping[str, float]
    ppo: PpoProfile | None = None
    source: str | None = None
    omitted_unitree_terms: Mapping[str, str] | None = None


def _profile_a_weights() -> dict[str, float]:
    weights = {name: 0.0 for name in MIMICLITE_LOCOMOTION_WEIGHTS}
    for name in (
        "linvel_exp",
        "linvel_projection",
        "angvel_z_exp",
        "backward_velocity_progress",
        "turn_rate_progress",
        "body_upright",
        "energy_l1",
        "action_rate_l2",
        "action_rate2_l2",
        "joint_vel_l2",
    ):
        weights[name] = MIMICLITE_LOCOMOTION_WEIGHTS[name]
    return weights


_AMP_REWARD_PROFILES = {
    PROFILE_A_MIMICLITE_SPEED_SAFETY: AmpRewardProfile(
        name=PROFILE_A_MIMICLITE_SPEED_SAFETY,
        description="Direct command locomotion with MimicLite velocity and safety terms, without AMP.",
        default_expert_dataset=DEFAULT_MOTION_DATASET,
        env_reward_weight=1.0,
        locomotion_reward_weight=1.0,
        amp_weight=0.0,
        latent_prior_weight=0.0,
        uses_amp=False,
        uses_latent_prior=False,
        locomotion_weights=_profile_a_weights(),
    ),
    PROFILE_B_AMP_23DOF_REFERENCE: AmpRewardProfile(
        name=PROFILE_B_AMP_23DOF_REFERENCE,
        description="22DoF adaptation of the AMP and latent-prior reference profile.",
        default_expert_dataset=DEFAULT_H0W_AMP_WALKING_EXPERT_DATASET,
        env_reward_weight=1.0,
        locomotion_reward_weight=1.1,
        amp_weight=0.06,
        latent_prior_weight=0.02,
        uses_amp=True,
        uses_latent_prior=True,
        locomotion_weights=MIMICLITE_LOCOMOTION_WEIGHTS,
    ),
    PROFILE_C_UNITREE_VELOCITY_H0W_COMPAT: AmpRewardProfile(
        name=PROFILE_C_UNITREE_VELOCITY_H0W_COMPAT,
        description="H0W-compatible Unitree velocity reward adaptation; angular momentum, G1 collision, and G1 posture terms are omitted because H0W does not expose them.",
        default_expert_dataset=DEFAULT_MOTION_DATASET,
        env_reward_weight=1.0,
        locomotion_reward_weight=1.0,
        amp_weight=0.0,
        latent_prior_weight=0.0,
        uses_amp=False,
        uses_latent_prior=False,
        locomotion_weights=UNITREE_VELOCITY_WEIGHTS,
        ppo=UNITREE_PPO_PROFILE,
        source=UNITREE_REWARD_SOURCE,
        omitted_unitree_terms=UNITREE_OMITTED_TERMS,
    ),
}


def get_amp_reward_profile(name: str) -> AmpRewardProfile:
    try:
        return _AMP_REWARD_PROFILES[name]
    except KeyError as exc:
        raise ValueError(f"Unknown AMP reward profile {name!r}; expected one of {sorted(_AMP_REWARD_PROFILES)}") from exc


def _apply_reward_profile_defaults(args: argparse.Namespace) -> AmpRewardProfile:
    profile = get_amp_reward_profile(args.reward_profile)
    if args.motion_dataset is None:
        args.motion_dataset = DEFAULT_MOTION_DATASET
    if args.expert_dataset is None:
        args.expert_dataset = profile.default_expert_dataset
    for name, value in {
        "env_reward_weight": profile.env_reward_weight,
        "locomotion_reward_weight": profile.locomotion_reward_weight,
        "amp_weight": profile.amp_weight,
        "latent_prior_weight": profile.latent_prior_weight,
    }.items():
        if getattr(args, name) is None:
            setattr(args, name, value)
    if not profile.uses_amp:
        args.amp_weight = 0.0
        args.latent_prior_weight = 0.0
    return profile


def _profile_ppo(profile: AmpRewardProfile) -> PpoProfile | None:
    return profile.ppo


def checkpoint_task_is_compatible(metadata: Mapping[str, object]) -> bool:
    return metadata.get("task") in {None, AMP_STAGE2_TASK}


def expected_h0w_amp_feature_dim(num_dof: int, history_length: int, key_body_count: int) -> int:
    return 3 + 3 * int(key_body_count) + int(history_length) * int(num_dof)


def build_online_amp_feature(core: Any, joint_history: torch.Tensor, key_body_indices: torch.Tensor) -> torch.Tensor:
    """Build the online AMP feature from a terminal-safe H0W transition state."""
    root_position = core.robot_root_states[:, :3]
    relative_positions = core.body_pos[:, key_body_indices] - root_position[:, None, :]
    key_count = int(key_body_indices.numel())
    rotations = core.base_quat[:, None, :].expand(-1, key_count, -1).reshape(-1, 4)
    local_positions = quat_rotate_inverse(rotations, relative_positions.reshape(-1, 3), w_last=True).reshape_as(relative_positions)
    return torch.cat((core.base_lin_vel, local_positions.flatten(1), joint_history.flatten(1)), dim=-1)


def _motion_joint_positions(motion: Mapping[str, Any], joint_names: tuple[str, ...]) -> np.ndarray:
    dof = np.asarray(motion.get("dof", motion.get("dof_pos")), dtype=np.float64)
    source_names = [str(name) for name in motion["joint_names"]]
    if dof.ndim != 2 or dof.shape[1] != len(source_names):
        raise ValueError(f"AMP expert dof shape {dof.shape} is inconsistent with its joint_names")
    missing = [name for name in joint_names if name not in source_names]
    if missing:
        raise ValueError(f"AMP expert is missing policy joints: {missing}")
    lookup = {name: index for index, name in enumerate(source_names)}
    return dof[:, [lookup[name] for name in joint_names]]


def _inverse_rotate_np(quaternion_xyzw: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion_xyzw, dtype=np.float64)
    vectors = np.asarray(vectors, dtype=np.float64)
    quaternion = quaternion / np.linalg.norm(quaternion, axis=-1, keepdims=True).clip(min=1.0e-8)
    x, y, z, w = np.moveaxis(quaternion, -1, 0)
    rotation = np.stack(
        (
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - w * z),
            2.0 * (x * z + w * y),
            2.0 * (x * y + w * z),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - w * x),
            2.0 * (x * z - w * y),
            2.0 * (y * z + w * x),
            1.0 - 2.0 * (x * x + y * y),
        ),
        axis=-1,
    ).reshape(quaternion.shape[:-1] + (3, 3))
    return np.einsum("...ji,...j->...i", rotation, vectors)


class PiPlusAmpExpertDataset:
    """Precomputed H0W expert windows for Profile B's 194D discriminator."""

    def __init__(self, features: np.ndarray, *, history_length: int, motion_count: int) -> None:
        expected = expected_h0w_amp_feature_dim(H0W_ACTION_DIM, history_length, len(H0W_KEY_BODIES))
        if features.ndim != 2 or features.shape[1] != expected or not np.isfinite(features).all():
            raise ValueError(f"AMP expert features must be finite [N, {expected}], got {features.shape}")
        self.features = torch.as_tensor(features, dtype=torch.float32)
        self.history_length = int(history_length)
        self.feature_dim = expected
        self.motion_count = int(motion_count)

    @classmethod
    def from_pkl(cls, path: str | Path, *, robot_config: str | Path, history_length: int) -> "PiPlusAmpExpertDataset":
        data = joblib.load(path)
        if not isinstance(data, Mapping) or not data:
            raise ValueError("AMP expert data must be a non-empty motion mapping")
        spec = h0w_robot_training_spec(robot_config)
        model = mujoco.MjModel.from_xml_path(str(spec.robot.xml_path))
        mj_data = mujoco.MjData(model)
        base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, spec.robot.base_body)
        key_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in H0W_KEY_BODIES]
        if base_id < 0 or any(index < 0 for index in key_ids):
            raise ValueError("H0W XML is missing AMP base or key body names")
        rows: list[np.ndarray] = []
        for motion in data.values():
            root_position = np.asarray(motion.get("root_trans_offset", motion.get("root_pos")), dtype=np.float64)
            root_rotation = np.asarray(motion.get("root_rot", motion.get("root_quat")), dtype=np.float64)
            dof = _motion_joint_positions(motion, tuple(spec.robot.control_joint_names))
            frame_count = root_position.shape[0]
            if frame_count < history_length or root_rotation.shape != (frame_count, 4):
                continue
            fps = float(np.asarray(motion.get("fps", 30.0)).reshape(-1)[0])
            root_velocity = np.zeros_like(root_position)
            if frame_count > 1:
                root_velocity[:-1] = np.diff(root_position, axis=0) * fps
                root_velocity[-1] = root_velocity[-2]
            local_velocity = _inverse_rotate_np(root_rotation, root_velocity)
            local_key_positions = np.empty((frame_count, len(key_ids), 3), dtype=np.float64)
            for index in range(frame_count):
                qpos = np.zeros(model.nq, dtype=np.float64)
                qpos[:3] = root_position[index]
                qpos[3:7] = root_rotation[index][[3, 0, 1, 2]]
                for joint_index, joint_name in enumerate(spec.robot.control_joint_names):
                    model_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
                    qpos[int(model.jnt_qposadr[model_joint_id])] = dof[index, joint_index]
                mj_data.qpos[:] = qpos
                mujoco.mj_forward(model, mj_data)
                relative = mj_data.xpos[key_ids] - mj_data.xpos[base_id]
                local_key_positions[index] = _inverse_rotate_np(np.repeat(root_rotation[index][None], len(key_ids), axis=0), relative)
            for index in range(history_length - 1, frame_count):
                rows.append(
                    np.concatenate((local_velocity[index], local_key_positions[index].reshape(-1), dof[index - history_length + 1 : index + 1].reshape(-1)))
                    .astype(np.float32, copy=False)
                )
        expected = expected_h0w_amp_feature_dim(H0W_ACTION_DIM, history_length, len(H0W_KEY_BODIES))
        features = np.stack(rows) if rows else np.zeros((0, expected), dtype=np.float32)
        return cls(features, history_length=history_length, motion_count=len(data))

    def sample(self, batch_size: int, device: torch.device) -> torch.Tensor:
        if not len(self.features):
            raise ValueError("AMP expert data contains no valid history windows")
        return self.features[torch.randint(len(self.features), (int(batch_size),))].to(device)


class AmpDiscriminator(nn.Module):
    """Gradient-penalized feature discriminator used only by Profile B."""

    def __init__(self, feature_dim: int, gradient_penalty_weight: float = 5.0) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(feature_dim, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1),
        )
        self.gradient_penalty_weight = float(gradient_penalty_weight)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)

    def loss(self, expert: torch.Tensor, policy: torch.Tensor) -> dict[str, torch.Tensor]:
        expert_score = self(expert)
        policy_score = self(policy)
        epsilon = torch.rand(expert.shape[0], 1, device=expert.device)
        mixed = (epsilon * expert + (1.0 - epsilon) * policy).requires_grad_(True)
        mixed_score = self(mixed)
        gradients = torch.autograd.grad(mixed_score, mixed, torch.ones_like(mixed_score), create_graph=True)[0]
        gradient_penalty = (gradients.flatten(1).norm(2, dim=1) - 1.0).square().mean()
        loss = functional.mse_loss(expert_score, torch.ones_like(expert_score)) + functional.mse_loss(
            policy_score, torch.zeros_like(policy_score)
        ) + self.gradient_penalty_weight * gradient_penalty
        return {
            "loss": loss,
            "expert_score": expert_score.mean(),
            "policy_score": policy_score.mean(),
            "gradient_penalty": gradient_penalty,
        }

    @torch.no_grad()
    def reward(self, features: torch.Tensor) -> torch.Tensor:
        return (1.0 - (self(features) - 1.0).square()).clamp_min(0.0)


class MimicLiteLocomotionRewardState:
    """Stateful command-locomotion rewards copied for the H0W body contract."""

    def __init__(
        self,
        *,
        num_envs: int,
        num_dof: int,
        dt: float,
        feet_indices: torch.Tensor,
        torso_index: int,
        joint_vel_indices: torch.Tensor,
        joint_deviation_indices: torch.Tensor,
        device: torch.device,
        weights: Mapping[str, float],
    ) -> None:
        if set(weights) != set(MIMICLITE_LOCOMOTION_WEIGHTS):
            raise ValueError("MimicLite profile weights must include exactly the known reward terms")
        self.dt = float(dt)
        self.weights = {name: float(weights[name]) for name in MIMICLITE_LOCOMOTION_WEIGHTS}
        self.feet_indices = feet_indices.to(device=device, dtype=torch.long)
        self.torso_index = int(torso_index)
        self.joint_vel_indices = joint_vel_indices.to(device=device, dtype=torch.long)
        self.joint_deviation_indices = joint_deviation_indices.to(device=device, dtype=torch.long)
        self.previous_actions = torch.zeros(num_envs, num_dof, device=device)
        self.previous_previous_actions = torch.zeros_like(self.previous_actions)
        self.previous_dof_vel = torch.zeros_like(self.previous_actions)
        self.contact_time = torch.zeros(num_envs, len(self.feet_indices), device=device)
        self.air_time = torch.zeros_like(self.contact_time)
        self.previous_contact = torch.zeros_like(self.contact_time, dtype=torch.bool)

    def reset(self, env_ids: torch.Tensor) -> None:
        self.previous_actions[env_ids] = 0.0
        self.previous_previous_actions[env_ids] = 0.0
        self.previous_dof_vel[env_ids] = 0.0
        self.contact_time[env_ids] = 0.0
        self.air_time[env_ids] = 0.0
        self.previous_contact[env_ids] = False

    def compute(self, core: Any, commands: torch.Tensor, actions: torch.Tensor, done: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        contact = core.contact_forces[:, self.feet_indices, 2] > 1.0
        first_contact = contact & ~self.previous_contact
        last_air_time = self.air_time
        contact_time = torch.where(contact, self.contact_time + self.dt, torch.zeros_like(self.contact_time))
        air_time = torch.where(contact, torch.zeros_like(self.air_time), self.air_time + self.dt)
        standing = (commands[:, :2].norm(dim=-1) < 0.1) & (commands[:, 2].abs() < 0.1)
        command_speed = commands[:, :2].norm(dim=-1)
        linear_error = (core.base_lin_vel[:, :2] - commands[:, :2]).square().sum(dim=-1)
        yaw_error = (core.base_ang_vel[:, 2] - commands[:, 2]).square()
        torso_gravity = quat_rotate_inverse(
            core.body_rot[:, self.torso_index],
            torch.tensor([0.0, 0.0, -1.0], device=core.device).expand(core.num_envs, -1),
            w_last=True,
        )
        torso_angular_velocity = quat_rotate_inverse(
            core.body_rot[:, self.torso_index], core.body_ang_vel[:, self.torso_index], w_last=True
        )
        default_dof_pos = core.default_dof_pos + core.default_dof_pos_offset
        action_delta = actions - self.previous_actions
        components = {
            "linvel_exp": torch.exp(-linear_error / 0.16),
            "linvel_projection": (core.base_lin_vel[:, :2] * commands[:, :2]).sum(dim=-1).clamp_max(command_speed),
            "angvel_z_exp": torch.exp(-yaw_error / 0.25),
            "backward_velocity_progress": torch.where(
                commands[:, 0] < -0.05,
                (core.base_lin_vel[:, 0] * commands[:, 0].sign() / commands[:, 0].abs().clamp_min(0.15)).clamp(-1.0, 1.0),
                torch.zeros_like(command_speed),
            ),
            "turn_rate_progress": torch.where(
                commands[:, 2].abs() >= 0.1,
                (core.base_ang_vel[:, 2] * commands[:, 2].sign() / commands[:, 2].abs().clamp_min(0.15)).clamp(-1.0, 1.0),
                torch.zeros_like(command_speed),
            ),
            "single_foot_contact": torch.where(
                (contact_time > 0.1).sum(dim=-1) == 1,
                torch.zeros_like(command_speed),
                -torch.ones_like(command_speed),
            ),
            "angvel_xy_l2": -torso_angular_velocity[:, :2].square().sum(dim=-1),
            "body_upright": 1.0 - torso_gravity[:, :2].square().sum(dim=-1),
            "stand_still": torch.where(
                standing, -(core.dof_pos - default_dof_pos).abs().sum(dim=-1), torch.zeros_like(command_speed)
            ),
            "feet_air_time": ((last_air_time - 0.5).clamp_max(0.0) * first_contact).sum(dim=-1),
            "feet_clearance": torch.zeros_like(command_speed),
            "energy_l1": -(core.torques * core.dof_vel).abs().sum(dim=-1),
            "joint_acc_l2": -((core.dof_vel - self.previous_dof_vel) / max(self.dt, 1.0e-6)).square().sum(dim=-1),
            "action_rate_l2": -action_delta.square().sum(dim=-1),
            "action_rate2_l2": -(actions - 2.0 * self.previous_actions + self.previous_previous_actions).square().sum(dim=-1),
            "joint_vel_l2": -core.dof_vel[:, self.joint_vel_indices].square().sum(dim=-1),
            "joint_deviation_l2": -(
                core.dof_pos[:, self.joint_deviation_indices] - default_dof_pos[:, self.joint_deviation_indices]
            ).square().sum(dim=-1),
        }
        components["single_foot_contact"] = torch.where(
            standing, torch.zeros_like(command_speed), components["single_foot_contact"]
        )
        components["feet_air_time"] = torch.where(standing, torch.zeros_like(command_speed), components["feet_air_time"])
        feet_height = core.body_pos[:, self.feet_indices, 2]
        swing = ~contact
        swing_count = swing.sum(dim=-1)
        clearance_score = torch.exp(-((feet_height - FEET_CLEARANCE_TARGET) / FEET_CLEARANCE_SIGMA).square())
        clearance = (clearance_score * swing).sum(dim=-1) / swing_count.clamp_min(1)
        valid_swing = ((contact_time > 0.1).sum(dim=-1) == 1) & (swing_count == 1) & ~standing
        components["feet_clearance"] = torch.where(valid_swing, clearance, torch.zeros_like(clearance))
        weighted = {name: self.dt * self.weights[name] * value for name, value in components.items()}
        self.previous_previous_actions = self.previous_actions.clone()
        self.previous_actions = actions.detach().clone()
        self.previous_dof_vel = core.dof_vel.detach().clone()
        self.contact_time = contact_time
        self.air_time = air_time
        self.previous_contact = contact
        if torch.any(done):
            self.reset(done.nonzero(as_tuple=False).flatten())
        return sum(weighted.values()), weighted


def _transition_core(live_core: Any, info: Mapping[str, Any]) -> Any:
    terminal_state = info.get("terminal_state")
    if isinstance(terminal_state, Mapping):
        return SimpleNamespace(num_envs=live_core.num_envs, device=live_core.device, **terminal_state)
    return live_core


def _load_latent_reference(path: str | Path, *, required: bool) -> torch.Tensor | None:
    reference_path = Path(path).expanduser().resolve()
    if not reference_path.is_file():
        if required:
            raise FileNotFoundError(f"Profile B requires a H0W projected latent reference bank: {reference_path}")
        return None
    values = np.asarray(joblib.load(reference_path), dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 256 or not len(values):
        raise ValueError(f"H0W latent reference must have shape [N, 256], got {values.shape}")
    bank = torch.as_tensor(values, dtype=torch.float32)
    return functional.normalize(bank, dim=-1).mul(float(bank.shape[-1]) ** 0.5)


def _latent_prior(projected_latent: torch.Tensor, bank: torch.Tensor) -> torch.Tensor:
    normalized_bank = functional.normalize(bank, dim=-1)
    similarity = functional.normalize(projected_latent, dim=-1) @ normalized_bank.T
    return 1.0 - similarity.max(dim=-1).values


def _distributed_context(args: argparse.Namespace) -> tuple[argparse.Namespace, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size <= 1:
        return args, rank, world_size
    if local_rank >= 4 or not torch.cuda.is_available():
        raise RuntimeError("PiPlus H0W DDP can use only CUDA physical GPUs 0-3")
    from datetime import timedelta

    torch.cuda.set_device(local_rank)
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl", init_method="env://", timeout=timedelta(hours=2))
    args.device = f"cuda:{local_rank}"
    args.seed += rank
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(local_rank)
    return args, rank, world_size


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bfm-model", type=Path, default=DEFAULT_BFM_MODEL)
    parser.add_argument("--decoder-path", type=Path, default=DEFAULT_DECODER_PATH)
    parser.add_argument("--robot-config", type=Path, default=DEFAULT_ROBOT_CONFIG)
    parser.add_argument("--motion-dataset", type=Path, default=None)
    parser.add_argument("--expert-dataset", type=Path, default=None)
    parser.add_argument("--latent-reference", type=Path, default=DEFAULT_LATENT_REFERENCE)
    parser.add_argument("--reward-profile", choices=tuple(_AMP_REWARD_PROFILES), default=PROFILE_A_MIMICLITE_SPEED_SAFETY)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu-ids", choices=("single", "all"), default="single")
    parser.add_argument("--work-dir", type=Path, default=PROJECT_ROOT / "runs/amp_stage2_piplus_h0w")
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--rollout-steps", type=int, default=None)
    parser.add_argument("--history-length", type=int, default=8)
    parser.add_argument("--ppo-epochs", type=int, default=5)
    parser.add_argument("--minibatch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--discriminator-learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--amp-weight", type=float, default=None)
    parser.add_argument("--latent-prior-weight", type=float, default=None)
    parser.add_argument("--env-reward-weight", type=float, default=None)
    parser.add_argument("--locomotion-reward-weight", type=float, default=None)
    parser.add_argument("--value-coef", type=float, default=None)
    parser.add_argument("--entropy-coef", type=float, default=None)
    parser.add_argument("--clip-value-loss", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--desired-kl", type=float, default=None)
    parser.add_argument("--clip-ratio", type=float, default=None)
    parser.add_argument("--max-grad-norm", type=float, default=None)
    parser.add_argument("--command-stand-prob", type=float, default=0.05)
    parser.add_argument("--command-turn-prob", type=float, default=0.20)
    parser.add_argument("--command-smoothing", type=float, default=0.02)
    parser.add_argument("--max-episode-length-s", type=float, default=20.0)
    parser.add_argument("--disable-domain-randomization", action="store_true")
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main(parsed_args: argparse.Namespace | None = None) -> None:
    args = _parse_args() if parsed_args is None else parsed_args
    profile = _apply_reward_profile_defaults(args)
    ppo_profile = profile.ppo if profile.ppo is not None else type(UNITREE_PPO_PROFILE)(
        value_coef=0.5,
        entropy_coef=0.003,
        clip_value_loss=False,
        desired_kl=None,
        schedule="fixed",
        learning_rate=1.0e-4,
        rollout_steps=32,
        num_minibatches=1,
        clip_ratio=0.2,
        gamma=0.99,
        gae_lambda=0.95,
        max_grad_norm=1.0,
    )
    for name in ("rollout_steps", "learning_rate", "value_coef", "entropy_coef", "clip_value_loss", "desired_kl", "clip_ratio", "max_grad_norm"):
        value = getattr(args, name, None)
        if value is None:
            setattr(args, name, getattr(ppo_profile, name))
    args.discount = args.discount if args.discount is not None else ppo_profile.gamma
    args.gae_lambda = args.gae_lambda if args.gae_lambda is not None else ppo_profile.gae_lambda
    if args.minibatch_size is None:
        args.minibatch_size = 1024 if profile.ppo is None else max(1, args.num_envs * args.rollout_steps // ppo_profile.num_minibatches)
    if args.history_length <= 0 or args.num_envs <= 0:
        raise ValueError("history-length and num-envs must be positive")
    if args.history_length != 8:
        raise ValueError("PiPlus H0W AMP export currently requires --history-length 8")
    if not 0.0 <= args.command_smoothing <= 1.0:
        raise ValueError("--command-smoothing must be in [0, 1]")
    expert_path = args.expert_dataset.expanduser().resolve()
    if profile.uses_amp and not expert_path.is_file():
        raise FileNotFoundError(
            "Profile B requires a dedicated H0W walking/run-with-stand AMP dataset; "
            f"missing {expert_path}"
        )
    contract = validate_h0w_assets(args.robot_config, args.bfm_model, args.decoder_path)
    latent_reference = _load_latent_reference(args.latent_reference, required=profile.uses_latent_prior)
    if args.dry_run:
        result = {
            **contract,
            "task": AMP_STAGE2_TASK,
            "reward_profile": profile.name,
            "reward_profile_description": profile.description,
            "uses_amp": profile.uses_amp,
            "amp_feature_dim": H0W_AMP_FEATURE_DIM,
            "reward_source": profile.source,
            "omitted_unitree_terms": dict(profile.omitted_unitree_terms or {}),
            "reward_source_versions": UNITREE_REWARD_SOURCE_VERSIONS if profile.source else None,
            "ppo": {name: getattr(ppo_profile, name) for name in ppo_profile.__dataclass_fields__},
            "effective_reward": {
                "env_reward_weight": args.env_reward_weight,
                "locomotion_reward_weight": args.locomotion_reward_weight,
                "amp_weight": args.amp_weight,
                "latent_prior_weight": args.latent_prior_weight,
                "locomotion_weights": dict(profile.locomotion_weights),
            },
        }
        if profile.uses_amp:
            expert = PiPlusAmpExpertDataset.from_pkl(expert_path, robot_config=args.robot_config, history_length=args.history_length)
            result.update({"expert_windows": int(len(expert.features)), "expert_motion_count": expert.motion_count})
        print(json.dumps(result, indent=2, sort_keys=True), flush=True)
        return
    args, rank, world_size = _distributed_context(args)
    if args.gpu_ids == "all" and world_size == 1:
        raise ValueError("--gpu-ids all requires torchrun with CUDA_VISIBLE_DEVICES=0,1,2,3")
    if args.smoke:
        args.num_envs = min(args.num_envs, 2)
        args.iterations = 1
        args.rollout_steps = min(args.rollout_steps, 4)
        args.ppo_epochs = 1
        args.minibatch_size = min(args.minibatch_size, args.num_envs * args.rollout_steps)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    decoder = load_decoder(args.bfm_model, args.decoder_path, device)
    environment, robot_training = build_h0w_locomotion_env(
        device=str(device),
        robot_config=args.robot_config,
        motion_dataset=args.motion_dataset,
        num_envs=args.num_envs,
        seed=args.seed,
        max_episode_length_s=args.max_episode_length_s,
        disable_domain_randomization=args.disable_domain_randomization,
    )
    try:
        observation, _ = environment.reset(to_numpy=False)
        observation_t = to_torch_observation(observation, device)
        commands = torch.zeros(args.num_envs, 3, device=device)
        command_targets = sample_commands(
            args.num_envs,
            device,
            low=(-0.5, -0.5, -1.0),
            high=(1.0, 0.5, 1.0),
            stand_probability=args.command_stand_prob,
            turn_probability=args.command_turn_prob,
        )
        policy = CommandEncoderPolicy(flatten_encoder_observation(observation_t, commands).shape[-1], decoder.z_dim).to(device)
        if latent_reference is not None and profile.name != PROFILE_C_UNITREE_VELOCITY_H0W_COMPAT:
            bank = latent_reference.to(device)
            with torch.no_grad():
                policy.latent_mean.bias.copy_(bank.mean(dim=0))
                policy.latent_log_std.copy_(bank.std(dim=0, unbiased=False).clamp_min(1.0e-3).log())
        optimizer = torch.optim.Adam(policy.parameters(), lr=args.learning_rate)
        expert = (
            PiPlusAmpExpertDataset.from_pkl(expert_path, robot_config=args.robot_config, history_length=args.history_length)
            if profile.uses_amp
            else None
        )
        discriminator = AmpDiscriminator(H0W_AMP_FEATURE_DIM).to(device) if expert is not None else None
        discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=args.discriminator_learning_rate) if discriminator else None
        broadcast_module_state(policy)
        if discriminator:
            broadcast_module_state(discriminator)
        start_iteration = 0
        if args.resume is not None:
            resume_path = args.resume.expanduser().resolve()
            if not resume_path.is_file():
                raise FileNotFoundError(f"PiPlus H0W AMP checkpoint does not exist: {resume_path}")
            checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
            checkpoint_metadata = checkpoint.get("metadata", {})
            if not checkpoint_task_is_compatible(checkpoint_metadata):
                raise ValueError("Cannot resume a non-AMP PiPlus H0W stage-2 checkpoint")
            checkpoint_profile = checkpoint_metadata.get("reward_profile")
            if checkpoint_profile not in {None, profile.name}:
                raise ValueError(f"Cannot resume reward profile {checkpoint_profile!r} as {profile.name!r}")
            policy.load_state_dict(checkpoint["policy"])
            if "policy_optimizer" in checkpoint:
                optimizer.load_state_dict(checkpoint["policy_optimizer"])
            if discriminator is not None and discriminator_optimizer is not None and "discriminator" in checkpoint:
                discriminator.load_state_dict(checkpoint["discriminator"])
                if "discriminator_optimizer" in checkpoint:
                    discriminator_optimizer.load_state_dict(checkpoint["discriminator_optimizer"])
            start_iteration = int(checkpoint.get("iteration", 0))
        body_names = tuple(environment._env.body_names)
        missing_bodies = [name for name in H0W_KEY_BODIES if name not in body_names]
        if missing_bodies:
            raise ValueError(f"H0W simulator is missing AMP key bodies: {missing_bodies}")
        key_body_indices = torch.tensor([body_names.index(name) for name in H0W_KEY_BODIES], device=device)
        spec = h0w_robot_training_spec(args.robot_config)
        torso_index = body_names.index(spec.torso_name)
        joint_vel_indices = torch.tensor(
            [index for index, name in enumerate(spec.robot.control_joint_names) if "shoulder" in name], device=device
        )
        joint_deviation_indices = torch.tensor(
            [
                index
                for index, name in enumerate(spec.robot.control_joint_names)
                if any(token in name for token in ("shoulder", "elbow", "hip"))
            ],
            device=device,
        )
        locomotion_state = (
            UnitreeVelocityLocomotionRewardState(
                num_envs=args.num_envs,
                num_dof=H0W_ACTION_DIM,
                dt=environment._env.dt,
                feet_indices=environment._env.feet_indices,
                torso_index=torso_index,
                device=device,
                weights=profile.locomotion_weights,
            )
            if profile.name == PROFILE_C_UNITREE_VELOCITY_H0W_COMPAT
            else MimicLiteLocomotionRewardState(
            num_envs=args.num_envs,
            num_dof=H0W_ACTION_DIM,
            dt=environment._env.dt,
            feet_indices=environment._env.feet_indices,
            torso_index=torso_index,
            joint_vel_indices=joint_vel_indices,
            joint_deviation_indices=joint_deviation_indices,
            device=device,
            weights=profile.locomotion_weights,
            )
        )
        joint_history = environment._env.dof_pos.to(device).unsqueeze(1).repeat(1, args.history_length, 1)
        work_dir = args.work_dir.expanduser().resolve()
        metadata = {
            "task": AMP_STAGE2_TASK,
            "amp": profile.uses_amp,
            "reward_profile": profile.name,
            "reward_profile_description": profile.description,
            "robot": robot_training.robot.name,
            "robot_config": str(args.robot_config.expanduser().resolve()),
            "motion_dataset": str(args.motion_dataset.expanduser().resolve()),
            "expert_dataset": str(expert_path),
            "bfm_model": str(args.bfm_model.expanduser().resolve()),
            "decoder_path": str(args.decoder_path.expanduser().resolve()),
            "contract": contract,
            "z_dim": policy.z_dim,
            "encoder_input_dim": policy.input_dim,
            "amp_feature_dim": H0W_AMP_FEATURE_DIM if profile.uses_amp else None,
            "distributed_world_size": world_size,
            "resume": str(args.resume.expanduser().resolve()) if args.resume is not None else None,
            "resume_iteration": start_iteration,
            "reward_source": profile.source,
            "reward_source_versions": UNITREE_REWARD_SOURCE_VERSIONS if profile.source else None,
            "omitted_unitree_terms": dict(profile.omitted_unitree_terms or {}),
            "ppo": {name: getattr(ppo_profile, name) for name in ppo_profile.__dataclass_fields__},
            "effective_reward": {
                "env_reward_weight": args.env_reward_weight,
                "locomotion_reward_weight": args.locomotion_reward_weight,
                "amp_weight": args.amp_weight,
                "latent_prior_weight": args.latent_prior_weight,
                "locomotion_weights": dict(profile.locomotion_weights),
            },
        }
        if rank == 0:
            work_dir.mkdir(parents=True, exist_ok=True)
            (work_dir / "config.json").write_text(json.dumps(metadata, indent=2) + "\n")
        barrier()
        for iteration in range(start_iteration, args.iterations):
            parts: list[dict[str, torch.Tensor]] = []
            amp_features: list[torch.Tensor] = []
            locomotion_rewards: list[torch.Tensor] = []
            environment_rewards: list[torch.Tensor] = []
            for _ in range(args.rollout_steps):
                features = flatten_encoder_observation(observation_t, commands)
                with torch.no_grad():
                    raw_z, old_log_prob, values = policy.sample(features)
                    actions = decoder.act(observation_t, decoder.project_z(raw_z))
                next_observation, environment_reward, terminated, truncated, info = environment.step(
                    actions.to(environment._env.device), to_numpy=False
                )
                transition = _transition_core(environment._env, info)
                transition = SimpleNamespace(
                    **{name: value.to(device) if torch.is_tensor(value) else value for name, value in vars(transition).items()}
                )
                terminated = terminated.to(device=device, dtype=torch.bool)
                truncated = truncated.to(device=device, dtype=torch.bool)
                done = terminated | truncated
                locomotion_reward, _components = locomotion_state.compute(transition, commands, actions, done)
                joint_history = torch.cat((joint_history[:, 1:], transition.dof_pos[:, None]), dim=1)
                amp_features.append(build_online_amp_feature(transition, joint_history, key_body_indices).detach())
                parts.append(
                    {
                        "features": features.detach(),
                        "raw_z": raw_z.detach(),
                        "old_log_prob": old_log_prob.detach(),
                        "values": values.detach(),
                        "rewards": torch.zeros_like(values),
                        "terminated": terminated,
                        "truncated": truncated,
                    }
                )
                locomotion_rewards.append(locomotion_reward.detach())
                environment_rewards.append(environment_reward.to(device))
                observation_t = to_torch_observation(next_observation, device)
                commands = commands + args.command_smoothing * (command_targets - commands)
                if torch.any(done):
                    commands[done] = 0.0
                    command_targets[done] = sample_commands(
                        int(done.sum()),
                        device,
                        low=(-0.5, -0.5, -1.0),
                        high=(1.0, 0.5, 1.0),
                        stand_probability=args.command_stand_prob,
                        turn_probability=args.command_turn_prob,
                    )
                    joint_history[done] = environment._env.dof_pos.to(device)[done, None]
            rollout = PpoRollout(**{name: torch.stack([part[name] for part in parts]) for name in parts[0]})
            locomotion_tensor = torch.stack(locomotion_rewards)
            environment_tensor = torch.stack(environment_rewards)
            feature_tensor = torch.stack(amp_features)
            amp_reward = torch.zeros_like(rollout.rewards)
            discriminator_metrics: dict[str, torch.Tensor] | None = None
            if expert is not None and discriminator is not None and discriminator_optimizer is not None:
                policy_features = feature_tensor.flatten(0, 1)
                discriminator_metrics = discriminator.loss(expert.sample(policy_features.shape[0], device), policy_features.detach())
                discriminator_optimizer.zero_grad(set_to_none=True)
                discriminator_metrics["loss"].backward()
                average_gradients(discriminator.parameters())
                discriminator_optimizer.step()
                amp_reward = discriminator.reward(policy_features).reshape_as(rollout.rewards)
            projected = decoder.project_z(rollout.raw_z.flatten(0, 1))
            if profile.uses_latent_prior:
                assert latent_reference is not None
                latent_penalty = _latent_prior(projected, latent_reference.to(device)).reshape_as(rollout.rewards)
            else:
                latent_penalty = torch.zeros_like(rollout.rewards)
            rollout.rewards = (
                args.env_reward_weight * environment_tensor
                + args.locomotion_reward_weight * locomotion_tensor
                + args.amp_weight * amp_reward
                - args.latent_prior_weight * latent_penalty
            )
            with torch.no_grad():
                last_value = policy(flatten_encoder_observation(observation_t, commands))[2]
            advantages, returns = compute_gae(
                rollout.rewards,
                rollout.values,
                rollout.terminated,
                rollout.truncated,
                last_value,
                discount=args.discount,
                gae_lambda=args.gae_lambda,
            )
            metrics = ppo_update(
                policy,
                rollout,
                advantages,
                returns,
                optimizer,
                epochs=args.ppo_epochs,
                minibatch_size=args.minibatch_size,
                entropy_coef=args.entropy_coef,
                value_coef=args.value_coef,
                clip_value_loss=args.clip_value_loss,
                desired_kl=args.desired_kl,
                schedule=ppo_profile.schedule,
            )
            metrics.update(
                {
                    "iteration": float(iteration + 1),
                    "reward_mean": float(rollout.rewards.mean()),
                    "locomotion_reward_mean": float(locomotion_tensor.mean()),
                    "amp_reward_mean": float(amp_reward.mean()),
                    "latent_prior_penalty": float(latent_penalty.mean()),
                    "termination_rate": float(rollout.terminated.float().mean()),
                    "discriminator_loss": float(discriminator_metrics["loss"]) if discriminator_metrics is not None else 0.0,
                }
            )
            metrics = average_metrics(metrics)
            if rank == 0:
                serializable = {name: float(value) if torch.is_tensor(value) else value for name, value in metrics.items()}
                print(json.dumps(serializable, sort_keys=True), flush=True)
                if (iteration + 1) % args.save_every == 0 or iteration + 1 == args.iterations:
                    payload: dict[str, Any] = {
                        "policy": policy.state_dict(),
                        "policy_optimizer": optimizer.state_dict(),
                        "iteration": iteration + 1,
                        "metadata": metadata,
                    }
                    if discriminator is not None and discriminator_optimizer is not None:
                        payload["discriminator"] = discriminator.state_dict()
                        payload["discriminator_optimizer"] = discriminator_optimizer.state_dict()
                    torch.save(payload, work_dir / f"checkpoint_{iteration + 1}.pt")
            barrier()
    finally:
        environment.close()
        if is_distributed():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
