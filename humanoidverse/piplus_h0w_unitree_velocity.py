"""Unitree-inspired H0W velocity rewards adapted from the Unitree MJLab snapshot."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

import torch

from humanoidverse.utils.torch_utils import quat_rotate_inverse

UNITREE_REWARD_SOURCE = "unitreerobotics/unitree_rl_mjlab@1425b15f73bd4095f0df53709d7c389c3eb9e790"
UNITREE_REWARD_SOURCE_VERSIONS = {"mjlab": "1.2.0", "rsl_rl": "5.0.1"}
UNITREE_OMITTED_TERMS = {
    "angular_momentum": "H0W core does not expose the Unitree angular-momentum observation.",
    "g1_self_collision": "H0W has no G1 self-collision sensor contract.",
    "g1_variable_posture": "H0W has no G1 posture regex/std map.",
}

UNITREE_VELOCITY_WEIGHTS: Mapping[str, float] = {
    "tracking_lin_vel": 1.0,
    "tracking_ang_vel": 1.0,
    "orientation_l2": -1.0,
    "ang_vel_xy_l2": -0.05,
    "termination": -200.0,
    "dof_acc_l2": -2.5e-7,
    "dof_pos_limits": -10.0,
    "action_rate_l2": -0.05,
    "gait_phase": 0.5,
    "feet_clearance": -1.0,
    "feet_slip": -0.25,
    "soft_landing": -1.0e-3,
    "stand_still": -1.0,
}


class UnitreeRewardCore(Protocol):
    num_envs: int
    base_lin_vel: torch.Tensor
    base_ang_vel: torch.Tensor
    body_ang_vel: torch.Tensor
    body_pos: torch.Tensor
    body_rot: torch.Tensor
    contact_forces: torch.Tensor
    dof_pos: torch.Tensor
    dof_vel: torch.Tensor
    default_dof_pos: torch.Tensor
    default_dof_pos_offset: torch.Tensor
    body_vel: torch.Tensor | None
    dof_pos_limits: torch.Tensor | None


class UnitreeVelocityLocomotionRewardState:
    """Stateful Unitree velocity rewards using the H0W body-array contract."""

    def __init__(
        self,
        *,
        num_envs: int,
        num_dof: int,
        dt: float,
        device: torch.device,
        feet_indices: torch.Tensor | None = None,
        torso_index: int = 0,
        weights: Mapping[str, float] = UNITREE_VELOCITY_WEIGHTS,
    ) -> None:
        self.dt = float(dt)
        self.device = device
        self.torso_index = int(torso_index)
        self.feet_indices = torch.arange(2, device=device, dtype=torch.long) if feet_indices is None else feet_indices.to(device=device, dtype=torch.long)
        self.weights = {name: float(weights[name]) for name in UNITREE_VELOCITY_WEIGHTS}
        self.previous_actions = torch.zeros(num_envs, num_dof, device=device)
        self.previous_dof_vel = torch.zeros_like(self.previous_actions)
        self.previous_contact = torch.zeros(num_envs, len(self.feet_indices), dtype=torch.bool, device=device)
        self.episode_steps = torch.zeros(num_envs, dtype=torch.long, device=device)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Reset temporal reward state for completed environments."""
        self.previous_actions[env_ids] = 0.0
        self.previous_dof_vel[env_ids] = 0.0
        self.previous_contact[env_ids] = False
        self.episode_steps[env_ids] = 0

    def compute(
        self,
        core: UnitreeRewardCore,
        commands: torch.Tensor,
        actions: torch.Tensor,
        done: torch.Tensor,
        episode_steps: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute raw signed terms and their dt-scaled weighted reward."""
        available_feet = self.feet_indices[self.feet_indices < core.contact_forces.shape[1]]
        feet_forces = core.contact_forces[:, available_feet]
        contact = feet_forces.norm(dim=-1) > 1.0
        first_contact = contact & ~self.previous_contact[:, : contact.shape[1]]
        if episode_steps is None:
            current_steps = self.episode_steps
        else:
            current_steps = episode_steps.to(device=commands.device, dtype=torch.long)
        command_active = commands[:, :2].norm(dim=-1) + commands[:, 2].abs() > 0.1
        standing = ~command_active
        linear_error = (core.base_lin_vel[:, :2] - commands[:, :2]).square().sum(dim=-1) + 2.0 * core.base_lin_vel[:, 2].square()
        angular_error = (core.base_ang_vel[:, 2] - commands[:, 2]).square() + 0.05 * core.body_ang_vel[:, self.torso_index, :2].square().sum(dim=-1)
        torso_gravity = quat_rotate_inverse(
            core.body_rot[:, self.torso_index],
            core.body_rot.new_tensor([0.0, 0.0, -1.0]).expand(core.num_envs, -1),
            w_last=True,
        )
        body_velocity = getattr(core, "body_vel", None)
        feet_velocity = (
            body_velocity[:, available_feet]
            if body_velocity is not None
            else torch.zeros_like(core.body_pos[:, available_feet])
        )
        feet_height = core.body_pos[:, available_feet, 2]
        foot_speed_xy = feet_velocity[..., :2].norm(dim=-1)
        offsets = core.body_pos.new_tensor([0.0, 0.5])[: contact.shape[1]]
        gait_phase = (current_steps.to(dtype=commands.dtype)[:, None] * self.dt / 0.6 + offsets) % 1.0
        desired_stance = gait_phase < 0.56
        gait_agreement = (
            torch.where(desired_stance == contact, torch.ones_like(feet_height), -torch.ones_like(feet_height)).mean(dim=-1)
            if contact.shape[1]
            else torch.zeros_like(commands[:, 0])
        )
        swing = ~contact
        clearance = (feet_height - 0.10).abs() * feet_velocity[..., :2].norm(dim=-1)
        force_magnitude = feet_forces.norm(dim=-1)
        limits = getattr(core, "dof_pos_limits", None)
        if limits is None:
            limit_cost = torch.zeros_like(commands[:, 0])
        else:
            if limits.ndim == 3 and limits.shape[0] == 1:
                limits = limits[0]
            lower = limits[..., 0]
            upper = limits[..., 1]
            excess = torch.maximum(lower - core.dof_pos, core.dof_pos - upper).clamp_min(0.0)
            limit_cost = excess.square().sum(dim=-1)
        default_dof_pos = core.default_dof_pos + core.default_dof_pos_offset
        action_delta = actions - self.previous_actions
        components = {
            "tracking_lin_vel": torch.exp(-linear_error / 0.25),
            "tracking_ang_vel": torch.exp(-angular_error / 0.5),
            "orientation_l2": -torso_gravity[:, :2].square().sum(dim=-1),
            "ang_vel_xy_l2": -core.body_ang_vel[:, self.torso_index, :2].square().sum(dim=-1),
            # Adapted from Unitree ``is_terminated``: Stage2 separates terminated/truncated after reward computation.
            "termination": -done.to(dtype=commands.dtype),
            "dof_acc_l2": -((core.dof_vel - self.previous_dof_vel) / max(self.dt, 1.0e-6)).square().sum(dim=-1),
            "dof_pos_limits": -limit_cost,
            "action_rate_l2": -action_delta.square().sum(dim=-1),
            "gait_phase": torch.where(command_active, gait_agreement, torch.zeros_like(gait_agreement)),
            "feet_clearance": torch.where(command_active, -(clearance * swing).sum(dim=-1), torch.zeros_like(clearance[:, 0])),
            "feet_slip": torch.where(command_active, -(foot_speed_xy.square() * contact).sum(dim=-1), torch.zeros_like(clearance[:, 0])),
            "soft_landing": torch.where(command_active, -(force_magnitude * first_contact).sum(dim=-1), torch.zeros_like(clearance[:, 0])),
            "stand_still": torch.where(standing, -(core.dof_pos - default_dof_pos).square().sum(dim=-1), torch.zeros_like(commands[:, 0])),
        }
        cost_terms = {
            "orientation_l2",
            "ang_vel_xy_l2",
            "termination",
            "dof_acc_l2",
            "dof_pos_limits",
            "action_rate_l2",
            "feet_clearance",
            "feet_slip",
            "soft_landing",
            "stand_still",
        }
        weighted = {
            name: self.dt * (abs(self.weights[name]) if name in cost_terms else self.weights[name]) * value
            for name, value in components.items()
        }
        self.previous_actions = actions.detach().clone()
        self.previous_dof_vel = core.dof_vel.detach().clone()
        self.previous_contact = contact
        self.episode_steps = current_steps + 1
        if torch.any(done):
            self.reset(done.nonzero(as_tuple=False).flatten())
        return sum(weighted.values()), weighted | components
