"""Command, observation history, curriculum, and rewards for ball kicking."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from humanoidverse.utils.torch_utils import my_quat_rotate, quat_rotate_inverse

KICK_REWARD_WEIGHTS = {
    "ball_velocity": 20.0,
    "correct_foot": 5.0,
    "wrong_foot": -5.0,
    "extra_touch": -1.0,
    "approach_progress": 5.0,
    "post_kick_stabilize": -2.0,
}


@dataclass(frozen=True)
class KickCurriculumState:
    ball_radius_range: tuple[float, float]
    ball_angle_range: tuple[float, float]
    ball_speed_range: tuple[float, float]
    target_speed_range: tuple[float, float]
    target_direction_range: tuple[float, float]
    foot_mode_probabilities: tuple[float, float, float]


def _lerp(start: float, end: float, alpha: float) -> float:
    return start + (end - start) * alpha


def kick_curriculum(iteration: int, total_iterations: int) -> KickCurriculumState:
    """Expand from a fixed forward kick to the full task distribution."""
    progress = max(0.0, min(1.0, iteration / max(total_iterations - 1, 1)))
    direction_progress = max(0.0, min(1.0, (progress - 0.10) / 0.65))
    moving_ball_progress = max(0.0, min(1.0, (progress - 0.50) / 0.50))
    foot_progress = max(0.0, min(1.0, progress / 0.25))
    left = 0.4 * foot_progress
    right = 0.4 * foot_progress
    return KickCurriculumState(
        ball_radius_range=(0.25, _lerp(0.32, 0.40, progress)),
        ball_angle_range=(-_lerp(math.pi / 6.0, math.pi, direction_progress), _lerp(math.pi / 6.0, math.pi, direction_progress)),
        ball_speed_range=(0.0, _lerp(0.0, 0.2, moving_ball_progress)),
        target_speed_range=(1.0, _lerp(2.0, 3.0, progress)),
        target_direction_range=(-_lerp(math.pi / 6.0, math.pi, direction_progress), _lerp(math.pi / 6.0, math.pi, direction_progress)),
        foot_mode_probabilities=(left, right, 1.0 - left - right),
    )


class KickCommandState:
    """Per-environment world-fixed target velocity and requested kicking foot."""

    def __init__(self, num_envs: int, device: torch.device) -> None:
        self.device = device
        self.target_velocity_w = torch.zeros(num_envs, 3, device=device)
        self.foot_encoding = torch.ones(num_envs, 2, device=device)

    def sample(self, env_ids: torch.Tensor, base_quat: torch.Tensor, curriculum: KickCurriculumState) -> None:
        if env_ids.numel() == 0:
            return
        count = int(env_ids.numel())
        speed = torch.empty(count, device=self.device).uniform_(*curriculum.target_speed_range)
        direction = torch.empty(count, device=self.device).uniform_(*curriculum.target_direction_range)
        forward_b = torch.zeros(count, 3, device=self.device)
        forward_b[:, 0] = 1.0
        forward_w = my_quat_rotate(base_quat[env_ids], forward_b)
        heading = torch.atan2(forward_w[:, 1], forward_w[:, 0])
        world_direction = heading + direction
        self.target_velocity_w[env_ids] = torch.stack(
            (speed * torch.cos(world_direction), speed * torch.sin(world_direction), torch.zeros_like(speed)), dim=-1
        )
        mode = torch.multinomial(torch.tensor(curriculum.foot_mode_probabilities, device=self.device), count, replacement=True)
        encodings = torch.tensor(((1.0, 0.0), (0.0, 1.0), (1.0, 1.0)), device=self.device)
        self.foot_encoding[env_ids] = encodings[mode]

    def command(self, base_quat: torch.Tensor) -> torch.Tensor:
        target_velocity_b = quat_rotate_inverse(base_quat, self.target_velocity_w, w_last=True)
        return torch.cat((target_velocity_b, self.foot_encoding), dim=-1)


class KickObservationHistory:
    """History of deployable kick commands and ball observations."""

    def __init__(self, num_envs: int, history_length: int, device: torch.device) -> None:
        if history_length < 1:
            raise ValueError("history_length must be positive")
        self.values = torch.zeros(num_envs, history_length, 11, device=device)

    @staticmethod
    def frame(command: torch.Tensor, ball_position_b: torch.Tensor, ball_velocity_b: torch.Tensor) -> torch.Tensor:
        return torch.cat((command, ball_position_b, ball_velocity_b), dim=-1)

    def reset(self, env_ids: torch.Tensor, command: torch.Tensor, ball_position_b: torch.Tensor, ball_velocity_b: torch.Tensor) -> None:
        if env_ids.numel() == 0:
            return
        frame = self.frame(command, ball_position_b, ball_velocity_b)
        self.values[env_ids] = frame[env_ids].unsqueeze(1)

    def append(self, command: torch.Tensor, ball_position_b: torch.Tensor, ball_velocity_b: torch.Tensor) -> None:
        frame = self.frame(command, ball_position_b, ball_velocity_b)
        self.values = torch.cat((self.values[:, 1:], frame.unsqueeze(1)), dim=1)

    def flattened(self) -> torch.Tensor:
        return self.values.reshape(self.values.shape[0], -1)


def flatten_kick_encoder_observation(obs: dict[str, torch.Tensor], history: KickObservationHistory | torch.Tensor) -> torch.Tensor:
    pieces = [obs["state"]]
    if "last_action" in obs:
        pieces.append(obs["last_action"])
    if "history_actor" in obs:
        pieces.append(obs["history_actor"])
    history_values = history.flattened() if isinstance(history, KickObservationHistory) else history.reshape(history.shape[0], -1)
    pieces.append(history_values)
    return torch.cat(pieces, dim=-1)


class KickRewardState:
    """Phase-aware kick reward with event rewards for the first ball contact."""

    def __init__(
        self,
        *,
        num_envs: int,
        dt: float,
        feet_indices: torch.Tensor,
        contact_force_threshold: float,
        velocity_std: float,
        device: torch.device,
    ) -> None:
        self.dt = float(dt)
        self.feet_indices = feet_indices.to(device=device, dtype=torch.long)
        self.contact_force_threshold = float(contact_force_threshold)
        self.velocity_std = float(velocity_std)
        self.prev_contact = torch.zeros(num_envs, 2, dtype=torch.bool, device=device)
        self.has_touched = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.has_valid_kick = torch.zeros_like(self.has_touched)
        self.prev_selected_foot_distance = torch.zeros(num_envs, device=device)

    def _selected_foot_distance(self, core, command: torch.Tensor) -> torch.Tensor:
        distance = torch.linalg.vector_norm(core.body_pos[:, self.feet_indices] - core.ball_pos_w.unsqueeze(1), dim=-1)
        allowed = command[:, 3:5] > 0.5
        return torch.where(allowed, distance, torch.full_like(distance, float("inf"))).amin(dim=-1)

    def reset(self, env_ids: torch.Tensor, core, command: torch.Tensor) -> None:
        if env_ids.numel() == 0:
            return
        self.prev_contact[env_ids] = False
        self.has_touched[env_ids] = False
        self.has_valid_kick[env_ids] = False
        self.prev_selected_foot_distance[env_ids] = self._selected_foot_distance(core, command)[env_ids]

    def compute(self, core, command: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        contact = core.foot_ball_contact_force > self.contact_force_threshold
        rising = contact & ~self.prev_contact
        any_rising = rising.any(dim=-1)
        first_touch = any_rising & ~self.has_touched
        allowed = command[:, 3:5] > 0.5
        correct = first_touch & (rising & allowed).any(dim=-1) & ~(rising & ~allowed).any(dim=-1)
        wrong = first_touch & (rising & ~allowed).any(dim=-1)
        extra = any_rising & self.has_touched
        self.has_touched |= first_touch
        self.has_valid_kick |= correct

        velocity_error = (core.ball_linear_velocity_b - command[:, :3]).square().sum(dim=-1)
        ball_velocity = torch.exp(-velocity_error / max(self.velocity_std**2, 1.0e-6)) * self.has_valid_kick.float()
        selected_distance = self._selected_foot_distance(core, command)
        approach_progress = (self.prev_selected_foot_distance - selected_distance).clamp(-0.05, 0.05) * (~self.has_touched).float()
        default_position = core.default_dof_pos + core.default_dof_pos_offset
        post_kick_stabilize = (core.dof_pos - default_position).square().mean(dim=-1) * self.has_touched.float()

        weighted = {
            "ball_velocity": self.dt * KICK_REWARD_WEIGHTS["ball_velocity"] * ball_velocity,
            "correct_foot": KICK_REWARD_WEIGHTS["correct_foot"] * correct.float(),
            "wrong_foot": KICK_REWARD_WEIGHTS["wrong_foot"] * wrong.float(),
            "extra_touch": KICK_REWARD_WEIGHTS["extra_touch"] * extra.float(),
            "approach_progress": KICK_REWARD_WEIGHTS["approach_progress"] * approach_progress,
            "post_kick_stabilize": self.dt * KICK_REWARD_WEIGHTS["post_kick_stabilize"] * post_kick_stabilize,
        }
        diagnostics = {
            "contact": contact.any(dim=-1),
            "correct_touch": correct,
            "wrong_touch": wrong,
            "extra_touch": extra,
            "has_valid_kick": self.has_valid_kick.clone(),
            "ball_velocity_error": velocity_error.sqrt(),
        }
        self.prev_contact = contact
        self.prev_selected_foot_distance = selected_distance
        return sum(weighted.values()), weighted, diagnostics
