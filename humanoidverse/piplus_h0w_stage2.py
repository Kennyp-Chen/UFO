"""Shared PPO primitives for PiPlus H0W 22DoF stage-2 fine tuning."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import torch
from torch import nn
from torch.distributions import Normal

from humanoidverse.distributed import average_gradients
from humanoidverse.piplus_h0w_onnx_decoder import H0W_ACTION_DIM, H0W_HISTORY_DIM, H0W_LATENT_DIM, H0W_STATE_DIM

COMMAND_LOW = (-0.8, -0.5, -0.8)
COMMAND_HIGH = (0.8, 0.5, 0.8)
COMMAND_SCALE = (1.25, 5.0, 1.25)
DOF_VELOCITY_SCALE = 0.05


@dataclass(frozen=True, slots=True)
class PpoProfile:
    value_coef: float
    entropy_coef: float
    clip_value_loss: bool
    desired_kl: float | None
    schedule: str
    learning_rate: float
    rollout_steps: int
    num_minibatches: int
    clip_ratio: float
    gamma: float
    gae_lambda: float
    max_grad_norm: float


UNITREE_PPO_PROFILE = PpoProfile(
    value_coef=1.0,
    entropy_coef=0.01,
    clip_value_loss=True,
    desired_kl=0.01,
    schedule="adaptive",
    learning_rate=1.0e-3,
    rollout_steps=24,
    num_minibatches=4,
    clip_ratio=0.2,
    gamma=0.99,
    gae_lambda=0.95,
    max_grad_norm=1.0,
)
UNITREE_PPO_CONFIG = MappingProxyType({
    "value_coef": UNITREE_PPO_PROFILE.value_coef,
    "entropy_coef": UNITREE_PPO_PROFILE.entropy_coef,
    "clip_value_loss": UNITREE_PPO_PROFILE.clip_value_loss,
    "desired_kl": UNITREE_PPO_PROFILE.desired_kl,
    "schedule": UNITREE_PPO_PROFILE.schedule,
    "learning_rate": UNITREE_PPO_PROFILE.learning_rate,
    "rollout_steps": UNITREE_PPO_PROFILE.rollout_steps,
    "num_minibatches": UNITREE_PPO_PROFILE.num_minibatches,
    "clip_ratio": UNITREE_PPO_PROFILE.clip_ratio,
    "gamma": UNITREE_PPO_PROFILE.gamma,
    "gae_lambda": UNITREE_PPO_PROFILE.gae_lambda,
    "max_grad_norm": UNITREE_PPO_PROFILE.max_grad_norm,
})


class CommandEncoderPolicy(nn.Module):
    """Stochastic velocity-command-to-latent policy with a PPO value head."""

    def __init__(self, input_dim: int, z_dim: int = H0W_LATENT_DIM, hidden_dim: int = 256, hidden_layers: int = 1) -> None:
        super().__init__()
        if hidden_layers < 1:
            raise ValueError("hidden_layers must be positive")
        self.input_dim = int(input_dim)
        self.z_dim = int(z_dim)
        layers: list[nn.Module] = [nn.Linear(self.input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.Tanh()]
        for _ in range(hidden_layers - 1):
            layers.extend((nn.Linear(hidden_dim, hidden_dim), nn.ReLU()))
        self.trunk = nn.Sequential(*layers)
        self.latent_mean = nn.Linear(hidden_dim, self.z_dim)
        self.latent_log_std = nn.Parameter(torch.full((self.z_dim,), -1.5))
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.trunk(features)
        mean = self.latent_mean(hidden)
        log_std = self.latent_log_std.clamp(-5.0, 1.0).expand_as(mean)
        return mean, log_std, self.value_head(hidden).squeeze(-1)

    def distribution(self, features: torch.Tensor) -> Normal:
        mean, log_std, _ = self(features)
        return Normal(mean, log_std.exp())

    def sample(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_std, value = self(features)
        distribution = Normal(mean, log_std.exp())
        raw_z = distribution.rsample()
        return raw_z, distribution.log_prob(raw_z).sum(dim=-1), value

    def deterministic_z(self, features: torch.Tensor) -> torch.Tensor:
        return self(features)[0]


def project_latent(raw_z: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.normalize(raw_z, dim=-1).mul(float(raw_z.shape[-1]) ** 0.5)


def _encoder_input_scale(observation: Mapping[str, torch.Tensor], commands: torch.Tensor) -> torch.Tensor:
    state = observation["state"]
    last_action = observation["last_action"]
    history_actor = observation["history_actor"]
    expected = (H0W_STATE_DIM, H0W_ACTION_DIM, H0W_HISTORY_DIM)
    actual = (state.shape[-1], last_action.shape[-1], history_actor.shape[-1])
    if actual != expected:
        raise ValueError(f"Expected H0W state/action/history dimensions {expected}, got {actual}")
    state_scale = torch.ones(H0W_STATE_DIM, device=state.device, dtype=state.dtype)
    state_scale[H0W_ACTION_DIM : 2 * H0W_ACTION_DIM] = DOF_VELOCITY_SCALE
    history_scale = torch.ones(H0W_HISTORY_DIM, device=history_actor.device, dtype=history_actor.dtype)
    history_length = H0W_HISTORY_DIM // (3 * H0W_ACTION_DIM + 6)
    velocity_start = history_length * (2 * H0W_ACTION_DIM + 3)
    history_scale[velocity_start : velocity_start + history_length * H0W_ACTION_DIM] = DOF_VELOCITY_SCALE
    return torch.cat((commands.new_tensor(COMMAND_SCALE), state_scale, torch.ones(H0W_ACTION_DIM, device=state.device), history_scale))


def flatten_encoder_observation(observation: Mapping[str, torch.Tensor], commands: torch.Tensor) -> torch.Tensor:
    if commands.ndim != 2 or commands.shape[-1] != 3:
        raise ValueError(f"Commands must have shape [batch, 3], got {tuple(commands.shape)}")
    features = torch.cat((commands, observation["state"], observation["last_action"], observation["history_actor"]), dim=-1)
    return features * _encoder_input_scale(observation, commands)


def speed_tracking_reward(
    base_lin_vel: torch.Tensor, base_ang_vel: torch.Tensor, commands: torch.Tensor
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    linear_error = (base_lin_vel[:, :2] - commands[:, :2]).square().sum(dim=-1)
    yaw_error = (base_ang_vel[:, 2] - commands[:, 2]).square()
    linear = torch.exp(-linear_error / 0.16)
    yaw = torch.exp(-yaw_error / 0.25)
    return linear + 0.5 * yaw, {"linear_velocity": linear, "yaw_velocity": yaw}


def sample_commands(
    num_envs: int,
    device: torch.device,
    *,
    low: tuple[float, float, float] = COMMAND_LOW,
    high: tuple[float, float, float] = COMMAND_HIGH,
    stand_probability: float = 0.15,
    turn_probability: float = 0.15,
) -> torch.Tensor:
    if not 0.0 <= stand_probability <= 1.0 or not 0.0 <= turn_probability <= 1.0:
        raise ValueError("Command stand and turn probabilities must be in [0, 1]")
    low_t = torch.tensor(low, device=device)
    high_t = torch.tensor(high, device=device)
    commands = low_t + torch.rand(num_envs, 3, device=device) * (high_t - low_t)
    turning = torch.rand(num_envs, device=device) < turn_probability
    commands[turning, :2] = 0.0
    standing = (torch.rand(num_envs, device=device) < stand_probability) | (
        (commands[:, :2].norm(dim=-1) < 0.1) & ~turning
    )
    commands[standing] = 0.0
    return commands


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    terminated: torch.Tensor,
    truncated: torch.Tensor,
    last_value: torch.Tensor,
    *,
    discount: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    advantages = torch.zeros_like(rewards)
    running_advantage = torch.zeros_like(last_value)
    next_value = last_value
    for step in reversed(range(rewards.shape[0])):
        not_done = (~(terminated[step] | truncated[step])).float()
        delta = rewards[step] + discount * next_value * not_done - values[step]
        running_advantage = delta + discount * gae_lambda * not_done * running_advantage
        advantages[step] = running_advantage
        next_value = values[step]
    return advantages, advantages + values


@dataclass
class PpoRollout:
    features: torch.Tensor
    raw_z: torch.Tensor
    old_log_prob: torch.Tensor
    values: torch.Tensor
    rewards: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor


def ppo_update(
    policy: CommandEncoderPolicy,
    rollout: PpoRollout,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    *,
    epochs: int,
    minibatch_size: int,
    clip_ratio: float = 0.2,
    value_coef: float = 0.5,
    entropy_coef: float = 0.001,
    max_grad_norm: float = 1.0,
    clip_value_loss: bool = False,
    desired_kl: float | None = None,
    schedule: str = "fixed",
    learning_rate: float | None = None,
) -> dict[str, float]:
    if learning_rate is not None:
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
    features = rollout.features.flatten(0, 1)
    raw_z = rollout.raw_z.flatten(0, 1)
    old_log_prob = rollout.old_log_prob.flatten()
    old_values = rollout.values.flatten()
    flat_advantages = advantages.flatten()
    flat_returns = returns.flatten()
    flat_advantages = (flat_advantages - flat_advantages.mean()) / flat_advantages.std(unbiased=False).clamp_min(1.0e-6)
    totals = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0}
    updates = 0
    for _ in range(epochs):
        for indices in torch.randperm(features.shape[0], device=features.device).split(minibatch_size):
            distribution = policy.distribution(features[indices])
            log_prob = distribution.log_prob(raw_z[indices]).sum(dim=-1)
            _, _, value = policy(features[indices])
            ratio = (log_prob - old_log_prob[indices]).clamp(-20.0, 20.0).exp()
            policy_loss = -torch.minimum(
                ratio * flat_advantages[indices],
                ratio.clamp(1.0 - clip_ratio, 1.0 + clip_ratio) * flat_advantages[indices],
            ).mean()
            value_error = (value - flat_returns[indices]).square()
            if clip_value_loss:
                clipped_value = old_values[indices] + (value - old_values[indices]).clamp(-clip_ratio, clip_ratio)
                value_error = torch.maximum(value_error, (clipped_value - flat_returns[indices]).square())
            value_loss = value_error.mean() if clip_value_loss else 0.5 * value_error.mean()
            entropy = distribution.entropy().mean()
            optimizer.zero_grad(set_to_none=True)
            (policy_loss + value_coef * value_loss - entropy_coef * entropy).backward()
            average_gradients(policy.parameters())
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
            optimizer.step()
            totals["policy_loss"] += float(policy_loss.detach())
            totals["value_loss"] += float(value_loss.detach())
            totals["entropy"] += float(entropy.detach())
            minibatch_kl = float((old_log_prob[indices] - log_prob).mean().detach())
            totals["approx_kl"] += minibatch_kl
            updates += 1
            if schedule == "adaptive" and desired_kl is not None:
                current_rate = float(optimizer.param_groups[0]["lr"])
                if minibatch_kl > 2.0 * desired_kl:
                    current_rate = max(1.0e-5, current_rate / 1.5)
                elif 0.0 < minibatch_kl < desired_kl / 2.0:
                    current_rate = min(1.0e-2, current_rate * 1.5)
                for group in optimizer.param_groups:
                    group["lr"] = current_rate
    return {name: value / max(updates, 1) for name, value in totals.items()} | {
        "ppo_updates": float(updates),
        "effective_learning_rate": float(optimizer.param_groups[0]["lr"]),
    }


def latest_checkpoint(model_folder: Path) -> Path:
    candidates: list[tuple[int, Path]] = []
    for path in model_folder.glob("checkpoint_*.pt"):
        match = re.fullmatch(r"checkpoint_(\d+)\.pt", path.name)
        if match:
            candidates.append((int(match.group(1)), path))
    if not candidates:
        raise FileNotFoundError(f"No checkpoint_<iteration>.pt files in {model_folder}")
    return max(candidates, key=lambda item: item[0])[1]


def to_torch_observation(observation: Mapping[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: torch.as_tensor(value, dtype=torch.float32, device=device) for key, value in observation.items() if key != "time"}
