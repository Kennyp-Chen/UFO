"""Train a command encoder on top of a frozen TeCH behavior model."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Mapping

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

from humanoidverse.agents.envs.tech_kick_mjlab import DEFAULT_BALL_MASS, DEFAULT_BALL_RADIUS, build_tech_kick_env
from humanoidverse.agents.load_utils import load_model_from_checkpoint_dir
from humanoidverse.distributed import average_gradients, barrier, broadcast_module_state
from humanoidverse.tasks.kick import (
    KickCommandState,
    KickObservationHistory,
    KickRewardState,
    flatten_kick_encoder_observation,
    kick_curriculum,
)

DEFAULT_TECH_CHECKPOINT = "runs/remote_tech_piplus_2h0w_8gpu_seed4728/checkpoint"
DEFAULT_ROBOT_CONFIG = "configs/robots/piplus_h0w.yaml"
DEFAULT_EXPERT_DATASET = (
    "/home/cato/Documents/HT_MotionData/lafan/PiPlus_S_12L8A0G2H0W_lafan_dataset_20260714/"
    "piplus_h0w_lafan_10s-clipped.pkl"
)


@dataclass
class KickRollout:
    encoder_features: torch.Tensor
    raw_z: torch.Tensor
    old_log_prob: torch.Tensor
    values: torch.Tensor
    rewards: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    task_components: dict[str, torch.Tensor]
    diagnostics: dict[str, torch.Tensor]


class CommandEncoderPolicy(nn.Module):
    """Stochastic task encoder and value function used by PPO."""

    def __init__(self, input_dim: int, z_dim: int, hidden_dim: int = 256, hidden_layers: int = 1) -> None:
        super().__init__()
        if hidden_layers < 1:
            raise ValueError("hidden_layers must be positive")
        self.input_dim = int(input_dim)
        self.z_dim = int(z_dim)
        layers: list[nn.Module] = [nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.Tanh()]
        for _ in range(hidden_layers - 1):
            layers.extend((nn.Linear(hidden_dim, hidden_dim), nn.ReLU()))
        self.trunk = nn.Sequential(*layers)
        self.latent_mean = nn.Linear(hidden_dim, z_dim)
        self.latent_log_std = nn.Parameter(torch.full((z_dim,), -1.5))
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
        mean, _, _ = self(features)
        return mean


def freeze_tech(model) -> None:
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)


def tech_action(model, obs: Mapping[str, torch.Tensor], raw_z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    projected_z = model.project_z(raw_z)
    return model.act(dict(obs), projected_z, mean=True), projected_z


def validate_latent_contract(policy: CommandEncoderPolicy, model) -> dict[str, object]:
    z_dim = int(model.cfg.archi.z_dim)
    if policy.z_dim != z_dim:
        raise ValueError(f"Encoder z_dim={policy.z_dim} does not match TeCH z_dim={z_dim}")
    probe = torch.randn(32, z_dim, device=next(policy.parameters()).device)
    with torch.no_grad():
        projected = model.project_z(probe)
    if projected.shape != probe.shape or not torch.isfinite(projected).all():
        raise ValueError("TeCH project_z returned an invalid latent")
    geometry = str(getattr(model.cfg.archi, "z_geometry", "hypersphere"))
    contract: dict[str, object] = {"z_dim": z_dim, "geometry": geometry, "norm_z": bool(model.cfg.archi.norm_z)}
    if geometry == "hypersphere" and bool(model.cfg.archi.norm_z):
        expected = z_dim**0.5
        actual = float(projected.norm(dim=-1).mean())
        if abs(actual - expected) > 1.0e-3:
            raise ValueError(f"Projected TeCH latent norm is {actual:.6f}, expected {expected:.6f}")
        contract.update({"mode": "checkpoint_project_z", "expected_norm": expected})
    else:
        contract["mode"] = "checkpoint_geometry"
    return contract


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
    next_advantage = torch.zeros_like(last_value)
    next_value = last_value
    for step in reversed(range(rewards.shape[0])):
        done = terminated[step] | truncated[step]
        not_done = (~done).float()
        delta = rewards[step] + discount * next_value * not_done - values[step]
        next_advantage = delta + discount * gae_lambda * not_done * next_advantage
        advantages[step] = next_advantage
        next_value = values[step]
    return advantages, advantages + values


def ppo_update(
    policy: CommandEncoderPolicy,
    rollout: KickRollout,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    *,
    epochs: int,
    minibatch_size: int,
    clip_ratio: float,
    value_coef: float,
    entropy_coef: float,
    optimizer: torch.optim.Optimizer,
    max_grad_norm: float,
) -> dict[str, float]:
    features = rollout.encoder_features.flatten(0, 1)
    raw_z = rollout.raw_z.flatten(0, 1)
    old_log_prob = rollout.old_log_prob.flatten()
    flat_advantages = advantages.flatten()
    flat_returns = returns.flatten()
    flat_advantages = (flat_advantages - flat_advantages.mean()) / flat_advantages.std(unbiased=False).clamp_min(1.0e-6)
    metrics: dict[str, float] = {}
    batch_size = features.shape[0]
    for _ in range(epochs):
        for indices in torch.randperm(batch_size, device=features.device).split(minibatch_size):
            mean, log_std, values = policy(features[indices])
            distribution = Normal(mean, log_std.exp())
            log_prob = distribution.log_prob(raw_z[indices]).sum(dim=-1)
            ratio = torch.exp(log_prob - old_log_prob[indices])
            unclipped = ratio * flat_advantages[indices]
            clipped = ratio.clamp(1.0 - clip_ratio, 1.0 + clip_ratio) * flat_advantages[indices]
            policy_loss = -torch.minimum(unclipped, clipped).mean()
            value_loss = 0.5 * (values - flat_returns[indices]).square().mean()
            entropy = distribution.entropy().mean()
            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            average_gradients(policy.parameters())
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
            optimizer.step()
            metrics = {
                "policy_loss": float(policy_loss.detach()),
                "value_loss": float(value_loss.detach()),
                "entropy": float(entropy.detach()),
                "approx_kl": float((old_log_prob[indices] - log_prob).mean().detach()),
            }
    return metrics


def _to_torch_obs(obs: Mapping[str, np.ndarray | torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        key: torch.as_tensor(value, dtype=torch.float32, device=device)
        for key, value in obs.items()
        if key != "time"
    }


def _transition_core(live_core, info: Mapping[str, object]):
    state = info.get("terminal_state")
    if not isinstance(state, Mapping):
        return live_core
    return SimpleNamespace(num_envs=live_core.num_envs, device=live_core.device, **state)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tech-checkpoint", default=DEFAULT_TECH_CHECKPOINT)
    parser.add_argument("--robot-config", default=DEFAULT_ROBOT_CONFIG)
    parser.add_argument("--expert-dataset", default=DEFAULT_EXPERT_DATASET)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu-ids", default="single")
    parser.add_argument("--work-dir", default="runs/tech_kick_stage2_piplus_h0w")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--iterations", type=int, default=30000)
    parser.add_argument("--rollout-steps", type=int, default=24)
    parser.add_argument("--kick-history-length", type=int, default=8)
    parser.add_argument("--ppo-epochs", type=int, default=5)
    parser.add_argument("--minibatch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--kick-reward-weight", type=float, default=1.0)
    parser.add_argument("--env-reward-weight", type=float, default=1.0)
    parser.add_argument("--max-episode-length-s", type=float, default=3.0)
    parser.add_argument("--ball-radius", type=float, default=DEFAULT_BALL_RADIUS)
    parser.add_argument("--ball-mass", type=float, default=DEFAULT_BALL_MASS)
    parser.add_argument("--contact-force-threshold", type=float, default=0.1)
    parser.add_argument("--ball-velocity-reward-std", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--disable-domain-randomization", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def _distributed_context(args: argparse.Namespace) -> tuple[argparse.Namespace, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size <= 1:
        return args, rank, world_size
    if not torch.cuda.is_available():
        raise RuntimeError("Distributed Stage2 training requires CUDA")
    from datetime import timedelta

    import torch.distributed as dist

    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://", timeout=timedelta(hours=2))
    args.device = f"cuda:{local_rank}"
    args.seed += rank
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(local_rank)
    return args, rank, world_size


def main(parsed_args: argparse.Namespace | None = None) -> None:
    args = _parse_args() if parsed_args is None else parsed_args
    args, rank, world_size = _distributed_context(args)
    rank0 = rank == 0
    if args.smoke:
        args.num_envs = min(args.num_envs, 2)
        args.iterations = 1
        args.rollout_steps = min(args.rollout_steps, 4)
        args.ppo_epochs = 1
        args.minibatch_size = min(args.minibatch_size, args.num_envs * args.rollout_steps)
    if args.ball_radius <= 0.0 or args.ball_mass <= 0.0:
        raise ValueError("ball radius and mass must be positive")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    env, robot_training = build_tech_kick_env(
        device=args.device,
        robot_config=args.robot_config,
        expert_dataset=args.expert_dataset,
        num_envs=args.num_envs,
        seed=args.seed,
        max_episode_length_s=args.max_episode_length_s,
        ball_radius=args.ball_radius,
        ball_mass=args.ball_mass,
        disable_domain_randomization=args.disable_domain_randomization,
    )
    model = load_model_from_checkpoint_dir(args.tech_checkpoint, device=device.type, strict=False)
    freeze_tech(model)
    if int(env.single_action_space.shape[0]) != int(model.action_dim):
        raise ValueError(f"TeCH action dimension {model.action_dim} does not match environment {env.single_action_space.shape[0]}")

    obs, _ = env.reset(to_numpy=False)
    obs_t = _to_torch_obs(obs, device)
    all_env_ids = torch.arange(args.num_envs, device=device)
    curriculum = kick_curriculum(0, args.iterations)
    env._env.set_ball_reset_distribution(
        radius_range=curriculum.ball_radius_range,
        angle_range=curriculum.ball_angle_range,
        speed_range=curriculum.ball_speed_range,
    )
    command_state = KickCommandState(args.num_envs, device)
    command_state.sample(all_env_ids, env._env.base_quat, curriculum)
    command = command_state.command(env._env.base_quat)
    history = KickObservationHistory(args.num_envs, args.kick_history_length, device)
    history.reset(all_env_ids, command, env._env.ball_position_b, env._env.ball_linear_velocity_b)
    encoder_features = flatten_kick_encoder_observation(obs_t, history)

    encoder_arch = model.cfg.archi.goal_encoder
    policy = CommandEncoderPolicy(
        encoder_features.shape[-1],
        int(model.cfg.archi.z_dim),
        hidden_dim=int(encoder_arch.hidden_dim),
        hidden_layers=int(encoder_arch.hidden_layers),
    ).to(device)
    latent_contract = validate_latent_contract(policy, model)
    broadcast_module_state(policy)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.learning_rate)
    reward_state = KickRewardState(
        num_envs=args.num_envs,
        dt=env._env.dt,
        feet_indices=env._env.feet_indices,
        contact_force_threshold=args.contact_force_threshold,
        velocity_std=args.ball_velocity_reward_std,
        device=device,
    )
    reward_state.reset(all_env_ids, env._env, command)

    metadata = {
        "task": "tech_kick_stage2",
        "tech_checkpoint": str(Path(args.tech_checkpoint).resolve()),
        "robot_config": str(Path(args.robot_config).resolve()),
        "expert_dataset": str(Path(args.expert_dataset).resolve()),
        "robot": robot_training.robot.name,
        "action_dim": int(model.action_dim),
        "encoder_input_dim": int(policy.input_dim),
        "z_dim": int(policy.z_dim),
        "kick_history_length": int(args.kick_history_length),
        "latent_contract": latent_contract,
        "ball_radius": float(args.ball_radius),
        "ball_mass": float(args.ball_mass),
    }
    if rank0:
        (work_dir / "config.json").write_text(json.dumps(metadata, indent=2) + "\n")
        print(json.dumps({"stage2_contract": metadata}, sort_keys=True), flush=True)

    start_iteration = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        policy.load_state_dict(checkpoint["policy"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_iteration = int(checkpoint["iteration"])

    try:
        for iteration in range(start_iteration, args.iterations):
            curriculum = kick_curriculum(iteration, args.iterations)
            env._env.set_ball_reset_distribution(
                radius_range=curriculum.ball_radius_range,
                angle_range=curriculum.ball_angle_range,
                speed_range=curriculum.ball_speed_range,
            )
            feature_steps, raw_z_steps, log_prob_steps, value_steps = [], [], [], []
            reward_steps, terminated_steps, truncated_steps = [], [], []
            component_steps: dict[str, list[torch.Tensor]] = {}
            diagnostic_steps: dict[str, list[torch.Tensor]] = {}

            for _ in range(args.rollout_steps):
                encoder_features = flatten_kick_encoder_observation(obs_t, history)
                with torch.no_grad():
                    raw_z, log_prob, value = policy.sample(encoder_features)
                    action, _ = tech_action(model, obs_t, raw_z)
                next_obs, env_reward, terminated, truncated, info = env.step(action, to_numpy=False)
                transition_core = _transition_core(env._env, info)
                task_reward, components, diagnostics = reward_state.compute(transition_core, command)
                total_reward = args.kick_reward_weight * task_reward + args.env_reward_weight * env_reward

                feature_steps.append(encoder_features)
                raw_z_steps.append(raw_z)
                log_prob_steps.append(log_prob)
                value_steps.append(value)
                reward_steps.append(total_reward)
                terminated_steps.append(terminated)
                truncated_steps.append(truncated)
                for name, values in components.items():
                    component_steps.setdefault(name, []).append(values)
                for name, values in diagnostics.items():
                    diagnostic_steps.setdefault(name, []).append(values.float())

                obs_t = _to_torch_obs(next_obs, device)
                done = terminated | truncated
                done_ids = done.nonzero(as_tuple=False).squeeze(-1)
                if done_ids.numel() > 0:
                    command_state.sample(done_ids, env._env.base_quat, curriculum)
                command = command_state.command(env._env.base_quat)
                history.append(command, env._env.ball_position_b, env._env.ball_linear_velocity_b)
                if done_ids.numel() > 0:
                    history.reset(done_ids, command, env._env.ball_position_b, env._env.ball_linear_velocity_b)
                    reward_state.reset(done_ids, env._env, command)

            with torch.no_grad():
                last_features = flatten_kick_encoder_observation(obs_t, history)
                _, _, last_value = policy(last_features)
            rollout = KickRollout(
                encoder_features=torch.stack(feature_steps),
                raw_z=torch.stack(raw_z_steps),
                old_log_prob=torch.stack(log_prob_steps),
                values=torch.stack(value_steps),
                rewards=torch.stack(reward_steps),
                terminated=torch.stack(terminated_steps),
                truncated=torch.stack(truncated_steps),
                task_components={name: torch.stack(values) for name, values in component_steps.items()},
                diagnostics={name: torch.stack(values) for name, values in diagnostic_steps.items()},
            )
            advantages, returns = compute_gae(
                rollout.rewards,
                rollout.values,
                rollout.terminated,
                rollout.truncated,
                last_value,
                discount=0.98,
                gae_lambda=0.95,
            )
            ppo_metrics = ppo_update(
                policy,
                rollout,
                advantages,
                returns,
                epochs=args.ppo_epochs,
                minibatch_size=args.minibatch_size,
                clip_ratio=0.2,
                value_coef=0.5,
                entropy_coef=0.001,
                optimizer=optimizer,
                max_grad_norm=1.0,
            )
            metrics = {
                "iteration": iteration + 1,
                "reward_mean": float(rollout.rewards.mean()),
                "termination_rate": float(rollout.terminated.float().mean()),
                "truncation_rate": float(rollout.truncated.float().mean()),
                **ppo_metrics,
                **{name + "_rate": float(values.mean()) for name, values in rollout.diagnostics.items()},
                **{f"reward/kick/{name}": float(values.mean()) for name, values in rollout.task_components.items()},
            }
            if world_size > 1:
                import torch.distributed as dist

                for key, item in metrics.items():
                    value = torch.tensor(float(item), device=device)
                    dist.all_reduce(value, op=dist.ReduceOp.SUM)
                    metrics[key] = float(value / world_size)
            if rank0:
                print(json.dumps(metrics, sort_keys=True), flush=True)
            if rank0 and ((iteration + 1) % args.save_every == 0 or iteration + 1 == args.iterations):
                torch.save(
                    {"policy": policy.state_dict(), "optimizer": optimizer.state_dict(), "iteration": iteration + 1, "metadata": metadata},
                    work_dir / f"checkpoint_{iteration + 1}.pt",
                )
            barrier()
    finally:
        env.close()
        if world_size > 1:
            import torch.distributed as dist

            if dist.is_initialized():
                dist.destroy_process_group()


def launch() -> None:
    args = _parse_args()
    if args.gpu_ids in (None, "single"):
        main(args)
        return
    existing_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if args.gpu_ids == "all":
        selected_gpus = None
        num_gpus = torch.cuda.device_count()
    else:
        requested = [int(value) for value in args.gpu_ids.split(",") if value.strip()]
        visible = [value.strip() for value in existing_visible.split(",")] if existing_visible else None
        selected_gpus = [visible[index] if visible else str(index) for index in requested]
        num_gpus = len(selected_gpus)
    if num_gpus <= 1:
        if selected_gpus is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(selected_gpus)
        main(args)
        return
    import torchrunx

    if selected_gpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(selected_gpus)
    os.environ.setdefault("TORCHRUNX_LOG_DIR", str(Path(args.work_dir) / "torchrunx"))
    from humanoidverse.tech_kick_stage2 import main as worker_main

    torchrunx.Launcher(hostnames=["localhost"], workers_per_host=num_gpus, backend=None).run(worker_main, args)


if __name__ == "__main__":
    launch()
