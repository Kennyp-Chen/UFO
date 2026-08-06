"""Train a velocity-command PPO encoder on a frozen PiPlus H0W BFM decoder."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import mujoco
import numpy as np
import torch
from safetensors import safe_open

from humanoidverse.distributed import average_metrics, barrier, broadcast_module_state, is_distributed
from humanoidverse.piplus_h0w_locomotion import build_h0w_locomotion_env, h0w_robot_training_spec
from humanoidverse.piplus_h0w_onnx_decoder import H0W_ACTION_DIM, H0W_ACTOR_OBSERVATION_DIM, load_decoder
from humanoidverse.piplus_h0w_stage2 import (
    CommandEncoderPolicy,
    PpoRollout,
    compute_gae,
    flatten_encoder_observation,
    ppo_update,
    sample_commands,
    speed_tracking_reward,
    to_torch_observation,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPEED_STAGE2_TASK = "speed_stage2_piplus_22dof"
DEFAULT_BFM_MODEL = PROJECT_ROOT / "model/piplus_h0w_bfm/model.safetensors"
DEFAULT_DECODER_PATH = (
    PROJECT_ROOT / "model/piplus_h0w_bfm/decoder/bfmzero-piplus-h0w-isaac-20260629_214205/exported/FBcprAuxModel.onnx"
)
DEFAULT_LATENT_REFERENCE = (
    PROJECT_ROOT / "model/piplus_h0w_bfm/decoder/bfmzero-piplus-h0w-isaac-20260629_214205/tracking_inference/zs_8.pkl"
)
DEFAULT_ROBOT_CONFIG = PROJECT_ROOT / "configs/robots/piplus_h0w.yaml"
DEFAULT_MOTION_DATASET = PROJECT_ROOT / "humanoidverse/data/piplus_h0w_lafan/piplus_h0w_lafan_10s-clipped.pkl"


def checkpoint_task_is_compatible(metadata: Mapping[str, object]) -> bool:
    return metadata.get("task") in {None, SPEED_STAGE2_TASK}


def _actuator_joint_names(model: mujoco.MjModel) -> list[str]:
    return [
        str(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, int(model.actuator_trnid[index, 0])))
        for index in range(model.nu)
    ]


def validate_h0w_assets(
    robot_config: str | Path,
    bfm_model: str | Path,
    decoder_path: str | Path,
) -> dict[str, object]:
    """Validate static H0W robot, state-dict, and decoder contracts."""
    spec = h0w_robot_training_spec(robot_config)
    model_path = Path(bfm_model).expanduser().resolve()
    decoder = Path(decoder_path).expanduser().resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"PiPlus H0W BFM state dict does not exist: {model_path}")
    if not decoder.is_file():
        raise FileNotFoundError(f"PiPlus H0W ONNX decoder does not exist: {decoder}")
    mjcf_model = mujoco.MjModel.from_xml_path(str(spec.robot.xml_path))
    if mjcf_model.nu != H0W_ACTION_DIM:
        raise ValueError(f"PiPlus H0W MJCF must expose {H0W_ACTION_DIM} actuators, got {mjcf_model.nu}")
    actuator_joints = _actuator_joint_names(mjcf_model)
    if actuator_joints != list(spec.robot.control_joint_names):
        raise ValueError("PiPlus H0W MJCF actuator order does not match the robot configuration")
    with safe_open(str(model_path), framework="pt", device="cpu") as artifact:
        output_key = "_actor.policy.6.mlp.1.weight"
        if output_key not in artifact.keys():
            raise ValueError(f"Expected GCR actor output tensor {output_key!r} in {model_path}")
        output_shape = tuple(artifact.get_tensor(output_key).shape)
        if output_shape[0] != H0W_ACTION_DIM:
            raise ValueError(f"PiPlus H0W BFM actor must produce {H0W_ACTION_DIM} actions, got {output_shape}")
    import onnx

    graph = onnx.load(str(decoder)).graph
    input_dim = int(graph.input[0].type.tensor_type.shape.dim[-1].dim_value)
    output_dim = int(graph.output[0].type.tensor_type.shape.dim[-1].dim_value)
    if (input_dim, output_dim) != (H0W_ACTOR_OBSERVATION_DIM, H0W_ACTION_DIM):
        raise ValueError(
            f"PiPlus H0W ONNX decoder must be {H0W_ACTOR_OBSERVATION_DIM}->{H0W_ACTION_DIM}, got {input_dim}->{output_dim}"
        )
    return {
        "robot_config": str(spec.config_path),
        "xml_path": str(spec.robot.xml_path),
        "action_dim": H0W_ACTION_DIM,
        "mjcf_nq": int(mjcf_model.nq),
        "mjcf_nv": int(mjcf_model.nv),
        "mjcf_nu": int(mjcf_model.nu),
        "bfm_model": str(model_path),
        "bfm_actor_output_shape": list(output_shape),
        "decoder_path": str(decoder),
        "decoder_input_dim": input_dim,
        "decoder_output_dim": output_dim,
    }


def _transition_core(live_core: Any, info: Mapping[str, Any]) -> Any:
    terminal_state = info.get("terminal_state")
    if isinstance(terminal_state, Mapping):
        return SimpleNamespace(num_envs=live_core.num_envs, device=live_core.device, **terminal_state)
    return live_core


def _distributed_context(args: argparse.Namespace) -> tuple[argparse.Namespace, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size <= 1:
        return args, rank, world_size
    if local_rank >= 4:
        raise RuntimeError("This server permits stage-2 training only on physical GPUs 0-3")
    if not torch.cuda.is_available():
        raise RuntimeError("Distributed PiPlus H0W stage-2 training requires CUDA")
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
    parser.add_argument("--motion-dataset", type=Path, default=DEFAULT_MOTION_DATASET)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu-ids", default="single", choices=("single", "all"))
    parser.add_argument("--work-dir", type=Path, default=PROJECT_ROOT / "runs/speed_stage2_piplus_h0w")
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--rollout-steps", type=int, default=32)
    parser.add_argument("--ppo-epochs", type=int, default=5)
    parser.add_argument("--minibatch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--discount", type=float, default=0.98)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--command-stand-prob", type=float, default=0.15)
    parser.add_argument("--command-turn-prob", type=float, default=0.15)
    parser.add_argument("--command-resample-steps", type=int, default=300)
    parser.add_argument("--command-resample-prob", type=float, default=0.75)
    parser.add_argument("--command-smoothing", type=float, default=0.15)
    parser.add_argument("--env-reward-weight", type=float, default=0.0)
    parser.add_argument("--max-episode-length-s", type=float, default=20.0)
    parser.add_argument("--disable-domain-randomization", action="store_true")
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--validate-assets", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main(parsed_args: argparse.Namespace | None = None) -> None:
    args = _parse_args() if parsed_args is None else parsed_args
    args, rank, world_size = _distributed_context(args)
    if args.gpu_ids == "all" and world_size == 1:
        raise ValueError("--gpu-ids all requires torchrun with CUDA_VISIBLE_DEVICES=0,1,2,3")
    if args.num_envs <= 0 or args.iterations <= 0 or args.rollout_steps <= 0 or args.minibatch_size <= 0:
        raise ValueError("num-envs, iterations, rollout-steps, and minibatch-size must be positive")
    if not 0.0 <= args.command_smoothing <= 1.0 or not 0.0 <= args.command_resample_prob <= 1.0:
        raise ValueError("command smoothing and resample probability must be in [0, 1]")
    if args.command_resample_steps <= 0:
        raise ValueError("--command-resample-steps must be positive")
    if args.smoke:
        args.num_envs = min(args.num_envs, 2)
        args.iterations = 1
        args.rollout_steps = min(args.rollout_steps, 4)
        args.ppo_epochs = 1
        args.minibatch_size = min(args.minibatch_size, args.num_envs * args.rollout_steps)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    contract = validate_h0w_assets(args.robot_config, args.bfm_model, args.decoder_path)
    if args.validate_assets:
        print(json.dumps(contract, indent=2, sort_keys=True), flush=True)
        return
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
        commands = sample_commands(
            args.num_envs,
            device,
            stand_probability=args.command_stand_prob,
            turn_probability=args.command_turn_prob,
        )
        policy = CommandEncoderPolicy(flatten_encoder_observation(observation_t, commands).shape[-1], decoder.z_dim).to(device)
        broadcast_module_state(policy)
        optimizer = torch.optim.Adam(policy.parameters(), lr=args.learning_rate)
        work_dir = args.work_dir.expanduser().resolve()
        metadata = {
            "task": SPEED_STAGE2_TASK,
            "amp": False,
            "robot": robot_training.robot.name,
            "robot_config": str(args.robot_config.expanduser().resolve()),
            "motion_dataset": str(args.motion_dataset.expanduser().resolve()),
            "decoder_path": str(args.decoder_path.expanduser().resolve()),
            "contract": contract,
            "z_dim": policy.z_dim,
            "encoder_input_dim": policy.input_dim,
            "distributed_world_size": world_size,
        }
        start_iteration = 0
        if args.resume is not None:
            checkpoint = torch.load(args.resume.expanduser().resolve(), map_location=device, weights_only=False)
            if not checkpoint_task_is_compatible(checkpoint.get("metadata", {})):
                raise ValueError("Cannot resume an AMP checkpoint as speed PPO")
            policy.load_state_dict(checkpoint["policy"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            start_iteration = int(checkpoint.get("iteration", 0))
        if rank == 0:
            work_dir.mkdir(parents=True, exist_ok=True)
            (work_dir / "config.json").write_text(json.dumps(metadata, indent=2) + "\n")
        barrier()
        episode_steps = torch.zeros(args.num_envs, dtype=torch.long, device=device)
        for iteration in range(start_iteration, args.iterations):
            parts: list[dict[str, torch.Tensor]] = []
            command_store: list[torch.Tensor] = []
            velocity_store: list[torch.Tensor] = []
            angular_velocity_store: list[torch.Tensor] = []
            for _ in range(args.rollout_steps):
                features = flatten_encoder_observation(observation_t, commands)
                with torch.no_grad():
                    raw_z, old_log_prob, value = policy.sample(features)
                    action = decoder.act(observation_t, decoder.project_z(raw_z))
                next_observation, environment_reward, terminated, truncated, info = environment.step(action.to(environment._env.device), to_numpy=False)
                transition_core = _transition_core(environment._env, info)
                base_lin_vel = transition_core.base_lin_vel.to(device)
                base_ang_vel = transition_core.base_ang_vel.to(device)
                terminated = terminated.to(device=device, dtype=torch.bool)
                truncated = truncated.to(device=device, dtype=torch.bool)
                speed_reward, _ = speed_tracking_reward(base_lin_vel, base_ang_vel, commands)
                parts.append(
                    {
                        "features": features.detach(),
                        "raw_z": raw_z.detach(),
                        "old_log_prob": old_log_prob.detach(),
                        "values": value.detach(),
                        "rewards": speed_reward + args.env_reward_weight * environment_reward.to(device),
                        "terminated": terminated,
                        "truncated": truncated,
                    }
                )
                command_store.append(commands.detach())
                velocity_store.append(base_lin_vel.detach())
                angular_velocity_store.append(base_ang_vel.detach())
                observation_t = to_torch_observation(next_observation, device)
                done = terminated | truncated
                episode_steps = torch.where(done, torch.zeros_like(episode_steps), episode_steps + 1)
                resample = (episode_steps > 0) & (episode_steps % args.command_resample_steps == 0)
                resample &= torch.rand(args.num_envs, device=device) < args.command_resample_prob
                resample |= done
                targets = sample_commands(
                    args.num_envs,
                    device,
                    stand_probability=args.command_stand_prob,
                    turn_probability=args.command_turn_prob,
                )
                commands = torch.where(
                    resample[:, None],
                    (1.0 - args.command_smoothing) * commands + args.command_smoothing * targets,
                    commands,
                )
            rollout = PpoRollout(**{name: torch.stack([part[name] for part in parts]) for name in parts[0]})
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
            )
            commands_tensor = torch.stack(command_store)
            velocities = torch.stack(velocity_store)
            angular_velocities = torch.stack(angular_velocity_store)
            metrics.update(
                {
                    "iteration": float(iteration + 1),
                    "reward_mean": float(rollout.rewards.mean()),
                    "speed/vx_mae": float((velocities[..., 0] - commands_tensor[..., 0]).abs().mean()),
                    "speed/vy_mae": float((velocities[..., 1] - commands_tensor[..., 1]).abs().mean()),
                    "speed/yaw_rate_mae": float((angular_velocities[..., 2] - commands_tensor[..., 2]).abs().mean()),
                    "termination_rate": float(rollout.terminated.float().mean()),
                }
            )
            metrics = average_metrics(metrics)
            if rank == 0:
                serializable_metrics = {name: float(value) if torch.is_tensor(value) else value for name, value in metrics.items()}
                print(json.dumps(serializable_metrics, sort_keys=True), flush=True)
                if (iteration + 1) % args.save_every == 0 or iteration + 1 == args.iterations:
                    torch.save(
                        {
                            "policy": policy.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "iteration": iteration + 1,
                            "metadata": metadata,
                        },
                        work_dir / f"checkpoint_{iteration + 1}.pt",
                    )
            barrier()
    finally:
        environment.close()
        if is_distributed():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
