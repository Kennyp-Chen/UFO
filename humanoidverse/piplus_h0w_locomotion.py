"""MJLab locomotion environment used by PiPlus H0W stage-2 algorithms."""

from __future__ import annotations

from pathlib import Path

from humanoidverse.agents.envs.humanoidverse_mjlab import HumanoidVerseMjlabConfig
from humanoidverse.utils.robot_spec import RobotTrainingSpec, load_robot_training_spec


class H0WLocomotionMjlabConfig(HumanoidVerseMjlabConfig):
    """Configuration marker that enables reference-free reset in the core."""

    locomotion_mode: bool = True


def build_h0w_locomotion_env(
    *,
    device: str,
    robot_config: str | Path,
    motion_dataset: str | Path,
    num_envs: int,
    seed: int,
    max_episode_length_s: float,
    disable_domain_randomization: bool,
    render_mode: str | None = None,
    render_size: int = 720,
):
    """Build the tracked H0W robot as a command locomotion environment."""
    dataset_path = Path(motion_dataset).expanduser().resolve()
    if not dataset_path.is_file():
        raise FileNotFoundError(f"PiPlus H0W motion dataset does not exist: {dataset_path}")
    robot_training = load_robot_training_spec(robot_config)
    if len(robot_training.robot.control_joint_names) != 22:
        raise ValueError(f"PiPlus H0W stage-2 requires 22 actions, got {len(robot_training.robot.control_joint_names)}")
    hydra_overrides = [
        f"robot={robot_training.hydra_robot}",
        f"robot.control.action_scale={robot_training.action_scale}",
        f"robot.control.action_clip_value={robot_training.action_clip_value}",
        f"robot.control.normalize_action_to={robot_training.normalize_action_to}",
        "env.config.resample_motion_when_training=False",
        "env.config.termination.terminate_when_motion_end=False",
        "env.config.termination.terminate_when_motion_far=False",
        *robot_training.hydra_overrides,
    ]
    config = H0WLocomotionMjlabConfig(
        device=device,
        lafan_tail_path=str(dataset_path),
        mjcf_path=str(robot_training.robot.xml_path),
        robot_config_path=str(robot_training.config_path),
        robot_training=robot_training.to_env_dict(),
        max_episode_length_s=max_episode_length_s,
        disable_obs_noise=False,
        disable_domain_randomization=disable_domain_randomization,
        relative_config_path="exp/bfm_zero/bfm_zero",
        include_last_action=True,
        include_history_actor=True,
        root_height_obs=True,
        auto_reset=False,
        seed=seed,
        render_mode=render_mode,
        render_size=render_size,
        hydra_overrides=hydra_overrides,
    )
    environment, creation = config.build(num_envs=num_envs)
    environment._creation_config = creation
    return environment, robot_training


def h0w_robot_training_spec(robot_config: str | Path) -> RobotTrainingSpec:
    """Load and validate the public H0W robot configuration."""
    spec = load_robot_training_spec(robot_config)
    if spec.robot.name != "piplus_h0w" or len(spec.robot.control_joint_names) != 22:
        raise ValueError("PiPlus H0W stage-2 requires configs/robots/piplus_h0w.yaml")
    return spec
