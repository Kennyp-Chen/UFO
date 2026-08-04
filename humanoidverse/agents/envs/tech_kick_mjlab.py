"""Robot-config-aware MJLab environment for frozen-TECH kick training."""

from __future__ import annotations

import math
from pathlib import Path

import mujoco
import torch
from mjlab.managers.event_manager import EventTermCfg, requires_model_fields

from humanoidverse.agents.envs.humanoidverse_mjlab import (
    HumanoidVerseMjlabCore,
    HumanoidVerseMjlabVectorEnv,
    _compose_humanoidverse_config,
    make_mjlab_ufo_env_cfg,
)
from humanoidverse.utils.robot_spec import load_robot_training_spec
from humanoidverse.utils.torch_utils import my_quat_rotate, quat_rotate_inverse, wxyz_to_xyzw

DEFAULT_BALL_RADIUS = 0.07
DEFAULT_BALL_MASS = 0.20
BALL_DYNAMIC_FRICTION = 0.10
BALL_STATIC_FRICTION = 0.20
# Isaac Sim keeps separate static/dynamic friction values. MuJoCo exposes one
# tangential friction cone, so use a calibrated lower value for rolling drag.
BALL_MUJOCO_SLIDING_FRICTION = 0.01
BALL_RESTITUTION = 0.82
BALL_LINEAR_DAMPING = 0.10
BALL_ANGULAR_DAMPING = 0.05
BALL_CONTACT_STIFFNESS = 10000.0
BALL_CONTACT_DAMPING_RATIO = 0.15
BALL_CONTACT_DAMPING = 2.0 * math.sqrt(BALL_CONTACT_STIFFNESS * DEFAULT_BALL_MASS) * BALL_CONTACT_DAMPING_RATIO
BALL_GEOM_PRIORITY = 1


@requires_model_fields("dof_damping")
def _set_ball_damping(env, env_ids: torch.Tensor | None) -> None:
    """Apply the Isaac Lab ball's separate linear and angular damping in MuJoCo.

    MuJoCo's free-joint XML attribute accepts one scalar and applies it to all six
    DOFs.  MJLab exposes ``dof_damping`` as a per-world model field, so set the
    translational and rotational components separately after model construction.
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
    else:
        env_ids = env_ids.to(device=env.device, dtype=torch.int)

    ball_dof_ids = env.scene["ball"].indexing.free_joint_v_adr
    damping = torch.tensor(
        [BALL_LINEAR_DAMPING] * 3 + [BALL_ANGULAR_DAMPING] * 3,
        device=env.device,
        dtype=env.sim.model.dof_damping.dtype,
    )
    env_grid = env_ids[:, None]
    dof_grid = ball_dof_ids[None, :]
    env.sim.model.dof_damping[env_grid, dof_grid] = damping


def _ball_entity_cfg(*, radius: float, mass: float):
    from mjlab.entity import EntityCfg

    def spec_fn():
        return mujoco.MjSpec.from_string(
            f"""
            <mujoco model="tech_kick_ball">
              <worldbody>
                <body name="ball">
                  <joint name="ball_freejoint" type="free" damping="{BALL_LINEAR_DAMPING}"/>
                  <geom name="ball_geom" type="sphere" size="{radius}" mass="{mass}"
                        priority="{BALL_GEOM_PRIORITY}" condim="3"
                        friction="{BALL_MUJOCO_SLIDING_FRICTION} 0.0 0.0"
                        solref="-{BALL_CONTACT_STIFFNESS} -{BALL_CONTACT_DAMPING}"
                        rgba="0.95 0.55 0.08 1"/>
                </body>
              </worldbody>
            </mujoco>
            """
        )

    return EntityCfg(
        spec_fn=spec_fn,
        init_state=EntityCfg.InitialStateCfg(pos=(0.30, 0.0, radius), rot=(1.0, 0.0, 0.0, 0.0), joint_pos={}),
    )


class TechKickMjlabCore(HumanoidVerseMjlabCore):
    """HumanoidVerse adapter with a floating ball and foot-ball contacts."""

    def __init__(self, hv_config, mjlab_env, *, creation_config, ball_radius: float) -> None:
        self.ball = mjlab_env.scene["ball"]
        self.ball_radius = float(ball_radius)
        self.ball_reset_radius_range = (0.25, 0.32)
        self.ball_reset_angle_range = (-math.pi / 6.0, math.pi / 6.0)
        self.ball_reset_speed_range = (0.0, 0.0)
        super().__init__(hv_config, mjlab_env, creation_config=creation_config)

    def _refresh_state(self) -> None:
        super()._refresh_state()
        ball_pose = self.ball.data.root_link_pose_w.clone()
        ball_velocity = self.ball.data.root_link_vel_w.clone()
        self.ball_pos_w = ball_pose[:, :3]
        self.ball_quat_w = wxyz_to_xyzw(ball_pose[:, 3:7])
        self.ball_lin_vel_w = ball_velocity[:, :3]
        self.ball_ang_vel_w = ball_velocity[:, 3:6]
        root_position = self.robot_root_states[:, :3]
        self.ball_position_b = quat_rotate_inverse(self.base_quat, self.ball_pos_w - root_position, w_last=True)
        self.ball_linear_velocity_b = quat_rotate_inverse(self.base_quat, self.ball_lin_vel_w, w_last=True)
        self.foot_ball_contact_force = self._read_foot_ball_contact_force()

    def _read_foot_ball_contact_force(self) -> torch.Tensor:
        sensor = self.mjlab_env.scene.sensors.get("foot_ball_contact")
        if sensor is None:
            return torch.zeros(self.num_envs, 2, device=self.device)
        data = sensor.data
        if data.force_history is not None:
            return torch.linalg.vector_norm(data.force_history, dim=-1).amax(dim=-1)
        if data.force is not None:
            return torch.linalg.vector_norm(data.force, dim=-1)
        if data.found is not None:
            return data.found.float()
        return torch.zeros(self.num_envs, 2, device=self.device)

    def set_ball_reset_distribution(
        self,
        *,
        radius_range: tuple[float, float],
        angle_range: tuple[float, float],
        speed_range: tuple[float, float],
    ) -> None:
        self.ball_reset_radius_range = tuple(float(value) for value in radius_range)
        self.ball_reset_angle_range = tuple(float(value) for value in angle_range)
        self.ball_reset_speed_range = tuple(float(value) for value in speed_range)

    def _reset_ball(self, env_ids: torch.Tensor) -> None:
        if env_ids.numel() == 0:
            return
        count = int(env_ids.numel())
        radius_min, radius_max = self.ball_reset_radius_range
        radius = torch.sqrt(torch.empty(count, device=self.device).uniform_(radius_min**2, radius_max**2))
        angle = torch.empty(count, device=self.device).uniform_(*self.ball_reset_angle_range)
        offset_b = torch.stack((radius * torch.cos(angle), radius * torch.sin(angle), torch.zeros_like(radius)), dim=-1)
        forward_w = my_quat_rotate(self.base_quat[env_ids], self.forward_vec[env_ids])
        heading = torch.atan2(forward_w[:, 1], forward_w[:, 0])
        cos_heading, sin_heading = torch.cos(heading), torch.sin(heading)
        offset_w = torch.stack(
            (
                cos_heading * offset_b[:, 0] - sin_heading * offset_b[:, 1],
                sin_heading * offset_b[:, 0] + cos_heading * offset_b[:, 1],
                offset_b[:, 2],
            ),
            dim=-1,
        )
        position = self.robot_root_states[env_ids, :3] + offset_w
        position[:, 2] = self.env_origins[env_ids, 2] + self.ball_radius

        speed = torch.empty(count, device=self.device).uniform_(*self.ball_reset_speed_range)
        velocity_angle = torch.empty(count, device=self.device).uniform_(-math.pi, math.pi)
        velocity_b = torch.stack((speed * torch.cos(velocity_angle), speed * torch.sin(velocity_angle), torch.zeros_like(speed)), dim=-1)
        velocity_w = torch.stack(
            (
                cos_heading * velocity_b[:, 0] - sin_heading * velocity_b[:, 1],
                sin_heading * velocity_b[:, 0] + cos_heading * velocity_b[:, 1],
                velocity_b[:, 2],
            ),
            dim=-1,
        )
        orientation_wxyz = torch.zeros(count, 4, device=self.device)
        orientation_wxyz[:, 0] = 1.0
        root_state = torch.cat((position, orientation_wxyz, velocity_w, torch.zeros(count, 3, device=self.device)), dim=-1)
        self.ball.write_root_state_to_sim(root_state, env_ids=env_ids)

    def reset_idx(self, env_ids: torch.Tensor, target_states: dict[str, torch.Tensor] | None = None) -> None:
        super().reset_idx(env_ids, target_states=target_states)
        self._reset_ball(env_ids)
        self.mjlab_env.scene.write_data_to_sim()
        self.mjlab_env.sim.forward()
        self._refresh_state()
        self.simulator.refresh()

    def _capture_transition_state(self) -> dict[str, torch.Tensor]:
        state = super()._capture_transition_state()
        state.update(
            {
                "ball_pos_w": self.ball_pos_w.detach().clone(),
                "ball_lin_vel_w": self.ball_lin_vel_w.detach().clone(),
                "ball_position_b": self.ball_position_b.detach().clone(),
                "ball_linear_velocity_b": self.ball_linear_velocity_b.detach().clone(),
                "foot_ball_contact_force": self.foot_ball_contact_force.detach().clone(),
            }
        )
        return state


def _configure_validation_viewer(
    viewer,
    *,
    render_size: int,
    camera_distance: float,
    camera_azimuth: float,
    camera_elevation: float,
    disable_shadows: bool,
) -> None:
    """Configure the camera for TeCH Kick rendering."""
    viewer.height = int(render_size)
    viewer.width = int(render_size)
    viewer.distance = float(camera_distance)
    viewer.azimuth = float(camera_azimuth)
    viewer.elevation = float(camera_elevation)
    viewer.origin_type = viewer.OriginType.ASSET_ROOT
    viewer.entity_name = "robot"
    viewer.max_extra_envs = 0
    if disable_shadows:
        # MuJoCo's EGL offscreen shadow map intermittently corrupts the checker
        # ground into black strips on this scene. This only affects video pixels,
        # so disable the unstable visual pass while preserving simulation physics.
        viewer.enable_shadows = False


def build_tech_kick_env(
    *,
    device: str,
    robot_config: str,
    expert_dataset: str,
    num_envs: int,
    seed: int,
    max_episode_length_s: float,
    ball_radius: float = DEFAULT_BALL_RADIUS,
    ball_mass: float = DEFAULT_BALL_MASS,
    disable_domain_randomization: bool = False,
    render_mode: str | None = None,
    render_size: int = 480,
    camera_distance: float = 2.5,
    camera_azimuth: float = 135.0,
    camera_elevation: float = -18.0,
):
    """Build a frozen-actor-compatible environment with a physical ball."""
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.sensor.contact_sensor import ContactMatch, ContactSensorCfg

    robot_training = load_robot_training_spec(robot_config)
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
    hv_config, unresolved = _compose_humanoidverse_config(
        num_envs=num_envs,
        relative_config_path="exp/bfm_zero/bfm_zero",
        hydra_overrides=hydra_overrides,
        lafan_tail_path=expert_dataset,
        data_mix_weights=None,
        disable_obs_noise=True,
        disable_domain_randomization=disable_domain_randomization,
        max_episode_length_s=max_episode_length_s,
        root_height_obs=True,
        robot_training=robot_training.to_env_dict(),
    )
    mjlab_cfg = make_mjlab_ufo_env_cfg(
        hv_config,
        num_envs=num_envs,
        seed=seed,
        mjcf_path=robot_training.robot.xml_path,
        auto_reset=False,
        robot_training=robot_training.to_env_dict(),
    )
    mjlab_cfg.scene.entities["ball"] = _ball_entity_cfg(radius=ball_radius, mass=ball_mass)
    mjlab_cfg.events["ball_damping"] = EventTermCfg(mode="startup", func=_set_ball_damping)
    mjlab_cfg.scene.sensors += (
        ContactSensorCfg(
            name="foot_ball_contact",
            primary=ContactMatch(mode="body", pattern=tuple(robot_training.robot.feet), entity="robot"),
            secondary=ContactMatch(mode="geom", pattern="ball_geom", entity="ball"),
            fields=("found", "force"),
            reduce="maxforce",
            secondary_policy="error",
            history_length=int(hv_config.simulator.config.sim.control_decimation),
        ),
    )
    if render_mode is not None:
        _configure_validation_viewer(
            mjlab_cfg.viewer,
            render_size=render_size,
            camera_distance=camera_distance,
            camera_azimuth=camera_azimuth,
            camera_elevation=camera_elevation,
            disable_shadows=render_mode == "rgb_array",
        )
    mjlab_env = ManagerBasedRlEnv(mjlab_cfg, device=device, render_mode=render_mode)
    creation_config = type("TechKickCreationConfig", (), {"locomotion_mode": True})()
    core = TechKickMjlabCore(hv_config, mjlab_env, creation_config=creation_config, ball_radius=ball_radius)
    env = HumanoidVerseMjlabVectorEnv(core, include_last_action=True, include_history_actor=True)
    env._creation_config = {
        "unresolved_conf": unresolved,
        "mjlab_env_cfg": mjlab_cfg,
        "robot_config": str(Path(robot_config).resolve()),
    }
    return env, robot_training
