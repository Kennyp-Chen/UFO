# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""HT motor actuator with identified torque-speed curve."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.actuators import DelayedPDActuator, DelayedPDActuatorCfg
from isaaclab.utils import configclass
from isaaclab.utils.types import ArticulationActions


class HTMotor(DelayedPDActuator):
    """HT DC motor model with identified non-linear torque-speed curve.

    This actuator implements a HT torque-speed relationship based on experimental
    motor identification. The model distinguishes between two operating modes:

    1. Motoring mode (torque and velocity have same sign):
       Speed is limited by the quadratic curve: |ω| ≤ -0.0141|T|² - 0.0709|T| + 6.2756

    2. Braking mode (torque and velocity have opposite signs):
       Speed is limited by rectangular boundaries: |T| < 20, |ω| < 6

    The torque limits are computed based on current joint velocity and applied to
    clip the computed efforts from the PD controller.
    """

    cfg: HTMotorCfg
    """The configuration for the actuator model."""

    def __init__(self, cfg: HTMotorCfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        self._joint_vel = torch.zeros_like(self.computed_effort)

        # Store curve parameters
        self._curve_a = cfg.curve_param_a  # -0.0141
        self._curve_b = cfg.curve_param_b  # -0.0709
        self._curve_c = cfg.curve_param_c  # 6.2756
        self._max_torque = cfg.max_torque  # 20.0
        self._max_velocity = cfg.max_velocity  # 6.0

        # Override parent class limits with identified values
        self._saturation_effort = self._max_torque
        # self.velocity_limit[:] = self._max_velocity
        # self.effort_limit[:] = self._max_torque

    """
    Helper functions.
    """

    def compute(
        self, control_action: ArticulationActions, joint_pos: torch.Tensor, joint_vel: torch.Tensor
    ) -> ArticulationActions:
        """Process the actuator group actions and compute the articulation actions.

        In case of implicit actuator, the control action is directly returned as the computed action.
        This function is a no-op and does not perform any computation on the input control action.
        However, it computes the approximate torques for the actuated joint since PhysX does not compute
        this quantity explicitly.

        Args:
            control_action: The joint action instance comprising of the desired joint positions, joint velocities
                and (feed-forward) joint efforts.
            joint_pos: The current joint positions of the joints in the group. Shape is (num_envs, num_joints).
            joint_vel: The current joint velocities of the joints in the group. Shape is (num_envs, num_joints).

        Returns:
            The computed desired joint positions, joint velocities and joint efforts.
        """
        self._joint_vel[:] = joint_vel
        # store approximate torques for reward computation
        error_pos = control_action.joint_positions - joint_pos
        error_vel = control_action.joint_velocities - joint_vel
        self.computed_effort = self.stiffness * error_pos + self.damping * error_vel + control_action.joint_efforts
        # clip the torques based on the motor limits
        self.applied_effort = self._clip_effort(self.computed_effort)

        control_action.joint_positions = None
        control_action.joint_velocities = None
        control_action.joint_efforts = self.applied_effort
        return control_action

    def _clip_effort(self, effort: torch.Tensor) -> torch.Tensor:
        """Clip efforts based on identified torque-speed curve.

        Args:
            effort: Computed effort from PD controller. Shape: (num_envs, num_joints)

        Returns:
            Clipped effort tensor with same shape as input.
        """
        # Get current joint velocities
        vel = self._joint_vel

        # Compute absolute values
        abs_vel = torch.abs(vel)
        abs_effort = torch.abs(effort)

        # Determine operating mode: True for motoring, False for braking
        # Motoring: torque and velocity have same sign (product > 0)
        # Braking: torque and velocity have opposite signs (product < 0)
        is_motoring = (effort * vel) >= 0.0

        # --- Compute maximum torque for motoring mode ---
        # Solve the curve equation for maximum torque at current velocity
        # Given: |ω| = -0.0141|T|² - 0.0709|T| + 6.2756
        # Rearrange to: 0.0141|T|² + 0.0709|T| + (|ω| - 6.2756) = 0
        # Use quadratic formula: T = (-b + sqrt(b² - 4ac)) / (2a)
        # Note: We take the positive root to get the valid positive solution

        a = -self._curve_a  # 0.0141
        b = -self._curve_b  # 0.0709
        c = abs_vel - self._curve_c  # |ω| - 6.2756

        # Discriminant
        discriminant = b * b - 4 * a * c
        # Clamp discriminant to avoid sqrt of negative numbers (numerical issues)
        discriminant = torch.clamp(discriminant, min=0.0)

        # Maximum torque from curve (use positive root to get valid solution)
        max_torque_motoring = (-b + torch.sqrt(discriminant)) / (2 * a)

        # Ensure within absolute limits
        max_torque_motoring = torch.clamp(max_torque_motoring, min=0.0, max=self._max_torque)

        # For velocities exceeding max velocity, torque should be zero in motoring mode
        max_torque_motoring = torch.where(
            abs_vel > self._max_velocity, torch.zeros_like(max_torque_motoring), max_torque_motoring
        )

        # --- Compute maximum torque for braking mode ---
        # In braking mode, simple rectangular limit applies
        max_torque_braking = torch.full_like(abs_vel, self._max_torque)

        # For velocities exceeding max velocity, allow full braking torque
        # (This is physically reasonable - you can brake harder at high speeds)

        # --- Select appropriate limit based on operating mode ---
        max_torque = torch.where(is_motoring, max_torque_motoring, max_torque_braking)

        # --- Apply the limits ---
        # Clip absolute effort, then restore sign
        effort_sign = torch.sign(effort)
        clipped_abs_effort = torch.clamp(abs_effort, max=max_torque)
        clamped_effort = effort_sign * clipped_abs_effort

        return clamped_effort


@configclass
class HTMotorCfg(DelayedPDActuatorCfg):
    """Configuration for HT motor with identified torque-speed curve.

    This configuration defines a motor model based on experimental identification
    of the torque-speed relationship. The model uses a quadratic curve for the
    motoring mode and rectangular limits for the braking mode.

    Identified torque-speed characteristics:
    - Motoring mode: |ω| ≤ -0.0141|T|² - 0.0709|T| + 6.2756
    - Braking mode: |T| < 20, |ω| < 6
    - Maximum torque: 20 N·m
    - Maximum velocity: 6 rad/s
    """

    class_type: type = HTMotor

    # Identified curve parameters for motoring mode
    # Curve equation: |ω| = a*|T|² + b*|T| + c
    curve_param_a: float = -0.0141
    """Quadratic coefficient of the torque-speed curve."""

    curve_param_b: float = -0.0709
    """Linear coefficient of the torque-speed curve."""

    curve_param_c: float = 6.2756
    """Constant term of the torque-speed curve (no-load speed in rad/s)."""

    # Physical limits from identification
    max_torque: float = 20.0
    """Maximum torque output in N·m (stall torque)."""

    max_velocity: float = 6.0
    """Maximum velocity in rad/s (no-load speed)."""

    # Override parent class defaults
    saturation_effort: float = 20.0
    """Peak motor torque. Defaults to identified max_torque."""

    # velocity_limit: float = 6.0
    # """Velocity limit. Defaults to identified max_velocity."""

    # effort_limit: float = 20.0
    # """Continuous torque limit. Defaults to identified max_torque."""


@configclass
class HTMotorCfg_5047(HTMotorCfg):
    curve_param_a = -0.0141
    curve_param_b = -0.0709
    curve_param_c = 6.2756
    max_torque = 20.0
    max_velocity = 6.0
    saturation_effort = 20.0


@configclass
class HTMotorCfg_5031(HTMotorCfg):
    curve_param_a = -0.0141
    curve_param_b = -0.0709
    curve_param_c = 6.2756
    max_torque = 20.0
    max_velocity = 6.0
    saturation_effort = 20.0


@configclass
class HTMotorCfg_5036(HTMotorCfg):
    curve_param_a = -0.006667
    curve_param_b = -0.113990
    curve_param_c = 7.732552
    max_torque = 20.0
    max_velocity = 6.0
    saturation_effort = 20.0


@configclass
class HTMotorCfg_4438(HTMotorCfg):
    curve_param_a = -0.128416
    curve_param_b = -0.699618
    curve_param_c = 19.833274
    max_torque = 10.0
    max_velocity = 20.0
    saturation_effort = 10.0


@configclass
class HTMotor40VCfg_3536(HTMotorCfg):
    curve_param_a = 2.860840068
    curve_param_b = -22.221680648
    curve_param_c = 41.753409187
    max_torque = 3.0
    max_velocity = 37.0
    saturation_effort = 3.0


@configclass
class HTMotor40VCfg_4438(HTMotorCfg):
    curve_param_a = -0.184120895
    curve_param_b = -0.637858724
    curve_param_c = 24.600869510
    max_torque = 10.0
    max_velocity = 24.0
    saturation_effort = 10.0


@configclass
class HTMotor40VCfg_5036(HTMotorCfg):
    curve_param_a = -0.021735007
    curve_param_b = -0.030980508
    curve_param_c = 14.063931256
    max_torque = 22.0
    max_velocity = 14.0
    saturation_effort = 22.0
