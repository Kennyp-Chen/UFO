"""Identified HT motor actuator for the MJLab backend.

The IsaacLab PiPlus tasks use an HT motor model which clips the PD effort to
an identified torque-speed curve.  MJLab has a different actuator interface;
this module applies the same clipping law after MJLab computes the PD effort.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import mujoco
import mujoco_warp as mjwarp
import torch
from mjlab.actuator.actuator import ActuatorCmd
from mjlab.actuator.pd_actuator import IdealPdActuator, IdealPdActuatorCfg

if TYPE_CHECKING:
    from mjlab.entity import Entity


@dataclass(kw_only=True)
class HTMotorActuatorCfg(IdealPdActuatorCfg):
    """Configuration for an identified HT motor torque-speed profile."""

    curve_param_a: float
    curve_param_b: float
    curve_param_c: float
    max_torque: float
    max_velocity: float

    def build(
        self, entity: Entity, target_ids: list[int], target_names: list[str]
    ) -> HTMotorActuator:
        return HTMotorActuator(self, entity, target_ids, target_names)


class HTMotorActuator(IdealPdActuator[HTMotorActuatorCfg]):
    """MJLab implementation of the HT motoring/braking clipping law."""

    def __init__(
        self,
        cfg: HTMotorActuatorCfg,
        entity: Entity,
        target_ids: list[int],
        target_names: list[str],
    ) -> None:
        super().__init__(cfg, entity, target_ids, target_names)
        self._joint_vel: torch.Tensor | None = None
        self._curve_a: torch.Tensor | None = None
        self._curve_b: torch.Tensor | None = None
        self._curve_c: torch.Tensor | None = None
        self._max_torque: torch.Tensor | None = None
        self._max_velocity: torch.Tensor | None = None

    def initialize(
        self,
        mj_model: mujoco.MjModel,
        model: mjwarp.Model,
        data: mjwarp.Data,
        device: str,
    ) -> None:
        super().initialize(mj_model, model, data, device)
        shape = (data.nworld, len(self._target_names))
        self._joint_vel = torch.zeros(shape, dtype=torch.float32, device=device)
        self._curve_a = torch.full(shape, self.cfg.curve_param_a, dtype=torch.float32, device=device)
        self._curve_b = torch.full(shape, self.cfg.curve_param_b, dtype=torch.float32, device=device)
        self._curve_c = torch.full(shape, self.cfg.curve_param_c, dtype=torch.float32, device=device)
        self._max_torque = torch.full(shape, self.cfg.max_torque, dtype=torch.float32, device=device)
        self._max_velocity = torch.full(shape, self.cfg.max_velocity, dtype=torch.float32, device=device)

    def compute(self, cmd: ActuatorCmd) -> torch.Tensor:
        if self._joint_vel is None:
            raise RuntimeError("HTMotorActuator must be initialized before compute().")
        self._joint_vel[:] = cmd.vel
        return super().compute(cmd)

    def _clip_effort(self, effort: torch.Tensor) -> torch.Tensor:
        values = (self._joint_vel, self._curve_a, self._curve_b, self._curve_c, self._max_torque, self._max_velocity)
        if any(value is None for value in values):
            raise RuntimeError("HTMotorActuator must be initialized before clipping effort.")
        assert self._joint_vel is not None
        assert self._curve_a is not None
        assert self._curve_b is not None
        assert self._curve_c is not None
        assert self._max_torque is not None
        assert self._max_velocity is not None

        abs_vel = torch.abs(self._joint_vel)
        abs_effort = torch.abs(effort)
        is_motoring = (effort * self._joint_vel) >= 0.0

        # Solve |w| = a*T^2 + b*T + c.  Select the smallest non-negative root
        # so profiles with either sign of the quadratic coefficient work.
        a = -self._curve_a
        b = -self._curve_b
        c = abs_vel - self._curve_c
        discriminant = torch.clamp(b * b - 4.0 * a * c, min=0.0)
        sqrt_discriminant = torch.sqrt(discriminant)
        safe_a = torch.where(torch.abs(a) > 1.0e-8, a, torch.ones_like(a))
        root_plus = (-b + sqrt_discriminant) / (2.0 * safe_a)
        root_minus = (-b - sqrt_discriminant) / (2.0 * safe_a)
        inf = torch.full_like(root_plus, float("inf"))
        root_plus = torch.where(root_plus >= 0.0, root_plus, inf)
        root_minus = torch.where(root_minus >= 0.0, root_minus, inf)
        max_torque_motoring = torch.minimum(root_plus, root_minus)
        linear_root = -c / torch.where(torch.abs(b) > 1.0e-8, b, torch.ones_like(b))
        max_torque_motoring = torch.where(torch.abs(a) > 1.0e-8, max_torque_motoring, linear_root)
        max_torque_motoring = torch.where(
            torch.isfinite(max_torque_motoring), max_torque_motoring, torch.zeros_like(max_torque_motoring)
        )
        max_torque_motoring = torch.minimum(torch.clamp(max_torque_motoring, min=0.0), self._max_torque)
        max_torque_motoring = torch.where(abs_vel > self._max_velocity, torch.zeros_like(max_torque_motoring), max_torque_motoring)

        # Braking uses the identified peak torque, matching HTMotor in soccer.
        max_torque = torch.where(is_motoring, max_torque_motoring, self._max_torque)
        return torch.sign(effort) * torch.clamp(abs_effort, max=max_torque)

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        super().reset(env_ids)
        if self._joint_vel is not None:
            if env_ids is None:
                self._joint_vel.zero_()
            else:
                self._joint_vel[env_ids] = 0.0
