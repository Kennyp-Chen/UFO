from __future__ import annotations

import unittest
from math import sqrt
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from humanoidverse.amp_stage2_piplus_22dof import (
    AMP_STAGE2_TASK,
    PROFILE_A_MIMICLITE_SPEED_SAFETY,
    PROFILE_B_AMP_23DOF_REFERENCE,
    PROFILE_C_UNITREE_VELOCITY_H0W_COMPAT,
    MimicLiteLocomotionRewardState,
    UnitreeVelocityLocomotionRewardState,
    _apply_reward_profile_defaults,
    build_online_amp_feature,
    checkpoint_task_is_compatible,
    expected_h0w_amp_feature_dim,
    get_amp_reward_profile,
)
from humanoidverse.amp_stage2_piplus_22dof_play import (
    _parse_args as parse_amp_playback_args,
)
from humanoidverse.amp_stage2_piplus_22dof_play import (
    checkpoint_task_is_compatible as amp_playback_task_is_compatible,
)
from humanoidverse.piplus_h0w_stage2 import UNITREE_PPO_CONFIG
from humanoidverse.speed_stage2 import SPEED_STAGE2_TASK


class AmpStage2PiPlus22DoFTest(unittest.TestCase):
    def _unitree_core(self) -> SimpleNamespace:
        return SimpleNamespace(
            num_envs=1,
            device=torch.device("cpu"),
            base_lin_vel=torch.tensor([[0.2, -0.1, 0.3]]),
            base_ang_vel=torch.tensor([[0.4, -0.2, 0.15]]),
            body_ang_vel=torch.tensor([[[0.4, -0.3, 0.2]]]),
            body_rot=torch.tensor([[[0.0, 0.3826834, 0.0, 0.9238795]]]),
            contact_forces=torch.zeros(1, 1, 3),
            body_pos=torch.zeros(1, 1, 3),
            torques=torch.zeros(1, 2),
            dof_vel=torch.tensor([[0.2, -0.1]]),
            dof_pos=torch.tensor([[0.25, -0.5]]),
            default_dof_pos=torch.zeros(1, 2),
            default_dof_pos_offset=torch.zeros(1, 2),
        )

    def test_unitree_profile_is_selectable_but_not_default_and_preserves_a_b(self) -> None:
        self.assertEqual(get_amp_reward_profile(PROFILE_C_UNITREE_VELOCITY_H0W_COMPAT).name, PROFILE_C_UNITREE_VELOCITY_H0W_COMPAT)
        self.assertEqual(get_amp_reward_profile(PROFILE_A_MIMICLITE_SPEED_SAFETY).name, PROFILE_A_MIMICLITE_SPEED_SAFETY)
        self.assertEqual(get_amp_reward_profile(PROFILE_B_AMP_23DOF_REFERENCE).name, PROFILE_B_AMP_23DOF_REFERENCE)
        with patch("sys.argv", ["amp_stage2_piplus_22dof.py"]):
            from humanoidverse.amp_stage2_piplus_22dof import _parse_args

            self.assertEqual(_parse_args().reward_profile, PROFILE_A_MIMICLITE_SPEED_SAFETY)

    def test_unitree_profile_disables_amp_and_latent_prior(self) -> None:
        profile = get_amp_reward_profile(PROFILE_C_UNITREE_VELOCITY_H0W_COMPAT)

        self.assertFalse(profile.uses_amp)
        self.assertFalse(profile.uses_latent_prior)
        self.assertEqual(profile.amp_weight, 0.0)
        self.assertEqual(profile.latent_prior_weight, 0.0)

    def test_unitree_linear_tracking_uses_xy_plus_twice_z_with_std_half(self) -> None:
        state = UnitreeVelocityLocomotionRewardState(num_envs=1, num_dof=2, dt=0.02, device=torch.device("cpu"))
        core = self._unitree_core()
        commands = torch.tensor([[0.0, 0.0, 0.0]])

        _reward, components = state.compute(core, commands, torch.zeros(1, 2), torch.zeros(1, dtype=torch.bool))

        error = 0.2**2 + (-0.1) ** 2 + 2.0 * 0.3**2
        self.assertTrue(torch.allclose(components["tracking_lin_vel"], torch.tensor([torch.exp(torch.tensor(-error / 0.5**2))])))

    def test_unitree_angular_tracking_uses_yaw_plus_point_zero_five_xy_rates(self) -> None:
        state = UnitreeVelocityLocomotionRewardState(num_envs=1, num_dof=2, dt=0.02, device=torch.device("cpu"))
        core = self._unitree_core()
        commands = torch.tensor([[0.0, 0.0, 0.0]])

        _reward, components = state.compute(core, commands, torch.zeros(1, 2), torch.zeros(1, dtype=torch.bool))

        error = 0.15**2 + 0.05 * (0.4**2 + (-0.3) ** 2)
        self.assertTrue(torch.allclose(components["tracking_ang_vel"], torch.tensor([torch.exp(torch.tensor(-error / sqrt(0.5) ** 2))])))

    def test_unitree_costs_and_standstill_gate_use_h0w_core_fields(self) -> None:
        state = UnitreeVelocityLocomotionRewardState(num_envs=1, num_dof=2, dt=0.02, device=torch.device("cpu"))
        core = self._unitree_core()
        actions = torch.tensor([[0.1, -0.2]])

        _reward, components = state.compute(core, torch.zeros(1, 3), actions, torch.zeros(1, dtype=torch.bool))

        self.assertTrue(torch.allclose(components["orientation_l2"], -torch.tensor([0.5]), atol=1.0e-5))
        self.assertTrue(torch.allclose(components["ang_vel_xy_l2"], -torch.tensor([0.4**2 + (-0.3) ** 2])))
        self.assertTrue(torch.allclose(components["action_rate_l2"], -actions.square().sum(dim=-1)))
        self.assertTrue(torch.allclose(components["dof_acc_l2"], -((core.dof_vel / 0.02).square().sum(dim=-1))))
        self.assertTrue(torch.allclose(components["stand_still"], -core.dof_pos.square().sum(dim=-1)))

        _standing_reward, standing_components = state.compute(
            core, torch.tensor([[0.2, 0.0, 0.0]]), actions, torch.zeros(1, dtype=torch.bool)
        )
        self.assertTrue(torch.allclose(standing_components["stand_still"], torch.zeros(1)))

    def test_unitree_profile_exposes_unitree_ppo_contract(self) -> None:
        self.assertEqual(UNITREE_PPO_CONFIG["value_coef"], 1.0)
        self.assertEqual(UNITREE_PPO_CONFIG["entropy_coef"], 0.01)
        self.assertTrue(UNITREE_PPO_CONFIG["clip_value_loss"])
        self.assertEqual(UNITREE_PPO_CONFIG["desired_kl"], 0.01)
        self.assertEqual(UNITREE_PPO_CONFIG["schedule"], "adaptive")
        self.assertEqual(UNITREE_PPO_CONFIG["learning_rate"], 1.0e-3)
        self.assertEqual(UNITREE_PPO_CONFIG["rollout_steps"], 24)
        self.assertEqual(UNITREE_PPO_CONFIG["num_minibatches"], 4)

    def test_amp_player_accepts_mp4_output_configuration(self) -> None:
        with patch(
            "sys.argv",
            [
                "amp_stage2_piplus_22dof_play.py",
                "--video-path",
                "/tmp/piplus_h0w_preview.mp4",
                "--render-every",
                "3",
                "--render-size",
                "320",
            ],
        ):
            args = parse_amp_playback_args()

        self.assertEqual(args.video_path, Path("/tmp/piplus_h0w_preview.mp4"))
        self.assertEqual(args.render_every, 3)
        self.assertEqual(args.render_size, 320)

    def test_amp_feature_contract_is_194_for_the_h0w_key_bodies(self) -> None:
        self.assertEqual(expected_h0w_amp_feature_dim(22, history_length=8, key_body_count=5), 194)
        core = SimpleNamespace(
            base_lin_vel=torch.zeros(1, 3),
            base_quat=torch.tensor([[0.0, 0.0, 0.0, 1.0]]),
            robot_root_states=torch.zeros(1, 13),
            body_pos=torch.zeros(1, 27, 3),
        )
        history = torch.zeros(1, 8, 22)

        feature = build_online_amp_feature(core, history, torch.tensor([0, 1, 2, 3, 4]))

        self.assertEqual(tuple(feature.shape), (1, 194))

    def test_profile_a_is_strictly_the_pure_locomotion_baseline(self) -> None:
        profile = get_amp_reward_profile(PROFILE_A_MIMICLITE_SPEED_SAFETY)
        args = SimpleNamespace(
            reward_profile=PROFILE_A_MIMICLITE_SPEED_SAFETY,
            motion_dataset=None,
            expert_dataset=None,
            amp_weight=0.25,
            latent_prior_weight=0.02,
            env_reward_weight=None,
            locomotion_reward_weight=None,
        )

        applied = _apply_reward_profile_defaults(args)

        self.assertEqual(AMP_STAGE2_TASK, "amp_stage2_piplus_22dof")
        self.assertEqual(applied, profile)
        self.assertFalse(profile.uses_amp)
        self.assertEqual(args.amp_weight, 0.0)
        self.assertEqual(args.latent_prior_weight, 0.0)
        self.assertIsInstance(args.motion_dataset, Path)
        self.assertIsInstance(args.expert_dataset, Path)

    def test_profile_a_prioritizes_planar_velocity_tracking(self) -> None:
        profile = get_amp_reward_profile(PROFILE_A_MIMICLITE_SPEED_SAFETY)

        self.assertEqual(profile.locomotion_weights["linvel_exp"], 3.7)

    def test_profile_b_requires_amp_and_player_rejects_speed(self) -> None:
        profile = get_amp_reward_profile(PROFILE_B_AMP_23DOF_REFERENCE)

        self.assertTrue(profile.uses_amp)
        self.assertTrue(profile.uses_latent_prior)
        self.assertTrue(checkpoint_task_is_compatible({"task": AMP_STAGE2_TASK}))
        self.assertFalse(checkpoint_task_is_compatible({"task": SPEED_STAGE2_TASK}))
        self.assertTrue(amp_playback_task_is_compatible({"task": AMP_STAGE2_TASK}))
        self.assertFalse(amp_playback_task_is_compatible({"task": SPEED_STAGE2_TASK}))

    def test_profile_b_mimiclite_rewards_penalize_double_support(self) -> None:
        state = MimicLiteLocomotionRewardState(
            num_envs=1,
            num_dof=2,
            dt=0.02,
            feet_indices=torch.tensor([1, 2]),
            torso_index=0,
            joint_vel_indices=torch.tensor([0]),
            joint_deviation_indices=torch.tensor([0]),
            device=torch.device("cpu"),
            weights=get_amp_reward_profile(PROFILE_B_AMP_23DOF_REFERENCE).locomotion_weights,
        )
        core = SimpleNamespace(
            num_envs=1,
            device=torch.device("cpu"),
            base_lin_vel=torch.zeros(1, 3),
            base_ang_vel=torch.zeros(1, 3),
            body_ang_vel=torch.zeros(1, 3, 3),
            body_pos=torch.tensor([[[0.0, 0.0, 0.4], [0.0, 0.0, 0.1], [0.0, 0.0, 0.1]]]),
            body_rot=torch.tensor([[[0.0, 0.0, 0.0, 1.0]]]).repeat(1, 3, 1),
            contact_forces=torch.tensor([[[0.0, 0.0, 0.0], [0.0, 0.0, 2.0], [0.0, 0.0, 2.0]]]),
            torques=torch.zeros(1, 2),
            dof_vel=torch.zeros(1, 2),
            dof_pos=torch.zeros(1, 2),
            default_dof_pos=torch.zeros(1, 2),
            default_dof_pos_offset=torch.zeros(1, 2),
        )

        _reward, components = state.compute(
            core, torch.tensor([[0.5, 0.0, 0.0]]), torch.zeros(1, 2), torch.zeros(1, dtype=torch.bool)
        )

        self.assertLess(float(components["single_foot_contact"]), 0.0)


if __name__ == "__main__":
    unittest.main()
