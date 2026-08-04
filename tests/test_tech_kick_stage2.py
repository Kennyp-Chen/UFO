import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import mujoco
import torch

from humanoidverse.agents.envs.tech_kick_mjlab import (
    BALL_ANGULAR_DAMPING,
    BALL_CONTACT_DAMPING,
    BALL_CONTACT_STIFFNESS,
    BALL_GEOM_PRIORITY,
    BALL_LINEAR_DAMPING,
    BALL_MUJOCO_SLIDING_FRICTION,
    DEFAULT_BALL_MASS,
    DEFAULT_BALL_RADIUS,
    _configure_validation_viewer,
    _set_ball_damping,
    _ball_entity_cfg,
)
from humanoidverse.tasks.kick import KickCommandState, KickObservationHistory, KickRewardState, kick_curriculum
from humanoidverse.tech_kick_stage2 import CommandEncoderPolicy, compute_gae, validate_latent_contract
from humanoidverse.tech_kick_stage2_play import stage2_checkpoints


class TechKickStage2Test(unittest.TestCase):
    def test_stage2_checkpoints_are_sorted_numerically(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            for name in ("checkpoint_10.pt", "checkpoint_2.pt", "checkpoint_bad.pt"):
                (folder / name).touch()

            checkpoints = stage2_checkpoints(folder)

        self.assertEqual([path.name for path in checkpoints], ["checkpoint_2.pt", "checkpoint_10.pt"])

    def test_curriculum_expands_task_distribution(self) -> None:
        start = kick_curriculum(0, 100)
        end = kick_curriculum(99, 100)

        self.assertEqual(start.foot_mode_probabilities, (0.0, 0.0, 1.0))
        self.assertAlmostEqual(sum(end.foot_mode_probabilities), 1.0)
        self.assertGreater(end.ball_radius_range[1], start.ball_radius_range[1])
        self.assertGreater(end.ball_speed_range[1], start.ball_speed_range[1])

    def test_command_target_stays_fixed_in_world_frame(self) -> None:
        state = KickCommandState(1, torch.device("cpu"))
        identity = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
        state.sample(torch.tensor([0]), identity, kick_curriculum(0, 1))
        target_w = state.target_velocity_w.clone()

        half_angle = math.pi / 4.0
        yaw_90 = torch.tensor([[0.0, 0.0, math.sin(half_angle), math.cos(half_angle)]])
        command = state.command(yaw_90)

        self.assertTrue(torch.equal(state.target_velocity_w, target_w))
        self.assertTrue(torch.allclose(command[:, :2].norm(dim=-1), target_w[:, :2].norm(dim=-1)))
        self.assertEqual(command.shape, (1, 5))

    def test_history_shape_matches_deployable_features(self) -> None:
        history = KickObservationHistory(2, 3, torch.device("cpu"))
        command = torch.zeros(2, 5)
        position = torch.ones(2, 3)
        velocity = torch.zeros(2, 3)
        history.reset(torch.tensor([0, 1]), command, position, velocity)

        self.assertEqual(history.flattened().shape, (2, 33))
        history.append(command, position * 2.0, velocity)
        self.assertTrue(torch.equal(history.values[:, -1, 5:8], position * 2.0))

    def test_policy_and_checkpoint_latent_contract(self) -> None:
        policy = CommandEncoderPolicy(input_dim=19, z_dim=8, hidden_dim=32)
        features = torch.randn(5, 19)
        raw_z, log_prob, value = policy.sample(features)
        model = SimpleNamespace(
            cfg=SimpleNamespace(archi=SimpleNamespace(z_dim=8, norm_z=True, z_geometry="hypersphere")),
            project_z=lambda z: math.sqrt(z.shape[-1]) * torch.nn.functional.normalize(z, dim=-1),
        )

        contract = validate_latent_contract(policy, model)

        self.assertEqual(raw_z.shape, (5, 8))
        self.assertEqual(log_prob.shape, (5,))
        self.assertEqual(value.shape, (5,))
        self.assertEqual(contract["mode"], "checkpoint_project_z")

    def test_gae_stops_at_termination_and_timeout(self) -> None:
        rewards = torch.tensor([[1.0], [2.0], [3.0]])
        values = torch.zeros_like(rewards)
        terminated = torch.tensor([[False], [True], [False]])
        truncated = torch.zeros_like(terminated)
        advantages, returns = compute_gae(
            rewards,
            values,
            terminated,
            truncated,
            torch.tensor([4.0]),
            discount=0.9,
            gae_lambda=1.0,
        )

        expected = torch.tensor([[1.0 + 0.9 * 2.0], [2.0], [3.0 + 0.9 * 4.0]])
        self.assertTrue(torch.allclose(advantages, expected))
        self.assertTrue(torch.allclose(returns, expected))

    def test_reward_velocity_is_gated_by_correct_first_touch(self) -> None:
        state = KickRewardState(
            num_envs=2,
            dt=0.02,
            feet_indices=torch.tensor([0, 1]),
            contact_force_threshold=0.1,
            velocity_std=0.8,
            device=torch.device("cpu"),
        )
        core = SimpleNamespace(
            body_pos=torch.tensor([[[0.3, 0.0, 0.1], [0.3, 0.2, 0.1]]] * 2),
            ball_pos_w=torch.tensor([[0.3, 0.0, 0.1]] * 2),
            ball_linear_velocity_b=torch.tensor([[2.0, 0.0, 0.0]] * 2),
            foot_ball_contact_force=torch.zeros(2, 2),
            default_dof_pos=torch.zeros(2, 3),
            default_dof_pos_offset=torch.zeros(2, 3),
            dof_pos=torch.zeros(2, 3),
        )
        command = torch.tensor([[2.0, 0.0, 0.0, 1.0, 0.0], [2.0, 0.0, 0.0, 0.0, 1.0]])
        state.reset(torch.tensor([0, 1]), core, command)
        core.foot_ball_contact_force[:, 0] = 1.0

        _, components, diagnostics = state.compute(core, command)

        self.assertTrue(torch.equal(diagnostics["correct_touch"], torch.tensor([True, False])))
        self.assertTrue(torch.equal(diagnostics["wrong_touch"], torch.tensor([False, True])))
        self.assertGreater(float(components["ball_velocity"][0]), 0.0)
        self.assertEqual(float(components["ball_velocity"][1]), 0.0)

    def test_ball_entity_is_a_floating_sphere(self) -> None:
        model = _ball_entity_cfg(radius=DEFAULT_BALL_RADIUS, mass=DEFAULT_BALL_MASS).spec_fn().compile()
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ball_geom")

        self.assertEqual(model.nq, 7)
        self.assertEqual(model.nv, 6)
        self.assertAlmostEqual(float(model.geom_size[geom_id, 0]), DEFAULT_BALL_RADIUS)
        self.assertAlmostEqual(float(model.body_mass[1]), DEFAULT_BALL_MASS)
        self.assertAlmostEqual(float(model.geom_friction[geom_id, 0]), BALL_MUJOCO_SLIDING_FRICTION)
        self.assertEqual(int(model.geom_priority[geom_id]), BALL_GEOM_PRIORITY)
        self.assertEqual(int(model.geom_condim[geom_id]), 3)
        self.assertAlmostEqual(float(model.geom_solref[geom_id, 0]), -BALL_CONTACT_STIFFNESS)
        self.assertAlmostEqual(float(model.geom_solref[geom_id, 1]), -BALL_CONTACT_DAMPING)

    def test_ball_damping_matches_reference_linear_and_angular_values(self) -> None:
        class Scene:
            def __getitem__(self, key):
                return SimpleNamespace(
                    indexing=SimpleNamespace(
                        free_joint_v_adr=torch.tensor([2, 3, 4, 5, 6, 7], dtype=torch.int32)
                    )
                )

        env = SimpleNamespace(
            num_envs=2,
            device="cpu",
            scene=Scene(),
            sim=SimpleNamespace(model=SimpleNamespace(dof_damping=torch.zeros(2, 10))),
        )
        _set_ball_damping(env, None)

        expected = torch.tensor([BALL_LINEAR_DAMPING] * 3 + [BALL_ANGULAR_DAMPING] * 3)
        self.assertTrue(torch.allclose(env.sim.model.dof_damping[:, 2:8], expected.expand(2, -1)))

    def test_validation_viewer_disables_unstable_offscreen_shadows(self) -> None:
        class OriginType:
            ASSET_ROOT = object()

        viewer = SimpleNamespace(OriginType=OriginType, enable_shadows=True)

        _configure_validation_viewer(
            viewer,
            render_size=480,
            camera_distance=2.5,
            camera_azimuth=135.0,
            camera_elevation=-18.0,
            disable_shadows=True,
        )

        self.assertEqual((viewer.height, viewer.width), (480, 480))
        self.assertEqual(viewer.origin_type, OriginType.ASSET_ROOT)
        self.assertEqual(viewer.entity_name, "robot")
        self.assertEqual(viewer.max_extra_envs, 0)
        self.assertFalse(viewer.enable_shadows)


if __name__ == "__main__":
    unittest.main()
