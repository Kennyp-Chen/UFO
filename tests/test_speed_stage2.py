from __future__ import annotations

import unittest

import mujoco
import torch

from humanoidverse.speed_stage2 import (
    SPEED_STAGE2_TASK,
    _actuator_joint_names,
    checkpoint_task_is_compatible,
    speed_tracking_reward,
)
from humanoidverse.speed_stage2_play import checkpoint_task_is_compatible as speed_playback_task_is_compatible
from humanoidverse.utils.robot_spec import load_robot_training_spec


class SpeedStage2Test(unittest.TestCase):
    def test_speed_reward_is_maximal_for_a_matched_velocity_command(self) -> None:
        reward, components = speed_tracking_reward(
            torch.tensor([[0.4, -0.2, 0.0]]),
            torch.tensor([[0.0, 0.0, 0.3]]),
            torch.tensor([[0.4, -0.2, 0.3]]),
        )

        self.assertEqual(SPEED_STAGE2_TASK, "speed_stage2_piplus_22dof")
        self.assertTrue(torch.allclose(reward, torch.tensor([1.5])))
        self.assertTrue(torch.allclose(components["linear_velocity"], torch.ones(1)))
        self.assertTrue(torch.allclose(components["yaw_velocity"], torch.ones(1)))

    def test_speed_stage_and_player_reject_amp_checkpoints(self) -> None:
        self.assertTrue(checkpoint_task_is_compatible({"task": SPEED_STAGE2_TASK}))
        self.assertTrue(checkpoint_task_is_compatible({}))
        self.assertFalse(checkpoint_task_is_compatible({"task": "amp_stage2_piplus_22dof"}))
        self.assertTrue(speed_playback_task_is_compatible({"task": SPEED_STAGE2_TASK}))
        self.assertFalse(speed_playback_task_is_compatible({"task": "amp_stage2_piplus_22dof"}))

    def test_h0w_actuator_names_match_the_robot_spec_order(self) -> None:
        spec = load_robot_training_spec("configs/robots/piplus_h0w.yaml")
        model = mujoco.MjModel.from_xml_path(str(spec.robot.xml_path))

        self.assertEqual(_actuator_joint_names(model), list(spec.robot.control_joint_names))


if __name__ == "__main__":
    unittest.main()
