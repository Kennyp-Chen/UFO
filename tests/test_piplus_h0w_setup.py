from __future__ import annotations

import hashlib
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import joblib
import mujoco
import numpy as np
from omegaconf import OmegaConf

from humanoidverse.agents.envs.humanoidverse_mjlab import _action_target_scale, _compose_humanoidverse_config
from humanoidverse.tools.validate_bfm_setup import validate_setup
from humanoidverse.utils.robot_spec import load_robot_training_spec

REPO_ROOT = Path(__file__).resolve().parents[1]
ROBOT_CONFIG = REPO_ROOT / "configs" / "robots" / "piplus_h0w.yaml"
HYDRA_ROBOT_CONFIG = REPO_ROOT / "humanoidverse" / "config" / "robot" / "piplus" / "piplus_h0w.yaml"

CONTROL_JOINTS = [
    "r_shoulder_pitch_joint",
    "r_shoulder_roll_joint",
    "r_upper_arm_joint",
    "r_elbow_joint",
    "l_shoulder_pitch_joint",
    "l_shoulder_roll_joint",
    "l_upper_arm_joint",
    "l_elbow_joint",
    "head_yaw_joint",
    "head_pitch_joint",
    "r_hip_pitch_joint",
    "r_hip_roll_joint",
    "r_thigh_joint",
    "r_calf_joint",
    "r_ankle_pitch_joint",
    "r_ankle_roll_joint",
    "l_hip_pitch_joint",
    "l_hip_roll_joint",
    "l_thigh_joint",
    "l_calf_joint",
    "l_ankle_pitch_joint",
    "l_ankle_roll_joint",
]


def _tiny_motion(frames: int = 3) -> dict[str, dict]:
    return {
        "tiny": {
            "root_trans_offset": np.zeros((frames, 3), dtype=np.float32),
            "root_rot": np.tile(np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32), (frames, 1)),
            "pose_aa": np.zeros((frames, 27, 3), dtype=np.float32),
            "dof": np.zeros((frames, len(CONTROL_JOINTS)), dtype=np.float32),
            "fps": 30,
            "joint_names": CONTROL_JOINTS,
        }
    }


class PiPlusH0wSetupTest(unittest.TestCase):
    def test_robot_training_spec_and_mjcf_layout(self) -> None:
        spec = load_robot_training_spec(ROBOT_CONFIG)
        model = mujoco.MjModel.from_xml_path(str(spec.robot.xml_path))

        self.assertEqual((model.nq, model.nv, model.nu, model.nbody - 1), (29, 28, 22, 27))
        self.assertEqual(spec.robot.control_joint_names, CONTROL_JOINTS)
        self.assertEqual(spec.contact_bodies, ["l_ankle_roll_link", "r_ankle_roll_link"])

        actuator_joints = [
            mujoco.mj_id2name(
                model,
                mujoco.mjtObj.mjOBJ_JOINT,
                int(model.actuator_trnid[actuator_id, 0]),
            )
            for actuator_id in range(model.nu)
        ]
        self.assertEqual(actuator_joints, CONTROL_JOINTS)

        xml_root = ET.parse(spec.robot.xml_path).getroot()
        root_body = xml_root.find("worldbody/body")
        self.assertIsNotNone(root_body)
        self.assertEqual(root_body.find("freejoint").attrib["name"], "floating_base_joint")
        self.assertIsNone(root_body.find("joint[@type='free']"))

    def test_hydra_composition_uses_h0w_dimensions_and_joint_groups(self) -> None:
        spec = load_robot_training_spec(ROBOT_CONFIG)
        cfg, _unresolved = _compose_humanoidverse_config(
            num_envs=2,
            relative_config_path="exp/bfm_zero/bfm_zero",
            hydra_overrides=[f"robot={spec.hydra_robot}", *spec.hydra_overrides],
            lafan_tail_path="/tmp/piplus-h0w-test.pkl",
            data_mix_weights=None,
            disable_obs_noise=False,
            disable_domain_randomization=False,
            max_episode_length_s=None,
            root_height_obs=True,
            robot_training=spec.to_env_dict(),
        )

        self.assertEqual((cfg.robot.actions_dim, cfg.robot.dof_obs_size, cfg.robot.num_bodies), (22, 22, 27))
        self.assertAlmostEqual(cfg.robot.init_state.pos[2], 0.351)
        self.assertAlmostEqual(cfg.robot.init_state.default_joint_angles.r_calf_joint, 0.65)
        target_scales = _action_target_scale(cfg)
        self.assertTrue(np.allclose(target_scales[:8].cpu().numpy(), 0.3691566, atol=1.0e-6))
        self.assertTrue(np.allclose(target_scales[8:10].cpu().numpy(), 0.09614226, atol=1.0e-6))
        self.assertTrue(np.allclose(target_scales[10:].cpu().numpy(), 0.11359521, atol=1.0e-6))

        robot_config = OmegaConf.load(HYDRA_ROBOT_CONFIG).robot
        self.assertEqual(robot_config.lower_body_actions_dim, 12)
        self.assertEqual(robot_config.upper_body_actions_dim, 10)
        self.assertEqual(len(robot_config.lower_dof_names), 12)
        self.assertEqual(len(robot_config.upper_dof_names), 10)
        self.assertEqual(list(robot_config.knee_dof_names), ["r_calf_joint", "l_calf_joint"])

    def test_setup_validator_accepts_h0w_motion_layout(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            motion_path = root / "motion.pkl"
            joblib.dump(_tiny_motion(), motion_path)
            motion_hash = hashlib.sha256(motion_path.read_bytes()).hexdigest()
            manifest = root / "manifest.yaml"
            manifest.write_text(
                "\n".join(
                    [
                        f"robot_config: {ROBOT_CONFIG}",
                        "datasets:",
                        "  - name: tiny_piplus_h0w",
                        "    format: ufo_pkl",
                        f"    train_path: {motion_path}",
                        f"    train_sha256: {motion_hash}",
                        "    weight: 1.0",
                    ]
                )
            )

            results = validate_setup(data_manifest=manifest)
            self.assertEqual([(item["dataset"], item["frames"]) for item in results], [("tiny_piplus_h0w", 3)])


if __name__ == "__main__":
    unittest.main()
