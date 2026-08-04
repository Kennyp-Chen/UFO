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

from humanoidverse.agents.envs.humanoidverse_mjlab import (
    _compose_humanoidverse_config,
)
from humanoidverse.tools.validate_bfm_setup import validate_motion_file, validate_setup
from humanoidverse.utils.robot_spec import load_robot_training_spec

REPO_ROOT = Path(__file__).resolve().parents[1]
ROBOT_CONFIG = REPO_ROOT / "configs" / "robots" / "piplus_bfm.yaml"
DATA_MANIFEST = REPO_ROOT / "configs" / "data" / "piplus_lafan.yaml"


def _tiny_piplus_motion(control_joints: list[str], *, frames: int = 3) -> dict[str, dict]:
    return {
        "tiny": {
            "root_trans_offset": np.zeros((frames, 3), dtype=np.float32),
            "root_rot": np.tile(np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32), (frames, 1)),
            "pose_aa": np.zeros((frames, 28, 3), dtype=np.float32),
            "dof": np.zeros((frames, len(control_joints)), dtype=np.float32),
            "fps": 30,
            "joint_names": control_joints,
        }
    }


class PiPlusBfmSetupTest(unittest.TestCase):
    def test_robot_training_spec_and_mjcf_layout(self) -> None:
        spec = load_robot_training_spec(ROBOT_CONFIG)
        model = mujoco.MjModel.from_xml_path(str(spec.robot.xml_path))

        self.assertEqual((model.nq, model.nv, model.nu, model.nbody - 1), (30, 29, 23, 28))
        self.assertEqual(len(spec.robot.control_joint_names), 23)
        self.assertEqual(len(spec.robot.body_names), 28)
        self.assertEqual(spec.contact_bodies, ["l_ankle_roll_link", "r_ankle_roll_link"])

        actuator_joints = []
        for actuator_id in range(model.nu):
            joint_id = int(model.actuator_trnid[actuator_id, 0])
            actuator_joints.append(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id))
        self.assertEqual(actuator_joints, spec.robot.control_joint_names)

        xml_root = ET.parse(spec.robot.xml_path).getroot()
        root_body = xml_root.find("worldbody/body")
        self.assertIsNotNone(root_body)
        self.assertEqual(root_body.find("freejoint").attrib["name"], "floating_base_joint")
        self.assertIsNone(root_body.find("joint[@type='free']"))

    def test_hydra_composition_preserves_piplus_reset_distribution(self) -> None:
        spec = load_robot_training_spec(ROBOT_CONFIG)
        cfg, _unresolved = _compose_humanoidverse_config(
            num_envs=2,
            relative_config_path="exp/bfm_zero/bfm_zero",
            hydra_overrides=[f"robot={spec.hydra_robot}", *spec.hydra_overrides],
            lafan_tail_path="/tmp/piplus-test.pkl",
            data_mix_weights=None,
            disable_obs_noise=False,
            disable_domain_randomization=False,
            max_episode_length_s=None,
            root_height_obs=True,
            robot_training=spec.to_env_dict(),
        )
        self.assertEqual(cfg.robot.actions_dim, 23)
        self.assertEqual(cfg.robot.num_bodies, 28)
        self.assertEqual(cfg.robot.algo_obs_dim_dict.actor_obs, 792)
        self.assertEqual(cfg.robot.algo_obs_dim_dict.critic_obs, 417)
        self.assertTrue(cfg.lie_down_init)
        self.assertAlmostEqual(cfg.lie_down_init_prob, 0.3)

    def test_manifest_is_portable_and_records_expected_hashes(self) -> None:
        config = OmegaConf.to_container(OmegaConf.load(DATA_MANIFEST), resolve=False)
        dataset = config["datasets"][0]
        self.assertEqual(config["robot_config"], "configs/robots/piplus_bfm.yaml")
        self.assertIn("PIPLUS_MOTION_DIR", dataset["train_path"])
        self.assertEqual(len(dataset["train_sha256"]), 64)
        self.assertEqual(len(dataset["inference_sha256"]), 64)

    def test_motion_validator_accepts_matching_layout_and_rejects_reordering(self) -> None:
        spec = load_robot_training_spec(ROBOT_CONFIG)
        control_joints = list(spec.robot.control_joint_names)
        with tempfile.TemporaryDirectory() as tmpdir:
            motion_path = Path(tmpdir) / "motion.pkl"
            motion = _tiny_piplus_motion(control_joints)
            joblib.dump(motion, motion_path)
            result = validate_motion_file(motion_path, control_joints=control_joints, body_count=28)
            self.assertEqual(result["motions"], 1)
            self.assertEqual(result["frames"], 3)

            motion["tiny"]["dof_pos"] = motion["tiny"].pop("dof")
            motion["tiny"]["root_quat"] = motion["tiny"].pop("root_rot")
            joblib.dump(motion, motion_path)
            alias_result = validate_motion_file(motion_path, control_joints=control_joints, body_count=28)
            self.assertEqual(alias_result["frames"], 3)

            motion["tiny"]["joint_names"] = list(reversed(control_joints))
            joblib.dump(motion, motion_path)
            with self.assertRaisesRegex(ValueError, "joint_names do not match"):
                validate_motion_file(motion_path, control_joints=control_joints, body_count=28)

    def test_setup_validator_checks_manifest_hash(self) -> None:
        spec = load_robot_training_spec(ROBOT_CONFIG)
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            motion_path = root / "motion.pkl"
            joblib.dump(_tiny_piplus_motion(list(spec.robot.control_joint_names)), motion_path)
            motion_hash = hashlib.sha256(motion_path.read_bytes()).hexdigest()
            manifest = root / "manifest.yaml"
            manifest.write_text(
                "\n".join(
                    [
                        f"robot_config: {ROBOT_CONFIG}",
                        "datasets:",
                        "  - name: tiny_piplus",
                        "    format: ufo_pkl",
                        f"    train_path: {motion_path}",
                        f"    train_sha256: {motion_hash}",
                        "    weight: 1.0",
                    ]
                )
            )
            results = validate_setup(data_manifest=manifest)
            self.assertEqual([(item["dataset"], item["split"]) for item in results], [("tiny_piplus", "train")])


if __name__ == "__main__":
    unittest.main()
