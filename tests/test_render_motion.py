from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import mujoco
import numpy as np

from humanoidverse.mjlab_inference_utils import MujocoQposRenderer
from humanoidverse.render_motion import frame_indices, motion_to_qpos, select_motion
from humanoidverse.utils.robot_spec import load_robot_spec

REPO_ROOT = Path(__file__).resolve().parents[1]


class RenderMotionTest(unittest.TestCase):
    def test_renderer_adds_floor_when_model_has_no_plane(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            xml_path = Path(tmpdir) / "no_floor.xml"
            xml_path.write_text(
                """
<mujoco model="no_floor">
  <worldbody>
    <body name="base" pos="0 0 0.5">
      <freejoint name="root"/>
      <geom name="body" type="sphere" size="0.1" mass="1"/>
    </body>
  </worldbody>
</mujoco>
""".strip()
            )
            renderer = MujocoQposRenderer(xml_path, render_size=64, expected_qpos_size=7)
            try:
                plane_names = [
                    mujoco.mj_id2name(renderer.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
                    for geom_id, geom_type in enumerate(renderer.model.geom_type)
                    if int(geom_type) == int(mujoco.mjtGeom.mjGEOM_PLANE)
                ]
                self.assertEqual(plane_names, ["inference_floor"])
            finally:
                renderer.close()

    def test_renderer_does_not_duplicate_existing_piplus_floor(self) -> None:
        for robot_name in ("piplus_bfm", "piplus_h0w"):
            with self.subTest(robot=robot_name):
                robot = load_robot_spec(REPO_ROOT / "configs" / "robots" / f"{robot_name}.yaml")
                renderer = MujocoQposRenderer(Path(robot.xml_path), render_size=64, expected_qpos_size=robot.nq)
                try:
                    plane_names = [
                        mujoco.mj_id2name(renderer.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
                        for geom_id, geom_type in enumerate(renderer.model.geom_type)
                        if int(geom_type) == int(mujoco.mjtGeom.mjGEOM_PLANE)
                    ]
                    self.assertEqual(plane_names, ["floor"])
                finally:
                    renderer.close()

    def test_both_piplus_models_map_named_motion_joints_to_qpos(self) -> None:
        for robot_name in ("piplus_bfm", "piplus_h0w"):
            with self.subTest(robot=robot_name):
                robot = load_robot_spec(REPO_ROOT / "configs" / "robots" / f"{robot_name}.yaml")
                model = mujoco.MjModel.from_xml_path(str(robot.xml_path))
                motion_joint_names = list(reversed(robot.control_joint_names))
                dof = np.tile(np.arange(len(motion_joint_names), dtype=np.float64), (2, 1))
                motion = {
                    "root_trans_offset": np.asarray([[1.0, 2.0, 0.8], [1.1, 2.0, 0.8]]),
                    "root_rot": np.asarray([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]]),
                    "dof": dof,
                    "joint_names": motion_joint_names,
                    "fps": 30,
                }

                qpos = motion_to_qpos(motion, robot, model, motion_key="test")

                self.assertEqual(qpos.shape, (2, robot.nq))
                root_addr = robot.joint_qpos_addr[robot.free_joint]
                np.testing.assert_allclose(qpos[0, root_addr : root_addr + 3], [1.0, 2.0, 0.8])
                np.testing.assert_allclose(qpos[0, root_addr + 3 : root_addr + 7], [1.0, 0.0, 0.0, 0.0])
                motion_index = {name: index for index, name in enumerate(motion_joint_names)}
                for joint_name in robot.control_joint_names:
                    self.assertEqual(qpos[0, robot.joint_qpos_addr[joint_name]], motion_index[joint_name])

    def test_motion_selection_and_frame_slicing_are_explicit(self) -> None:
        motions = {"first": {"fps": 30}, "second": {"fps": 60}}
        self.assertEqual(select_motion(motions, motion_key=None, motion_index=None)[0], "first")
        self.assertEqual(select_motion(motions, motion_key="second", motion_index=None)[0], "second")
        self.assertEqual(frame_indices(10, start_frame=1, max_frames=3, stride=2).tolist(), [1, 3, 5])
        with self.assertRaisesRegex(ValueError, "out of range"):
            select_motion(motions, motion_key=None, motion_index=2)
        with self.assertRaisesRegex(ValueError, "stride"):
            frame_indices(10, start_frame=0, max_frames=None, stride=0)


if __name__ == "__main__":
    unittest.main()
