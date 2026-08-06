from __future__ import annotations

import math
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from humanoidverse.piplus_h0w_locomotion import H0WLocomotionMjlabConfig
from humanoidverse.piplus_h0w_onnx_decoder import H0W_ACTOR_OBSERVATION_DIM, OnnxPiPlusH0WDecoder, build_actor_observation
from humanoidverse.piplus_h0w_stage2 import CommandEncoderPolicy, compute_gae, flatten_encoder_observation, project_latent
from humanoidverse.piplus_h0w_stage2_play import MujocoQposRenderer, resolve_playback_asset


class PiPlusH0WStage2CommonTest(unittest.TestCase):
    def test_h0w_environment_exposes_mjlab_rgb_array_render_mode(self) -> None:
        self.assertIn("render_mode", H0WLocomotionMjlabConfig.model_fields)

    def test_playback_uses_checkpoint_metadata_when_asset_override_is_absent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset_path = Path(directory) / "piplus_h0w_lafan.pkl"
            dataset_path.touch()

            resolved = resolve_playback_asset(None, {"motion_dataset": str(dataset_path)}, "motion_dataset")

        self.assertEqual(resolved, dataset_path.resolve())

    def test_qpos_renderer_produces_structured_frames(self) -> None:
        """Offscreen qpos rendering must not reproduce the MJLab rgb_array noise frames.

        The broken rgb_array path produced near-black frames (mean ~0.0) with sparse
        random bright spikes (std >> mean, min 0 / max 255). A structured robot+floor
        render has a clearly positive mean and limited per-channel standard deviation.
        """
        from humanoidverse.piplus_h0w_locomotion import h0w_robot_training_spec

        spec = h0w_robot_training_spec("configs/robots/piplus_h0w.yaml")
        renderer = MujocoQposRenderer(spec.robot.xml_path, render_size=64)
        try:
            frame = renderer.render_qpos(np.zeros(renderer.model.nq))
        finally:
            renderer.close()

        self.assertEqual(frame.shape, (64, 64, 3))
        self.assertEqual(frame.dtype, np.uint8)
        self.assertGreater(float(frame.mean()), 1.0)
        self.assertLess(float(frame.std()), 90.0)

    def test_cuda_onnx_provider_uses_the_requested_device_index(self) -> None:
        import onnx
        from onnx import TensorProto, helper

        graph = helper.make_graph(
            [
                helper.make_node(
                    "Constant",
                    inputs=[],
                    outputs=["action"],
                    value=helper.make_tensor("action_value", TensorProto.FLOAT, [1, 22], [0.0] * 22),
                )
            ],
            "decoder",
            [helper.make_tensor_value_info("actor_input", TensorProto.FLOAT, [1, 616])],
            [helper.make_tensor_value_info("action", TensorProto.FLOAT, [1, 22])],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
        captured: dict[str, object] = {}

        class FakeSession:
            def __init__(self, _serialized: bytes, *, providers: object) -> None:
                captured["providers"] = providers

        fake_runtime = SimpleNamespace(
            get_available_providers=lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"],
            InferenceSession=FakeSession,
        )
        with tempfile.TemporaryDirectory() as directory:
            decoder_path = Path(directory) / "decoder.onnx"
            onnx.save(model, decoder_path)
            with patch.dict(sys.modules, {"onnxruntime": fake_runtime}):
                OnnxPiPlusH0WDecoder(decoder_path, torch.device("cuda:2"))

        self.assertEqual(
            captured["providers"],
            [("CUDAExecutionProvider", {"device_id": 2}), "CPUExecutionProvider"],
        )

    def test_decoder_actor_observation_has_the_616d_contract(self) -> None:
        observation = {
            "state": torch.zeros(2, 50),
            "last_action": torch.zeros(2, 22),
            "history_actor": torch.zeros(2, 288),
        }
        actor_observation = build_actor_observation(observation, torch.zeros(2, 256))

        self.assertEqual(H0W_ACTOR_OBSERVATION_DIM, 616)
        self.assertEqual(tuple(actor_observation.shape), (2, 616))

    def test_command_encoder_features_keep_the_h0w_363d_contract(self) -> None:
        observation = {
            "state": torch.zeros(2, 50),
            "last_action": torch.zeros(2, 22),
            "history_actor": torch.zeros(2, 288),
        }
        features = flatten_encoder_observation(observation, torch.zeros(2, 3))
        policy = CommandEncoderPolicy(features.shape[-1], z_dim=256)
        raw_z, log_prob, value = policy.sample(features)

        self.assertEqual(tuple(features.shape), (2, 363))
        self.assertEqual(tuple(raw_z.shape), (2, 256))
        self.assertEqual(tuple(log_prob.shape), (2,))
        self.assertEqual(tuple(value.shape), (2,))

    def test_latent_projection_matches_the_frozen_decoder_norm(self) -> None:
        projected = project_latent(torch.randn(4, 256))

        self.assertTrue(torch.allclose(projected.norm(dim=-1), torch.full((4,), math.sqrt(256.0)), atol=1.0e-5))

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


if __name__ == "__main__":
    unittest.main()
