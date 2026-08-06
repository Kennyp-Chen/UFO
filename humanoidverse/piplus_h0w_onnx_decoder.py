"""Frozen ONNX decoder adapter for the 22DoF PiPlus H0W BFM export."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as functional

H0W_ACTION_DIM = 22
H0W_LATENT_DIM = 256
H0W_STATE_DIM = 50
H0W_HISTORY_DIM = 288
H0W_ACTOR_OBSERVATION_DIM = H0W_STATE_DIM + H0W_ACTION_DIM + H0W_HISTORY_DIM + H0W_LATENT_DIM


def build_actor_observation(observation: Mapping[str, torch.Tensor], latent: torch.Tensor) -> torch.Tensor:
    """Validate and concatenate the frozen decoder's actor-observation contract."""
    try:
        state = observation["state"]
        last_action = observation["last_action"]
        history_actor = observation["history_actor"]
    except KeyError as exc:
        raise ValueError("PiPlus H0W decoder requires state, last_action, and history_actor observations") from exc

    expected = (H0W_STATE_DIM, H0W_ACTION_DIM, H0W_HISTORY_DIM, H0W_LATENT_DIM)
    actual = (state.shape[-1], last_action.shape[-1], history_actor.shape[-1], latent.shape[-1])
    if actual != expected:
        raise ValueError(f"Unexpected PiPlus H0W decoder input dimensions: got {actual}, expected {expected}")
    batch_size = state.shape[0]
    if any(value.shape[0] != batch_size for value in (last_action, history_actor, latent)):
        raise ValueError("PiPlus H0W decoder inputs must share a batch dimension")
    return torch.cat((state, last_action, history_actor, latent), dim=-1)


class OnnxPiPlusH0WDecoder:
    """Execute a 22DoF H0W export while keeping its PyTorch-facing API frozen."""

    action_dim = H0W_ACTION_DIM
    z_dim = H0W_LATENT_DIM

    def __init__(self, decoder_path: str | Path, device: torch.device) -> None:
        import onnx

        self.decoder_path = Path(decoder_path).expanduser().resolve()
        if not self.decoder_path.is_file():
            raise FileNotFoundError(f"ONNX decoder does not exist: {self.decoder_path}")
        model = onnx.load(str(self.decoder_path))
        if len(model.graph.input) != 1 or len(model.graph.output) != 1:
            raise ValueError("PiPlus H0W ONNX decoder must expose exactly one input and one output")
        input_dim = int(model.graph.input[0].type.tensor_type.shape.dim[-1].dim_value)
        output_dim = int(model.graph.output[0].type.tensor_type.shape.dim[-1].dim_value)
        if input_dim != H0W_ACTOR_OBSERVATION_DIM or output_dim != H0W_ACTION_DIM:
            raise ValueError(
                f"PiPlus H0W ONNX decoder must be {H0W_ACTOR_OBSERVATION_DIM}->{H0W_ACTION_DIM}, "
                f"got {input_dim}->{output_dim}"
            )
        self.input_name = model.graph.input[0].name
        self.output_name = model.graph.output[0].name
        self._runtime_session: Any | None = None
        self._reference_session: Any | None = None
        try:
            import onnxruntime as ort
        except ImportError:
            for value_info in (*model.graph.input, *model.graph.output):
                batch_dim = value_info.type.tensor_type.shape.dim[0]
                batch_dim.ClearField("dim_value")
                batch_dim.dim_param = "batch"
            from onnx.reference import ReferenceEvaluator

            self._reference_session = ReferenceEvaluator(model)
        else:
            for value_info in (*model.graph.input, *model.graph.output):
                batch_dim = value_info.type.tensor_type.shape.dim[0]
                batch_dim.ClearField("dim_value")
                batch_dim.dim_param = "batch"
            onnx.checker.check_model(model)
            providers: list[str | tuple[str, dict[str, int]]] = ["CPUExecutionProvider"]
            if device.type == "cuda" and "CUDAExecutionProvider" in ort.get_available_providers():
                providers.insert(0, ("CUDAExecutionProvider", {"device_id": int(device.index or 0)}))
            self._runtime_session = ort.InferenceSession(model.SerializeToString(), providers=providers)

    def project_z(self, latent: torch.Tensor) -> torch.Tensor:
        return functional.normalize(latent, dim=-1).mul(float(self.z_dim) ** 0.5)

    def act(self, observation: Mapping[str, torch.Tensor], latent: torch.Tensor, mean: bool = True) -> torch.Tensor:
        del mean
        actor_observation = build_actor_observation(observation, latent)
        actor_observation_np = actor_observation.detach().to(device="cpu", dtype=torch.float32).numpy()
        if self._runtime_session is not None:
            actions = self._runtime_session.run([self.output_name], {self.input_name: actor_observation_np})[0]
        else:
            assert self._reference_session is not None
            # The exported graph is batch-compatible even though its on-disk
            # annotation is fixed at one. The constructor changes only the
            # in-memory annotation, preserving the immutable source artifact.
            with np.errstate(over="ignore"):
                actions = self._reference_session.run([self.output_name], {self.input_name: actor_observation_np})[0]
        return torch.as_tensor(actions, device=latent.device, dtype=torch.float32)


def load_decoder(_bfm_model_path: Path | None, decoder_path: Path, device: torch.device) -> OnnxPiPlusH0WDecoder:
    """Load the ONNX decoder; the BFM safetensors path is intentionally unused."""
    del _bfm_model_path
    return OnnxPiPlusH0WDecoder(decoder_path, device)
