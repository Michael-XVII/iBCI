"""Carrier-aware FALCON streaming decoder for the H1 M3 deployment successor.

This is additive and intentionally does not modify the generic ``SpintDecoder``.
Calibration payloads are fitted and frozen by the successor packager; reset
selects the payload matching the FALCON session key and predict always supplies
both identity and carrier to ``H1CarrierIdSpint``.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import torch

from falcon_challenge.config import FalconConfig, FalconTask
from falcon_challenge.interface import BCIDecoder

from src.h1_m4_cce_contract import state_hash
from src.models.components.h1_carrierid_spint import H1CarrierIdSpint


PACKAGE_SCHEMA = "h1_cal_aug_all_source_m3_deployment_v1_package"


class H1CarrierIdSpintDecoder(BCIDecoder):
    """Frozen M3 identity+carrier decoder using a causal W700 buffer."""

    def __init__(
        self,
        task_config: FalconConfig,
        package_path: str | Path,
        batch_size: int = 1,
        *,
        device: str | None = None,
    ) -> None:
        if task_config.task != FalconTask.h1:
            raise ValueError("H1CarrierIdSpintDecoder accepts FalconTask.h1 only")
        if int(batch_size) <= 0:
            raise ValueError("decoder batch size must be positive")
        super().__init__(task_config=task_config, batch_size=int(batch_size))
        self._task_config = task_config
        self.batch_size = int(batch_size)
        self.package_path = Path(package_path).resolve()
        payload = torch.load(self.package_path, map_location="cpu", weights_only=False)
        if payload.get("schema") != PACKAGE_SCHEMA or payload.get("task") != "h1":
            raise ValueError("H1 carrier package schema/task drift")
        if int(payload.get("window_size", -1)) != 700 or float(payload.get("prediction_divisor", -1)) != 20.0:
            raise ValueError("H1 carrier package W/scaling drift")
        sessions = payload.get("sessions")
        if not isinstance(sessions, dict) or len(sessions) != 27:
            raise ValueError("H1 carrier package requires all 27 calibration session keys")
        self.sessions = sessions
        self.window_size = 700
        self.prediction_divisor = 20.0
        self.checkpoint_sha256 = str(payload["checkpoint_sha256"])
        self.arm = str(payload["arm"])
        self.source_authority_sha256 = str(payload["source_authority_sha256"])
        self.model = H1CarrierIdSpint(**payload["model_kwargs"])
        self.model.load_state_dict(payload["state_dict"], strict=True)
        self.model.eval()
        self.expected_state_sha256 = str(payload["model_state_sha256"])
        if state_hash(self.model.state_dict()) != self.expected_state_sha256:
            raise ValueError("packaged H1 model state hash drift")
        selected = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        if str(selected).startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA decoder reload requested but CUDA is unavailable")
        self.device = torch.device(selected)
        self.model.to(self.device)
        self.observation_buffer = np.zeros(
            (self.window_size, self.batch_size, task_config.n_channels), dtype=np.float32
        )
        self.history_count = np.zeros(self.batch_size, dtype=np.int64)
        self.local_identity: torch.Tensor | None = None
        self.local_carrier: torch.Tensor | None = None
        self.local_keys: tuple[str, ...] = ()

    def set_batch_size(self, batch_size: int) -> None:
        if int(batch_size) <= 0:
            raise ValueError("decoder batch size must be positive")
        self.batch_size = int(batch_size)
        self.observation_buffer = np.zeros(
            (self.window_size, self.batch_size, self._task_config.n_channels), dtype=np.float32
        )
        self.history_count = np.zeros(self.batch_size, dtype=np.int64)
        self.local_identity = None
        self.local_carrier = None
        self.local_keys = ()

    def reset(self, dataset_tags: List[Path] = [Path("")]) -> None:
        keys = [self._task_config.hash_dataset(Path(tag).stem) for tag in dataset_tags]
        if not keys or len(keys) > self.batch_size:
            raise ValueError("reset requires one to batch_size dataset tags")
        for key in keys:
            if key not in self.sessions:
                raise ValueError(f"calibration payload missing for {key}")
        padded = keys + [keys[0]] * (self.batch_size - len(keys))
        identities = []
        carriers = []
        for key in padded:
            row = self.sessions[key]
            identity = np.asarray(row["identity"], dtype=np.float32)
            carrier = np.asarray(row["carrier"], dtype=np.float32)
            if identity.shape != (3, 1024, 176) or carrier.shape != (176, 4):
                raise ValueError(f"M3 identity/carrier shape drift for {key}")
            if not np.isfinite(identity).all() or not np.isfinite(carrier).all():
                raise ValueError(f"nonfinite M3 payload for {key}")
            identities.append(identity)
            carriers.append(carrier)
        self.local_identity = torch.as_tensor(np.stack(identities), dtype=torch.float32, device=self.device)
        self.local_carrier = torch.as_tensor(np.stack(carriers), dtype=torch.float32, device=self.device)
        self.local_keys = tuple(keys)
        self.observation_buffer.fill(0.0)
        self.history_count.fill(0)
        self.model.eval()

    def on_done(self, dones: np.ndarray) -> None:
        # H1 is continual; neither trials nor eval-mask gaps reset calibration or history.
        return None

    def observe(self, neural_observations: np.ndarray) -> None:
        values = np.asarray(neural_observations, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != self._task_config.n_channels:
            raise ValueError("H1 observations must be [batch,176]")
        active = values.shape[0]
        if active <= 0 or active > self.batch_size:
            raise ValueError("H1 observation batch drift")
        if active < self.batch_size:
            values = np.pad(values, ((0, self.batch_size - active), (0, 0)))
        self.observation_buffer = np.roll(self.observation_buffer, -1, axis=0)
        self.observation_buffer[-1] = values
        self.history_count[:active] += 1

    def predict(self, neural_observations: np.ndarray) -> np.ndarray:
        if self.local_identity is None or self.local_carrier is None:
            raise RuntimeError("decoder.reset must precede predict")
        active = int(np.asarray(neural_observations).shape[0])
        self.observe(neural_observations)
        neural = torch.as_tensor(
            self.observation_buffer.transpose(1, 0, 2), dtype=torch.float32, device=self.device
        )
        with torch.no_grad():
            output = self.model(
                neural,
                calib_trialized_neural_features=self.local_identity,
                carrier=self.local_carrier,
            )
        prediction = output[:, -1, :].detach().cpu().numpy().astype(np.float32) / np.float32(
            self.prediction_divisor
        )
        if not np.isfinite(prediction).all():
            raise RuntimeError("nonfinite H1 deployment prediction")
        return prediction[:active]

    def model_state_sha256(self) -> str:
        return state_hash(self.model.state_dict())


__all__ = ("H1CarrierIdSpintDecoder", "PACKAGE_SCHEMA")
