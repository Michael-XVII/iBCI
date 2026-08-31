"""Additive direct-H1 SPINT module for masked dense auxiliary supervision."""
from __future__ import annotations

import hashlib
from typing import Any

import torch
from torchmetrics import MeanMetric

from src.data.h1_window_mask_contract_v1 import masked_dense_aux_mse
from src.models.falcon_module import FalconLitModule


ALLOWED_LAMBDAS = (0.0, 0.1, 0.3, 1.0)


class H1MaskedDenseAuxLitModule(FalconLitModule):
    """Keep H1 inference/metrics last-bin-only while augmenting training loss."""

    def __init__(self, *args, dense_aux_lambda: float, **kwargs) -> None:
        value = float(dense_aux_lambda)
        if value not in ALLOWED_LAMBDAS:
            raise ValueError(f"dense_aux_lambda must be one of {ALLOWED_LAMBDAS}")
        super().__init__(*args, **kwargs)
        if self.hparams.task != "h1" or not self.hparams.decode_last_timestep_only:
            raise ValueError("masked dense auxiliary V1 is restricted to last-bin H1")
        self.dense_aux_lambda = value
        self.train_governing_loss = MeanMetric()
        self.train_dense_aux_loss = MeanMetric()
        self.last_train_r2_by_session: dict[str, float] = {}
        self.last_validation_r2_by_session: dict[str, float] = {}
        self.last_validation_r2_mean: float | None = None
        self.validation_r2_history: list[dict[str, Any]] = []
        self.last_training_terms: dict[str, float] = {}
        self.saw_nonzero_auxiliary = False
        self.all_training_terms_finite = True
        self.all_gradients_finite = True
        self.initial_net_state_sha256: str | None = None
        self.last_completed_epoch: int | None = None

    def _capture_initial_net_state(self) -> None:
        if self.initial_net_state_sha256 is not None:
            return
        digest = hashlib.sha256(b"h1-masked-dense-aux-net-state-v1\0")
        for name, value in sorted(self.net.state_dict().items()):
            tensor = value.detach().cpu().contiguous()
            digest.update(name.encode())
            digest.update(str(tensor.dtype).encode())
            digest.update(repr(tuple(tensor.shape)).encode())
            digest.update(tensor.numpy().tobytes())
        self.initial_net_state_sha256 = digest.hexdigest()

    def loss_step(self, batch, *, apply_aux: bool) -> dict[str, Any]:
        if len(batch) != 5:
            raise ValueError("masked dense H1 batch must contain exactly five fields")
        neural, behavior_target, calib, session_name, window_valid = batch
        full_pred = self.forward(
            x=neural, calib_trialized_neural_features=calib
        )
        self._capture_initial_net_state()
        if self.hparams.predict_scaled_behavior:
            full_pred = full_pred / self.hparams.behavior_scaling_factor
        last_pred = full_pred[:, -1:, :]
        last_target = behavior_target[:, -1:, :]
        governing = self.mse_loss(last_pred, last_target)
        auxiliary = masked_dense_aux_mse(full_pred, behavior_target, window_valid)
        loss = governing
        if apply_aux:
            loss = governing + self.dense_aux_lambda * auxiliary
        return {
            "loss": loss,
            "governing_loss": governing,
            "auxiliary_loss": auxiliary,
            "behavior_pred": last_pred,
            "behavior_target": last_target,
            "session_name": session_name,
        }

    @staticmethod
    def _one_session(session_names) -> str:
        unique = set(session_names)
        if len(unique) != 1:
            raise ValueError("all samples in one batch must belong to one session")
        return next(iter(unique))

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        out = self.loss_step(batch, apply_aux=True)
        terms = {
            key: float(out[key].detach().cpu())
            for key in ("loss", "governing_loss", "auxiliary_loss")
        }
        self.last_training_terms = terms
        self.saw_nonzero_auxiliary |= terms["auxiliary_loss"] > 0.0
        self.all_training_terms_finite &= all(
            torch.isfinite(out[key]).all().item()
            for key in ("loss", "governing_loss", "auxiliary_loss")
        )
        session = self._one_session(out["session_name"])
        self.train_loss(out["loss"])
        self.train_governing_loss(out["governing_loss"])
        self.train_dense_aux_loss(out["auxiliary_loss"])
        self.train_r2[session].update(
            out["behavior_pred"].flatten(0, 1), out["behavior_target"].flatten(0, 1)
        )
        self.log("train/loss", self.train_loss, on_epoch=True, prog_bar=True)
        self.log("train/last_bin_mse", self.train_governing_loss, on_epoch=True)
        self.log("train/masked_dense_aux_mse", self.train_dense_aux_loss, on_epoch=True)
        return out["loss"]

    def on_after_backward(self) -> None:
        for parameter in self.parameters():
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all().item():
                self.all_gradients_finite = False

    def on_train_epoch_end(self) -> None:
        self.last_completed_epoch = int(self.current_epoch)
        values: dict[str, float] = {}
        tensors = []
        for session, metric in self.train_r2.items():
            if metric.total > 2:
                value = metric.compute()
                tensors.append(value)
                values[session] = float(value.detach().cpu())
                self.log(f"train_{session}/r2", value)
            metric.reset()
        if tensors:
            self.log("train/r2_mean", torch.stack(tensors).mean(), prog_bar=True)
        self.last_train_r2_by_session = values

    def validation_step(self, batch, batch_idx: int, dataloader_idx: int = 0) -> None:
        if dataloader_idx != 0:
            raise RuntimeError("formal held-out validation is forbidden")
        out = self.loss_step(batch, apply_aux=False)
        session = self._one_session(out["session_name"])
        self.val_heldin_loss(out["governing_loss"])
        self.val_heldin_r2[session].update(
            out["behavior_pred"].flatten(0, 1), out["behavior_target"].flatten(0, 1)
        )
        self.log("val_heldin/loss", self.val_heldin_loss, on_epoch=True)

    def on_validation_epoch_end(self) -> None:
        values: dict[str, float] = {}
        tensors = []
        for session, metric in self.val_heldin_r2.items():
            if metric.total > 2:
                value = metric.compute()
                tensors.append(value)
                values[session] = float(value.detach().cpu())
                self.log(f"val_heldin_{session}/r2", value)
            metric.reset()
        if not tensors:
            return
        mean = torch.stack(tensors).mean()
        self.log("val_heldin/r2_mean", mean, prog_bar=True)
        self.val_heldin_r2_mean_best(mean)
        self.log("val_heldin/r2_mean_best", self.val_heldin_r2_mean_best.compute())
        self.last_validation_r2_by_session = values
        self.last_validation_r2_mean = float(mean.detach().cpu())
        self.validation_r2_history.append({
            "epoch_zero_based": int(self.current_epoch),
            "r2_by_recording": dict(values),
            "equal_recording_mean_r2": self.last_validation_r2_mean,
            "metric": "last_bin_variance_weighted_r2",
        })
