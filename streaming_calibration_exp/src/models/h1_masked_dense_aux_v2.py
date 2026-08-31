"""V2 Lightning route with masked training and legacy four-field validation."""
from __future__ import annotations

import torch

from src.models.h1_masked_dense_aux_v1 import H1MaskedDenseAuxLitModule


class H1MaskedDenseAuxV2LitModule(H1MaskedDenseAuxLitModule):
    def legacy_validation_outputs(self, batch):
        if len(batch) != 4:
            raise ValueError("V2 validation must preserve the established four-field batch")
        neural, behavior_target, calibration, session_name = batch
        full_pred = self.forward(x=neural, calib_trialized_neural_features=calibration)
        if self.hparams.predict_scaled_behavior:
            full_pred = full_pred / self.hparams.behavior_scaling_factor
        prediction = full_pred[:, -1:, :]
        target = behavior_target[:, -1:, :]
        return self.mse_loss(prediction, target), prediction, target, session_name

    def validation_step(self, batch, batch_idx: int, dataloader_idx: int = 0) -> None:
        if dataloader_idx != 0:
            raise RuntimeError("formal held-out validation is forbidden")
        loss, prediction, target, sessions = self.legacy_validation_outputs(batch)
        session = self._one_session(sessions)
        self.val_heldin_loss(loss)
        self.val_heldin_r2[session].update(prediction.flatten(0, 1), target.flatten(0, 1))
        self.log("val_heldin/loss", self.val_heldin_loss, on_epoch=True)
