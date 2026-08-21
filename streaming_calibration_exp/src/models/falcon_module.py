"""FALCON benchmark LightningModule - part of "SPINT: Spatial Permutation-Invariant Neural Transformer for Consistent Intracortical Motor Decoding".
Scaffolding adapted from the Hydra template (ashleve/lightning-hydra-template).
Copyright (c) 2024-2026 University of Washington. Developed in UW NeuroAI Lab by Trung Le.
"""
from typing import Any, Dict, Tuple

import torch
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.regression import R2Score
import torch.nn as nn
import lightning.pytorch as pl


DATASET_NAMES = {
    'm1': {
        'heldin': ['ses-20120924', 'ses-20120926', 'ses-20120927', 'ses-20120928'],
        'heldout': ['ses-20121004', 'ses-20121017', 'ses-20121024'],
    },
    'm2': {
        'heldin': ['ses-2020-10-19-Run1','ses-2020-10-19-Run2','ses-2020-10-20-Run1','ses-2020-10-20-Run2',
                   'ses-2020-10-27-Run1', 'ses-2020-10-27-Run2', 'ses-2020-10-28-Run1'],
        'heldout': ['ses-2020-10-30-Run1','ses-2020-10-30-Run2', 'ses-2020-11-18-Run1','ses-2020-11-19-Run1',
                    'ses-2020-11-24-Run1', 'ses-2020-11-24-Run2'],
    },
    'h1': {
        'heldin': ['ses-19250101T111740','ses-19250101T112404','ses-19250108T110520','ses-19250108T111022',
                   'ses-19250108T111455', 'ses-19250113T120811', 'ses-19250113T121303', 'ses-19250115T110633',
                   'ses-19250115T111328', 'ses-19250119T113543', 'ses-19250119T114045', 'ses-19250120T115044',
                   'ses-19250120T115537'],
        'heldout': ['ses-19250126T113454','ses-19250126T114029', 'ses-19250127T120333','ses-19250127T120826',
                    'ses-19250129T112555', 'ses-19250129T113059', 'ses-19250202T113958', 'ses-19250202T114452',
                    'ses-19250203T113515', 'ses-19250203T114018', 'ses-19250206T112219', 'ses-19250206T112712',
                    'ses-19250209T111826', 'ses-19250209T112327'],
    },
    'mc_maze': {
        'heldin': ['mc_maze_ses-full'],
        'heldout': ['mc_maze_ses-full'],
    }
}
DATASET_NUM_NEURONS = {
    'm1': 64,
    'm2': 96,
    'h1': 176,
}


class FalconLitModule(pl.LightningModule):
    """LightningModule for training neural decoders on the FALCON benchmark.

    Wraps a SpintModel with training/validation/test loops, tracking per-session
    R2 scores separately for held-in and held-out session splits.
    """

    def __init__(
        self,
        task: str,
        net: torch.nn.Module,
        decode_last_timestep_only: bool,
        predict_scaled_behavior: bool,
        behavior_scaling_factor: float,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
    ) -> None:
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.net = net

        # loss function
        self.mse_loss = torch.nn.MSELoss()

        # metric objects for calculating and averaging R2 across batches
        self.train_r2 = nn.ModuleDict({k: R2Score(multioutput="variance_weighted") for k in DATASET_NAMES[self.hparams.task]['heldin']})
        self.val_heldin_r2 = nn.ModuleDict({k: R2Score(multioutput="variance_weighted") for k in DATASET_NAMES[self.hparams.task]['heldin']})
        self.val_heldout_r2 = nn.ModuleDict({k: R2Score(multioutput="variance_weighted") for k in DATASET_NAMES[self.hparams.task]['heldout']})
        self.test_heldin_r2 = nn.ModuleDict({k: R2Score(multioutput="variance_weighted") for k in DATASET_NAMES[self.hparams.task]['heldin']})
        self.test_heldout_r2 = nn.ModuleDict({k: R2Score(multioutput="variance_weighted") for k in DATASET_NAMES[self.hparams.task]['heldout']})

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_heldin_loss = MeanMetric()
        self.val_heldout_loss = MeanMetric()
        self.test_heldin_loss = MeanMetric()
        self.test_heldout_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_heldin_r2_mean_best = MaxMetric()
        self.val_heldout_r2_mean_best = MaxMetric()

    def forward(
        self,
        x: torch.Tensor,
        calib_trialized_neural_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.net(x, calib_trialized_neural_features=calib_trialized_neural_features)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_heldin_loss.reset()
        self.val_heldout_loss.reset()
        self.val_heldin_r2_mean_best.reset()
        self.val_heldout_r2_mean_best.reset()
        for metric in self.val_heldin_r2.values():
            metric.reset()
        for metric in self.val_heldout_r2.values():
            metric.reset()

    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        neural, behavior_target, calib_trialized_neural_features, session_name = batch
        behavior_pred = self.forward(x=neural, calib_trialized_neural_features=calib_trialized_neural_features)
        # in continuous decoding, only consider the last time step
        if self.hparams.decode_last_timestep_only:
            behavior_pred = behavior_pred[:, -1:, :]
            behavior_target = behavior_target[:, -1:, :]
        if self.hparams.predict_scaled_behavior:
            behavior_pred = behavior_pred / self.hparams.behavior_scaling_factor
        loss = self.mse_loss(behavior_pred, behavior_target)
        return loss, behavior_pred, behavior_target, session_name

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        loss, behavior_pred, behavior_target, session_name = self.model_step(batch)
        if len(set(session_name)) == 1:
            session_name = session_name[0] # we already made sure by using batch sampler that all samples in the batch belong to the same session
        else:
            raise ValueError("All samples in the batch should belong to the same session") # just in case we missed something

        # update and log metrics
        self.train_loss(loss)
        self.train_r2[session_name].update(behavior_pred.flatten(start_dim=0, end_dim=1), behavior_target.flatten(start_dim=0, end_dim=1))
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        # return loss or backpropagation will fail
        return loss

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        train_r2s = []
        for sess_name, sess_r2 in self.train_r2.items():
            if sess_r2.total > 2:
                r2 = sess_r2.compute()
                train_r2s.append(r2)
                self.log(f"train_{sess_name}/r2", r2, sync_dist=True, add_dataloader_idx=False)
            sess_r2.reset()

        if train_r2s:
            train_r2_mean = torch.stack(train_r2s).mean()
            train_r2_std = torch.stack(train_r2s).std(unbiased=False)
            self.log("train/r2_mean", train_r2_mean, sync_dist=True, prog_bar=True)
            self.log("train/r2_std", train_r2_std, sync_dist=True, prog_bar=True)

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int, dataloader_idx: int=0) -> None:
        loss, behavior_pred, behavior_target, session_name = self.model_step(batch)
        if len(set(session_name)) == 1:
            session_name = session_name[0] # we already made sure by using batch sampler that all samples in the batch belong to the same session
        else:
            raise ValueError("All samples in the batch should belong to the same session") # just in case we missed something
        
        # update and log metrics
        if dataloader_idx == 0:
            self.val_heldin_loss(loss)
            self.val_heldin_r2[session_name].update(behavior_pred.flatten(start_dim=0, end_dim=1), behavior_target.flatten(start_dim=0, end_dim=1))
            self.log("val_heldin/loss", self.val_heldin_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, add_dataloader_idx=False)
        elif dataloader_idx == 1:
            self.val_heldout_loss(loss)
            self.val_heldout_r2[session_name].update(behavior_pred.flatten(start_dim=0, end_dim=1), behavior_target.flatten(start_dim=0, end_dim=1))
            self.log("val_heldout/loss", self.val_heldout_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, add_dataloader_idx=False)

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        val_heldin_r2s, val_heldout_r2s = [], []
        for sess_name, sess_r2 in self.val_heldin_r2.items():
            if sess_r2.total > 2:
                r2 = sess_r2.compute()
                val_heldin_r2s.append(r2)
                self.log(f"val_heldin_{sess_name}/r2", r2, sync_dist=True, add_dataloader_idx=False)
            sess_r2.reset()
        for sess_name, sess_r2 in self.val_heldout_r2.items():
            if sess_r2.total > 2:
                r2 = sess_r2.compute()
                val_heldout_r2s.append(r2)
                self.log(f"val_heldout_{sess_name}/r2", r2, sync_dist=True, add_dataloader_idx=False)
            sess_r2.reset()

        if val_heldin_r2s:
            val_heldin_r2_mean = torch.stack(val_heldin_r2s).mean()
            val_heldin_r2_std = torch.stack(val_heldin_r2s).std(unbiased=False)
            self.log("val_heldin/r2_mean", val_heldin_r2_mean, sync_dist=True, prog_bar=True)
            self.log("val_heldin/r2_std", val_heldin_r2_std, sync_dist=True, prog_bar=True)
            self.val_heldin_r2_mean_best(val_heldin_r2_mean)  # update best so far val r2
            self.log("val_heldin/r2_mean_best", self.val_heldin_r2_mean_best.compute(), sync_dist=True, prog_bar=True)
        if val_heldout_r2s:
            val_heldout_r2_mean = torch.stack(val_heldout_r2s).mean()
            val_heldout_r2_std = torch.stack(val_heldout_r2s).std(unbiased=False)
            self.log("val_heldout/r2_mean", val_heldout_r2_mean, sync_dist=True, prog_bar=True)
            self.log("val_heldout/r2_std", val_heldout_r2_std, sync_dist=True, prog_bar=True)
            self.val_heldout_r2_mean_best(val_heldout_r2_mean)  # update best so far val r2
            self.log("val_heldout/r2_mean_best", self.val_heldout_r2_mean_best.compute(), sync_dist=True, prog_bar=True)

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int, dataloader_idx: int=0) -> None:
        loss, behavior_pred, behavior_target, session_name = self.model_step(batch)
        if len(set(session_name)) == 1:
            session_name = session_name[0] # we already made sure by using batch sampler that all samples in the batch belong to the same session
        else:
            raise ValueError("All samples in the batch should belong to the same session") # just in case we missed something
        
        # update and log metrics
        if dataloader_idx == 0:
            self.test_heldin_loss(loss)
            self.test_heldin_r2[session_name].update(behavior_pred.flatten(start_dim=0, end_dim=1), behavior_target.flatten(start_dim=0, end_dim=1))
            self.log("test_heldin/loss", self.test_heldin_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, add_dataloader_idx=False)
        elif dataloader_idx == 1:
            self.test_heldout_loss(loss)
            self.test_heldout_r2[session_name].update(behavior_pred.flatten(start_dim=0, end_dim=1), behavior_target.flatten(start_dim=0, end_dim=1))
            self.log("test_heldout/loss", self.test_heldout_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, add_dataloader_idx=False)

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        test_heldin_r2s, test_heldout_r2s = [], []
        for sess_name, sess_r2 in self.test_heldin_r2.items():
            if sess_r2.total > 2:
                r2 = sess_r2.compute()
                test_heldin_r2s.append(r2)
                self.log(f"test_heldin_{sess_name}/r2", r2, sync_dist=True, add_dataloader_idx=False)
            sess_r2.reset()
        for sess_name, sess_r2 in self.test_heldout_r2.items():
            if sess_r2.total > 2:
                r2 = sess_r2.compute()
                test_heldout_r2s.append(r2)
                self.log(f"test_heldout_{sess_name}/r2", r2, sync_dist=True, add_dataloader_idx=False)
            sess_r2.reset()

        if test_heldin_r2s:
            test_heldin_r2_mean = torch.stack(test_heldin_r2s).mean()
            test_heldin_r2_std = torch.stack(test_heldin_r2s).std(unbiased=False)
            self.log("test_heldin/r2_mean", test_heldin_r2_mean, sync_dist=True, prog_bar=True)
            self.log("test_heldin/r2_std", test_heldin_r2_std, sync_dist=True, prog_bar=True)
        if test_heldout_r2s:
            test_heldout_r2_mean = torch.stack(test_heldout_r2s).mean()
            test_heldout_r2_std = torch.stack(test_heldout_r2s).std(unbiased=False)
            self.log("test_heldout/r2_mean", test_heldout_r2_mean, sync_dist=True, prog_bar=True)
            self.log("test_heldout/r2_std", test_heldout_r2_std, sync_dist=True, prog_bar=True)

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_heldout/r2_mean",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
