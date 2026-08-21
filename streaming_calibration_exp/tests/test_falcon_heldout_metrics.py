from __future__ import annotations

from pathlib import Path

import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.models.falcon_module import FalconLitModule


class _ResetOnlyMetric(nn.Module):
    def __init__(self, total: int) -> None:
        super().__init__()
        self.register_buffer("total", torch.tensor(total))
        self.reset_calls = 0

    def compute(self) -> torch.Tensor:
        raise AssertionError("metrics with <= 2 samples must not be computed")

    def reset(self) -> None:
        self.reset_calls += 1


class _ToyNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(2))

    def forward(
        self,
        x: torch.Tensor,
        calib_trialized_neural_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del calib_trialized_neural_features
        return self.bias.view(1, 1, 2).expand(x.shape[0], x.shape[1], 2)


def _optimizer(params):
    return torch.optim.SGD(params, lr=0.01)


def _module() -> FalconLitModule:
    return FalconLitModule(
        task="m1",
        net=_ToyNet(),
        decode_last_timestep_only=False,
        predict_scaled_behavior=False,
        behavior_scaling_factor=1.0,
        optimizer=_optimizer,
        scheduler=None,
        compile=False,
    )


def test_on_train_start_resets_actual_metric_maps_not_static_session_names():
    module = _module()
    heldin = _ResetOnlyMetric(total=0)
    heldout = _ResetOnlyMetric(total=0)
    module.val_heldin_r2 = nn.ModuleDict({"dynamic-heldin": heldin})
    module.val_heldout_r2 = nn.ModuleDict({"dynamic-heldout": heldout})

    module.on_train_start()

    assert heldin.reset_calls == 1
    assert heldout.reset_calls == 1


def test_empty_session_metrics_do_not_compute_or_emit_aggregate_r2():
    module = _module()
    metrics = [_ResetOnlyMetric(total=2) for _ in range(5)]
    module.train_r2 = nn.ModuleDict({"dynamic-train": metrics[0]})
    module.val_heldin_r2 = nn.ModuleDict({"heldin": metrics[1]})
    module.val_heldout_r2 = nn.ModuleDict({"heldout": metrics[2]})
    module.test_heldin_r2 = nn.ModuleDict({"heldin": metrics[3]})
    module.test_heldout_r2 = nn.ModuleDict({"heldout": metrics[4]})
    logged: list[str] = []
    module.log = lambda name, *args, **kwargs: logged.append(name)

    module.on_train_epoch_end()
    module.on_validation_epoch_end()
    module.on_test_epoch_end()

    assert all(metric.reset_calls == 1 for metric in metrics)
    assert not any(name.endswith("/r2_mean") for name in logged)


class _SessionDataset(Dataset):
    def __init__(self, session_name: str) -> None:
        self.session_name = session_name
        self.neural = torch.zeros(4, 1, 3)
        self.behavior = torch.arange(8, dtype=torch.float32).reshape(4, 1, 2)
        self.calibration = torch.zeros(4, 1, 1)

    def __len__(self) -> int:
        return len(self.neural)

    def __getitem__(self, index: int):
        return (
            self.neural[index],
            self.behavior[index],
            self.calibration[index],
            self.session_name,
        )


def test_heldout_monitor_is_available_to_checkpoint_and_early_stopping(tmp_path: Path):
    module = _module()
    checkpoint = ModelCheckpoint(
        dirpath=tmp_path,
        monitor="val_heldout/r2_mean",
        mode="max",
        save_top_k=1,
    )
    early_stopping = EarlyStopping(monitor="val_heldout/r2_mean", mode="max", patience=1)
    heldin_loader = DataLoader(_SessionDataset("ses-20120924"), batch_size=4)
    heldout_loader = DataLoader(_SessionDataset("ses-20121004"), batch_size=4)
    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=1,
        callbacks=[checkpoint, early_stopping],
        logger=False,
        enable_model_summary=False,
        enable_progress_bar=False,
        num_sanity_val_steps=0,
        limit_train_batches=1,
        limit_val_batches=1,
    )

    trainer.fit(module, train_dataloaders=heldin_loader, val_dataloaders=[heldin_loader, heldout_loader])

    value = trainer.callback_metrics["val_heldout/r2_mean"]
    assert torch.isfinite(value)
    assert checkpoint.best_model_score is not None
    assert torch.isfinite(checkpoint.best_model_score)
