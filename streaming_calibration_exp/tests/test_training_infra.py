"""Gradient and callback instantiation tests."""
from __future__ import annotations

from omegaconf import OmegaConf

import pytest
import torch

from src.models.components.spint import SpintModel
from src.models.components.streaming_encoders import EarlyPoolEncoder
from src.models.components.streaming_spint import StreamingSpintModel
from src.models.falcon_module import FalconLitModule
from src.models.streaming_calibration_module import StreamingCalibrationLitModule
from src.utils.instantiators import _collect_target_nodes, instantiate_callbacks


def _init_decoder(decoder: SpintModel) -> None:
    calib = torch.randn(1, 2, 100, 96)
    neural = torch.randn(1, 50, 96)
    decoder.eval()
    with torch.no_grad():
        decoder(neural, calib)


def test_task_only_backward_updates_encoder_not_decoder():
    decoder = SpintModel(
        model_dim=64,
        num_covariates=2,
        window_size=50,
        num_heads=4,
        num_layers=1,
        num_id_layers=3,
    )
    _init_decoder(decoder)
    encoder = EarlyPoolEncoder(100, 50, 32)
    model = StreamingSpintModel(decoder=decoder, id_encoder=encoder)
    model.freeze_decoder()
    model.train()

    neural = torch.randn(1, 50, 96)
    calib = torch.randn(1, 2, 100, 96)
    behavior, _ = model(neural, calib)
    loss = behavior.pow(2).mean()
    loss.backward()

    encoder_grads = [p.grad for p in encoder.parameters() if p.grad is not None]
    assert encoder_grads, "encoder should receive task gradients through frozen decoder"
    assert any(g.abs().sum() > 0 for g in encoder_grads)
    assert all(p.grad is None for p in decoder.parameters())


def test_collect_target_nodes_ignores_polluted_root():
    cfg = OmegaConf.create(
        {
            "_target_": "lightning.pytorch.callbacks.RichModelSummary",
            "best_checkpoint": {"_target_": "lightning.pytorch.callbacks.ModelCheckpoint"},
            "early_stopping": {"_target_": "lightning.pytorch.callbacks.EarlyStopping"},
        }
    )
    nodes = []
    _collect_target_nodes(cfg, nodes)
    targets = {node._target_ for node in nodes}
    assert "lightning.pytorch.callbacks.ModelCheckpoint" in targets
    assert "lightning.pytorch.callbacks.EarlyStopping" in targets
    assert "lightning.pytorch.callbacks.RichModelSummary" not in targets


def test_train_callbacks_compose_to_six_targets():
    from hydra import compose, initialize_config_dir
    from pathlib import Path

    cfg_dir = str(Path(__file__).resolve().parents[1] / "configs")
    with initialize_config_dir(version_base="1.3", config_dir=cfg_dir):
        cfg = compose(config_name="train.yaml", overrides=["experiment=b2_d128"])
    nodes = []
    _collect_target_nodes(cfg.callbacks, nodes)
    assert len(nodes) == 6
    targets = {node._target_ for node in nodes}
    assert "lightning.pytorch.callbacks.ModelCheckpoint" in targets
    assert "lightning.pytorch.callbacks.EarlyStopping" in targets
    assert "lightning.pytorch.callbacks.LearningRateMonitor" in targets
    assert "src.callbacks.override_epoch_step.OverrideEpochStepCallback" in targets


def test_drop_early_stopping_is_noop_when_disabled():
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, RichModelSummary

    from src.train import _drop_early_stopping

    callbacks = [ModelCheckpoint(), EarlyStopping(monitor="x"), RichModelSummary()]
    result = _drop_early_stopping(callbacks, enabled=False)
    assert result is callbacks
    assert len(result) == 3


def test_drop_early_stopping_removes_early_stopping_when_enabled():
    # M2 fixed-epoch-budget mode (sua_exploration/docs/CURRENT_RESULTS.md section H.2):
    # --no_early_stopping / no_early_stopping=true must drop EarlyStopping so every
    # variant trains exactly trainer.max_epochs, without touching other callbacks.
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, RichModelSummary

    from src.train import _drop_early_stopping

    checkpoint_cb = ModelCheckpoint()
    summary_cb = RichModelSummary()
    callbacks = [checkpoint_cb, EarlyStopping(monitor="x"), summary_cb]
    result = _drop_early_stopping(callbacks, enabled=True)
    assert result == [checkpoint_cb, summary_cb]
    assert not any(isinstance(cb, EarlyStopping) for cb in result)


def test_no_early_stopping_defaults_false_in_train_yaml():
    from hydra import compose, initialize_config_dir
    from pathlib import Path

    cfg_dir = str(Path(__file__).resolve().parents[1] / "configs")
    with initialize_config_dir(version_base="1.3", config_dir=cfg_dir):
        cfg = compose(config_name="train.yaml", overrides=["experiment=b2_d128"])
    assert cfg.get("no_early_stopping") is False


def test_support_prediction_consistency_weight_is_nonnegative():
    with pytest.raises(ValueError, match="must be >= 0"):
        StreamingCalibrationLitModule(
            task="m2",
            variant="B16ZF",
            teacher_ckpt_path="unused.ckpt",
            window_size=50,
            optimizer=None,
            support_prediction_consistency_weight=-0.1,
        )


def test_falcon_module_validation_allows_source_only_loader():
    module = FalconLitModule(
        task="m1",
        net=torch.nn.Identity(),
        decode_last_timestep_only=True,
        predict_scaled_behavior=False,
        behavior_scaling_factor=1.0,
        optimizer=None,
        scheduler=None,
        compile=False,
    )
    module.log = lambda *args, **kwargs: None
    pred = torch.randn(4, 1, 16)
    target = torch.randn(4, 1, 16)
    module.val_heldin_r2["ses-20120924"].update(pred.flatten(0, 1), target.flatten(0, 1))

    module.on_validation_epoch_end()


def test_falcon_module_test_allows_source_only_loader():
    module = FalconLitModule(
        task="m1",
        net=torch.nn.Identity(),
        decode_last_timestep_only=True,
        predict_scaled_behavior=False,
        behavior_scaling_factor=1.0,
        optimizer=None,
        scheduler=None,
        compile=False,
    )
    module.log = lambda *args, **kwargs: None
    pred = torch.randn(4, 1, 16)
    target = torch.randn(4, 1, 16)
    module.test_heldin_r2["ses-20120924"].update(pred.flatten(0, 1), target.flatten(0, 1))

    module.on_test_epoch_end()


def test_training_step_adds_two_support_prediction_consistency():
    class _SupportSensitiveStudent(torch.nn.Module):
        def forward(self, neural, calib_trials):
            value = calib_trials.mean(dim=(1, 2, 3)).view(-1, 1, 1)
            prediction = value.expand(-1, neural.shape[1], 2)
            identity = torch.zeros(
                neural.shape[0], neural.shape[-1], neural.shape[1]
            )
            return prediction, identity

    module = StreamingCalibrationLitModule(
        task="m2",
        variant="B16ZF",
        teacher_ckpt_path="unused.ckpt",
        window_size=50,
        optimizer=None,
        support_prediction_consistency_weight=1.0,
    )
    module.student = _SupportSensitiveStudent()
    module.log = lambda *args, **kwargs: None
    primary_loss = torch.tensor(1.0, requires_grad=True)
    module.model_step = lambda batch: {
        "loss": primary_loss,
        "behavior_pred": torch.zeros(2, 1, 2),
        "session_name": ["same-session", "same-session"],
    }
    neural = torch.zeros(2, 50, 3)
    behavior = torch.zeros(2, 50, 2)
    calib = torch.stack([torch.zeros(2, 4, 3), torch.ones(2, 4, 3)])

    loss = module.training_step(
        (neural, behavior, calib, ["same-session", "same-session"]), 0
    )

    assert loss > primary_loss
