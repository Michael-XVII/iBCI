from __future__ import annotations

from types import SimpleNamespace

import torch

from src.models.streaming_calibration_module import StreamingCalibrationLitModule


class _Metric:
    def __init__(self, value: float):
        self.total = torch.tensor(3)
        self.value = torch.tensor(value)
        self.reset_called = False

    def compute(self):
        return self.value

    def reset(self):
        self.reset_called = True


class _Best:
    def __call__(self, value):
        self.value = value

    def compute(self):
        return self.value


def test_validation_epoch_logs_heldout_mean_for_heldout_selection():
    heldin, heldout = _Metric(.25), _Metric(.75)
    calls = []
    module = SimpleNamespace(
        val_heldin_r2={"in": heldin},
        val_heldout_r2={"out": heldout},
        val_heldin_r2_mean_best=_Best(),
        log=lambda name, value, **kwargs: calls.append((name, value)),
    )
    StreamingCalibrationLitModule.on_validation_epoch_end(module)
    values = {name: float(value) for name, value in calls}
    assert values["val_heldout/r2_mean"] == .75
    assert heldin.reset_called and heldout.reset_called
