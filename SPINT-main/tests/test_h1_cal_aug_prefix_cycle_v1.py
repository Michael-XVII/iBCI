from __future__ import annotations

import json
from pathlib import Path
import random
import stat

import numpy as np
import pytest

from src.h1_cal_aug_prefix_cycle_v1 import (
    BUDGETS,
    C1_CYCLE,
    EXPERIMENT3_COMMIT,
    MAX_PREFIX,
    MODEL_PARAMETERS,
    SCHEMA,
    STATUS_NO_TRANSFER,
    STATUS_PASS,
    _m7_schedule,
    _prefix_cycle,
    common_config,
    create_attempt,
    dry_plan,
    transfer_decision,
)
from src.h1_hc_date_lodo_regen_v1 import verify_sidecar
from src.h1_m4_cce_contract import CONFIRMATORY_DATES


def test_dry_plan_is_zero_access_and_complete() -> None:
    plan = dry_plan()
    assert plan["status"] == "DRY_NO_WRITE_NO_DATA_NO_CUDA"
    assert tuple(plan["outer_dates"]) == CONFIRMATORY_DATES
    assert plan["models"] == 10 and plan["target_access"] == 0
    assert tuple(plan["c1_cycle"]) == C1_CYCLE == (7, 5, 4)
    assert BUDGETS == (4, 5, 7) and MAX_PREFIX == 7


def test_prefix_cycle_balanced_and_does_not_consume_rng() -> None:
    random.seed(123)
    before = random.getstate()
    schedule = _prefix_cycle("19250108", 50, 101)
    after = random.getstate()
    assert before == after
    assert schedule.shape == (50, 101)
    assert set(np.unique(schedule)) == {4, 5, 7}
    for row in schedule:
        counts = [int(np.sum(row == value)) for value in C1_CYCLE]
        assert max(counts) - min(counts) <= 1
    assert np.array_equal(schedule, _prefix_cycle("19250108", 50, 101))


class _Record:
    def __init__(self, count: int):
        self.trial_values = tuple(range(count))


class _Dataset:
    def __init__(self, counts: dict[str, int]):
        self.records = {name: _Record(count) for name, count in counts.items()}
        self.windows = []
        for name in counts:
            self.windows.extend([(name, index) for index in range(64)])


def test_m7_schedule_uses_only_legal_starts_and_is_repeatable() -> None:
    dataset = _Dataset({"ses-a": 8, "ses-b": 15})
    order = np.concatenate((np.arange(64), np.arange(64, 128))).astype(np.int64)
    schedule = _m7_schedule(dataset, order, "19250108")
    assert schedule.shape == (50, 128)
    assert schedule[:, :64].min() >= 0 and schedule[:, :64].max() < 2
    assert schedule[:, 64:].min() >= 0 and schedule[:, 64:].max() < 9
    assert np.array_equal(schedule, _m7_schedule(dataset, order, "19250108"))


def test_m7_rejects_recording_without_later_query() -> None:
    dataset = _Dataset({"ses-a": 7})
    order = np.arange(64, dtype=np.int64)
    with pytest.raises(RuntimeError, match="causal query"):
        _m7_schedule(dataset, order, "19250108")


def _metrics(deltas: dict[int, list[float]]) -> dict[str, dict]:
    result = {}
    for index, date in enumerate(CONFIRMATORY_DATES):
        result[date] = {
            "budgets": {
                str(budget): {"delta_c1_minus_t0": values[index]}
                for budget, values in deltas.items()
            }
        }
    return result


def test_transfer_gate_passes_exact_registered_boundary() -> None:
    decision = transfer_decision(_metrics({
        4: [0.01, 0.01, 0.01, 0.02, -0.00],
        5: [-0.01] * 5,
        7: [-0.01] * 5,
    }))
    assert decision["verdict"] == "PASS_TRANSFER"
    assert decision["m4_positive_dates"] == 4


@pytest.mark.parametrize("deltas", [
    {4: [0.009] * 5, 5: [0.0] * 5, 7: [0.0] * 5},
    {4: [0.02, 0.02, 0.02, -0.01, -0.01], 5: [0.0] * 5, 7: [0.0] * 5},
    {4: [0.02] * 5, 5: [-0.011] * 5, 7: [0.0] * 5},
    {4: [0.02] * 5, 5: [0.0] * 5, 7: [-0.011] * 5},
])
def test_transfer_gate_rejects_failed_primary_or_safety(deltas: dict[int, list[float]]) -> None:
    assert transfer_decision(_metrics(deltas))["verdict"] == "COMPLETE_NO_TRANSFER"


def test_common_config_is_fresh_epoch49_hc() -> None:
    config = common_config()
    assert config["warm_start"] is False and config["terminal_epoch_zero_based"] == 49
    assert config["t0_prefix"] == 7 and config["c1_cycle"] == [7, 5, 4]
    assert config["base"]["seed"] == 42 and config["base"]["epochs"] == 50


def test_attempt_is_publish_once_0444_with_sidecar(tmp_path: Path) -> None:
    root = tmp_path / "result"
    body = create_attempt(root, {"x": "0" * 64}, EXPERIMENT3_COMMIT)
    path = root / "attempt.json"
    assert body["schema"] == SCHEMA
    assert stat.S_IMODE(path.stat().st_mode) == 0o444
    verify_sidecar(path)
    with pytest.raises(RuntimeError, match="not fresh"):
        create_attempt(root, {"x": "0" * 64}, EXPERIMENT3_COMMIT)


def test_model_supports_variable_prefix_without_topology_change() -> None:
    import torch
    from src.models.components.h1_carrierid_spint import H1CarrierIdSpint

    kwargs = common_config()["base"]["model_kwargs"]
    torch.manual_seed(42)
    model = H1CarrierIdSpint(**kwargs).eval()
    assert sum(parameter.numel() for parameter in model.parameters()) == MODEL_PARAMETERS
    carrier = torch.zeros(1, 176, 4)
    with torch.no_grad():
        identities = [model.carrierid_identity_projection(torch.zeros(1, m, 1024, 176), carrier) for m in BUDGETS]
    assert all(value.shape == (1, 176, 700) for value in identities)


def test_status_constants_are_scientific_terminal_labels() -> None:
    assert STATUS_PASS == "PASS_H1_CAL_AUG_PREFIX_CYCLE_TRANSFER"
    assert STATUS_NO_TRANSFER == "COMPLETE_H1_CAL_AUG_PREFIX_CYCLE_NO_TRANSFER"


def test_runner_attempt_precedes_source_loader_and_cuda_textually() -> None:
    runner = Path(__file__).resolve().parents[2] / "tfpd_exploration/h1_series_20260830/scripts/run_h1_cal_aug_prefix_cycle_v1.py"
    text = runner.read_text(encoding="utf-8")
    prepare = text.index("create_attempt(args.result_root")
    cpu = text.index("_cpu_gate(args.result_root)", prepare)
    source = text.index("prepare_source_authority(args.data_root", cpu)
    assert prepare < cpu < source
    assert "tqdm" not in text and "progress" not in text.lower()
