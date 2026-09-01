from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys

import numpy as np
import pytest
import torch

import src.h1_causal_output_ema_v1 as ema
from src.data.h1_m4_cce_date_lodo import target_sessions_for_date
from src.h1_hc_date_lodo_regen_v1 import verify_sidecar
from src.h1_m4_cce_contract import CONFIRMATORY_DATES


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER = REPO_ROOT / "tfpd_exploration/h1_series_20260830/scripts/run_h1_causal_output_ema_v1.py"


def test_dry_run_is_zero_write_zero_data_zero_cuda(tmp_path: Path) -> None:
    result = tmp_path / "must-not-exist"
    environment = dict(os.environ)
    environment.update({"PYTHONNOUSERSITE": "1", "CUDA_VISIBLE_DEVICES": ""})
    completed = subprocess.run(
        [sys.executable, str(RUNNER), "--dry-run", "--result-root", str(result)],
        cwd=REPO_ROOT,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert not result.exists()
    body = json.loads(completed.stdout)
    assert body["status"] == "DRY_NO_WRITE_NO_DATA_NO_CUDA"
    assert body["writes"] == body["target_access"] == body["cuda_queries"] == 0


def test_attempt_is_published_before_first_cuda_query(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "result"
    ema.create_attempt(root, {"synthetic.py": "0" * 64}, "1" * 40)
    cell_attempt = root / "smoke" / "attempt.json"

    def cuda_query() -> bool:
        assert verify_sidecar(cell_attempt)
        return False

    monkeypatch.setattr(torch.cuda, "is_available", cuda_query)
    with pytest.raises(ema.EmaExperimentError, match="requires CUDA"):
        ema.run_smoke_cell(tmp_path / "predecessor", root, 0)
    assert verify_sidecar(cell_attempt)
    assert verify_sidecar(cell_attempt.parent / "failure.json")


def test_attempt_refuses_nonfresh_root_and_is_immutable(tmp_path: Path) -> None:
    root = tmp_path / "result"
    ema.create_attempt(root, {"x": "0" * 64}, "1" * 40)
    path = root / "attempt.json"
    assert verify_sidecar(path)
    assert path.stat().st_mode & 0o777 == 0o444
    with pytest.raises(ema.EmaExperimentError, match="not fresh"):
        ema.create_attempt(root, {"x": "0" * 64}, "1" * 40)


def test_registered_ema_formula_float64_and_alpha_zero_identity() -> None:
    prediction = np.asarray([[1, 2, 3, 4, 5, 6, 7], [3, 4, 5, 6, 7, 8, 9], [7, 8, 9, 10, 11, 12, 13]], dtype=np.float32)
    raw = ema.causal_ema(prediction, 0.0)
    filtered = ema.causal_ema(prediction, 0.7)
    assert raw.dtype == filtered.dtype == np.float64
    assert np.array_equal(raw, prediction.astype(np.float64))
    assert np.array_equal(filtered[0], prediction[0].astype(np.float64))
    assert np.allclose(filtered[1], 0.7 * prediction[0] + 0.3 * prediction[1], atol=1e-15)
    assert np.allclose(filtered[2], 0.7 * filtered[1] + 0.3 * prediction[2], atol=1e-15)
    with pytest.raises(ema.EmaExperimentError, match="not pre-registered"):
        ema.causal_ema(prediction, 0.2)


def _cache(raw_rows: list[np.ndarray], target_rows: list[np.ndarray], masks: list[np.ndarray]) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for index, (raw, target, mask) in enumerate(zip(raw_rows, target_rows, masks, strict=True)):
        result[f"raw_{index}"] = np.asarray(raw, dtype=np.float32)
        result[f"target_{index}"] = np.asarray(target, dtype=np.float32)
        result[f"score_mask_{index}"] = np.asarray(mask, dtype=bool)
        result[f"output_bins_{index}"] = np.arange(700, 700 + len(raw), dtype=np.int64)
    return result


def test_score_cache_filters_through_mask_gap_and_resets_per_recording() -> None:
    target_a = np.arange(35, dtype=np.float32).reshape(5, 7)
    raw_a = target_a.copy()
    raw_a[1] += 20.0
    target_b = np.arange(35, 70, dtype=np.float32).reshape(5, 7)
    raw_b = target_b.copy()
    masks = [np.asarray([True, False, True, True, True]), np.ones(5, dtype=bool)]
    cache = _cache([raw_a, raw_b], [target_a, target_b], masks)
    metrics = ema.score_cache(cache, ("a", "b"))
    filtered_a = ema.causal_ema(raw_a, 0.7)
    assert np.array_equal(filtered_a[0], raw_a[0].astype(np.float64))
    assert not np.array_equal(filtered_a[2], raw_a[2].astype(np.float64))
    filtered_b = ema.causal_ema(raw_b, 0.7)
    assert np.array_equal(filtered_b[0], raw_b[0].astype(np.float64))
    assert metrics["alpha_order"] == list(ema.ALPHAS)
    assert metrics["selection_performed"] is False
    assert metrics["per_recording"]["a"]["scored_outputs"] == 4


def test_score_cache_uses_float64_variance_weighted_r2() -> None:
    generator = np.random.default_rng(3)
    target = generator.normal(size=(20, 7)).astype(np.float32)
    raw = target + generator.normal(scale=0.1, size=(20, 7)).astype(np.float32)
    mask = np.ones(20, dtype=bool)
    metrics = ema.score_cache(_cache([raw], [target], [mask]), ("session",))
    observed = metrics["per_recording"]["session"]["r2_by_alpha"]["0.0"]
    expected = 1.0 - float(np.square(target.astype(np.float64) - raw.astype(np.float64)).sum()) / float(
        np.square(target.astype(np.float64) - target.astype(np.float64).mean(axis=0, keepdims=True)).sum()
    )
    assert observed == pytest.approx(expected, abs=1e-15)


def _date_rows(deltas: list[float]) -> list[dict[str, object]]:
    rows = []
    for date, delta in zip(CONFIRMATORY_DATES, deltas, strict=True):
        rows.append(
            {
                "outer_date": date,
                "metrics": {
                    "alpha_metrics": {
                        "0.7": {"delta_equal_recording_mean_r2_vs_raw": delta},
                    }
                },
            }
        )
    return rows


def test_transfer_gate_requires_magnitude_and_four_positive_dates() -> None:
    passed = ema.transfer_decision(_date_rows([0.014, 0.013, 0.015, 0.014, -0.001]))
    assert passed["verdict"] == "PASS_TRANSFER"
    insufficient_breadth = ema.transfer_decision(_date_rows([0.03, 0.03, 0.03, -0.001, -0.001]))
    assert insufficient_breadth["verdict"] == "COMPLETE_NO_TRANSFER"
    insufficient_magnitude = ema.transfer_decision(_date_rows([0.005, 0.005, 0.005, 0.005, -0.001]))
    assert insufficient_magnitude["verdict"] == "COMPLETE_NO_TRANSFER"


def test_target_loader_opens_only_requested_outer_date(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    date = CONFIRMATORY_DATES[0]
    expected = target_sessions_for_date(date)
    paths: dict[str, Path] = {}
    for session in expected:
        path = tmp_path / f"{session}.nwb"
        path.write_bytes(b"1234")
        paths[session] = path
    opened: list[str] = []

    def fake_load(path: Path):
        session = path.stem
        opened.append(session)
        return SimpleNamespace(session_name=session, date=date, input_sha256="a" * 64)

    monkeypatch.setattr(ema, "reject_nonpublic_heldin_scope", lambda _path: None)
    monkeypatch.setattr(ema, "index_heldin_calib", lambda _path: paths)
    access = {"target_recordings_opened": 0, "target_bytes_read": 0, "target_sessions_opened": [], "files": []}
    records = ema._target_records(tmp_path, date, access, record_loader=fake_load)
    assert tuple(records) == expected == tuple(opened)
    assert access["target_recordings_opened"] == len(expected)
    assert access["target_bytes_read"] == 4 * len(expected)


def test_predecessor_table_is_complete_and_exact_epoch49_authority() -> None:
    assert tuple(ema.PREDECESSOR) == CONFIRMATORY_DATES
    assert len({row["checkpoint_sha256"] for row in ema.PREDECESSOR.values()}) == 5
    assert all(len(row["terminal_sha256"]) == len(row["terminal_state_sha256"]) == 64 for row in ema.PREDECESSOR.values())
    assert all(int(row["global_step"]) > 0 for row in ema.PREDECESSOR.values())
    assert len(ema.PREDECESSOR_TERMINAL_SHA256) == 64


def test_cli_exposes_registered_public_phases() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    for flag in ("--dry-run", "--cpu-gate", "--smoke", "--evaluate", "--verify-terminal", "--detached-supervisor", "--data-root", "--predecessor-root", "--result-root", "--physical-gpus"):
        assert flag in source


def test_runtime_has_no_progress_bar_or_training_operations() -> None:
    module_source = Path(ema.__file__).read_text(encoding="utf-8")
    runner_source = RUNNER.read_text(encoding="utf-8")
    assert "tqdm" not in module_source + runner_source
    body = module_source[module_source.index("def run_evaluation_cell(") : module_source.index("def _load_cache(")]
    assert ".backward(" not in body
    assert "torch.optim" not in body
    assert "model.train(" not in body


def test_malformed_cache_and_incomplete_date_set_fail_closed() -> None:
    raw = np.zeros((4, 7), dtype=np.float32)
    target = np.arange(28, dtype=np.float32).reshape(4, 7)
    malformed = _cache([raw], [target], [np.ones(4, dtype=bool)])
    malformed["output_bins_0"][2] += 4
    with pytest.raises(ema.EmaExperimentError, match="noncontinuous"):
        ema.score_cache(malformed, ("session",))
    with pytest.raises(ema.EmaExperimentError, match="all five dates"):
        ema.transfer_decision(_date_rows([0.1] * 5)[:4])
