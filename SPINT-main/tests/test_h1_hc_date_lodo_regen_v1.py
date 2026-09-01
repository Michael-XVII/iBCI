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

import src.data.h1_carrierid_date_lodo_source as source_loader
import src.h1_hc_date_lodo_regen_v1 as regen
from src.data.h1_m4_cce_date_lodo import source_sessions_for_date, target_sessions_for_date
from src.data.h1_m4_eb_pilot import (
    EXPECTED_NEURONS,
    H1_HELDIN_SESSIONS,
    TrialBlocks,
    fit_frozen_carrier,
    session_date,
)
from src.h1_m4_cce_contract import CONFIRMATORY_DATES, state_hash
from src.models.components.h1_carrierid_spint import H1CarrierIdSpint


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER = REPO_ROOT / "tfpd_exploration/h1_series_20260830/scripts/run_h1_hc_date_lodo_regen_v1.py"


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
    assert body["target_access"] == 0


def test_attempt_is_published_before_first_cuda_query(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "result"
    regen.create_attempt(root, {"synthetic.py": "0" * 64}, "1" * 40)
    cell_attempt = root / "smoke" / f"smoke_{CONFIRMATORY_DATES[0]}" / "attempt.json"

    def cuda_query() -> bool:
        assert regen.verify_sidecar(cell_attempt)
        return False

    monkeypatch.setattr(torch.cuda, "is_available", cuda_query)
    with pytest.raises(regen.RegenError, match="requires CUDA"):
        regen.run_cell(Path("/unused"), root, CONFIRMATORY_DATES[0], 0, smoke=True, smoke_steps=1)
    assert regen.verify_sidecar(cell_attempt)
    assert regen.verify_sidecar(cell_attempt.parent / "failure.json")


def test_source_loader_never_opens_outer_date(monkeypatch: pytest.MonkeyPatch) -> None:
    outer = CONFIRMATORY_DATES[0]
    paths = {name: Path(f"/synthetic/{name}.nwb") for name in H1_HELDIN_SESSIONS}
    opened: list[str] = []

    def fake_load(path: Path):
        name = path.stem
        opened.append(name)
        return SimpleNamespace(date=session_date(name))

    monkeypatch.setattr(source_loader, "reject_nonpublic_heldin_scope", lambda _path: None)
    monkeypatch.setattr(source_loader, "index_heldin_calib", lambda _path: paths)
    records, audit = source_loader.load_source_records_with_target_filename_index(
        "/synthetic", outer, record_loader=fake_load
    )
    assert tuple(records) == source_sessions_for_date(outer)
    assert tuple(opened) == source_sessions_for_date(outer)
    assert not (set(opened) & set(target_sessions_for_date(outer)))
    assert audit.manifest()["target_recordings_opened"] == 0
    assert audit.manifest()["target_bytes_read"] == 0


def test_candidate_grid_and_registered_tie_break() -> None:
    rows = [
        {
            "q": q,
            "lambda": lam,
            "equal_date_mean_r2": 0.25,
            "worst_date_r2": -0.1,
        }
        for q in regen.Q_GRID
        for lam in regen.LAMBDA_GRID
    ]
    selected = regen.select_candidate(rows)
    assert selected["q"] == 4
    assert selected["lambda"] == 10.0
    rows[-1]["equal_date_mean_r2"] = 0.2500000000001
    assert regen.select_candidate(rows) == rows[-1]
    with pytest.raises(regen.RegenError, match="candidate grid"):
        regen.select_candidate(rows[:-1])


def test_float64_variance_weighted_r2() -> None:
    truth = np.asarray([[0, 1], [1, 3], [2, 5]], dtype=np.float32)
    perfect = regen.variance_weighted_r2(truth, truth)
    shifted = regen.variance_weighted_r2(truth, truth + np.float32(1.0))
    assert isinstance(perfect, float) and perfect == 1.0
    expected = 1.0 - 6.0 / float(np.square(truth.astype(np.float64) - truth.mean(axis=0, dtype=np.float64)).sum())
    assert shifted == pytest.approx(expected, abs=1e-15)


class _SyntheticRecord:
    def __init__(self, name: str, seed: int):
        self.session_name = name
        self.input_sha256 = f"{seed:064x}"
        self.num_neurons = EXPECTED_NEURONS
        self.trial_values = tuple(float(value) for value in range(7))
        generator = np.random.default_rng(seed)
        self._blocks = {}
        weights = generator.normal(size=(EXPECTED_NEURONS, 7)) * 0.01
        for value in self.trial_values:
            rates = generator.normal(loc=value * 0.1, scale=1.0, size=(8, EXPECTED_NEURONS))
            velocity = rates @ weights + generator.normal(scale=0.01, size=(8, 7))
            self._blocks[value] = TrialBlocks(
                trial_number=value,
                rates=np.asarray(rates, np.float64),
                velocity=np.asarray(velocity, np.float64),
                block_indices=np.zeros((8, 5), dtype=np.int64),
            )

    def blocks_for(self, value: float) -> TrialBlocks:
        return self._blocks[float(value)]


def test_refit_eb_carrier_is_finite() -> None:
    records = {
        name: _SyntheticRecord(name, index + 1)
        for index, name in enumerate(H1_HELDIN_SESSIONS[:4])
    }
    selected = {"q": 4, "lambda": 0.1}
    plan = regen._make_final_plan(records, CONFIRMATORY_DATES[0], selected, "a" * 64)
    assert plan.U.shape == (7, 4)
    assert np.isfinite(plan.tau2) and plan.tau2 > 0
    first = next(iter(records.values()))
    carrier = fit_frozen_carrier(first, plan, first.trial_values[:4])["carrier"]
    assert carrier.shape == (EXPECTED_NEURONS, 4)
    assert np.isfinite(carrier).all()


def test_schedule_is_repeatable_and_batch_complete() -> None:
    names = H1_HELDIN_SESSIONS[:2]
    records = {
        name: SimpleNamespace(eval_mask=np.ones(64, dtype=bool))
        for name in names
    }
    starts = {name: (0, 1, 2, 3) for name in names}
    first = regen._build_schedule(records, starts, CONFIRMATORY_DATES[0])
    second = regen._build_schedule(records, starts, CONFIRMATORY_DATES[0])
    assert np.array_equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])
    assert first[1].shape == (50, len(first[0]))
    assert len(first[0]) % regen.BATCH_SIZE == 0


def _model_kwargs() -> dict[str, object]:
    return dict(regen.model_config()["model_kwargs"])


def test_h32_model_parameter_count_and_forward_semantics() -> None:
    torch.manual_seed(42)
    model = H1CarrierIdSpint(**_model_kwargs()).eval()
    assert sum(parameter.numel() for parameter in model.parameters()) == regen.MODEL_PARAMETERS
    neural = torch.zeros(1, 700, EXPECTED_NEURONS)
    identity = torch.zeros(1, 4, 1024, EXPECTED_NEURONS)
    carrier = torch.zeros(1, EXPECTED_NEURONS, 4)
    with torch.no_grad():
        output = model(neural, calib_trialized_neural_features=identity, carrier=carrier)
    assert output.shape == (1, 700, 7)
    with pytest.raises(ValueError, match="hidden_dim=32"):
        H1CarrierIdSpint(**{**_model_kwargs(), "carrier_hidden_dim": 64})
    with pytest.raises(ValueError, match="requires calibration identity"):
        model(neural)


def _valid_metadata() -> tuple[dict[str, object], dict[str, object]]:
    metadata: dict[str, object] = {
        "schema": regen.CHECKPOINT_SCHEMA,
        "outer_date": CONFIRMATORY_DATES[0],
        "fresh_seed": 42,
        "checkpoint_epoch_zero_based": 49,
        "epochs_completed": 50,
        "selected_by": "fixed_terminal_epoch_no_validation_or_target_selection",
        "warm_start": False,
        "config_sha256": "c" * 64,
        "source_authority_sha256": "s" * 64,
        "experiment_attempt_sha256": "a" * 64,
        "code_closure_sha256": "z" * 64,
        "initial_state_sha256": "i" * 64,
        "terminal_state_sha256": "t" * 64,
        "global_step": 100,
        "target_recordings_opened": 0,
        "target_bytes_read": 0,
        "target_optimizer_steps": 0,
        "target_backward_steps": 0,
    }
    terminal = {
        "config_sha256": "c" * 64,
        "initial_state_sha256": "i" * 64,
        "terminal_state_sha256": "t" * 64,
        "global_step": 100,
    }
    return metadata, terminal


@pytest.mark.parametrize(
    ("field", "value"),
    (("warm_start", True), ("checkpoint_epoch_zero_based", 48), ("target_bytes_read", 1), ("global_step", 0)),
)
def test_checkpoint_contract_rejects_invalid_terminal(field: str, value: object) -> None:
    metadata, terminal = _valid_metadata()
    metadata[field] = value
    with pytest.raises(regen.RegenError):
        regen.validate_checkpoint_contract(
            metadata, terminal, CONFIRMATORY_DATES[0], "a" * 64, "z" * 64, "s" * 64
        )


def test_receipt_sidecar_rejects_missing_writable_and_wrong_sha(tmp_path: Path) -> None:
    root = tmp_path / "authority"
    regen.create_attempt(root, {"x": "0" * 64}, "1" * 40)
    path = root / "attempt.json"
    assert len(regen.verify_sidecar(path)) == 64
    path.chmod(0o644)
    with pytest.raises(regen.RegenError, match="not 0444"):
        regen.verify_sidecar(path)
    path.chmod(0o444)
    sidecar = path.with_name(path.name + ".sha256")
    sidecar.chmod(0o644)
    sidecar.write_text(f"{'f' * 64}  {path.name}\n", encoding="ascii")
    sidecar.chmod(0o444)
    with pytest.raises(regen.RegenError, match="mismatch"):
        regen.verify_sidecar(path)
    with pytest.raises(regen.RegenError, match="not fresh"):
        regen.create_attempt(root, {"x": "0" * 64}, "1" * 40)


def test_state_hash_changes_with_model_state() -> None:
    first = {"weight": torch.zeros(2, 2)}
    second = {"weight": torch.ones(2, 2)}
    assert state_hash(first) != state_hash(second)


def test_completed_runtime_log_is_immutable_and_verified(tmp_path: Path) -> None:
    path = tmp_path / "cell.log"
    path.write_text("terminal output\n", encoding="utf-8")
    digest = regen.seal_existing_log(path)
    assert regen.verify_sidecar(path) == digest
    assert path.stat().st_mode & 0o777 == 0o444


def test_training_loop_has_no_progress_print_or_tqdm() -> None:
    source = Path(regen.__file__).read_text(encoding="utf-8")
    body = source[source.index("def run_cell("):source.index("def validate_checkpoint_contract(")]
    assert "tqdm" not in body and "print(" not in body


def test_malformed_source_batch_is_rejected() -> None:
    dataset = object.__new__(regen.SourceDataset)
    dataset.windows = (("a", 0), ("b", 0))
    with pytest.raises(regen.RegenError, match="mixes source sessions"):
        dataset.batch((0, 1), (0, 0))
