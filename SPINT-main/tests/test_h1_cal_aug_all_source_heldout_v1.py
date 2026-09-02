from __future__ import annotations

from dataclasses import fields
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.data.h1_cal_aug_all_source_heldout_v1 import (
    HELDOUT_DIRECTORY,
    H1_HELDOUT_SESSIONS,
    HeldoutMetadataError,
    HeldoutTrialMetadata,
    index_heldout_calib,
    read_heldout_trial_metadata,
)
import src.h1_cal_aug_all_source_heldout_v1 as audit_module
from src.h1_cal_aug_all_source_heldout_v1 import (
    PASS_STATUS,
    STOP_STATUS,
    AllSourceHeldoutError,
    create_attempt,
    dry_plan,
    load_attempt,
    run_metadata_feasibility_audit,
    verify_metadata_terminal,
)


EXPECTED_HELDOUT = (
    "ses-19250126T113454", "ses-19250126T114029",
    "ses-19250127T120333", "ses-19250127T120826",
    "ses-19250129T112555", "ses-19250129T113059",
    "ses-19250202T113958", "ses-19250202T114452",
    "ses-19250203T113515", "ses-19250203T114018",
    "ses-19250206T112219", "ses-19250206T112712",
    "ses-19250209T111826", "ses-19250209T112327",
)


class _TrackedData:
    def __init__(self, value: np.ndarray, reads: list[str], name: str) -> None:
        self.value = value
        self.reads = reads
        self.name = name

    def __getitem__(self, key: object) -> np.ndarray:
        self.reads.append(self.name)
        return self.value[key]


class _Acquisitions(dict[str, SimpleNamespace]):
    def __init__(self, arrays: dict[str, np.ndarray], reads: list[str]) -> None:
        super().__init__({
            name: SimpleNamespace(data=_TrackedData(value, reads, name))
            for name, value in arrays.items()
        })


class _FakeIO:
    def __init__(self, acquisitions: _Acquisitions) -> None:
        self.acquisitions = acquisitions

    def __enter__(self) -> "_FakeIO":
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def read(self) -> SimpleNamespace:
        return SimpleNamespace(acquisition=self.acquisitions)


def _make_roster(tmp_path: Path) -> tuple[Path, dict[str, Path]]:
    root = tmp_path / "000954"
    directory = root / HELDOUT_DIRECTORY
    directory.mkdir(parents=True)
    paths = {}
    for session in H1_HELDOUT_SESSIONS:
        path = directory / f"sub-HumanPitt_{session}.nwb"
        path.touch()
        paths[session] = path
    return root, paths


def _metadata_rows(count: int = 5) -> tuple[HeldoutTrialMetadata, ...]:
    return tuple(
        HeldoutTrialMetadata(session, count, count >= 5, "eval_mask")
        for session in H1_HELDOUT_SESSIONS
    )


def _seed_gate_root(tmp_path: Path) -> Path:
    result_root = tmp_path / "result"
    create_attempt(result_root, {"source.py": "a" * 64}, "b" * 40)
    audit_module.publish_json(result_root / "predecessor_authority.json", {
        "schema": f"{audit_module.SCHEMA}_predecessor_authority",
        "status": "PASS_H1_ALL_SOURCE_HELDOUT_V1_PREDECESSOR",
    })
    return result_root


def test_registered_roster_is_exact_and_stable() -> None:
    assert H1_HELDOUT_SESSIONS == EXPECTED_HELDOUT
    assert len(H1_HELDOUT_SESSIONS) == len(set(H1_HELDOUT_SESSIONS)) == 14


def test_dry_plan_is_zero_write_zero_data_zero_cuda() -> None:
    plan = dry_plan()
    assert plan["status"] == "DRY_NO_WRITE_NO_NWB_NO_CUDA"
    assert plan["registered_heldout_sessions"] == list(EXPECTED_HELDOUT)
    assert plan["minimum_legal_trial_count"] == 5
    for name in ("neural_reads", "behavior_reads", "carrier_fits", "model_inference", "r2_calculations", "gpu_training"):
        assert plan[name] == 0


def test_attempt_is_immutable_and_must_be_first(tmp_path: Path) -> None:
    result_root = tmp_path / "result"
    create_attempt(result_root, {"source.py": "a" * 64}, "b" * 40)
    attempt = result_root / "attempt.json"
    sidecar = result_root / "attempt.json.sha256"
    assert attempt.stat().st_mode & 0o777 == 0o444
    assert sidecar.stat().st_mode & 0o777 == 0o444
    assert load_attempt(result_root)["heldout_files_opened"] == 0
    with pytest.raises(AllSourceHeldoutError, match="not fresh"):
        create_attempt(result_root, {}, "c" * 40)


def test_index_accepts_only_exact_registered_heldout_roster(tmp_path: Path) -> None:
    root, paths = _make_roster(tmp_path)
    assert index_heldout_calib(root) == paths
    (root / HELDOUT_DIRECTORY / "sub-HumanPitt_ses-19990101T000000.nwb").touch()
    with pytest.raises(HeldoutMetadataError, match="roster drift"):
        index_heldout_calib(root)


def test_metadata_loader_reads_only_trialnum_and_eval_mask(tmp_path: Path) -> None:
    root, paths = _make_roster(tmp_path)
    reads: list[str] = []
    arrays = {
        "TrialNum": np.repeat(np.arange(1, 6), 3),
        "eval_mask": np.ones(15, dtype=bool),
        "OpenLoopKinematicsVelocity": np.arange(105).reshape(15, 7),
        "ElectricalSeries": np.arange(15),
    }
    row = read_heldout_trial_metadata(
        paths[EXPECTED_HELDOUT[0]], root,
        io_factory=lambda *_args, **_kwargs: _FakeIO(_Acquisitions(arrays, reads)),
    )
    assert reads == ["TrialNum", "eval_mask"]
    assert row == HeldoutTrialMetadata(EXPECTED_HELDOUT[0], 5, True, "eval_mask")
    assert tuple(item.name for item in fields(row)) == (
        "session_name", "legal_trial_count", "m4_evaluable", "validity_field",
    )


def test_metadata_loader_supports_blacklist_without_other_access(tmp_path: Path) -> None:
    root, paths = _make_roster(tmp_path)
    reads: list[str] = []
    arrays = {
        "TrialNum": np.repeat(np.arange(1, 6), 2),
        "Blacklist": np.zeros(10, dtype=bool),
        "OpenLoopKinematicsVelocity": np.zeros((10, 7)),
    }
    row = read_heldout_trial_metadata(
        paths[EXPECTED_HELDOUT[1]], root,
        io_factory=lambda *_args, **_kwargs: _FakeIO(_Acquisitions(arrays, reads)),
    )
    assert reads == ["TrialNum", "Blacklist"]
    assert row.legal_trial_count == 5 and row.m4_evaluable
    assert row.validity_field == "Blacklist"


def test_metadata_loader_rejects_non_heldout_scope_and_bad_arrays(tmp_path: Path) -> None:
    root, paths = _make_roster(tmp_path)
    heldin = root / "sub-HumanPitt-held-in-calib" / f"sub-HumanPitt_{EXPECTED_HELDOUT[0]}.nwb"
    heldin.parent.mkdir()
    heldin.touch()
    with pytest.raises(HeldoutMetadataError, match="only direct held-out-calib"):
        read_heldout_trial_metadata(heldin, root, io_factory=lambda *_a, **_k: None)
    arrays = {"TrialNum": np.asarray([1, 2, 1, 3, 4, 5]), "eval_mask": np.ones(6, dtype=bool)}
    with pytest.raises(HeldoutMetadataError, match="not chronological"):
        read_heldout_trial_metadata(
            paths[EXPECTED_HELDOUT[0]], root,
            io_factory=lambda *_a, **_k: _FakeIO(_Acquisitions(arrays, [])),
        )


@pytest.mark.parametrize("short_count, expected_status", [(None, PASS_STATUS), (4, STOP_STATUS)])
def test_aggregate_gate_is_all_14_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, short_count: int | None, expected_status: str) -> None:
    rows = list(_metadata_rows(5))
    if short_count is not None:
        rows[-1] = HeldoutTrialMetadata(EXPECTED_HELDOUT[-1], short_count, False, "eval_mask")
    monkeypatch.setattr(audit_module, "audit_registered_heldout_metadata", lambda _root: tuple(rows))
    result_root = _seed_gate_root(tmp_path)
    terminal = run_metadata_feasibility_audit(tmp_path / "unused", result_root)
    assert terminal["status"] == expected_status
    assert terminal["continuation_to_training_allowed"] is (short_count is None)
    assert terminal["gpu_training_started"] is False
    assert verify_metadata_terminal(result_root)["status"] == expected_status
    feasibility = json.loads((result_root / "metadata_feasibility.json").read_text())
    assert feasibility["automatic_m3_fallback"] is False
    assert feasibility["m4_evaluable_recordings"] == (14 if short_count is None else 13)
    if short_count is not None:
        stopped = json.loads((result_root / "terminal.json").read_text())
        assert stopped["status"] == STOP_STATUS
        assert stopped["claim"] == "M4 protocol infeasible before training; no automatic M3 fallback"


def test_cli_exposes_no_training_prediction_or_scoring_phase() -> None:
    root = Path(__file__).resolve().parents[2]
    runner = (root / "tfpd_exploration/h1_series_20260830/scripts/run_h1_cal_aug_all_source_heldout_v1.py").read_text()
    assert runner.index("create_attempt(args.result_root") < runner.index("run_metadata_feasibility_audit(")
    assert "--audit-heldout-metadata" in runner and "--verify-metadata-terminal" in runner
    for forbidden in ("--train", "--evaluate", "--smoke", "--physical-gpus", "--checkpoint"):
        assert forbidden not in runner


def test_work_order_records_metadata_exception_and_no_m3_fallback() -> None:
    root = Path(__file__).resolve().parents[2]
    work_order = (root / "tfpd_exploration/h1_series_20260830/H1_CAL_AUG_ALL_SOURCE_HELDOUT_V1_WORK_ORDER.md").read_text()
    amendment = (root / "tfpd_exploration/h1_series_20260830/docs/AMENDMENT_H1_CAL_AUG_ALL_SOURCE_HELDOUT_V1_METADATA_FEASIBILITY.md").read_text()
    assert "metadata-only feasibility access is the sole pre-training exception" in work_order
    assert "STOP_H1_ALL_SOURCE_HELDOUT_M4_PROTOCOL_INFEASIBLE" in work_order
    assert "do not fall back to M3" in work_order
    assert "TrialNum" in amendment and "eval_mask" in amendment and "Blacklist" in amendment
    assert "launches no gpu training" in amendment.lower()
