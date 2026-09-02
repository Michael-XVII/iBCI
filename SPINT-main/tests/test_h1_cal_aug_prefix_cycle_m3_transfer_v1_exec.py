from __future__ import annotations

from pathlib import Path
import stat

import pytest

from src.h1_cal_aug_prefix_cycle_m3_transfer_v1 import (
    EXPERIMENT4_A1_TERMINAL_SHA256,
    M4_AUDIT_TERMINAL_SHA256,
    M4_METADATA_TERMINAL_SHA256,
)
from src.h1_cal_aug_prefix_cycle_m3_transfer_v1_exec import (
    VERDICT_NO_CLEAR,
    VERDICT_STRONG,
    VERDICT_SUPPORT,
    M3ExecutionError,
    collect_predecessor_authority,
    create_attempt,
    interpretation,
    load_attempt,
)
from src.h1_hc_date_lodo_regen_v1 import verify_sidecar
from src.h1_m4_cce_contract import CONFIRMATORY_DATES


def test_formal_attempt_is_after_cpu_and_before_cuda_target(tmp_path: Path) -> None:
    root = tmp_path / "result"
    body = create_attempt(root, {"reviewed.py": "a" * 64}, "b" * 40, "c" * 64)
    assert body["status"] == "ATTEMPT_AFTER_CPU_GATE_BEFORE_AUTHORITY_CUDA_AND_TARGET"
    assert body["cpu_gate_stdout_sha256"] == "c" * 64
    assert body["outer_dates"] == list(CONFIRMATORY_DATES)
    assert body["cuda_initialized"] is False
    for name in (
        "target_recordings_opened", "target_bytes_read", "heldout_calib_recordings_opened",
        "optimizer_steps", "backward_steps", "parameter_updates", "warm_starts", "checkpoint_selections",
    ):
        assert body[name] == 0
    assert stat.S_IMODE((root / "attempt.json").stat().st_mode) == 0o444
    assert load_attempt(root) == body
    verify_sidecar(root / "attempt.json")
    with pytest.raises(M3ExecutionError, match="not fresh"):
        create_attempt(root, {}, "d" * 40, "e" * 64)


def test_predecessor_authority_fails_closed_before_missing_checkpoint_access(tmp_path: Path) -> None:
    with pytest.raises((FileNotFoundError, RuntimeError)):
        collect_predecessor_authority(
            tmp_path / "training", tmp_path / "eval-a1", tmp_path / "m4",
            tmp_path / "regen", tmp_path / "experiment3",
        )


@pytest.mark.parametrize(
    "mean,positive,expected",
    [
        (0.0100000000, 4, VERDICT_STRONG),
        (0.0099999999, 4, VERDICT_SUPPORT),
        (0.1000000000, 3, VERDICT_NO_CLEAR),
        (0.0000000000, 5, VERDICT_NO_CLEAR),
        (-0.0100000000, 5, VERDICT_NO_CLEAR),
    ],
)
def test_preregistered_interpretation(mean: float, positive: int, expected: str) -> None:
    assert interpretation(mean, positive) == expected


def test_authority_constants_bind_sealed_terminals() -> None:
    assert EXPERIMENT4_A1_TERMINAL_SHA256 == "dc9e7ab44954d3d193f67f9bf8936aafdaf2b05be9968d5e0091c0b0ecf092fd"
    assert M4_AUDIT_TERMINAL_SHA256 == "3ff971dc576958b13ace990bcca8aea2e8b999e2af2ed50f418296d05f8d5cfc"
    assert M4_METADATA_TERMINAL_SHA256 == "e692db3b744a7831610c2338ce34504ccedc4c8696e4d73333de899ed295b563"


def test_runner_orders_cpu_attempt_authority_evaluation_and_terminal() -> None:
    root = Path(__file__).resolve().parents[2]
    runner = root / "tfpd_exploration/h1_series_20260830/scripts/run_h1_cal_aug_prefix_cycle_m3_transfer_v1_exec.py"
    text = runner.read_text(encoding="utf-8")
    cpu = text.index("_run_cpu_gate_no_write()")
    attempt = text.index("create_attempt(args.result_root", cpu)
    authority = text.index("prepare_predecessor_authority(", attempt)
    evaluation = text.index("_evaluate_all(args, gpus)", authority)
    terminal = text.index("verify_terminal(", evaluation)
    assert cpu < attempt < authority < evaluation < terminal
    assert "run_arm" not in text and 'phases.add_argument("--train"' not in text
    assert "tqdm" not in text and "progress" not in text.lower()


def test_execution_uses_reviewed_m3_primitives_and_no_heldout_loader() -> None:
    module = Path(__file__).resolve().parents[1] / "src/h1_cal_aug_prefix_cycle_m3_transfer_v1_exec.py"
    text = module.read_text(encoding="utf-8")
    for required in (
        "materialize_m3_calibration", "m3_causal_surface", "scale_last_bin_prediction", "score_m3",
        "aggregate_m3_results", "collect_training_authority", "_load_plan_normalizer", "_load_arm_model",
    ):
        assert required in text
    for forbidden in ("index_heldout", "audit_registered_heldout_metadata", "--train", "optimizer.step", ".backward("):
        assert forbidden not in text
    assert '"heldout_calib_recordings_opened": 0' in text
    assert '"checkpoint_selections": 0' in text


def test_cell_attempt_precedes_cuda_models_and_target_loader() -> None:
    module = Path(__file__).resolve().parents[1] / "src/h1_cal_aug_prefix_cycle_m3_transfer_v1_exec.py"
    text = module.read_text(encoding="utf-8")
    start = text.index("def run_evaluation_cell(")
    attempt = text.index('publish_json(directory / "attempt.json"', start)
    cuda = text.index("torch.cuda.is_available()", attempt)
    model = text.index("_load_arm_model(", cuda)
    target = text.index("_target_records(", model)
    assert attempt < cuda < model < target
