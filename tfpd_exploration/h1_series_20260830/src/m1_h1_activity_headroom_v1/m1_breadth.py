"""Frozen official-checkpoint breadth evaluation for M1 activity headroom."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import sys
from typing import Any

import numpy as np

from .core import ARM_ORDER, ActivityHeadroomError, array_digest
from .m1 import (
    BATCH_SIZE,
    GROWING_CAP,
    OUTPUTS,
    SUPPORT_TRIALS,
    UNITS,
    WINDOW,
    _evaluate_arm,
    _evaluate_direct_static,
)


@dataclass(frozen=True)
class OfficialM1FoldSpec:
    fold: int
    target_session: str
    target_relative: str
    target_sha256: str
    checkpoint_relative: str
    checkpoint_sha256: str
    manifest_relative: str

    def __post_init__(self) -> None:
        if (
            self.fold not in {0, 1, 2}
            or self.target_session not in {"20120924", "20120926", "20120927"}
            or any(not isinstance(value, str) or not value for value in (
                self.target_relative,
                self.checkpoint_relative,
                self.manifest_relative,
            ))
            or len(self.target_sha256) != 64
            or len(self.checkpoint_sha256) != 64
        ):
            raise ActivityHeadroomError("official M1 fold specification drift")


_RUN0 = (
    "streaming_calibration_exp/logs/m1_afc4_source_decoder_fold0/runs/"
    "2026-08-06-16-15-55-070150_rid-m1_afc4_source_decoder_fold0_dev20_resume_e1r1_fNone_s42"
)
_RUN1 = (
    "streaming_calibration_exp/logs/m1_afc4_source_decoder_fold1/runs/"
    "2026-08-06-16-20-17-515482_rid-m1_afc4_source_decoder_fold1_dev20_f1_s42"
)
_RUN2 = "streaming_calibration_exp/logs/m1_afc4_source_decoder_fold2_remote/runs/remote_fold2_source_epoch019"

OFFICIAL_M1_FOLDS: dict[int, OfficialM1FoldSpec] = {
    0: OfficialM1FoldSpec(
        fold=0,
        target_session="20120924",
        target_relative=(
            "SPINT-main/data/000941/sub-MonkeyL-held-in-calib/"
            "sub-MonkeyL-held-in-calib_ses-20120924_behavior+ecephys.nwb"
        ),
        target_sha256="63ee25782c62ff2275dcfbdcaa56552ec4c26fcde00f5a74e5be54785b5c25eb",
        checkpoint_relative=f"{_RUN0}/checkpoints/best_ckpt/epoch_018.ckpt",
        checkpoint_sha256="f2921cabea819fed58b15e169f9cb899472416d30ee5a9b12c4c2087e96cb6be",
        manifest_relative=f"{_RUN0}/source_only_decoder_manifest.json",
    ),
    1: OfficialM1FoldSpec(
        fold=1,
        target_session="20120926",
        target_relative=(
            "SPINT-main/data/000941/sub-MonkeyL-held-in-calib/"
            "sub-MonkeyL-held-in-calib_ses-20120926_behavior+ecephys.nwb"
        ),
        target_sha256="9c72512308194b93cc19b51733514eb9721152ffe4dd357ee93aabd4be5caa91",
        checkpoint_relative=f"{_RUN1}/checkpoints/best_ckpt/epoch_019.ckpt",
        checkpoint_sha256="b15edc9f66ff9acded6b78fe0b7a2041b359f8f831db22f3ab87f6083e8f50f3",
        manifest_relative=f"{_RUN1}/source_only_decoder_manifest.json",
    ),
    2: OfficialM1FoldSpec(
        fold=2,
        target_session="20120927",
        target_relative=(
            "SPINT-main/data/000941/sub-MonkeyL-held-in-calib/"
            "sub-MonkeyL-held-in-calib_ses-20120927_behavior+ecephys.nwb"
        ),
        target_sha256="2d2fdc9be5ccb7a47969894ff1da43a994da11bfff298353f94d489c4af37b3a",
        checkpoint_relative=f"{_RUN2}/checkpoints/best_ckpt/epoch_019.ckpt",
        checkpoint_sha256="925fba67a6a4338ee6e399751e80d53c5e74a22a7dd3ae0329d9c8da9e0291e2",
        manifest_relative=f"{_RUN2}/source_only_decoder_manifest.json",
    ),
}


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise ActivityHeadroomError(message)


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _load_dataset(root: Path, spec: OfficialM1FoldSpec) -> tuple[Any, np.ndarray, dict[str, Any]]:
    from tfpd_exploration.src.cross_session_worst_group_v1 import source_reader

    target = root / spec.target_relative
    _need(target.is_file() and not target.is_symlink() and _file_sha(target) == spec.target_sha256,
          "official M1 target descriptor/body drift")
    manifest_path = root / spec.manifest_relative
    _need(manifest_path.is_file() and not manifest_path.is_symlink(), "official M1 manifest is absent")
    import json
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _need(
        manifest.get("source_only") is True
        and manifest.get("outer_fold") == spec.fold
        and manifest.get("outer_left_out") == f"ses-{spec.target_session}"
        and manifest.get("heldout_opened") is False
        and manifest.get("minival_opened") is False
        and manifest.get("formal") is False,
        "official M1 source-only manifest semantics drift",
    )
    runtime = source_reader.load_native_m1_runtime(root)
    recipe = source_reader.NATIVE_M1_READER_RECIPE
    datamodule = runtime.falcon_datamodule_type(**recipe.datamodule_kwargs(data_dir=root))
    record = datamodule.prepare_session_data(
        target,
        runtime.task,
        standardize_covariates=recipe.standardize_covariates,
        covariates_mean=None,
        covariates_std=None,
        use_intertrials=recipe.use_intertrials,
        include_trial_targets=False,
        include_trial_obj_ids=False,
    )
    dataset = runtime.falcon_dataset_type(
        sessions_dict={spec.target_session: record},
        calib_sessions_dict={spec.target_session: record},
        **recipe.dataset_kwargs(),
    )
    trials = np.ascontiguousarray(
        dataset.calib_trialized_neural_features[spec.target_session], dtype=np.float32,
    )
    _need(
        trials.ndim == 3 and trials.shape[1:] == (1024, UNITS)
        and trials.shape[0] >= GROWING_CAP + 1
        and int(dataset.calib_n_trials[spec.target_session]) == SUPPORT_TRIALS
        and len(dataset) > 0,
        "official M1 trialized activity/support topology drift",
    )
    _need(_file_sha(target) == spec.target_sha256, "official M1 target changed during parse")
    return dataset, trials, {
        "target_path": str(target),
        "target_sha256": spec.target_sha256,
        "manifest_relative": spec.manifest_relative,
        "manifest_sha256": _file_sha(manifest_path),
    }


def _load_model(root: Path, spec: OfficialM1FoldSpec, device: str) -> tuple[Any, str]:
    import torch
    from tfpd_exploration.src.cross_session_worst_group_v1 import source_physical

    checkpoint = root / spec.checkpoint_relative
    _need(
        checkpoint.is_file() and not checkpoint.is_symlink()
        and _file_sha(checkpoint) == spec.checkpoint_sha256,
        "official M1 checkpoint descriptor/body drift",
    )
    streaming_root = root / "streaming_calibration_exp"
    _need(streaming_root.is_dir(), "streaming calibration source root is absent")
    occupied = sys.modules.get("src")
    if occupied is not None:
        occupied_file = str(getattr(occupied, "__file__", ""))
        occupied_paths = tuple(str(item) for item in getattr(occupied, "__path__", ()))
        _need(
            str(streaming_root) in occupied_file or any(str(streaming_root) in item for item in occupied_paths),
            "top-level src namespace is occupied by a non-streaming package",
        )
    if str(streaming_root) not in sys.path:
        sys.path.insert(0, str(streaming_root))
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    hparams = payload.get("hyper_parameters", {})
    model = hparams.get("net")
    state = payload.get("state_dict", {})
    _need(
        callable(model)
        and hparams.get("task") == "m1"
        and hparams.get("decode_last_timestep_only") is True
        and hparams.get("predict_scaled_behavior") is False
        and float(hparams.get("behavior_scaling_factor")) == 1.0
        and isinstance(state, dict) and len(state) > 0
        and all(isinstance(key, str) and key.startswith("net.") for key in state),
        "official M1 checkpoint model/output contract drift",
    )
    stripped = {key.removeprefix("net."): value for key, value in state.items()}
    observed = model.load_state_dict(stripped, strict=True)
    _need(not observed.missing_keys and not observed.unexpected_keys, "official M1 strict state load drift")
    model = model.to(device).eval()
    state_sha = source_physical._state_digest(model)
    return model, state_sha


def _output_trial_indices(dataset: Any, session: str) -> tuple[int, ...]:
    starts = np.asarray(dataset.trial_start_indices[session], dtype=np.int64)
    result = []
    for current_session, start in dataset.window_indices:
        _need(current_session == session, "official M1 query session drift")
        result.append(int(np.searchsorted(starts, int(start) + WINDOW - 1, side="right") - 1))
    _need(len(result) == len(dataset), "official M1 output-trial mapping drift")
    return tuple(result)


def run_official_fold(root: Path, *, fold: int, device: str) -> dict[str, Any]:
    import torch
    from tfpd_exploration.src.cross_session_worst_group_v1 import source_physical

    root = Path(root).resolve()
    _need(fold in OFFICIAL_M1_FOLDS, "unsupported official M1 fold")
    _need(device.startswith("cuda") and torch.cuda.is_available(), "M1 breadth requires CUDA")
    spec = OFFICIAL_M1_FOLDS[fold]
    dataset, trials, authority = _load_dataset(root, spec)
    model, state_before = _load_model(root, spec, device)
    output_trials = _output_trial_indices(dataset, spec.target_session)
    cached = [
        _evaluate_arm(
            model=model,
            dataset=dataset,
            trial_activity=trials,
            output_trials=output_trials,
            arm=arm,
            device=device,
        )
        for arm in ARM_ORDER
    ]
    direct = _evaluate_direct_static(model, dataset, trials, device=device)
    max_abs = float(np.max(np.abs(
        direct["_prediction"].astype(np.float64) - cached[0]["_prediction"].astype(np.float64)
    )))
    r2_abs = abs(float(direct["r2"]) - float(cached[0]["r2"]))
    _need(max_abs <= 2.0e-6 and r2_abs <= 2.0e-7, "official M1 cached-identity parity failed")
    direct["cached_identity_parity"] = {
        "max_abs_prediction": max_abs,
        "abs_r2": r2_abs,
        "tolerance_prediction": 2.0e-6,
        "tolerance_r2": 2.0e-7,
        "pass": True,
    }
    results = [direct, *cached[1:]]
    for row in results:
        row.pop("_prediction", None)
        row.pop("_target", None)
    state_after = source_physical._state_digest(model)
    _need(state_before == state_after, "official M1 model state changed during breadth evaluation")
    return {
        "schema": "m1_official_activity_headroom_breadth_v1",
        "status": "COMPLETE_FROZEN_WEIGHT_ACTIVITY_HEADROOM",
        "dataset": "M1",
        "surface": f"held_in_calibration_official_fold_{fold}_{spec.target_session}",
        "device": device,
        "authority": {
            **authority,
            "fold": fold,
            "checkpoint_relative": spec.checkpoint_relative,
            "checkpoint_sha256": spec.checkpoint_sha256,
            "checkpoint_state_sha256": state_before,
        },
        "contract": {
            "support_trials": SUPPORT_TRIALS,
            "growing_cap": GROWING_CAP,
            "trial_count": int(trials.shape[0]),
            "target_optimizer_backward_update": 0,
            "full_session_oracle_is_noncausal": True,
            "official_checkpoint_breadth_only": True,
        },
        "results": results,
        "model_state_before_sha256": state_before,
        "model_state_after_sha256": state_after,
        "model_state_immutable": True,
        "cuda_initialized": bool(torch.cuda.is_initialized()),
    }


__all__ = ("OFFICIAL_M1_FOLDS", "OfficialM1FoldSpec", "run_official_fold")
