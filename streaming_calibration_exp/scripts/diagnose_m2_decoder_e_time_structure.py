#!/usr/bin/env python
"""Frozen-consumer E-time-structure diagnostic for M2 decoder-lane experiments."""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import hydra
import rootutils
import torch
from omegaconf import OmegaConf

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.models.streaming_calibration_module import StreamingCalibrationLitModule


@dataclass
class R2Accumulator:
    n: int = 0
    sum_y: torch.Tensor | None = None
    sum_y2: torch.Tensor | None = None
    sse: torch.Tensor | None = None

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        pred = pred.detach().cpu().float().reshape(-1, pred.shape[-1])
        target = target.detach().cpu().float().reshape(-1, target.shape[-1])
        if self.sum_y is None:
            width = target.shape[-1]
            self.sum_y = torch.zeros(width, dtype=torch.float64)
            self.sum_y2 = torch.zeros(width, dtype=torch.float64)
            self.sse = torch.zeros(width, dtype=torch.float64)
        assert self.sum_y2 is not None and self.sse is not None
        target64 = target.double()
        pred64 = pred.double()
        self.n += int(target64.shape[0])
        self.sum_y += target64.sum(dim=0)
        self.sum_y2 += (target64 * target64).sum(dim=0)
        residual = target64 - pred64
        self.sse += (residual * residual).sum(dim=0)

    def compute(self) -> float:
        if self.n <= 1 or self.sum_y is None or self.sum_y2 is None or self.sse is None:
            return float("nan")
        sst = self.sum_y2 - (self.sum_y * self.sum_y) / max(self.n, 1)
        denom = torch.clamp(sst.sum(), min=1.0e-12)
        return float(1.0 - self.sse.sum() / denom)


def identity_variants(identity: torch.Tensor, *, permutation_seed: int) -> dict[str, torch.Tensor]:
    """Return matched E perturbations for the decoder-lane waveform diagnostic."""
    if identity.ndim != 3:
        raise ValueError(f"Expected identity [B,N,W], got {tuple(identity.shape)}")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(permutation_seed))
    order = torch.randperm(identity.shape[-1], generator=generator, device="cpu").to(identity.device)
    return {
        "original": identity,
        "mean_time": identity.mean(dim=-1, keepdim=True).expand_as(identity),
        "permute_time": identity.index_select(-1, order),
    }


def _git_text(args: list[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], text=True).strip()
    except Exception:
        return ""


def _checkpoint_from_artifact(artifact_dir: Path) -> Path:
    manifest_path = artifact_dir / "checkpoint_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        ckpt = Path(manifest.get("artifact_checkpoint_path") or "")
        if ckpt.exists():
            return ckpt
        ckpt = Path(manifest.get("source_checkpoint_path") or "")
        if ckpt.exists():
            return ckpt
    ckpt = artifact_dir / "checkpoints" / "best.ckpt"
    if ckpt.exists():
        return ckpt
    last_ckpt = artifact_dir / "checkpoints" / "best_ckpt" / "last.ckpt"
    if last_ckpt.exists():
        checkpoint = torch.load(last_ckpt, map_location="cpu", weights_only=False)
        for callback_state in checkpoint.get("callbacks", {}).values():
            if not isinstance(callback_state, dict):
                continue
            if callback_state.get("monitor") != "val_heldout/r2_mean":
                continue
            best_model_path = callback_state.get("best_model_path")
            if best_model_path and Path(best_model_path).exists():
                return Path(best_model_path)
        candidates = sorted((artifact_dir / "checkpoints" / "best_ckpt").glob("epoch_*.ckpt"))
        if candidates:
            return candidates[-1]
    raise FileNotFoundError(f"No best checkpoint found under {artifact_dir}")


def _config_from_artifact(artifact_dir: Path) -> Path:
    for relative in ("resolved_config.yaml", ".hydra/config.yaml"):
        cfg_path = artifact_dir / relative
        if cfg_path.exists():
            return cfg_path
    raise FileNotFoundError(f"Missing resolved_config.yaml or .hydra/config.yaml in {artifact_dir}")


def _move_batch(batch: tuple, device: torch.device) -> tuple:
    moved = []
    for item in batch:
        moved.append(item.to(device, non_blocking=True) if isinstance(item, torch.Tensor) else item)
    return tuple(moved)


def _batch_parts(batch: tuple) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Iterable[str], torch.Tensor | None, torch.Tensor | None]:
    if len(batch) == 6:
        neural, target, calib, session_name, side_features, electrode_ids = batch
    elif len(batch) == 5:
        neural, target, calib, session_name, side_features = batch
        electrode_ids = None
    elif len(batch) == 4:
        neural, target, calib, session_name = batch
        side_features = None
        electrode_ids = None
    else:
        raise ValueError(f"Unexpected batch length {len(batch)}")
    return neural, target, calib, session_name, side_features, electrode_ids


def evaluate_artifact(
    artifact_dir: Path,
    *,
    data_dir: str | None,
    include_heldout: bool,
    device: torch.device,
    permutation_seed: int,
    limit_batches: int | None,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    cfg = OmegaConf.load(_config_from_artifact(artifact_dir))
    if data_dir is not None:
        cfg.data.data_dir = data_dir
    cfg.data.include_heldout_in_fit = False
    cfg.data.include_heldout_in_test = bool(include_heldout)
    cfg.data.num_workers = 0
    cfg.data.pin_memory = False

    ckpt_path = _checkpoint_from_artifact(artifact_dir)
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    module = StreamingCalibrationLitModule(**checkpoint["hyper_parameters"])
    module.setup("test")
    module.load_state_dict(checkpoint["state_dict"], strict=True)
    if module.student is None:
        raise RuntimeError("Streaming student was not initialized")
    if module.student.decoder_mode != "coupled":
        raise ValueError("E-time diagnostic currently supports coupled decoder checkpoints only")
    module.eval().to(device)

    datamodule = hydra.utils.instantiate(cfg.data)
    datamodule.setup("test")
    loaders = datamodule.test_dataloader()
    if not isinstance(loaders, list):
        loaders = [loaders]
    split_names = ["test_heldin", "test_heldout"][: len(loaders)]

    acc: dict[tuple[str, str, str], R2Accumulator] = defaultdict(R2Accumulator)
    with torch.no_grad():
        for split_name, loader in zip(split_names, loaders):
            for batch_idx, batch in enumerate(loader):
                if limit_batches is not None and batch_idx >= limit_batches:
                    break
                batch = _move_batch(batch, device)
                neural, target, calib, session_names, side_features, electrode_ids = _batch_parts(batch)
                session_list = list(session_names)
                if len(set(session_list)) != 1:
                    raise ValueError("Expected session-homogeneous batches")
                session = session_list[0]
                identity = module.student.compute_identity(
                    calib, side_features=side_features, electrode_ids=electrode_ids
                )
                for variant_name, e_variant in identity_variants(identity, permutation_seed=permutation_seed).items():
                    pred = module.student.decode_with_identity(neural, e_variant)
                    pred, sliced_target = module._slice_last_timestep(pred, target)
                    acc[(split_name, "__mean__", variant_name)].update(pred, sliced_target)
                    acc[(split_name, session, variant_name)].update(pred, sliced_target)

    rows: list[dict[str, object]] = []
    for (split_name, session, variant_name), metric in sorted(acc.items()):
        rows.append({
            "artifact_dir": str(artifact_dir),
            "run_id": str(cfg.get("run_id", artifact_dir.name)),
            "arm": str(cfg.data.get("side_feature_group", "none")),
            "variant": str(cfg.model.get("variant", "")),
            "seed": int(cfg.get("seed", 0)),
            "fold_id": cfg.data.get("loso_fold"),
            "split": split_name,
            "session": session,
            "e_condition": variant_name,
            "n_samples": metric.n,
            "r2_variance_weighted": metric.compute(),
        })

    originals = {
        (row["split"], row["session"]): row["r2_variance_weighted"]
        for row in rows if row["e_condition"] == "original"
    }
    for row in rows:
        base = originals.get((row["split"], row["session"]))
        row["delta_vs_original"] = (
            float(row["r2_variance_weighted"]) - float(base) if base is not None else float("nan")
        )

    meta = {
        "artifact_dir": str(artifact_dir),
        "checkpoint_path": str(ckpt_path),
        "data_dir": str(cfg.data.data_dir),
        "include_heldout": include_heldout,
        "permutation_seed": permutation_seed,
        "split_manifest": json.loads((artifact_dir / "split_manifest.json").read_text()) if (artifact_dir / "split_manifest.json").exists() else {},
    }
    return rows, meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", action="append", required=True, help="Training artifact directory; repeat for matched arms.")
    parser.add_argument("--data-dir", default=None, help="Override data.data_dir from the saved config.")
    parser.add_argument("--include-heldout", action="store_true", help="Also evaluate official held-out-calib as test_heldout.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--permutation-seed", type=int, default=20260813)
    parser.add_argument("--limit-batches", type=int, default=None)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, object]] = []
    artifacts_meta = []
    for item in args.artifact_dir:
        rows, meta = evaluate_artifact(
            Path(item),
            data_dir=args.data_dir,
            include_heldout=args.include_heldout,
            device=device,
            permutation_seed=args.permutation_seed,
            limit_batches=args.limit_batches,
        )
        all_rows.extend(rows)
        artifacts_meta.append(meta)

    csv_path = output_dir / "M2_DECODER_E_TIME_STRUCTURE_DIAGNOSTIC.csv"
    fields = [
        "artifact_dir", "run_id", "arm", "variant", "seed", "fold_id", "split",
        "session", "e_condition", "n_samples", "r2_variance_weighted", "delta_vs_original",
    ]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)

    json_path = output_dir / "M2_DECODER_E_TIME_STRUCTURE_DIAGNOSTIC.json"
    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "branch": _git_text(["branch", "--show-current"]),
        "git_sha": _git_text(["rev-parse", "HEAD"]),
        "device": str(device),
        "limit_batches": args.limit_batches,
        "artifacts": artifacts_meta,
        "rows": all_rows,
    }
    json_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {csv_path}")
    print(f"wrote {json_path}")


if __name__ == "__main__":
    main()
