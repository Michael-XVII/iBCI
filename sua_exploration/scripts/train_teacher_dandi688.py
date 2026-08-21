"""Train a DANDI 000688 multi-session SPINT teacher checkpoint."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from functools import partial
from pathlib import Path

import lightning.pytorch as pl
import torch
import torch.nn as nn
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from torchmetrics.regression import R2Score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mc_maze.multisession_datamodule import Dandi688MultiSessionDataModule

_sce_root = Path(__file__).resolve().parents[2] / "streaming_calibration_exp"
sys.path.insert(0, str(_sce_root))
from src.metrics.run_artifacts import assert_run_dir_is_fresh
from src.models.components.spint import SpintModel
from src.models.falcon_module import FalconLitModule


def parse_split_counts(text: str) -> tuple[int, int, int]:
    parts = [int(part.strip()) for part in text.split(",")]
    if len(parts) != 3 or any(part < 0 for part in parts):
        raise ValueError("split_counts must be three comma-separated non-negative integers")
    return parts[0], parts[1], parts[2]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def configure_multisession_metrics(model: FalconLitModule, dm: Dandi688MultiSessionDataModule) -> None:
    model.train_r2 = nn.ModuleDict(
        {name: R2Score(multioutput="variance_weighted") for name in dm.session_splits["train"]}
    )
    model.val_heldin_r2 = nn.ModuleDict(
        {name: R2Score(multioutput="variance_weighted") for name in dm.session_splits["val"]}
    )
    model.val_heldout_r2 = nn.ModuleDict(
        {name: R2Score(multioutput="variance_weighted") for name in dm.session_splits["val"]}
    )
    model.test_heldin_r2 = nn.ModuleDict(
        {name: R2Score(multioutput="variance_weighted") for name in dm.session_splits["test"]}
    )
    model.test_heldout_r2 = nn.ModuleDict(
        {name: R2Score(multioutput="variance_weighted") for name in dm.session_splits["test"]}
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/home/ial-mohd/dataset/ial-mohd/000688/sub-C")
    parser.add_argument("--task", type=str, default="CO", choices=["CO", "RT"])
    parser.add_argument("--split_counts", type=str, default="37,8,8")
    parser.add_argument("--out_name", type=str, default="teacher_dandi688_co_heldout_spint")
    parser.add_argument("--max_epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--calibration_n_trials", type=int, default=50)
    parser.add_argument("--limit_train_batches", type=float, default=None)
    parser.add_argument("--limit_val_batches", type=float, default=None)
    parser.add_argument("--limit_test_batches", type=float, default=None)
    parser.add_argument("--require_gpu", action="store_true")
    parser.add_argument("--accelerator", choices=["auto", "cpu", "gpu"], default="auto")
    parser.add_argument("--disable_progress_bar", action="store_true")
    args = parser.parse_args()

    if args.require_gpu and not torch.cuda.is_available():
        raise RuntimeError("--require_gpu was set but CUDA is unavailable")
    if args.accelerator == "gpu" and not torch.cuda.is_available():
        raise RuntimeError("--accelerator gpu was set but CUDA is unavailable")
    pl.seed_everything(args.seed, workers=True)

    data_dir = Path(args.data_dir).expanduser().resolve()
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")
    split_counts = parse_split_counts(args.split_counts)
    output_dir = Path(__file__).resolve().parents[1] / "checkpoints" / args.out_name
    results_dir = Path(__file__).resolve().parents[1] / "results"
    assert_run_dir_is_fresh(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    dm = Dandi688MultiSessionDataModule(
        data_dir=str(data_dir),
        task=args.task,
        split_counts=split_counts,
        batch_size=args.batch_size,
        window_size=50,
        calibration_n_trials=args.calibration_n_trials,
        max_trial_length=100,
        bin_size_ms=20,
        num_workers=args.num_workers,
        seed=args.seed,
        cache_dir=args.cache_dir,
        side_feature_group=None,
    )
    dm.setup("fit")

    net = SpintModel(
        model_dim=512,
        num_covariates=2,
        window_size=50,
        num_heads=64,
        num_layers=1,
        num_id_layers=3,
        use_learnable_id=True,
        learnable_id_type="mlp",
        learnable_rep=True,
        dropout_rate=0.0,
        dynamic_dropout=True,
        dynamic_dropout_low=0.0,
        dynamic_dropout_high=1.0,
        tf_drop_rate=0.1,
        readin_layer_type="mlp",
    )
    optimizer = partial(torch.optim.Adam, lr=5.0e-5, weight_decay=0.0)
    model = FalconLitModule(
        net=net,
        optimizer=optimizer,
        scheduler=None,
        compile=False,
        task="mc_maze",
        decode_last_timestep_only=True,
        predict_scaled_behavior=True,
        behavior_scaling_factor=5.0,
    )
    configure_multisession_metrics(model, dm)

    checkpoint_cb = ModelCheckpoint(
        dirpath=str(output_dir),
        filename="best-{epoch:03d}-{val_heldout/r2_mean:.4f}",
        monitor="val_heldout/r2_mean",
        mode="max",
        save_top_k=1,
    )
    callbacks = [checkpoint_cb, EarlyStopping(monitor="val_heldout/r2_mean", mode="max", patience=args.patience)]
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator=("gpu" if args.accelerator == "gpu" or (args.accelerator == "auto" and torch.cuda.is_available()) else "cpu"),
        devices=1,
        callbacks=callbacks,
        check_val_every_n_epoch=1,
        log_every_n_steps=50,
        default_root_dir=str(output_dir),
        deterministic=True,
        enable_progress_bar=not args.disable_progress_bar,
        limit_train_batches=args.limit_train_batches if args.limit_train_batches is not None else 1.0,
        limit_val_batches=args.limit_val_batches if args.limit_val_batches is not None else 1.0,
        limit_test_batches=args.limit_test_batches if args.limit_test_batches is not None else 1.0,
    )
    trainer.fit(model, datamodule=dm)
    test_metrics = trainer.test(model, datamodule=dm, ckpt_path=checkpoint_cb.best_model_path, verbose=False)

    metadata = {
        "schema_version": 1,
        "status": "completed",
        "created_at": datetime.now().astimezone().isoformat(),
        "purpose": "local_dandi688_teacher_for_template_ridge_db_experiment",
        "data_dir": str(data_dir),
        "task": args.task,
        "split_counts": list(split_counts),
        "seed": args.seed,
        "session_splits": dm.session_splits,
        "session_unit_counts": dm.session_unit_counts,
        "checkpoint_monitor": "val_heldout/r2_mean",
        "best_checkpoint": str(Path(checkpoint_cb.best_model_path).resolve()),
        "best_checkpoint_sha256": sha256_file(Path(checkpoint_cb.best_model_path)),
        "best_checkpoint_validation_r2": float(checkpoint_cb.best_model_score),
        "test_metrics": test_metrics,
        "heldout_selected": True,
        "heldout_backward_gradients": False,
        "training": {
            "max_epochs": args.max_epochs,
            "patience": args.patience,
            "batch_size": args.batch_size,
            "calibration_n_trials": args.calibration_n_trials,
            "limit_train_batches": args.limit_train_batches,
            "limit_val_batches": args.limit_val_batches,
            "limit_test_batches": args.limit_test_batches,
        },
    }
    write_json(output_dir / "run_metadata.json", metadata)
    write_json(results_dir / f"teacher_{args.out_name}_seed{args.seed}.json", metadata)
    print(f"Best teacher checkpoint: {checkpoint_cb.best_model_path}")
    print(f"Best val_heldout/r2_mean: {checkpoint_cb.best_model_score:.4f}")


if __name__ == "__main__":
    main()
