"""Aggregate Template-Ridge D-b held-out-selected DANDI688 runs."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

ARMS = ("t4", "ts4", "tr4", "trs4", "trls4", "trz4")
SIDE_BY_ARM = {
    "t4": "t4",
    "ts4": "ts4",
    "tr4": "tr4",
    "trs4": "trs4",
    "trls4": "trls4",
    "trz4": "trz4",
}
PAIRS = {
    "TR4-T4": ("tr4", "t4"),
    "TR4-TS4": ("tr4", "ts4"),
    "TR4-TRS4": ("tr4", "trs4"),
    "TR4-TRLS4": ("tr4", "trls4"),
    "TR4-TRZ4": ("tr4", "trz4"),
}


def parse_seeds(text: str) -> list[int]:
    seeds = [int(part.strip()) for part in text.split(",") if part.strip()]
    if not seeds:
        raise ValueError("at least one seed is required")
    if len(set(seeds)) != len(seeds):
        raise ValueError("duplicate seed")
    return seeds


def finite_float(value, *, field: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} is not finite: {value!r}")
    return result


def result_path(results_dir: Path, arm: str, seed: int) -> Path:
    out_name = f"template_ridge_db_heldout_spint_{arm}_s{seed}"
    return results_dir / f"p3_{out_name}_seed{seed}.json"


def load_run(results_dir: Path, arm: str, seed: int) -> dict:
    path = result_path(results_dir, arm, seed)
    if not path.is_file():
        raise FileNotFoundError(f"missing run summary for arm={arm} seed={seed}: {path}")
    data = json.loads(path.read_text())
    if data.get("variant") != "B3S":
        raise ValueError(f"{path}: expected variant B3S, got {data.get('variant')!r}")
    side = (data.get("side_features") or {}).get("group")
    if side != SIDE_BY_ARM[arm]:
        raise ValueError(f"{path}: expected side_features {SIDE_BY_ARM[arm]!r}, got {side!r}")
    selection = data.get("heldout_spint_selection") or {}
    if not selection.get("heldout_selected") or selection.get("heldout_backward_gradients") is not False:
        raise ValueError(f"{path}: run is not marked held-out-selected/gradient-free")
    if selection.get("checkpoint_monitor") != "val_heldout/r2_mean":
        raise ValueError(f"{path}: wrong checkpoint monitor {selection.get('checkpoint_monitor')!r}")
    if not data.get("held_out_test_evaluated"):
        raise ValueError(f"{path}: held-out test metrics were not written")
    metrics_list = data.get("test_metrics")
    if not isinstance(metrics_list, list) or not metrics_list:
        raise ValueError(f"{path}: missing test_metrics")
    metric_matches = [
        metrics
        for metrics in metrics_list
        if isinstance(metrics, dict) and "test_heldout/r2_mean" in metrics
    ]
    if not metric_matches:
        raise ValueError(f"{path}: missing test_heldout/r2_mean")
    metrics = metric_matches[-1]
    test_heldout_mean = finite_float(metrics.get("test_heldout/r2_mean"), field=f"{path}: test_heldout/r2_mean")
    val_heldout = finite_float(data.get("best_checkpoint_validation_r2"), field=f"{path}: best val heldout")
    test_heldin = finite_float(metrics.get("test_heldin/r2_mean"), field=f"{path}: test_heldin/r2_mean")
    sessions = data.get("session_splits", {}).get("test") or []
    per_session = {}
    for session in sessions:
        key = f"test_heldout_{session}/r2"
        per_session[session] = finite_float(metrics.get(key), field=f"{path}: {key}")
    return {
        "path": str(path),
        "arm": arm,
        "seed": seed,
        "val_heldout_r2_mean": val_heldout,
        "test_heldout_r2_mean": test_heldout_mean,
        "test_heldin_r2_mean": test_heldin,
        "per_session": per_session,
        "best_checkpoint": data.get("best_checkpoint"),
    }


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def sample_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = mean(values)
    return math.sqrt(sum((value - m) ** 2 for value in values) / (len(values) - 1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("sua_exploration/results"))
    parser.add_argument("--out-dir", type=Path, default=Path("sua_exploration/results/template_ridge_db_heldout_spint_v1"))
    parser.add_argument("--seeds", default="42,43,44")
    args = parser.parse_args()
    seeds = parse_seeds(args.seeds)
    rows = []
    by_arm_seed = {}
    for arm in ARMS:
        for seed in seeds:
            run = load_run(args.results_dir, arm, seed)
            by_arm_seed[(arm, seed)] = run
            for session, r2 in run["per_session"].items():
                rows.append({
                    "arm": arm,
                    "seed": seed,
                    "session": session,
                    "val_heldout_r2_mean": run["val_heldout_r2_mean"],
                    "test_heldout_r2_mean": run["test_heldout_r2_mean"],
                    "test_heldin_r2_mean": run["test_heldin_r2_mean"],
                    "test_heldout_session_r2": r2,
                    "best_checkpoint": run["best_checkpoint"],
                })
    arm_summary = {}
    for arm in ARMS:
        values = [by_arm_seed[(arm, seed)]["test_heldout_r2_mean"] for seed in seeds]
        arm_summary[arm] = {
            "mean_test_heldout_r2": mean(values),
            "std_test_heldout_r2": sample_std(values),
            "seed_values": {str(seed): by_arm_seed[(arm, seed)]["test_heldout_r2_mean"] for seed in seeds},
        }
    paired = {}
    sessions = sorted(by_arm_seed[("tr4", seeds[0])]["per_session"])
    for name, (left, right) in PAIRS.items():
        seed_deltas = [
            by_arm_seed[(left, seed)]["test_heldout_r2_mean"] - by_arm_seed[(right, seed)]["test_heldout_r2_mean"]
            for seed in seeds
        ]
        session_deltas = []
        for seed in seeds:
            if sorted(by_arm_seed[(left, seed)]["per_session"]) != sessions or sorted(by_arm_seed[(right, seed)]["per_session"]) != sessions:
                raise ValueError(f"session mismatch for pair {name} seed {seed}")
            for session in sessions:
                session_deltas.append(
                    by_arm_seed[(left, seed)]["per_session"][session]
                    - by_arm_seed[(right, seed)]["per_session"][session]
                )
        paired[name] = {
            "mean_delta": mean(seed_deltas),
            "std_delta": sample_std(seed_deltas),
            "seed_deltas": {str(seed): delta for seed, delta in zip(seeds, seed_deltas)},
            "positive_seed_count": sum(delta > 0.0 for delta in seed_deltas),
            "positive_session_seed_count": sum(delta > 0.0 for delta in session_deltas),
            "session_seed_count": len(session_deltas),
        }
    primary = paired["TR4-T4"]
    null_gate = paired["TR4-TRLS4"]
    promotion_gate = {
        "threshold": 0.03,
        "requires": "TR4-T4 >= +0.03 mean held-out R2, all seed means positive, all held-out session means positive; null gate TR4-TRLS4 same direction",
        "tr4_t4_pass": (
            primary["mean_delta"] >= 0.03
            and primary["positive_seed_count"] == len(seeds)
            and primary["positive_session_seed_count"] == primary["session_seed_count"]
        ),
        "null_gate_pass": (
            null_gate["mean_delta"] >= 0.03
            and null_gate["positive_seed_count"] == len(seeds)
            and null_gate["positive_session_seed_count"] == null_gate["session_seed_count"]
        ),
    }
    promotion_gate["pass"] = promotion_gate["tr4_t4_pass"] and promotion_gate["null_gate_pass"]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "TEMPLATE_RIDGE_DB_HELDOUT_SPINT_PER_SEED_SESSION.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    aggregate = {
        "schema_version": 1,
        "protocol": "template_ridge_db_heldout_spint_v1",
        "heldout_selected": True,
        "heldout_backward_gradients": False,
        "seeds": seeds,
        "arms": list(ARMS),
        "arm_summary": arm_summary,
        "paired_deltas": paired,
        "promotion_gate": promotion_gate,
        "per_seed_session_csv": str(csv_path),
    }
    json_path = args.out_dir / "TEMPLATE_RIDGE_DB_HELDOUT_SPINT_AGGREGATE.json"
    json_path.write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json_path)
    print(csv_path)


if __name__ == "__main__":
    main()
