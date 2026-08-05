#!/usr/bin/env python
"""Rerun ADNI non-MESC baselines with the MESC-matched optimizer schedule.

The default ``core`` suite runs CE, DAST, and SORD for five seeds (15 runs).
The ``all`` suite additionally runs Label Smoothing, CORAL, and CORN (30 runs).
Only final Test metrics are written to the result CSV.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import statistics
import subprocess
import sys
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASELINE_TRAINER = PROJECT_ROOT / "Alzheimer_MRI_Loss" / "ResNet_baseline.py"
ORDINAL_TRAINER = PROJECT_ROOT / "tools" / "train_ordinal_baseline.py"
DEFAULT_SEEDS = (1, 4, 5, 6, 9)
PERCENT_METRICS = {"top1", "top2", "top3", "acc", "bal_acc"}
TEST_METRICS = (
    "test_loss",
    "top1",
    "top2",
    "top3",
    "acc",
    "bal_acc",
    "macro_f1",
    "qwk",
    "mae",
    "weighted_f1",
    "precision_macro",
    "recall_macro",
    "ovr_roc_auc_macro",
    "ovr_pr_auc_macro",
)
METRIC_RE = re.compile(r"([A-Za-z0-9_]+)=(-?[0-9]+(?:\.[0-9]+)?)%?")
CHECKPOINT_RE = re.compile(r"^Checkpoint:\s*(.+\.pt)\s*$")


@dataclass(frozen=True)
class Experiment:
    key: str
    method: str
    criterion: str
    trainer: str


EXPERIMENTS = {
    "ce": Experiment("ce", "ResNet50", "ce", "baseline"),
    "dast": Experiment("dast", "ResNet50 + DAST", "dast", "baseline"),
    "sord_ce": Experiment("sord_ce", "ResNet50 + SORD", "sord_ce", "baseline"),
    "label_smoothing_ce": Experiment(
        "label_smoothing_ce",
        "ResNet50 + Label Smoothing",
        "label_smoothing_ce",
        "baseline",
    ),
    "coral": Experiment("coral", "ResNet50 + CORAL", "coral", "ordinal"),
    "corn": Experiment("corn", "ResNet50 + CORN", "corn", "ordinal"),
}
CORE_EXPERIMENTS = ("ce", "dast", "sord_ce")
ALL_EXPERIMENTS = (*CORE_EXPERIMENTS, "label_smoothing_ce", "coral", "corn")
CSV_FIELDS = (
    "dataset",
    "experiment",
    "method",
    "criterion",
    "seed",
    "optimizer_profile",
    "base_lr",
    "epochs",
    "patience",
    *TEST_METRICS,
    "checkpoint",
    "run_tag",
    "completed_at",
)


def parse_seeds(raw: str) -> list[int]:
    seeds = [int(value.strip()) for value in raw.split(",") if value.strip()]
    if not seeds:
        raise ValueError("at least one seed is required")
    if len(seeds) != len(set(seeds)):
        raise ValueError("seeds contain duplicates")
    return seeds


def parse_experiments(raw: str) -> list[Experiment]:
    normalized = raw.strip().lower()
    if normalized == "core":
        keys = list(CORE_EXPERIMENTS)
    elif normalized == "all":
        keys = list(ALL_EXPERIMENTS)
    else:
        keys = [value.strip().lower() for value in raw.split(",") if value.strip()]
    if not keys:
        raise ValueError("at least one experiment is required")
    unknown = [key for key in keys if key not in EXPERIMENTS]
    if unknown:
        raise ValueError(f"unknown experiments: {unknown}; choices: {list(EXPERIMENTS)}")
    if len(keys) != len(set(keys)):
        raise ValueError("experiments contain duplicates")
    return [EXPERIMENTS[key] for key in keys]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    parser.add_argument(
        "--experiments",
        default="core",
        help="core, all, or comma-separated keys: ce,dast,sord_ce,label_smoothing_ce,coral,corn",
    )
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--early-delta", type=float, default=1e-4)
    parser.add_argument("--base-lr", type=float, default=1e-4)
    parser.add_argument("--dast-tau", type=float, default=1.0)
    parser.add_argument("--dast-gamma", type=float, default=1.5)
    parser.add_argument("--sord-tau", type=float, default=1.0)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--conda-env", default="none")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--tag-prefix", default="")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--skip-runs", type=int, default=0)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--verbose-trainer", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    try:
        args.seed_values = parse_seeds(args.seeds)
        args.experiment_values = parse_experiments(args.experiments)
    except ValueError as exc:
        parser.error(str(exc))

    args.data_root = args.data_root or os.getenv("ALZHEIMER_DATA_ROOT")
    if not args.data_root:
        parser.error("provide --data-root or set ALZHEIMER_DATA_ROOT")
    args.data_root = Path(args.data_root).expanduser().resolve()

    positive = (
        args.epochs,
        args.patience,
        args.batch_size,
        args.image_size,
        args.base_lr,
        args.dast_tau,
        args.sord_tau,
    )
    if any(value <= 0 for value in positive):
        parser.error("epochs, patience, sizes, base LR, and tau values must be positive")
    if args.num_workers < 0 or args.skip_runs < 0 or args.early_delta < 0 or args.dast_gamma < 0:
        parser.error("workers, skip-runs, early-delta, and DAST gamma cannot be negative")
    if not 0 <= args.label_smoothing < 1:
        parser.error("--label-smoothing must satisfy 0 <= value < 1")
    return args


def python_prefix(args: argparse.Namespace) -> list[str]:
    env_name = args.conda_env.strip()
    if env_name.lower() not in {"", "none", "null", "false", "0"}:
        return ["conda", "run", "--no-capture-output", "-n", env_name, "python", "-u"]
    return [args.python, "-u"]


def build_env(args: argparse.Namespace, experiment: Experiment, seed: int, tag: str) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "PYTHONUNBUFFERED": "1",
            "GLOBAL_EXPERIMENT_SEED": str(seed),
            "ALZHEIMER_SEED": str(seed),
            "ALZHEIMER_DATA_ROOT": str(args.data_root),
            "ALZHEIMER_EPOCHS": str(args.epochs),
            "ALZHEIMER_PATIENCE": str(args.patience),
            "ALZHEIMER_BATCH_SIZE": str(args.batch_size),
            "ALZHEIMER_IMAGE_SIZE": str(args.image_size),
            "ALZHEIMER_NUM_WORKERS": str(args.num_workers),
            "ALZHEIMER_EARLY_DELTA": str(args.early_delta),
            "ALZHEIMER_BASE_LR": str(args.base_lr),
            "ALZHEIMER_DAST_TAU": str(args.dast_tau),
            "ALZHEIMER_DAST_GAMMA": str(args.dast_gamma),
            "ALZHEIMER_SORD_TAU": str(args.sord_tau),
            "ALZHEIMER_LABEL_SMOOTHING": str(args.label_smoothing),
            "ALZHEIMER_RUN_TAG": tag,
            "ALZHEIMER_BASELINE_LR_PROFILE": "matched_mesc",
            "ALZHEIMER_LOSSES": experiment.criterion,
        }
    )
    omp_threads = env.get("OMP_NUM_THREADS", "").strip()
    if not omp_threads.isdigit() or int(omp_threads) <= 0:
        env["OMP_NUM_THREADS"] = str(max(1, args.num_workers))
    return env


def build_command(args: argparse.Namespace, experiment: Experiment, seed: int, tag: str) -> list[str]:
    prefix = python_prefix(args)
    if experiment.trainer == "baseline":
        return [*prefix, str(BASELINE_TRAINER)]
    return [
        *prefix,
        str(ORDINAL_TRAINER),
        "--dataset",
        "adni",
        "--method",
        experiment.criterion,
        "--data-root",
        str(args.data_root),
        "--seed",
        str(seed),
        "--epochs",
        str(args.epochs),
        "--patience",
        str(args.patience),
        "--batch-size",
        str(args.batch_size),
        "--image-size",
        str(args.image_size),
        "--num-workers",
        str(args.num_workers),
        "--early-delta",
        str(args.early_delta),
        "--optimizer-profile",
        "matched_mesc",
        "--base-lr",
        str(args.base_lr),
        "--run-tag",
        tag,
    ]


def parse_test_metrics(line: str) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for name, raw_value in METRIC_RE.findall(line):
        value = float(raw_value)
        if name in PERCENT_METRICS:
            value /= 100.0
        metrics["test_loss" if name == "loss" else name] = value
    return metrics


def expected_checkpoint(experiment: Experiment, tag: str) -> Path:
    checkpoint_dir = PROJECT_ROOT / "Alzheimer_MRI_Loss" / "checkpoints"
    if experiment.trainer == "ordinal":
        return checkpoint_dir / f"best_resnet50_{experiment.criterion}_{tag}.pt"
    return checkpoint_dir / f"best_ResNet_baseline_{experiment.criterion}_{tag}.pt"


def write_rows(path: Path, rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def run_one(
    args: argparse.Namespace,
    experiment: Experiment,
    seed: int,
    tag: str,
) -> dict[str, object] | None:
    command = build_command(args, experiment, seed, tag)
    env = build_env(args, experiment, seed, tag)
    print(f"[RUN] ADNI | {experiment.method} | seed={seed}", flush=True)
    if args.dry_run:
        print("      " + " ".join(command), flush=True)
        return {}

    process = subprocess.Popen(
        command,
        cwd=PROJECT_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    tail: deque[str] = deque(maxlen=60)
    test_metrics: dict[str, float] | None = None
    checkpoint = expected_checkpoint(experiment, tag)
    assert process.stdout is not None
    for raw_line in process.stdout:
        line = raw_line.rstrip("\r\n")
        tail.append(line)
        if args.verbose_trainer:
            print(line, flush=True)
        stripped = line.strip()
        if stripped.startswith("Test |"):
            parsed = parse_test_metrics(stripped)
            if parsed:
                test_metrics = parsed
        elif test_metrics is not None and (
            "ovr_roc_auc_macro=" in stripped or "ovr_pr_auc_macro=" in stripped
        ):
            test_metrics.update(parse_test_metrics(stripped))
        checkpoint_match = CHECKPOINT_RE.match(stripped)
        if checkpoint_match:
            checkpoint = Path(checkpoint_match.group(1))

    return_code = process.wait()
    required = {"test_loss", "acc", "bal_acc", "macro_f1", "qwk", "mae"}
    if return_code != 0 or test_metrics is None or not required.issubset(test_metrics):
        reason = f"return code {return_code}" if return_code else "final Test metrics were not found"
        print(f"[FAIL] {experiment.key}, seed={seed}: {reason}", file=sys.stderr)
        print("------ trainer output tail ------", file=sys.stderr)
        for line in tail:
            print(line, file=sys.stderr)
        print("---------------------------------", file=sys.stderr)
        return None

    row: dict[str, object] = {
        "dataset": "ADNI",
        "experiment": experiment.key,
        "method": experiment.method,
        "criterion": experiment.criterion,
        "seed": seed,
        "optimizer_profile": "matched_mesc",
        "base_lr": args.base_lr,
        "epochs": args.epochs,
        "patience": args.patience,
        **test_metrics,
        "checkpoint": str(checkpoint),
        "run_tag": tag,
        "completed_at": datetime.now().isoformat(timespec="seconds"),
    }
    display = " | ".join(
        f"{name}={float(row[name]):.4f}"
        for name in ("acc", "bal_acc", "macro_f1", "qwk", "mae")
    )
    print(f"[TEST] {experiment.key} | seed={seed} | {display}", flush=True)
    return row


def print_summary(rows: Sequence[dict[str, object]]) -> None:
    groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[str(row["experiment"])].append(row)
    if not groups:
        return
    print("\nSummary (mean +/- sample SD)")
    for key in ALL_EXPERIMENTS:
        group = groups.get(key)
        if not group:
            continue
        print(f"  {key} (n={len(group)})")
        for metric in ("acc", "bal_acc", "macro_f1", "qwk", "mae"):
            values = [float(row[metric]) for row in group]
            sd = statistics.stdev(values) if len(values) > 1 else 0.0
            print(f"    {metric}: {statistics.mean(values):.4f} +/- {sd:.4f}")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.dry_run and not args.data_root.is_dir():
        raise FileNotFoundError(f"ADNI data root not found: {args.data_root}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = args.tag_prefix or f"adni_matched_baselines_{timestamp}"
    output = args.output or PROJECT_ROOT / "analysis_tables" / f"{prefix}_test_results.csv"
    if not output.is_absolute():
        output = (PROJECT_ROOT / output).resolve()

    total = len(args.seed_values) * len(args.experiment_values)
    print(f"Experiments : {[experiment.key for experiment in args.experiment_values]}")
    print(f"Seeds       : {args.seed_values}")
    print(f"Runs        : {total}")
    print("LR profile  : matched_mesc")
    print(f"Results     : {output}")

    rows: list[dict[str, object]] = []
    scheduled = 0
    for seed in args.seed_values:
        for experiment in args.experiment_values:
            scheduled += 1
            if scheduled <= args.skip_runs:
                print(f"[SKIP] {experiment.key} | seed={seed}")
                continue
            tag = f"{prefix}_{experiment.key}_seed{seed}"
            row = run_one(args, experiment, seed, tag)
            if args.dry_run:
                continue
            if row is None:
                write_rows(output, rows)
                if not args.continue_on_error:
                    print(f"Stopped after failure. Completed rows: {output}", file=sys.stderr)
                    return 1
                continue
            rows.append(row)
            write_rows(output, rows)

    if not args.dry_run:
        print_summary(rows)
        print(f"Results: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
