#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Run KOA ResNet50 + MESC for multiple seeds and retain only Test metrics.

The underlying trainer still performs normal training and validation so that
early stopping can select the best checkpoint. Its epoch-by-epoch output is
consumed but neither printed nor written to a log file. This wrapper prints one
Test row per seed and writes only those final rows to a CSV file.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import statistics
import subprocess
import sys
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = PROJECT_ROOT / "Knee" / "ResNet_layer3+MECS+CE.py"
DEFAULT_SEEDS = (42, 777, 1234, 2024, 3407)
METRIC_RE = re.compile(r"([A-Za-z0-9_]+)=(-?[0-9]+(?:\.[0-9]+)?)%?")
PERCENT_METRICS = {"top1", "top2", "top3", "acc", "bal_acc"}
CSV_FIELDS = (
    "seed",
    "loss",
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
SUMMARY_METRICS = ("acc", "bal_acc", "macro_f1", "qwk", "mae")


def parse_seeds(raw: str) -> List[int]:
    seeds = [int(value.strip()) for value in raw.split(",") if value.strip()]
    if not seeds:
        raise argparse.ArgumentTypeError("--seeds must contain at least one integer")
    if len(seeds) != len(set(seeds)):
        raise argparse.ArgumentTypeError("--seeds contains duplicate values")
    return seeds


def parse_metrics(line: str) -> Dict[str, float]:
    return {key.lower(): float(value) for key, value in METRIC_RE.findall(line)}


def python_command(args: argparse.Namespace) -> List[str]:
    conda_env = args.conda_env.strip()
    if conda_env.lower() not in {"", "none", "null", "false", "0"}:
        return ["conda", "run", "--no-capture-output", "-n", conda_env, "python", "-u", str(TRAIN_SCRIPT)]
    return [args.python, "-u", str(TRAIN_SCRIPT)]


def build_env(seed: int, args: argparse.Namespace, run_id: str) -> Dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "PYTHONUNBUFFERED": "1",
            "GLOBAL_EXPERIMENT_SEED": str(seed),
            "KNEE_SEED": str(seed),
            "KNEE_EPOCHS": str(args.epochs),
            "KNEE_PATIENCE": str(args.patience),
            "KNEE_RUN_TAG": f"test_only_{run_id}_koa_resnet50_plus_mesc_seed{seed}",
        }
    )
    if args.batch_size is not None:
        env["KNEE_BATCH_SIZE"] = str(args.batch_size)
    if args.image_size is not None:
        env["KNEE_IMAGE_SIZE"] = str(args.image_size)
    if args.num_workers is not None:
        env["KNEE_NUM_WORKERS"] = str(args.num_workers)
    if args.early_delta is not None:
        env["KNEE_EARLY_DELTA"] = str(args.early_delta)
    return env


def run_seed(seed: int, args: argparse.Namespace, run_id: str) -> Optional[Dict[str, float]]:
    cmd = python_command(args)
    env = build_env(seed, args, run_id)
    tail: deque[str] = deque(maxlen=40)
    test_metrics: Optional[Dict[str, float]] = None
    waiting_for_test_auc = False

    print(f"[RUN] seed={seed} (epochs={args.epochs}, patience={args.patience})", flush=True)
    try:
        process = subprocess.Popen(
            cmd,
            cwd=PROJECT_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
    except OSError as exc:
        print(f"[FAIL] seed={seed}: could not start trainer: {exc}", file=sys.stderr)
        return None

    assert process.stdout is not None
    for raw_line in process.stdout:
        line = raw_line.rstrip("\r\n")
        tail.append(line)
        stripped = line.strip()

        # Match the standard test set only. In particular, do not accept AutoTest.
        if stripped.startswith("Test |"):
            test_metrics = parse_metrics(stripped)
            waiting_for_test_auc = True
            continue
        if waiting_for_test_auc and "ovr_roc_auc_macro=" in stripped:
            assert test_metrics is not None
            test_metrics.update(parse_metrics(stripped))
            waiting_for_test_auc = False
        elif stripped:
            waiting_for_test_auc = False

    return_code = process.wait()
    required = {"acc", "bal_acc", "macro_f1", "qwk", "mae"}
    if return_code != 0 or test_metrics is None or not required.issubset(test_metrics):
        reason = f"return code {return_code}" if return_code else "standard Test metrics were not found"
        print(f"[FAIL] seed={seed}: {reason}", file=sys.stderr)
        print("------ trainer output tail ------", file=sys.stderr)
        for line in tail:
            print(line, file=sys.stderr)
        print("---------------------------------", file=sys.stderr)
        return None

    return test_metrics


def format_metric(key: str, value: float) -> str:
    if key in PERCENT_METRICS:
        return f"{value:.2f}%"
    return f"{value:.4f}"


def print_result(seed: int, metrics: Dict[str, float]) -> None:
    ordered = (
        "loss",
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
    values = [f"{key}={format_metric(key, metrics[key])}" for key in ordered if key in metrics]
    print(f"[TEST] seed={seed} | " + " | ".join(values), flush=True)


def write_results(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        return
    print("\nSummary (mean +/- sample SD)")
    for key in SUMMARY_METRICS:
        values = [float(row[key]) for row in rows]
        mean = statistics.mean(values)
        sd = statistics.stdev(values) if len(values) > 1 else 0.0
        suffix = "%" if key in PERCENT_METRICS else ""
        print(f"  {key}: {mean:.4f} +/- {sd:.4f}{suffix}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--early-delta", type=float, default=None)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--conda-env", default="none", help="Use 'none' inside an already activated environment.")
    parser.add_argument("--output", type=Path, default=None, help="CSV path; defaults to analysis_tables/test_only_<timestamp>.csv")
    args = parser.parse_args()

    if args.epochs <= 0:
        parser.error("--epochs must be positive")
    if args.patience <= 0:
        parser.error("--patience must be positive")
    try:
        seeds = parse_seeds(args.seeds)
    except (ValueError, argparse.ArgumentTypeError) as exc:
        parser.error(str(exc))

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = args.output or PROJECT_ROOT / "analysis_tables" / f"koa_mesc_test_only_{run_id}.csv"
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path

    rows: List[Dict[str, object]] = []
    for seed in seeds:
        metrics = run_seed(seed, args, run_id)
        if metrics is None:
            print("Stopped after failure; completed Test rows have still been saved.", file=sys.stderr)
            write_results(output_path, rows)
            print(f"Results: {output_path}")
            return 1
        row: Dict[str, object] = {"seed": seed, **metrics}
        rows.append(row)
        print_result(seed, metrics)
        write_results(output_path, rows)

    print_summary(rows)
    print(f"\nResults: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
