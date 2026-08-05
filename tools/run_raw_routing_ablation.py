#!/usr/bin/env python
"""Run the strict raw-logit vs median-anchored routing control on KOA and ADNI.

This runner launches only the missing raw-logit arm with fixed DAST.  Existing
MESC-A+DAST and median-anchored MESC-B+DAST results can then be compared on the
same seeds.  Child training output is consumed; only final standard Test rows
are printed and written to CSV.
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
from typing import Iterable, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ADAPTER = PROJECT_ROOT / "tools" / "run_training_with_mecsb.py"
TRAIN_SCRIPTS = {
    "koa": PROJECT_ROOT / "Knee" / "ResNet_layer3+MECS+Loss4.py",
    "adni": PROJECT_ROOT / "Alzheimer_MRI_Loss" / "ResNet_layer3+MECS.py",
}
DEFAULT_SEEDS = (1, 9, 6, 5, 4)
METRIC_RE = re.compile(r"([A-Za-z0-9_]+)=(-?[0-9]+(?:\.[0-9]+)?)%?")
PERCENT_METRICS = {"top1", "top2", "top3", "acc", "bal_acc"}
CSV_FIELDS = (
    "dataset",
    "method",
    "attention",
    "loss",
    "seed",
    "tau",
    "gamma",
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
    "checkpoint",
    "run_tag",
    "completed_at",
)


def comma_list(raw: str) -> list[str]:
    return [part.strip().lower() for part in raw.split(",") if part.strip()]


def parse_seeds(raw: str) -> list[int]:
    seeds = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not seeds:
        raise ValueError("at least one seed is required")
    if len(seeds) != len(set(seeds)):
        raise ValueError("seeds contain duplicates")
    return seeds


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", default="koa,adni")
    parser.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    parser.add_argument("--koa-data-root", type=Path, default=None)
    parser.add_argument("--adni-data-root", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--early-delta", type=float, default=None)
    parser.add_argument("--dast-tau", type=float, default=1.0)
    parser.add_argument("--dast-gamma", type=float, default=1.5)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--conda-env", default="none")
    parser.add_argument("--tag-prefix", default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--skip-runs", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    args.dataset_names = comma_list(args.datasets)
    if not args.dataset_names or any(name not in TRAIN_SCRIPTS for name in args.dataset_names):
        parser.error("--datasets supports koa, adni, or koa,adni.")
    if len(args.dataset_names) != len(set(args.dataset_names)):
        parser.error("--datasets contains duplicates.")
    try:
        args.seed_values = parse_seeds(args.seeds)
    except ValueError as exc:
        parser.error(str(exc))
    if args.epochs <= 0 or args.patience <= 0 or args.skip_runs < 0:
        parser.error("epochs/patience must be positive and skip-runs cannot be negative.")
    if args.dast_tau <= 0 or args.dast_gamma < 0:
        parser.error("DAST tau must be positive and gamma cannot be negative.")

    args.koa_data_root = args.koa_data_root or os.getenv("KNEE_DATA_ROOT")
    args.adni_data_root = args.adni_data_root or os.getenv("ALZHEIMER_DATA_ROOT")
    for name in args.dataset_names:
        attr = f"{name}_data_root"
        value = getattr(args, attr)
        if not value:
            parser.error(f"{name.upper()} requires --{name}-data-root or its data-root environment variable.")
        setattr(args, attr, Path(value).expanduser().resolve())
    return args


def validate_data(args: argparse.Namespace) -> None:
    if "koa" in args.dataset_names:
        missing = [name for name in ("train", "val", "test") if not (args.koa_data_root / name).is_dir()]
        if missing:
            raise FileNotFoundError(f"KOA root {args.koa_data_root} is missing {missing}.")
    if "adni" in args.dataset_names and not args.adni_data_root.is_dir():
        raise FileNotFoundError(f"ADNI root not found: {args.adni_data_root}")


def python_prefix(args: argparse.Namespace) -> list[str]:
    env_name = args.conda_env.strip()
    if env_name.lower() not in {"", "none", "null", "false", "0"}:
        return ["conda", "run", "--no-capture-output", "-n", env_name, "python", "-u"]
    return [args.python, "-u"]


def build_command(dataset_name: str, args: argparse.Namespace) -> list[str]:
    return python_prefix(args) + [
        str(ADAPTER),
        "--variant",
        "raw",
        "--script",
        str(TRAIN_SCRIPTS[dataset_name]),
    ]


def build_env(dataset_name: str, seed: int, tag: str, args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    omp = env.get("OMP_NUM_THREADS", "").strip()
    if not omp.isdigit() or int(omp) <= 0:
        env["OMP_NUM_THREADS"] = str(max(1, args.num_workers or 4))
    prefix = "KNEE" if dataset_name == "koa" else "ALZHEIMER"
    data_root = args.koa_data_root if dataset_name == "koa" else args.adni_data_root
    env["GLOBAL_EXPERIMENT_SEED"] = str(seed)
    env[f"{prefix}_SEED"] = str(seed)
    env[f"{prefix}_DATA_ROOT"] = str(data_root)
    env[f"{prefix}_EPOCHS"] = str(args.epochs)
    env[f"{prefix}_PATIENCE"] = str(args.patience)
    env[f"{prefix}_RUN_TAG"] = tag
    env[f"{prefix}_DAST_TAU"] = str(args.dast_tau)
    env[f"{prefix}_DAST_GAMMA"] = str(args.dast_gamma)
    env["MESC_IMPLEMENTATION"] = "MECS_RawRouting"
    if dataset_name == "adni":
        env["ALZHEIMER_LOSSES"] = "dast"
    if args.batch_size is not None:
        env[f"{prefix}_BATCH_SIZE"] = str(args.batch_size)
    if args.image_size is not None:
        env[f"{prefix}_IMAGE_SIZE"] = str(args.image_size)
    if args.num_workers is not None:
        env[f"{prefix}_NUM_WORKERS"] = str(args.num_workers)
    if args.early_delta is not None:
        env[f"{prefix}_EARLY_DELTA"] = str(args.early_delta)
    return env


def parse_metrics(line: str) -> dict[str, float]:
    return {name.lower(): float(value) for name, value in METRIC_RE.findall(line)}


def expected_checkpoint(dataset_name: str, tag: str) -> Path:
    if dataset_name == "koa":
        return PROJECT_ROOT / "Knee" / "checkpoints" / f"best_resnet50_mecs_layer3_dast_knee_oa_new_{tag}.pt"
    return PROJECT_ROOT / "Alzheimer_MRI_Loss" / "checkpoints" / f"best_ResNet_layer3+MECS_dast_{tag}.pt"


def run_one(dataset_name: str, seed: int, tag_prefix: str, args: argparse.Namespace):
    tag = f"{tag_prefix}_{dataset_name}_mesc_raw_routing_plus_dast_seed{seed}"
    command = build_command(dataset_name, args)
    env = build_env(dataset_name, seed, tag, args)
    print(f"[RUN] {dataset_name.upper()} | MESC-Raw + DAST | seed={seed}", flush=True)
    if args.dry_run:
        print("      " + " ".join(command), flush=True)
        return {}

    tail: deque[str] = deque(maxlen=50)
    test_metrics = None
    waiting_for_auc = False
    try:
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
    except OSError as exc:
        print(f"[FAIL] could not start trainer: {exc}", file=sys.stderr)
        return None

    assert process.stdout is not None
    for raw_line in process.stdout:
        line = raw_line.rstrip("\r\n")
        tail.append(line)
        stripped = line.strip()
        if stripped.startswith("Test |"):
            test_metrics = parse_metrics(stripped)
            waiting_for_auc = True
            continue
        if waiting_for_auc and "ovr_roc_auc_macro=" in stripped:
            assert test_metrics is not None
            test_metrics.update(parse_metrics(stripped))
            waiting_for_auc = False
        elif stripped:
            waiting_for_auc = False

    return_code = process.wait()
    required = {"loss", "acc", "bal_acc", "macro_f1", "qwk", "mae"}
    if return_code != 0 or test_metrics is None or not required.issubset(test_metrics):
        reason = f"return code {return_code}" if return_code else "standard Test metrics were not found"
        print(f"[FAIL] {dataset_name.upper()}, seed={seed}: {reason}", file=sys.stderr)
        print("------ trainer output tail ------", file=sys.stderr)
        for line in tail:
            print(line, file=sys.stderr)
        print("---------------------------------", file=sys.stderr)
        return None

    print_values = []
    ordered = (
        "loss", "top1", "top2", "top3", "acc", "bal_acc", "macro_f1", "qwk", "mae",
        "weighted_f1", "precision_macro", "recall_macro", "ovr_roc_auc_macro", "ovr_pr_auc_macro",
    )
    for name in ordered:
        if name not in test_metrics:
            continue
        suffix = "%" if name in PERCENT_METRICS else ""
        print_values.append(f"{name}={test_metrics[name]:.4f}{suffix}")
    print(
        f"[TEST] {dataset_name}_mesc_raw_routing_dast | seed={seed} | " + " | ".join(print_values),
        flush=True,
    )

    row: dict[str, object] = {
        "dataset": dataset_name.upper(),
        "method": "ResNet50 + MESC Raw Routing + DAST",
        "attention": "MESC-Raw",
        "loss": "DAST",
        "seed": seed,
        "tau": args.dast_tau,
        "gamma": args.dast_gamma,
        "test_loss": test_metrics.pop("loss"),
        "checkpoint": str(expected_checkpoint(dataset_name, tag)),
        "run_tag": tag,
        "completed_at": datetime.now().isoformat(timespec="seconds"),
    }
    for name, value in test_metrics.items():
        row[name] = value / 100.0 if name in PERCENT_METRICS else value
    return row


def write_rows(path: Path, rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows: Sequence[dict[str, object]]) -> None:
    print("\nSummary (mean +/- sample SD)")
    for dataset_name in dict.fromkeys(str(row["dataset"]) for row in rows):
        selected = [row for row in rows if row["dataset"] == dataset_name]
        print(f"  {dataset_name} (n={len(selected)})")
        for metric in ("acc", "bal_acc", "macro_f1", "qwk", "mae"):
            values = [float(row[metric]) for row in selected]
            sd = statistics.stdev(values) if len(values) > 1 else 0.0
            print(f"    {metric}: {statistics.mean(values):.4f} +/- {sd:.4f}")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.dry_run:
        validate_data(args)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag_prefix = args.tag_prefix or f"raw_routing_ablation_{timestamp}"
    output = args.output or PROJECT_ROOT / "analysis_tables" / f"{tag_prefix}_test_results.csv"
    if not output.is_absolute():
        output = PROJECT_ROOT / output
    total = len(args.dataset_names) * len(args.seed_values)
    print(f"Datasets : {args.dataset_names}")
    print(f"Seeds    : {args.seed_values}")
    print(f"Runs     : {total}")
    print(f"DAST     : tau={args.dast_tau}, gamma={args.dast_gamma}")
    print(f"Results  : {output}")

    rows: list[dict[str, object]] = []
    scheduled = 0
    for seed in args.seed_values:
        for dataset_name in args.dataset_names:
            scheduled += 1
            if scheduled <= args.skip_runs:
                print(f"[SKIP] {dataset_name.upper()} | seed={seed}")
                continue
            row = run_one(dataset_name, seed, tag_prefix, args)
            if args.dry_run:
                continue
            if row is None:
                write_rows(output, rows)
                print(f"Stopped after failure. Completed rows: {output}", file=sys.stderr)
                return 1
            rows.append(row)
            write_rows(output, rows)
    if not args.dry_run:
        print_summary(rows)
        print(f"\nResults: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
