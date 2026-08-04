#!/usr/bin/env python
"""Run the requested KOA/ADNI MESC-A, SORD, CORAL, and CORN experiments.

The child trainers still perform validation and early stopping, but this
orchestrator only prints and stores each run's final standard Test result.
KOA AutoTest lines are deliberately ignored.
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
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MEC_S_B_ADAPTER = PROJECT_ROOT / "tools" / "run_training_with_mecsb.py"
ORDINAL_TRAINER = PROJECT_ROOT / "tools" / "train_ordinal_baseline.py"
DEFAULT_SEEDS = (1, 9, 6, 5, 4)
METRIC_RE = re.compile(r"([A-Za-z0-9_]+)=(-?[0-9]+(?:\.[0-9]+)?)%?")
CHECKPOINT_RE = re.compile(r"(?:[A-Za-z]:)?[^\s]*\.pt")
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
CSV_FIELDS = (
    "dataset",
    "experiment",
    "method",
    "fusion",
    "criterion",
    "seed",
    "tau",
    "gamma",
    *TEST_METRICS,
    "checkpoint",
    "run_tag",
    "completed_at",
)


@dataclass(frozen=True)
class Experiment:
    key: str
    group: str
    dataset: str
    method: str
    fusion: str
    criterion: str
    launcher: str
    script: Path


EXPERIMENTS = (
    Experiment(
        "koa_mesc_a_ce", "mesc-a", "koa", "ResNet50 + MESC-A", "equal_average", "ce", "direct",
        PROJECT_ROOT / "Knee" / "ResNet_layer3+MECS+CE.py",
    ),
    Experiment(
        "koa_mesc_a_dast", "mesc-a", "koa", "ResNet50 + MESC-A + DAST", "equal_average", "dast", "direct",
        PROJECT_ROOT / "Knee" / "ResNet_layer3+MECS+Loss4.py",
    ),
    Experiment(
        "adni_mesc_a_ce", "mesc-a", "adni", "ResNet50 + MESC-A", "equal_average", "ce", "direct",
        PROJECT_ROOT / "Alzheimer_MRI_Loss" / "ResNet_layer3+MECS.py",
    ),
    Experiment(
        "adni_mesc_a_dast", "mesc-a", "adni", "ResNet50 + MESC-A + DAST", "equal_average", "dast", "direct",
        PROJECT_ROOT / "Alzheimer_MRI_Loss" / "ResNet_layer3+MECS.py",
    ),
    Experiment(
        "koa_baseline_sord", "sord", "koa", "ResNet50 + SORD", "none", "sord", "direct",
        PROJECT_ROOT / "Knee" / "ResNet_baseline_loss_compare.py",
    ),
    Experiment(
        "koa_mesc_b_sord", "sord", "koa", "ResNet50 + MESC-B + SORD", "dynamic_router", "sord", "mecsb",
        PROJECT_ROOT / "Knee" / "ResNet_layer3+MECS+CE.py",
    ),
    Experiment(
        "adni_baseline_sord", "sord", "adni", "ResNet50 + SORD", "none", "sord", "direct",
        PROJECT_ROOT / "Alzheimer_MRI_Loss" / "ResNet_baseline.py",
    ),
    Experiment(
        "adni_mesc_b_sord", "sord", "adni", "ResNet50 + MESC-B + SORD", "dynamic_router", "sord", "mecsb",
        PROJECT_ROOT / "Alzheimer_MRI_Loss" / "ResNet_layer3+MECS.py",
    ),
    Experiment(
        "koa_coral", "ordinal", "koa", "ResNet50 + CORAL", "none", "coral", "ordinal", ORDINAL_TRAINER,
    ),
    Experiment(
        "koa_corn", "ordinal", "koa", "ResNet50 + CORN", "none", "corn", "ordinal", ORDINAL_TRAINER,
    ),
    Experiment(
        "adni_coral", "ordinal", "adni", "ResNet50 + CORAL", "none", "coral", "ordinal", ORDINAL_TRAINER,
    ),
    Experiment(
        "adni_corn", "ordinal", "adni", "ResNet50 + CORN", "none", "corn", "ordinal", ORDINAL_TRAINER,
    ),
)
EXPERIMENT_BY_KEY = {experiment.key: experiment for experiment in EXPERIMENTS}


def comma_list(raw: str) -> list[str]:
    return [part.strip().lower() for part in raw.split(",") if part.strip()]


def parse_seeds(raw: str) -> list[int]:
    seeds = [int(value.strip()) for value in raw.split(",") if value.strip()]
    if not seeds:
        raise ValueError("at least one seed is required")
    if len(seeds) != len(set(seeds)):
        raise ValueError("seeds contain duplicates")
    return seeds


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--groups", default="all", help="mesc-a,sord,ordinal, or all")
    parser.add_argument("--experiments", default="", help="Optional exact experiment keys, comma-separated.")
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
    parser.add_argument("--sord-tau", type=float, default=1.0)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--conda-env", default="none")
    parser.add_argument("--tag-prefix", default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--skip-runs", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    try:
        args.seed_values = parse_seeds(args.seeds)
    except ValueError as exc:
        parser.error(str(exc))
    args.dataset_values = comma_list(args.datasets)
    if not args.dataset_values or any(value not in {"koa", "adni"} for value in args.dataset_values):
        parser.error("--datasets supports koa, adni, or koa,adni.")
    if len(args.dataset_values) != len(set(args.dataset_values)):
        parser.error("--datasets contains duplicates.")

    if args.experiments:
        keys = comma_list(args.experiments)
        unknown = [key for key in keys if key not in EXPERIMENT_BY_KEY]
        if unknown:
            parser.error(f"unknown experiment keys: {unknown}")
        selected = [EXPERIMENT_BY_KEY[key] for key in keys]
    else:
        groups = comma_list(args.groups)
        if groups == ["all"]:
            groups = ["mesc-a", "sord", "ordinal"]
        if not groups or any(group not in {"mesc-a", "sord", "ordinal"} for group in groups):
            parser.error("--groups supports mesc-a, sord, ordinal, or all.")
        selected = [experiment for experiment in EXPERIMENTS if experiment.group in groups]
    args.selected_experiments = [
        experiment for experiment in selected if experiment.dataset in args.dataset_values
    ]
    if not args.selected_experiments:
        parser.error("the experiment and dataset filters selected no runs.")

    if args.epochs <= 0 or args.patience <= 0 or args.skip_runs < 0:
        parser.error("--epochs and --patience must be positive; --skip-runs cannot be negative.")
    if args.dast_tau <= 0 or args.sord_tau <= 0 or args.dast_gamma < 0:
        parser.error("tau values must be positive and DAST gamma cannot be negative.")

    args.koa_data_root = args.koa_data_root or os.getenv("KNEE_DATA_ROOT")
    args.adni_data_root = args.adni_data_root or os.getenv("ALZHEIMER_DATA_ROOT")
    for name in ("koa", "adni"):
        if name not in args.dataset_values:
            continue
        attr = f"{name}_data_root"
        value = getattr(args, attr)
        if not value:
            parser.error(f"{name.upper()} requires --{name}-data-root or its dataset environment variable.")
        setattr(args, attr, Path(value).expanduser().resolve())
    return args


def validate_roots(args: argparse.Namespace) -> None:
    if "koa" in args.dataset_values:
        missing = [name for name in ("train", "val", "test") if not (args.koa_data_root / name).is_dir()]
        if missing:
            raise FileNotFoundError(f"KOA root {args.koa_data_root} is missing {missing}.")
    if "adni" in args.dataset_values and not args.adni_data_root.is_dir():
        raise FileNotFoundError(f"ADNI root not found: {args.adni_data_root}")


def python_prefix(args: argparse.Namespace) -> list[str]:
    env_name = args.conda_env.strip()
    if env_name.lower() not in {"", "none", "null", "false", "0"}:
        return ["conda", "run", "--no-capture-output", "-n", env_name, "python", "-u"]
    return [args.python, "-u"]


def run_tag(prefix: str, experiment: Experiment, seed: int) -> str:
    return f"{prefix}_{experiment.key}_seed{seed}"


def direct_env(experiment: Experiment, seed: int, args: argparse.Namespace, tag: str) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    omp_threads = env.get("OMP_NUM_THREADS", "").strip()
    if not omp_threads.isdigit() or int(omp_threads) <= 0:
        env["OMP_NUM_THREADS"] = str(max(1, args.num_workers or 4))
    env["GLOBAL_EXPERIMENT_SEED"] = str(seed)

    prefix = "KNEE" if experiment.dataset == "koa" else "ALZHEIMER"
    env[f"{prefix}_SEED"] = str(seed)
    env[f"{prefix}_EPOCHS"] = str(args.epochs)
    env[f"{prefix}_PATIENCE"] = str(args.patience)
    env[f"{prefix}_RUN_TAG"] = tag
    env[f"{prefix}_DATA_ROOT"] = str(
        args.koa_data_root if experiment.dataset == "koa" else args.adni_data_root
    )
    if args.batch_size is not None:
        env[f"{prefix}_BATCH_SIZE"] = str(args.batch_size)
    if args.image_size is not None:
        env[f"{prefix}_IMAGE_SIZE"] = str(args.image_size)
    if args.num_workers is not None:
        env[f"{prefix}_NUM_WORKERS"] = str(args.num_workers)
    if args.early_delta is not None:
        env[f"{prefix}_EARLY_DELTA"] = str(args.early_delta)

    if experiment.criterion == "dast":
        env[f"{prefix}_DAST_TAU"] = str(args.dast_tau)
        env[f"{prefix}_DAST_GAMMA"] = str(args.dast_gamma)
    elif experiment.criterion == "sord":
        env[f"{prefix}_SORD_TAU"] = str(args.sord_tau)
        # Record the exact DAST ablation interpretation as well.
        env[f"{prefix}_DAST_TAU"] = str(args.sord_tau)
        env[f"{prefix}_DAST_GAMMA"] = "0"

    if experiment.dataset == "adni":
        env["ALZHEIMER_LOSSES"] = "sord_ce" if experiment.criterion == "sord" else experiment.criterion
    elif experiment.key == "koa_mesc_a_ce":
        env["KNEE_MESC_LOSS"] = "ce"
    elif experiment.key == "koa_baseline_sord":
        env["KNEE_LOSS"] = "sord_ce"
    elif experiment.key == "koa_mesc_b_sord":
        env["KNEE_MESC_LOSS"] = "sord_ce"
    if experiment.launcher == "mecsb":
        env["MESC_IMPLEMENTATION"] = "MECS_VersionB"
    return env


def build_command(experiment: Experiment, seed: int, args: argparse.Namespace, tag: str) -> list[str]:
    prefix = python_prefix(args)
    if experiment.launcher == "mecsb":
        return prefix + [str(MEC_S_B_ADAPTER), "--script", str(experiment.script)]
    if experiment.launcher == "ordinal":
        data_root = args.koa_data_root if experiment.dataset == "koa" else args.adni_data_root
        command = prefix + [
            str(ORDINAL_TRAINER),
            "--dataset", experiment.dataset,
            "--method", experiment.criterion,
            "--data-root", str(data_root),
            "--seed", str(seed),
            "--epochs", str(args.epochs),
            "--patience", str(args.patience),
            "--run-tag", tag,
        ]
        if args.batch_size is not None:
            command += ["--batch-size", str(args.batch_size)]
        if args.image_size is not None:
            command += ["--image-size", str(args.image_size)]
        if args.num_workers is not None:
            command += ["--num-workers", str(args.num_workers)]
        if args.early_delta is not None:
            command += ["--early-delta", str(args.early_delta)]
        return command
    return prefix + [str(experiment.script)]


def parse_metrics(line: str) -> dict[str, float]:
    return {name.lower(): float(value) for name, value in METRIC_RE.findall(line)}


def expected_checkpoint(experiment: Experiment, tag: str) -> Path:
    if experiment.launcher == "ordinal":
        parent = "Knee" if experiment.dataset == "koa" else "Alzheimer_MRI_Loss"
        return PROJECT_ROOT / parent / "checkpoints" / f"best_resnet50_{experiment.criterion}_{tag}.pt"
    if experiment.dataset == "adni":
        stem = "ResNet_baseline" if "baseline" in experiment.key else "ResNet_layer3+MECS"
        loss = "sord_ce" if experiment.criterion == "sord" else experiment.criterion
        return PROJECT_ROOT / "Alzheimer_MRI_Loss" / "checkpoints" / f"best_{stem}_{loss}_{tag}.pt"
    koa_names = {
        "koa_mesc_a_ce": "best_resnet50_mecs_layer3_knee_oa",
        "koa_mesc_a_dast": "best_resnet50_mecs_layer3_dast_knee_oa_new",
        "koa_baseline_sord": "best_resnet50_knee_oa_sord_ce",
        "koa_mesc_b_sord": "best_resnet50_mecs_layer3_knee_oa_sord_ce",
    }
    return PROJECT_ROOT / "Knee" / "checkpoints" / f"{koa_names[experiment.key]}_{tag}.pt"


def run_one(experiment: Experiment, seed: int, args: argparse.Namespace, prefix: str):
    tag = run_tag(prefix, experiment, seed)
    env = direct_env(experiment, seed, args, tag)
    command = build_command(experiment, seed, args, tag)
    print(f"[RUN] {experiment.dataset.upper()} | {experiment.method} | seed={seed}", flush=True)
    if args.dry_run:
        print("      " + " ".join(command), flush=True)
        return {}

    tail: deque[str] = deque(maxlen=50)
    test_metrics = None
    checkpoint = ""
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
        if ".pt" in stripped and any(word in stripped.lower() for word in ("saved", "checkpoint")):
            matches = CHECKPOINT_RE.findall(stripped)
            if matches:
                checkpoint = matches[-1]

    return_code = process.wait()
    required = {"loss", "acc", "bal_acc", "macro_f1", "qwk", "mae"}
    if return_code != 0 or test_metrics is None or not required.issubset(test_metrics):
        reason = f"return code {return_code}" if return_code else "standard Test metrics were not found"
        print(f"[FAIL] {experiment.key}, seed={seed}: {reason}", file=sys.stderr)
        print("------ trainer output tail ------", file=sys.stderr)
        for line in tail:
            print(line, file=sys.stderr)
        print("---------------------------------", file=sys.stderr)
        return None

    test_metrics["test_loss"] = test_metrics.pop("loss")
    if not checkpoint:
        checkpoint = str(expected_checkpoint(experiment, tag))
    tau = args.dast_tau if experiment.criterion == "dast" else (
        args.sord_tau if experiment.criterion == "sord" else ""
    )
    gamma = args.dast_gamma if experiment.criterion == "dast" else (
        0.0 if experiment.criterion == "sord" else ""
    )
    row = {
        "dataset": experiment.dataset.upper(),
        "experiment": experiment.key,
        "method": experiment.method,
        "fusion": experiment.fusion,
        "criterion": experiment.criterion.upper(),
        "seed": seed,
        "tau": tau,
        "gamma": gamma,
        **test_metrics,
        "checkpoint": checkpoint,
        "run_tag": tag,
        "completed_at": datetime.now().isoformat(timespec="seconds"),
    }
    ordered = (
        "test_loss", "top1", "top2", "top3", "acc", "bal_acc", "macro_f1", "qwk", "mae",
        "weighted_f1", "precision_macro", "recall_macro", "ovr_roc_auc_macro", "ovr_pr_auc_macro",
    )
    values = []
    for name in ordered:
        if name not in row:
            continue
        display_name = "loss" if name == "test_loss" else name
        suffix = "%" if name in PERCENT_METRICS else ""
        values.append(f"{display_name}={float(row[name]):.4f}{suffix}")
    print(f"[TEST] {experiment.key} | seed={seed} | " + " | ".join(values), flush=True)
    return row


def write_rows(path: Path, rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows: Sequence[dict[str, object]]) -> None:
    if not rows:
        return
    print("\nSummary (mean +/- sample SD)")
    keys = list(dict.fromkeys(str(row["experiment"]) for row in rows))
    for key in keys:
        selected = [row for row in rows if row["experiment"] == key]
        print(f"  {key} (n={len(selected)})")
        for metric in ("acc", "bal_acc", "macro_f1", "qwk", "mae"):
            values = [float(row[metric]) for row in selected]
            sd = statistics.stdev(values) if len(values) > 1 else 0.0
            suffix = "%" if metric in PERCENT_METRICS else ""
            print(f"    {metric}: {statistics.mean(values):.4f} +/- {sd:.4f}{suffix}")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.dry_run:
        validate_roots(args)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = args.tag_prefix or f"requested_ordinal_{timestamp}"
    output = args.output or PROJECT_ROOT / "analysis_tables" / f"{prefix}_test_results.csv"
    if not output.is_absolute():
        output = PROJECT_ROOT / output

    total = len(args.selected_experiments) * len(args.seed_values)
    print(f"Experiments : {[experiment.key for experiment in args.selected_experiments]}")
    print(f"Seeds       : {args.seed_values}")
    print(f"Runs        : {total}")
    print(f"Epochs      : {args.epochs}")
    print(f"Patience    : {args.patience}")
    print(f"DAST        : tau={args.dast_tau}, gamma={args.dast_gamma}")
    print(f"SORD        : tau={args.sord_tau}, gamma=0")
    print(f"Results     : {output}")

    rows: list[dict[str, object]] = []
    scheduled = 0
    for seed in args.seed_values:
        for experiment in args.selected_experiments:
            scheduled += 1
            if scheduled <= args.skip_runs:
                print(f"[SKIP] {experiment.key} | seed={seed}")
                continue
            row = run_one(experiment, seed, args, prefix)
            if args.dry_run:
                continue
            if row is None:
                write_rows(output, rows)
                print(f"Stopped after failure. Completed Test rows: {output}", file=sys.stderr)
                return 1
            rows.append(row)
            write_rows(output, rows)

    if not args.dry_run:
        print_summary(rows)
        print(f"\nResults: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
