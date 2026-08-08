#!/usr/bin/env python
"""Run the ResNet50 DASL experiments on KOA and Alzheimer MRI.

DASL has no tunable loss hyperparameters.  The runner reuses the established
ResNet50 data splits, augmentation, optimizer schedules, validation monitors,
and early stopping.  It stores final Test rows only in the result CSV, while
keeping the complete child output and best checkpoints for auditing.
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
ADAPTER = PROJECT_ROOT / "tools" / "run_training_with_mecsb.py"
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
CHECKPOINT_RE = re.compile(r"(?:[A-Za-z]:)?[^\s]*\.pt")


@dataclass(frozen=True)
class Experiment:
    key: str
    method: str
    attention: str
    launcher: str


EXPERIMENTS = {
    "baseline_dasl": Experiment(
        "baseline_dasl",
        "ResNet50 + DASL",
        "none",
        "direct",
    ),
    "mesc_equal_dasl": Experiment(
        "mesc_equal_dasl",
        "ResNet50 + MESC-Equal + DASL",
        "MECS_VersionA",
        "direct",
    ),
    "mesc_direct_dasl": Experiment(
        "mesc_direct_dasl",
        "ResNet50 + MESC-Direct + DASL",
        "MECS_RawRouting",
        "raw",
    ),
    "mesc_dasl": Experiment(
        "mesc_dasl",
        "ResNet50 + MESC (Ours) + DASL",
        "MECS_VersionB",
        "median",
    ),
}
SUITES = {
    "core": ("baseline_dasl", "mesc_dasl"),
    "routing": ("mesc_equal_dasl", "mesc_direct_dasl", "mesc_dasl"),
    "full": (
        "baseline_dasl",
        "mesc_equal_dasl",
        "mesc_direct_dasl",
        "mesc_dasl",
    ),
}
TRAIN_SCRIPTS = {
    "koa": {
        "baseline": PROJECT_ROOT / "Knee" / "ResNet_baseline_loss_compare.py",
        "mesc": PROJECT_ROOT / "Knee" / "ResNet_layer3+MECS+CE.py",
    },
    "adni": {
        "baseline": PROJECT_ROOT / "Alzheimer_MRI_Loss" / "ResNet_baseline.py",
        "mesc": PROJECT_ROOT / "Alzheimer_MRI_Loss" / "ResNet_layer3+MECS.py",
    },
}
CSV_FIELDS = (
    "dataset",
    "backbone",
    "experiment",
    "method",
    "attention",
    "insertion_point",
    "criterion",
    "loss_formula",
    "loss_hyperparameters",
    "seed",
    "epochs",
    "patience",
    "batch_size",
    "image_size",
    "num_workers",
    "early_delta",
    "optimizer_profile",
    "koa_backbone_lr",
    "koa_head_lr",
    "weight_decay",
    "adni_base_lr",
    "adni_test_ratio",
    "adni_val_ratio",
    "data_root",
    *TEST_METRICS,
    "checkpoint",
    "log",
    "run_tag",
    "completed_at",
)


def comma_list(raw: str) -> list[str]:
    return [part.strip().lower() for part in raw.split(",") if part.strip()]


def parse_seeds(raw: str) -> list[int]:
    seeds = [int(part) for part in comma_list(raw)]
    if not seeds:
        raise ValueError("at least one seed is required")
    if len(seeds) != len(set(seeds)):
        raise ValueError("seeds contain duplicates")
    return seeds


def sanitize_tag(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "._-" else "-" for ch in value.strip())
    while "--" in cleaned:
        cleaned = cleaned.replace("--", "-")
    return cleaned.strip("._-")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=tuple(SUITES), default="core")
    parser.add_argument(
        "--experiments",
        default="",
        help="Optional comma-separated experiment keys; overrides --suite.",
    )
    parser.add_argument("--datasets", default="koa,adni")
    parser.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    parser.add_argument("--koa-data-root", type=Path, default=None)
    parser.add_argument("--adni-data-root", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--early-delta", type=float, default=1e-4)
    parser.add_argument("--koa-backbone-lr", type=float, default=1e-4)
    parser.add_argument("--koa-head-lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--adni-base-lr", type=float, default=1e-4)
    parser.add_argument("--adni-test-ratio", type=float, default=0.2)
    parser.add_argument("--adni-val-ratio", type=float, default=0.1)
    parser.add_argument("--conda-env", default="none")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--tag-prefix", default="")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--verbose-trainer", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    try:
        args.seed_values = parse_seeds(args.seeds)
    except ValueError as exc:
        parser.error(str(exc))
    args.dataset_values = comma_list(args.datasets)
    if not args.dataset_values or any(name not in TRAIN_SCRIPTS for name in args.dataset_values):
        parser.error("--datasets supports koa, adni, or koa,adni")
    if len(args.dataset_values) != len(set(args.dataset_values)):
        parser.error("datasets contain duplicates")

    experiment_keys = comma_list(args.experiments) if args.experiments else list(SUITES[args.suite])
    unknown = [key for key in experiment_keys if key not in EXPERIMENTS]
    if unknown:
        parser.error(f"unknown experiments: {unknown}; choices: {list(EXPERIMENTS)}")
    if len(experiment_keys) != len(set(experiment_keys)):
        parser.error("experiments contain duplicates")
    args.experiment_values = [EXPERIMENTS[key] for key in experiment_keys]

    positive = (
        args.epochs,
        args.patience,
        args.batch_size,
        args.image_size,
        args.koa_backbone_lr,
        args.koa_head_lr,
        args.adni_base_lr,
    )
    if any(value <= 0 for value in positive):
        parser.error("epochs, patience, sizes, and learning rates must be positive")
    if args.num_workers is not None and args.num_workers < 0:
        parser.error("num workers cannot be negative")
    if args.early_delta < 0 or args.weight_decay < 0:
        parser.error("early delta and weight decay cannot be negative")
    if not 0 < args.adni_test_ratio < 1 or not 0 < args.adni_val_ratio < 1:
        parser.error("ADNI split ratios must be between 0 and 1")
    if args.resume and args.output is None:
        parser.error("--resume requires --output")

    args.koa_data_root = args.koa_data_root or os.getenv("KNEE_DATA_ROOT")
    args.adni_data_root = args.adni_data_root or os.getenv("ALZHEIMER_DATA_ROOT")
    for dataset in args.dataset_values:
        attribute = f"{dataset}_data_root"
        value = getattr(args, attribute)
        if not value:
            parser.error(f"{dataset.upper()} requires its data-root argument or environment variable")
        setattr(args, attribute, Path(value).expanduser().resolve())
    return args


def validate_data_root(dataset: str, root: Path) -> None:
    if not root.is_dir():
        raise FileNotFoundError(f"{dataset.upper()} data root not found: {root}")
    if dataset == "koa":
        missing = [name for name in ("train", "val", "test") if not (root / name).is_dir()]
        if missing:
            raise FileNotFoundError(f"KOA root is missing split directories: {missing}")


def python_prefix(args: argparse.Namespace) -> list[str]:
    env_name = args.conda_env.strip()
    if env_name.lower() not in {"", "none", "null", "false", "0"}:
        return ["conda", "run", "--no-capture-output", "-n", env_name, "python", "-u"]
    return [args.python, "-u"]


def training_script(dataset: str, experiment: Experiment) -> Path:
    model_type = "baseline" if experiment.key == "baseline_dasl" else "mesc"
    return TRAIN_SCRIPTS[dataset][model_type]


def build_command(
    args: argparse.Namespace,
    dataset: str,
    experiment: Experiment,
) -> list[str]:
    script = training_script(dataset, experiment)
    prefix = python_prefix(args)
    if experiment.launcher in {"median", "raw"}:
        return [
            *prefix,
            str(ADAPTER),
            "--variant",
            experiment.launcher,
            "--script",
            str(script),
        ]
    return [*prefix, str(script)]


def build_environment(
    args: argparse.Namespace,
    dataset: str,
    experiment: Experiment,
    seed: int,
    run_tag: str,
) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    workers = args.num_workers if args.num_workers is not None else (4 if dataset == "koa" else 2)
    omp_threads = env.get("OMP_NUM_THREADS", "").strip()
    if not omp_threads.isdigit() or int(omp_threads) <= 0:
        env["OMP_NUM_THREADS"] = str(max(1, workers))
    env["GLOBAL_EXPERIMENT_SEED"] = str(seed)

    prefix = "KNEE" if dataset == "koa" else "ALZHEIMER"
    env[f"{prefix}_SEED"] = str(seed)
    env[f"{prefix}_DATA_ROOT"] = str(getattr(args, f"{dataset}_data_root"))
    env[f"{prefix}_EPOCHS"] = str(args.epochs)
    env[f"{prefix}_PATIENCE"] = str(args.patience)
    env[f"{prefix}_BATCH_SIZE"] = str(args.batch_size)
    env[f"{prefix}_IMAGE_SIZE"] = str(args.image_size)
    env[f"{prefix}_NUM_WORKERS"] = str(workers)
    env[f"{prefix}_EARLY_DELTA"] = str(args.early_delta)
    env[f"{prefix}_RUN_TAG"] = run_tag

    if dataset == "koa":
        env["KNEE_LR_BACKBONE"] = str(args.koa_backbone_lr)
        env["KNEE_LR_HEAD"] = str(args.koa_head_lr)
        env["KNEE_WEIGHT_DECAY"] = str(args.weight_decay)
        if experiment.key == "baseline_dasl":
            env["KNEE_LOSS"] = "dasl"
        else:
            env["KNEE_MESC_LOSS"] = "dasl"
    else:
        env["ALZHEIMER_LOSSES"] = "dasl"
        env["ALZHEIMER_BASE_LR"] = str(args.adni_base_lr)
        env["ALZHEIMER_TEST_RATIO"] = str(args.adni_test_ratio)
        env["ALZHEIMER_VAL_RATIO"] = str(args.adni_val_ratio)
        env["ALZHEIMER_BASELINE_LR_PROFILE"] = "matched_mesc"
    return env


def parse_metrics(line: str) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for name, raw_value in METRIC_RE.findall(line):
        value = float(raw_value)
        if name in PERCENT_METRICS:
            value /= 100.0
        metrics["test_loss" if name == "loss" else name] = value
    return metrics


def expected_checkpoint(dataset: str, experiment: Experiment, run_tag: str) -> Path:
    if dataset == "koa":
        if experiment.key == "baseline_dasl":
            filename = f"best_resnet50_knee_oa_dasl_{run_tag}.pt"
        else:
            filename = f"best_resnet50_mecs_layer3_knee_oa_dasl_{run_tag}.pt"
        return PROJECT_ROOT / "Knee" / "checkpoints" / filename
    stem = "ResNet_baseline" if experiment.key == "baseline_dasl" else "ResNet_layer3+MECS"
    return (
        PROJECT_ROOT
        / "Alzheimer_MRI_Loss"
        / "checkpoints"
        / f"best_{stem}_dasl_{run_tag}.pt"
    )


def write_rows(path: Path, rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def read_rows(path: Path) -> list[dict[str, object]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def validate_resume_rows(
    rows: Sequence[dict[str, object]],
    args: argparse.Namespace,
) -> None:
    common_expected = {
        "backbone": "ResNet50",
        "criterion": "DASL",
        "loss_hyperparameters": "none",
        "epochs": str(args.epochs),
        "patience": str(args.patience),
        "batch_size": str(args.batch_size),
        "image_size": str(args.image_size),
        "early_delta": str(args.early_delta),
        "koa_backbone_lr": str(args.koa_backbone_lr),
        "koa_head_lr": str(args.koa_head_lr),
        "weight_decay": str(args.weight_decay),
        "adni_base_lr": str(args.adni_base_lr),
        "adni_test_ratio": str(args.adni_test_ratio),
        "adni_val_ratio": str(args.adni_val_ratio),
    }
    for row_number, row in enumerate(rows, start=2):
        for field, expected in common_expected.items():
            if str(row.get(field, "")) != expected:
                raise ValueError(
                    f"Cannot resume: row {row_number} has {field}={row.get(field)!r}, "
                    f"expected {expected!r}."
                )
        dataset = str(row.get("dataset", "")).lower()
        if dataset not in {"koa", "adni"}:
            raise ValueError(f"Cannot resume: row {row_number} has unknown dataset {dataset!r}.")
        expected_workers = args.num_workers if args.num_workers is not None else (4 if dataset == "koa" else 2)
        if str(row.get("num_workers", "")) != str(expected_workers):
            raise ValueError(f"Cannot resume: row {row_number} uses different num_workers.")
        expected_root = str(getattr(args, f"{dataset}_data_root"))
        if str(row.get("data_root", "")) != expected_root:
            raise ValueError(f"Cannot resume: row {row_number} uses a different data root.")


def run_one(
    args: argparse.Namespace,
    dataset: str,
    experiment: Experiment,
    seed: int,
    run_tag: str,
    log_path: Path,
) -> dict[str, object] | None:
    command = build_command(args, dataset, experiment)
    environment = build_environment(args, dataset, experiment, seed, run_tag)
    print(f"[RUN] {dataset.upper()} | {experiment.method} | seed={seed}", flush=True)
    if args.dry_run:
        print("      " + " ".join(command), flush=True)
        return {}

    log_path.parent.mkdir(parents=True, exist_ok=True)
    process = subprocess.Popen(
        command,
        cwd=PROJECT_ROOT,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    tail: deque[str] = deque(maxlen=80)
    test_metrics: dict[str, float] | None = None
    checkpoint = ""
    assert process.stdout is not None
    with log_path.open("w", encoding="utf-8", newline="") as log_handle:
        log_handle.write("COMMAND: " + " ".join(command) + "\n" + "=" * 90 + "\n")
        for raw_line in process.stdout:
            log_handle.write(raw_line)
            line = raw_line.rstrip("\r\n")
            tail.append(line)
            if args.verbose_trainer:
                print(line, flush=True)
            stripped = line.strip()
            if stripped.startswith("Test |"):
                parsed = parse_metrics(stripped)
                if parsed:
                    test_metrics = parsed
            elif test_metrics is not None and (
                "ovr_roc_auc_macro=" in stripped or "ovr_pr_auc_macro=" in stripped
            ):
                test_metrics.update(parse_metrics(stripped))
            if ".pt" in stripped and any(
                marker in stripped.lower() for marker in ("saved", "checkpoint")
            ):
                matches = CHECKPOINT_RE.findall(stripped)
                if matches:
                    checkpoint = matches[-1]

    return_code = process.wait()
    required = {"test_loss", "acc", "bal_acc", "macro_f1", "qwk", "mae"}
    if return_code != 0 or test_metrics is None or not required.issubset(test_metrics):
        reason = f"return code {return_code}" if return_code else "final Test metrics were not found"
        print(f"[FAIL] {dataset}/{experiment.key}/seed{seed}: {reason}", file=sys.stderr)
        print("------ trainer output tail ------", file=sys.stderr)
        for line in tail:
            print(line, file=sys.stderr)
        print(f"Full log: {log_path}", file=sys.stderr)
        print("---------------------------------", file=sys.stderr)
        return None

    if not checkpoint:
        checkpoint = str(expected_checkpoint(dataset, experiment, run_tag))
    workers = args.num_workers if args.num_workers is not None else (4 if dataset == "koa" else 2)
    row: dict[str, object] = {
        "dataset": dataset.upper(),
        "backbone": "ResNet50",
        "experiment": experiment.key,
        "method": experiment.method,
        "attention": experiment.attention,
        "insertion_point": "after_layer3" if experiment.key != "baseline_dasl" else "none",
        "criterion": "DASL",
        "loss_formula": "-log(p_y)-log(1-E_p[|k-y|/(K-1)]+eps)",
        "loss_hyperparameters": "none",
        "seed": seed,
        "epochs": args.epochs,
        "patience": args.patience,
        "batch_size": args.batch_size,
        "image_size": args.image_size,
        "num_workers": workers,
        "early_delta": args.early_delta,
        "optimizer_profile": "koa_adamw_2group" if dataset == "koa" else "adni_matched_mesc_adam",
        "koa_backbone_lr": args.koa_backbone_lr,
        "koa_head_lr": args.koa_head_lr,
        "weight_decay": args.weight_decay,
        "adni_base_lr": args.adni_base_lr,
        "adni_test_ratio": args.adni_test_ratio,
        "adni_val_ratio": args.adni_val_ratio,
        "data_root": str(getattr(args, f"{dataset}_data_root")),
        **test_metrics,
        "checkpoint": checkpoint,
        "log": str(log_path),
        "run_tag": run_tag,
        "completed_at": datetime.now().isoformat(timespec="seconds"),
    }
    display = " | ".join(
        f"{metric}={float(row[metric]):.4f}"
        for metric in ("acc", "bal_acc", "macro_f1", "qwk", "mae")
    )
    print(f"[TEST] {dataset}_{experiment.key} | seed={seed} | {display}", flush=True)
    return row


def print_summary(rows: Sequence[dict[str, object]]) -> None:
    groups: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["dataset"]), str(row["experiment"]))].append(row)
    if not groups:
        return
    print("\nSummary (mean +/- sample SD)")
    for (dataset, experiment), group in sorted(groups.items()):
        print(f"  {dataset} | {experiment} (n={len(group)})")
        for metric in ("acc", "bal_acc", "macro_f1", "qwk", "mae"):
            values = [float(row[metric]) for row in group]
            sample_sd = statistics.stdev(values) if len(values) > 1 else 0.0
            print(f"    {metric}: {statistics.mean(values):.4f} +/- {sample_sd:.4f}")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.dry_run:
        for dataset in args.dataset_values:
            validate_data_root(dataset, getattr(args, f"{dataset}_data_root"))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fallback_prefix = f"resnet50_dasl_{args.suite}_{timestamp}"
    prefix = sanitize_tag(args.tag_prefix) or fallback_prefix
    output = args.output or PROJECT_ROOT / "analysis_tables" / f"{prefix}_test_results.csv"
    if not output.is_absolute():
        output = (PROJECT_ROOT / output).resolve()
    log_dir = PROJECT_ROOT / "batch_logs" / prefix

    if args.resume:
        if not output.is_file():
            raise FileNotFoundError(f"resume output not found: {output}")
        rows = read_rows(output)
        validate_resume_rows(rows, args)
    else:
        if output.exists() and not args.dry_run:
            raise FileExistsError(f"output exists; use --resume or another --output: {output}")
        rows: list[dict[str, object]] = []

    completed = {
        (str(row["dataset"]).lower(), int(row["seed"]), str(row["experiment"]))
        for row in rows
    }
    schedule = [
        (dataset, seed, experiment)
        for dataset in args.dataset_values
        for seed in args.seed_values
        for experiment in args.experiment_values
    ]
    counts = defaultdict(int)
    for dataset, _, _ in schedule:
        counts[dataset] += 1
    print("Backbone    : ResNet50")
    print("Loss        : DASL (no tunable hyperparameters)")
    print(f"Suite       : {args.suite}")
    print(f"Experiments : {[experiment.key for experiment in args.experiment_values]}")
    print(f"Datasets    : {args.dataset_values}")
    print(f"Seeds       : {args.seed_values}")
    print(f"Runs        : {len(schedule)} | by dataset={dict(counts)}")
    print(f"Already done: {len(completed)}")
    print(f"Results     : {output}")
    print(f"Logs        : {log_dir}")

    for dataset, seed, experiment in schedule:
        completion_key = (dataset, seed, experiment.key)
        if completion_key in completed:
            print(f"[SKIP completed] {dataset}_{experiment.key} | seed={seed}")
            continue
        run_tag = f"{prefix}_{dataset}_{experiment.key}_seed{seed}"
        log_path = log_dir / f"{dataset}_{experiment.key}_seed{seed}.log"
        row = run_one(args, dataset, experiment, seed, run_tag, log_path)
        if args.dry_run:
            continue
        if row is None:
            write_rows(output, rows)
            if not args.continue_on_error:
                print(f"Stopped after failure. Completed rows: {output}", file=sys.stderr)
                return 1
            continue
        rows.append(row)
        completed.add(completion_key)
        write_rows(output, rows)

    if not args.dry_run:
        print_summary(rows)
        print(f"Results: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
