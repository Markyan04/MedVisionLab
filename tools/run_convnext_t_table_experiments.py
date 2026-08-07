#!/usr/bin/env python
"""Run the paper-table experiments with a ConvNeXt-T backbone.

Repeated rows across the main, routing, DAST-mechanism, attention, and loss
tables are trained once and tagged with every table that consumes them.  KOA
DAST sensitivity adds six non-default configurations; the default
``tau=1.0, gamma=1.5`` row reuses ``baseline_dast``.  Results contain final Test
metrics only, while full stdout and checkpoints remain available for auditing.
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
TRAINER = PROJECT_ROOT / "tools" / "train_convnext_t_experiment.py"
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
    attention: str
    loss: str
    tables: tuple[str, ...]
    datasets: tuple[str, ...] = ("koa", "adni")
    tau: float = 1.0
    gamma: float = 1.5


EXPERIMENTS: dict[str, Experiment] = {
    "baseline_ce": Experiment(
        "baseline_ce", "ConvNeXt-T + CE", "none", "ce", ("main", "mechanism", "losses")
    ),
    "baseline_dast": Experiment(
        "baseline_dast",
        "ConvNeXt-T + DAST",
        "none",
        "dast",
        ("main", "mechanism", "losses", "sensitivity"),
    ),
    "mesc_ce": Experiment(
        "mesc_ce",
        "ConvNeXt-T + MESC + CE",
        "mesc",
        "ce",
        ("main", "mechanism", "attention"),
    ),
    "mesc_dast": Experiment(
        "mesc_dast",
        "ConvNeXt-T + MESC + DAST",
        "mesc",
        "dast",
        ("main", "routing", "mechanism"),
    ),
    "mesc_equal_dast": Experiment(
        "mesc_equal_dast",
        "ConvNeXt-T + MESC-Equal + DAST",
        "mesc_equal",
        "dast",
        ("routing",),
    ),
    "mesc_direct_dast": Experiment(
        "mesc_direct_dast",
        "ConvNeXt-T + MESC-Direct + DAST",
        "mesc_direct",
        "dast",
        ("routing",),
    ),
    "baseline_sord": Experiment(
        "baseline_sord",
        "ConvNeXt-T + SORD",
        "none",
        "sord_ce",
        ("mechanism", "losses"),
    ),
    "mesc_sord": Experiment(
        "mesc_sord",
        "ConvNeXt-T + MESC + SORD",
        "mesc",
        "sord_ce",
        ("mechanism",),
    ),
    "se_ce": Experiment("se_ce", "ConvNeXt-T + SE + CE", "se", "ce", ("attention",)),
    "cbam_ce": Experiment("cbam_ce", "ConvNeXt-T + CBAM + CE", "cbam", "ce", ("attention",)),
    "eca_ce": Experiment("eca_ce", "ConvNeXt-T + ECA + CE", "eca", "ce", ("attention",)),
    "msca_ce": Experiment("msca_ce", "ConvNeXt-T + MSCA + CE", "msca", "ce", ("attention",)),
    "baseline_label_smoothing": Experiment(
        "baseline_label_smoothing",
        "ConvNeXt-T + Label Smoothing",
        "none",
        "label_smoothing_ce",
        ("losses",),
    ),
    "baseline_coral": Experiment(
        "baseline_coral", "ConvNeXt-T + CORAL", "none", "coral", ("losses",)
    ),
}

for tau, gamma in ((0.5, 1.5), (1.0, 0.5), (1.0, 1.0), (1.0, 2.0), (1.5, 1.5), (2.0, 1.5)):
    tau_slug = str(tau).replace(".", "p")
    gamma_slug = str(gamma).replace(".", "p")
    key = f"sensitivity_tau{tau_slug}_gamma{gamma_slug}"
    EXPERIMENTS[key] = Experiment(
        key,
        f"ConvNeXt-T + DAST (tau={tau:g}, gamma={gamma:g})",
        "none",
        "dast",
        ("sensitivity",),
        datasets=("koa",),
        tau=tau,
        gamma=gamma,
    )


SUITE_ORDER = {
    "core": ("baseline_ce", "baseline_dast", "mesc_ce", "mesc_dast"),
    "main": ("baseline_ce", "baseline_dast", "mesc_ce", "mesc_dast"),
    "routing": ("mesc_equal_dast", "mesc_direct_dast", "mesc_dast"),
    "mechanism": (
        "baseline_ce",
        "baseline_sord",
        "baseline_dast",
        "mesc_ce",
        "mesc_sord",
        "mesc_dast",
    ),
    "attention": ("se_ce", "cbam_ce", "eca_ce", "msca_ce", "mesc_ce"),
    "losses": (
        "baseline_ce",
        "baseline_label_smoothing",
        "baseline_sord",
        "baseline_coral",
        "baseline_dast",
    ),
    "sensitivity": (
        "sensitivity_tau0p5_gamma1p5",
        "sensitivity_tau1p0_gamma0p5",
        "sensitivity_tau1p0_gamma1p0",
        "baseline_dast",
        "sensitivity_tau1p0_gamma2p0",
        "sensitivity_tau1p5_gamma1p5",
        "sensitivity_tau2p0_gamma1p5",
    ),
}


def ordered_union(groups: Iterable[Sequence[str]]) -> tuple[str, ...]:
    values: list[str] = []
    for group in groups:
        for value in group:
            if value not in values:
                values.append(value)
    return tuple(values)


SUITE_ORDER["all"] = ordered_union(
    SUITE_ORDER[name]
    for name in ("main", "routing", "mechanism", "attention", "losses", "sensitivity")
)


CSV_FIELDS = (
    "dataset",
    "backbone",
    "insertion_point",
    "stage3_channels",
    "experiment",
    "method",
    "attention",
    "criterion",
    "tables",
    "seed",
    "dast_tau",
    "dast_gamma",
    "sord_tau",
    "label_smoothing",
    "pretrained",
    "optimizer_profile",
    "base_lr",
    "koa_backbone_lr",
    "koa_head_lr",
    "weight_decay",
    "epochs",
    "patience",
    "batch_size",
    "image_size",
    "data_root",
    *TEST_METRICS,
    "checkpoint",
    "log",
    "run_tag",
    "completed_at",
)


def parse_csv(raw: str) -> list[str]:
    return [value.strip().lower() for value in raw.split(",") if value.strip()]


def sanitize_tag(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "._-" else "-" for ch in value.strip())
    while "--" in cleaned:
        cleaned = cleaned.replace("--", "-")
    return cleaned.strip("._-")


def parse_seeds(raw: str) -> list[int]:
    seeds = [int(value) for value in parse_csv(raw)]
    if not seeds:
        raise ValueError("at least one seed is required")
    if len(seeds) != len(set(seeds)):
        raise ValueError("seeds contain duplicates")
    return seeds


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=tuple(SUITE_ORDER), default="core")
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
    parser.add_argument("--base-lr", type=float, default=1e-4)
    parser.add_argument("--koa-backbone-lr", type=float, default=1e-4)
    parser.add_argument("--koa-head-lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dast-tau", type=float, default=1.0)
    parser.add_argument("--dast-gamma", type=float, default=1.5)
    parser.add_argument("--sord-tau", type=float, default=1.0)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--conda-env", default="none")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--tag-prefix", default="")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--verbose-trainer", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    try:
        args.seed_values = parse_seeds(args.seeds)
    except ValueError as exc:
        parser.error(str(exc))
    args.dataset_names = parse_csv(args.datasets)
    if not args.dataset_names or any(value not in {"koa", "adni"} for value in args.dataset_names):
        parser.error("--datasets supports koa, adni, or koa,adni")
    if len(args.dataset_names) != len(set(args.dataset_names)):
        parser.error("datasets contain duplicates")

    experiment_keys = parse_csv(args.experiments) if args.experiments else list(SUITE_ORDER[args.suite])
    unknown = [key for key in experiment_keys if key not in EXPERIMENTS]
    if unknown:
        parser.error(f"unknown experiments: {unknown}; choices: {list(EXPERIMENTS)}")
    if len(experiment_keys) != len(set(experiment_keys)):
        parser.error("experiments contain duplicates")
    args.experiment_values = [EXPERIMENTS[key] for key in experiment_keys]

    args.koa_data_root = args.koa_data_root or os.getenv("KNEE_DATA_ROOT")
    args.adni_data_root = args.adni_data_root or os.getenv("ALZHEIMER_DATA_ROOT")
    for dataset_name in args.dataset_names:
        value = getattr(args, f"{dataset_name}_data_root")
        if not value:
            parser.error(f"{dataset_name.upper()} requires its data-root argument or environment variable")
        setattr(args, f"{dataset_name}_data_root", Path(value).expanduser().resolve())

    positive = (
        args.epochs,
        args.patience,
        args.batch_size,
        args.image_size,
        args.base_lr,
        args.koa_backbone_lr,
        args.koa_head_lr,
        args.dast_tau,
        args.sord_tau,
    )
    if any(value <= 0 for value in positive):
        parser.error("epochs, patience, sizes, learning rates, and tau values must be positive")
    if args.early_delta < 0 or args.weight_decay < 0 or args.dast_gamma < 0:
        parser.error("early delta, weight decay, and DAST gamma cannot be negative")
    if not 0.0 <= args.label_smoothing < 1.0:
        parser.error("label smoothing must satisfy 0 <= value < 1")
    if args.num_workers is not None and args.num_workers < 0:
        parser.error("num workers cannot be negative")
    if args.resume and args.output is None:
        parser.error("--resume requires --output")
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


def configured_tau_gamma(args: argparse.Namespace, experiment: Experiment) -> tuple[float, float]:
    if experiment.key.startswith("sensitivity_"):
        return experiment.tau, experiment.gamma
    return args.dast_tau, args.dast_gamma


def build_command(
    args: argparse.Namespace,
    dataset: str,
    experiment: Experiment,
    seed: int,
    run_tag: str,
) -> list[str]:
    tau, gamma = configured_tau_gamma(args, experiment)
    data_root = getattr(args, f"{dataset}_data_root")
    workers = args.num_workers if args.num_workers is not None else (4 if dataset == "koa" else 2)
    command = [
        *python_prefix(args),
        str(TRAINER),
        "--dataset",
        dataset,
        "--data-root",
        str(data_root),
        "--attention",
        experiment.attention,
        "--loss",
        experiment.loss,
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
        str(workers),
        "--early-delta",
        str(args.early_delta),
        "--base-lr",
        str(args.base_lr),
        "--koa-backbone-lr",
        str(args.koa_backbone_lr),
        "--koa-head-lr",
        str(args.koa_head_lr),
        "--weight-decay",
        str(args.weight_decay),
        "--dast-tau",
        str(tau),
        "--dast-gamma",
        str(gamma),
        "--sord-tau",
        str(args.sord_tau),
        "--label-smoothing",
        str(args.label_smoothing),
        "--run-tag",
        run_tag,
    ]
    if args.no_pretrained:
        command.append("--no-pretrained")
    return command


def parse_test_metrics(line: str) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for name, raw_value in METRIC_RE.findall(line):
        value = float(raw_value)
        if name in PERCENT_METRICS:
            value /= 100.0
        metrics["test_loss" if name == "loss" else name] = value
    return metrics


def write_rows(path: Path, rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def read_rows(path: Path) -> list[dict[str, object]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def validate_resume_rows(rows: Sequence[dict[str, object]], args: argparse.Namespace) -> None:
    expected = {
        "backbone": "ConvNeXt-T",
        "insertion_point": "after_stage3_features_5",
        "stage3_channels": "384",
        "pretrained": str(not args.no_pretrained),
        "base_lr": str(args.base_lr),
        "koa_backbone_lr": str(args.koa_backbone_lr),
        "koa_head_lr": str(args.koa_head_lr),
        "weight_decay": str(args.weight_decay),
        "epochs": str(args.epochs),
        "patience": str(args.patience),
        "batch_size": str(args.batch_size),
        "image_size": str(args.image_size),
    }
    for index, row in enumerate(rows, start=2):
        for field, expected_value in expected.items():
            if str(row.get(field, "")) != expected_value:
                raise ValueError(
                    f"Cannot resume: row {index} has {field}={row.get(field)!r}, "
                    f"expected {expected_value!r}."
                )
        dataset = str(row.get("dataset", "")).lower()
        expected_root = str(getattr(args, f"{dataset}_data_root"))
        if str(row.get("data_root", "")) != expected_root:
            raise ValueError(f"Cannot resume: row {index} uses a different {dataset} data root.")


def run_one(
    args: argparse.Namespace,
    dataset: str,
    experiment: Experiment,
    seed: int,
    run_tag: str,
    log_path: Path,
) -> dict[str, object] | None:
    command = build_command(args, dataset, experiment, seed, run_tag)
    print(f"[RUN] {dataset.upper()} | {experiment.method} | seed={seed}", flush=True)
    if args.dry_run:
        print("      " + " ".join(command), flush=True)
        return {}

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    omp_threads = env.get("OMP_NUM_THREADS", "").strip()
    if not omp_threads.isdigit() or int(omp_threads) <= 0:
        env["OMP_NUM_THREADS"] = str(max(1, args.num_workers or 2))

    log_path.parent.mkdir(parents=True, exist_ok=True)
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
                parsed = parse_test_metrics(stripped)
                if parsed:
                    test_metrics = parsed
            elif test_metrics is not None and (
                "ovr_roc_auc_macro=" in stripped or "ovr_pr_auc_macro=" in stripped
            ):
                test_metrics.update(parse_test_metrics(stripped))
            checkpoint_match = CHECKPOINT_RE.match(stripped)
            if checkpoint_match:
                checkpoint = checkpoint_match.group(1)

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

    tau, gamma = configured_tau_gamma(args, experiment)
    row: dict[str, object] = {
        "dataset": dataset.upper(),
        "backbone": "ConvNeXt-T",
        "insertion_point": "after_stage3_features_5",
        "stage3_channels": 384,
        "experiment": experiment.key,
        "method": experiment.method,
        "attention": experiment.attention,
        "criterion": experiment.loss,
        "tables": "|".join(experiment.tables),
        "seed": seed,
        "dast_tau": tau if experiment.loss == "dast" else "",
        "dast_gamma": gamma if experiment.loss == "dast" else "",
        "sord_tau": args.sord_tau if experiment.loss == "sord_ce" else "",
        "label_smoothing": args.label_smoothing if experiment.loss == "label_smoothing_ce" else "",
        "pretrained": not args.no_pretrained,
        "optimizer_profile": "koa_adamw_2group" if dataset == "koa" else "adni_progressive_adam",
        "base_lr": args.base_lr,
        "koa_backbone_lr": args.koa_backbone_lr,
        "koa_head_lr": args.koa_head_lr,
        "weight_decay": args.weight_decay,
        "epochs": args.epochs,
        "patience": args.patience,
        "batch_size": args.batch_size,
        "image_size": args.image_size,
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
            sd = statistics.stdev(values) if len(values) > 1 else 0.0
            print(f"    {metric}: {statistics.mean(values):.4f} +/- {sd:.4f}")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.dry_run:
        for dataset in args.dataset_names:
            validate_data_root(dataset, getattr(args, f"{dataset}_data_root"))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = sanitize_tag(args.tag_prefix) or f"convnext_t_{args.suite}_{timestamp}"
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
        for dataset in args.dataset_names
        for seed in args.seed_values
        for experiment in args.experiment_values
        if dataset in experiment.datasets
        and not (args.suite == "sensitivity" and not args.experiments and dataset != "koa")
    ]
    if not schedule:
        raise ValueError("No runs remain after applying dataset support to the selected experiments.")

    counts = defaultdict(int)
    for dataset, _, _ in schedule:
        counts[dataset] += 1
    print("Backbone       : ConvNeXt-T")
    print("MESC insertion : after stage3 (features[5], 384 channels)")
    print(f"Suite          : {args.suite}")
    print(f"Experiments    : {[experiment.key for experiment in args.experiment_values]}")
    print(f"Datasets       : {args.dataset_names}")
    print(f"Seeds          : {args.seed_values}")
    print(f"Runs           : {len(schedule)} | by dataset={dict(counts)}")
    print(f"Already done   : {len(completed)}")
    print(f"Results        : {output}")
    print(f"Logs           : {log_dir}")

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
