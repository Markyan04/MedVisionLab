#!/usr/bin/env python
"""Compare MESC-B after ResNet layer1/layer2/layer3/layer4 on KOA and ADNI.

The default command runs the complete proposed setting (MESC-B + DAST) on
both datasets with seed 9. Every run has a unique checkpoint and full log;
held-out Test metrics are also written incrementally to one CSV file.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ADAPTER = PROJECT_ROOT / "tools" / "run_training_with_mecsb.py"
TRAIN_SCRIPTS = {
    "koa": {
        "ce": PROJECT_ROOT / "Knee" / "ResNet_layer3+MECS+CE.py",
        "dast": PROJECT_ROOT / "Knee" / "ResNet_layer3+MECS+Loss4.py",
    },
    "adni": {
        "ce": PROJECT_ROOT / "Alzheimer_MRI_Loss" / "ResNet_layer3+MECS.py",
        "dast": PROJECT_ROOT / "Alzheimer_MRI_Loss" / "ResNet_layer3+MECS.py",
    },
}
LAYER_CHANNELS = {
    "layer1": 256,
    "layer2": 512,
    "layer3": 1024,
    "layer4": 2048,
}
METRIC_RE = re.compile(r"([A-Za-z0-9_]+)=(-?[0-9]+(?:\.[0-9]+)?)%?")
PERCENT_METRICS = {"top1", "top2", "top3", "acc", "bal_acc"}
CSV_FIELDS = (
    "dataset", "method", "attention", "insert_after", "channels", "loss",
    "seed", "tau", "gamma", "test_loss", "top1", "top2", "top3", "acc",
    "bal_acc", "macro_f1", "qwk", "mae", "weighted_f1",
    "precision_macro", "recall_macro", "ovr_roc_auc_macro",
    "ovr_pr_auc_macro", "checkpoint", "log", "run_tag", "completed_at",
)


def comma_list(raw: str) -> list[str]:
    return [part.strip().lower() for part in raw.split(",") if part.strip()]


def parse_seeds(raw: str) -> list[int]:
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("at least one seed is required")
    if len(values) != len(set(values)):
        raise ValueError("seeds contain duplicates")
    return values


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", default="koa,adni")
    parser.add_argument("--layers", default="layer1,layer2,layer3,layer4")
    parser.add_argument("--seeds", default="9")
    parser.add_argument("--loss", choices=("ce", "dast"), default="dast")
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
    args.layer_names = comma_list(args.layers)
    if not args.dataset_names or any(name not in TRAIN_SCRIPTS for name in args.dataset_names):
        parser.error("--datasets supports koa, adni, or koa,adni.")
    if len(args.dataset_names) != len(set(args.dataset_names)):
        parser.error("--datasets contains duplicates.")
    if not args.layer_names or any(name not in LAYER_CHANNELS for name in args.layer_names):
        parser.error("--layers supports layer1,layer2,layer3,layer4.")
    if len(args.layer_names) != len(set(args.layer_names)):
        parser.error("--layers contains duplicates.")
    try:
        args.seed_values = parse_seeds(args.seeds)
    except ValueError as exc:
        parser.error(str(exc))
    if args.epochs <= 0 or args.patience <= 0:
        parser.error("--epochs and --patience must be positive.")
    if args.skip_runs < 0:
        parser.error("--skip-runs cannot be negative.")
    if args.dast_tau <= 0 or args.dast_gamma < 0:
        parser.error("DAST tau must be positive and gamma cannot be negative.")

    args.koa_data_root = args.koa_data_root or os.getenv("KNEE_DATA_ROOT")
    args.adni_data_root = args.adni_data_root or os.getenv("ALZHEIMER_DATA_ROOT")
    for name in args.dataset_names:
        attr = f"{name}_data_root"
        value = getattr(args, attr)
        if not value:
            env_name = "KNEE_DATA_ROOT" if name == "koa" else "ALZHEIMER_DATA_ROOT"
            parser.error(f"{name.upper()} requires --{name}-data-root or {env_name}.")
        setattr(args, attr, Path(value).expanduser().resolve())
    return args


def validate_data(args: argparse.Namespace) -> None:
    if "koa" in args.dataset_names:
        missing = [
            name for name in ("train", "val", "test")
            if not (args.koa_data_root / name).is_dir()
        ]
        if missing:
            raise FileNotFoundError(
                f"KOA root {args.koa_data_root} is missing split directories: {missing}"
            )
    if "adni" in args.dataset_names and not args.adni_data_root.is_dir():
        raise FileNotFoundError(f"ADNI root not found: {args.adni_data_root}")


def python_prefix(args: argparse.Namespace) -> list[str]:
    env_name = args.conda_env.strip()
    if env_name.lower() not in {"", "none", "null", "false", "0"}:
        return ["conda", "run", "--no-capture-output", "-n", env_name, "python", "-u"]
    return [args.python, "-u"]


def build_command(dataset_name: str, layer_name: str, args: argparse.Namespace) -> list[str]:
    return python_prefix(args) + [
        str(ADAPTER), "--variant", "median", "--insert-after", layer_name,
        "--script", str(TRAIN_SCRIPTS[dataset_name][args.loss]),
    ]


def build_env(
    dataset_name: str,
    layer_name: str,
    seed: int,
    tag: str,
    args: argparse.Namespace,
) -> dict[str, str]:
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
    env["MESC_IMPLEMENTATION"] = "MECS_VersionB"
    env["MESC_INSERT_AFTER"] = layer_name
    if dataset_name == "adni":
        env["ALZHEIMER_LOSSES"] = args.loss
    if args.loss == "dast":
        env[f"{prefix}_DAST_TAU"] = str(args.dast_tau)
        env[f"{prefix}_DAST_GAMMA"] = str(args.dast_gamma)
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


def expected_checkpoint(
    dataset_name: str,
    layer_name: str,
    tag: str,
    loss: str,
) -> Path:
    if dataset_name == "koa":
        if loss == "ce":
            filename = f"best_resnet50_mecs_{layer_name}_knee_oa_{tag}.pt"
        else:
            filename = f"best_resnet50_mecs_{layer_name}_dast_knee_oa_new_{tag}.pt"
        return PROJECT_ROOT / "Knee" / "checkpoints" / filename
    filename = f"best_ResNet_{layer_name}+MECS_{loss}_{tag}.pt"
    return PROJECT_ROOT / "Alzheimer_MRI_Loss" / "checkpoints" / filename


def write_rows(path: Path, rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_rows(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def run_one(
    dataset_name: str,
    layer_name: str,
    seed: int,
    tag_prefix: str,
    log_dir: Path,
    args: argparse.Namespace,
) -> dict[str, object] | None:
    tag = f"{tag_prefix}_{dataset_name}_mecsb_{layer_name}_{args.loss}_seed{seed}"
    command = build_command(dataset_name, layer_name, args)
    env = build_env(dataset_name, layer_name, seed, tag, args)
    log_path = log_dir / f"{tag}.log"
    print(
        f"[RUN] {dataset_name.upper()} | MESC-B after {layer_name} + "
        f"{args.loss.upper()} | seed={seed} -> {log_path}",
        flush=True,
    )
    if args.dry_run:
        print("      " + " ".join(command), flush=True)
        return {}

    tail: deque[str] = deque(maxlen=50)
    test_metrics = None
    waiting_for_auc = False
    log_path.parent.mkdir(parents=True, exist_ok=True)
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
    with log_path.open("w", encoding="utf-8", newline="") as log_handle:
        log_handle.write("COMMAND: " + " ".join(command) + "\n")
        log_handle.write(f"RUN_TAG: {tag}\n")
        log_handle.write(f"INSERT_AFTER: {layer_name}\n")
        log_handle.write("=" * 90 + "\n")
        for raw_line in process.stdout:
            log_handle.write(raw_line)
            log_handle.flush()
            line = raw_line.rstrip("\r\n")
            tail.append(line)
            stripped = line.strip()
            if stripped.startswith("Epoch "):
                print(
                    f"[{dataset_name.upper()}/{layer_name}] {stripped}",
                    flush=True,
                )
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
        reason = f"return code {return_code}" if return_code else "Test metrics were not found"
        print(
            f"[FAIL] {dataset_name.upper()}, {layer_name}, seed={seed}: {reason}",
            file=sys.stderr,
        )
        print(f"       Full log: {log_path}", file=sys.stderr)
        for line in tail:
            print(line, file=sys.stderr)
        return None

    ordered = (
        "loss", "top1", "top2", "top3", "acc", "bal_acc", "macro_f1",
        "qwk", "mae", "weighted_f1", "precision_macro", "recall_macro",
        "ovr_roc_auc_macro", "ovr_pr_auc_macro",
    )
    printable = []
    for name in ordered:
        if name in test_metrics:
            suffix = "%" if name in PERCENT_METRICS else ""
            printable.append(f"{name}={test_metrics[name]:.4f}{suffix}")
    print(
        f"[TEST] {dataset_name} | {layer_name} | seed={seed} | " + " | ".join(printable),
        flush=True,
    )

    row: dict[str, object] = {
        "dataset": dataset_name.upper(),
        "method": f"ResNet50 + MESC-B after {layer_name} + {args.loss.upper()}",
        "attention": "MESC-B",
        "insert_after": layer_name,
        "channels": LAYER_CHANNELS[layer_name],
        "loss": args.loss.upper(),
        "seed": seed,
        "tau": args.dast_tau if args.loss == "dast" else "",
        "gamma": args.dast_gamma if args.loss == "dast" else "",
        "test_loss": test_metrics.pop("loss"),
        "checkpoint": str(expected_checkpoint(dataset_name, layer_name, tag, args.loss)),
        "log": str(log_path),
        "run_tag": tag,
        "completed_at": datetime.now().isoformat(timespec="seconds"),
    }
    for name, value in test_metrics.items():
        row[name] = value / 100.0 if name in PERCENT_METRICS else value
    return row


def print_summary(rows: Sequence[dict[str, object]]) -> None:
    print("\nLayer-position results")
    for dataset_name in dict.fromkeys(str(row["dataset"]) for row in rows):
        print(f"  {dataset_name}")
        selected = [row for row in rows if row["dataset"] == dataset_name]
        for row in selected:
            print(
                f"    {row['insert_after']}: acc={float(row['acc']):.4f}, "
                f"macro_f1={float(row['macro_f1']):.4f}, "
                f"qwk={float(row['qwk']):.4f}, mae={float(row['mae']):.4f}"
            )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.dry_run:
        validate_data(args)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag_prefix = args.tag_prefix or f"mecsb_layer_position_{timestamp}"
    output = args.output or PROJECT_ROOT / "analysis_tables" / f"{tag_prefix}_test_results.csv"
    if not output.is_absolute():
        output = PROJECT_ROOT / output
    log_dir = PROJECT_ROOT / "batch_logs" / tag_prefix
    total = len(args.seed_values) * len(args.dataset_names) * len(args.layer_names)
    print(f"Datasets : {args.dataset_names}")
    print(f"Layers   : {args.layer_names}")
    print(f"Seeds    : {args.seed_values}")
    print(f"Loss     : {args.loss}")
    print(f"Runs     : {total}")
    if args.loss == "dast":
        print(f"DAST     : tau={args.dast_tau}, gamma={args.dast_gamma}")
    print(f"Logs     : {log_dir}")
    print(f"Results  : {output}")

    rows: list[dict[str, object]] = read_rows(output) if args.skip_runs else []
    if rows:
        if len(rows) != args.skip_runs:
            raise ValueError(
                f"--skip-runs={args.skip_runs}, but {output} contains {len(rows)} rows. "
                "Use the number of completed CSV rows when resuming."
            )
        print(f"Resume   : loaded {len(rows)} completed rows from {output}")
    scheduled = 0
    for seed in args.seed_values:
        for dataset_name in args.dataset_names:
            for layer_name in args.layer_names:
                scheduled += 1
                if scheduled <= args.skip_runs:
                    print(f"[SKIP] {dataset_name.upper()} | {layer_name} | seed={seed}")
                    continue
                row = run_one(
                    dataset_name,
                    layer_name,
                    seed,
                    tag_prefix,
                    log_dir,
                    args,
                )
                if args.dry_run:
                    continue
                if row is None:
                    write_rows(output, rows)
                    print(f"Stopped after failure. Partial CSV: {output}", file=sys.stderr)
                    return 1
                rows.append(row)
                write_rows(output, rows)
    if not args.dry_run:
        print_summary(rows)
        print(f"\nResults: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
