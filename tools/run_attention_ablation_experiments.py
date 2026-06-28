#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Launch fair layer3 attention baselines and MESC internal ablations."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable, List


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = PROJECT_ROOT / "tools" / "train_layer3_attention.py"
DEFAULT_SEEDS = (42, 777, 1234, 2024, 3407)
ATTENTION_BASELINES = ("se", "cbam", "eca", "msca")
ALL_ATTENTION_MODULES = ("se", "cbam", "eca", "msca", "mesc")
MESC_VARIANTS = ("avg_only", "avg_max", "avg_max_median", "spatial_only", "channel_only", "full")
DISPLAY_DATASETS = {
    "koa": "KOA",
    "adni": "ADNI",
    "chest": "Chest X-ray Image",
    "ham10000": "HAM10000",
    "brain": "Brain Tumor MRI",
}


def parse_csv(raw: str) -> List[str]:
    return [x.strip().lower() for x in raw.split(",") if x.strip()]


def parse_ints(raw: str) -> List[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def python_command(args: argparse.Namespace) -> List[str]:
    conda_env = (args.conda_env or "").strip()
    if conda_env and conda_env.lower() not in {"none", "null", "false", "0"}:
        return ["conda", "run", "-n", conda_env, "python"]
    return [args.python]


def command(args: argparse.Namespace, dataset: str, attention: str, seed: int) -> List[str]:
    base = python_command(args)
    tag = f"{args.suite}_{dataset}_{attention}_{args.loss}_seed{seed}"
    cmd = base + [str(TRAIN_SCRIPT), "--dataset", dataset, "--attention", attention, "--loss", args.loss, "--seed", str(seed), "--run-tag", tag]
    if args.epochs is not None:
        cmd += ["--epochs", str(args.epochs)]
    if args.batch_size is not None:
        cmd += ["--batch-size", str(args.batch_size)]
    if args.image_size is not None:
        cmd += ["--image-size", str(args.image_size)]
    if args.num_workers is not None:
        cmd += ["--num-workers", str(args.num_workers)]
    if args.patience is not None:
        cmd += ["--patience", str(args.patience)]
    if args.early_delta is not None:
        cmd += ["--early-delta", str(args.early_delta)]
    return cmd


def write_header(fp, dataset: str, attention: str, seed: int, loss: str, cmd: List[str]) -> None:
    fp.write(f"DATASET: {DISPLAY_DATASETS.get(dataset, dataset)}\n")
    fp.write(f"METHOD: ResNet50 + layer3 {attention}\n")
    fp.write(f"SEED: {seed}\n")
    fp.write(f"ATTENTION: {attention}\n")
    fp.write(f"LOSS: {loss}\n")
    fp.write("COMMAND: " + " ".join(cmd) + "\n")
    fp.write("=" * 90 + "\n")
    fp.flush()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("attention", "mesc-internal"), default="attention")
    parser.add_argument("--datasets", default="koa,adni,chest")
    parser.add_argument(
        "--attentions",
        default=",".join(ATTENTION_BASELINES),
        help="Comma-separated modules. Use 'all' for se,cbam,eca,msca,mesc.",
    )
    parser.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    parser.add_argument("--loss", choices=("ce", "dast"), default="ce")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--early-delta", type=float, default=None)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--conda-env", default="Paper")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-runs", type=int, default=0)
    args = parser.parse_args()
    if args.suite == "attention" and args.loss != "ce":
        raise ValueError("The attention comparison suite is CE-only.")

    attentions = ALL_ATTENTION_MODULES if args.attentions.strip().lower() == "all" else tuple(parse_csv(args.attentions))
    if args.suite == "mesc-internal":
        attentions = MESC_VARIANTS
    out_dir = PROJECT_ROOT / "batch_logs" / f"{args.suite}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    rc = 0
    scheduled_index = 0
    for dataset in parse_csv(args.datasets):
        for seed in parse_ints(args.seeds):
            for attention in attentions:
                scheduled_index += 1
                if scheduled_index <= args.skip_runs:
                    print(f"[SKIP] {dataset} | {attention} | seed={seed}")
                    continue
                cmd = command(args, dataset, attention, seed)
                log_path = out_dir / f"{args.suite}_{dataset}_{attention}_{args.loss}_seed{seed}.log"
                print(f"[RUN] {dataset} | {attention} | seed={seed} -> {log_path}")
                if args.dry_run:
                    print("      " + " ".join(cmd))
                    continue
                with log_path.open("w", encoding="utf-8", newline="") as fp:
                    write_header(fp, dataset, attention, seed, args.loss, cmd)
                    proc = subprocess.run(cmd, cwd=PROJECT_ROOT, stdout=fp, stderr=subprocess.STDOUT, env=os.environ.copy())
                    fp.write("\n" + "=" * 90 + "\n")
                    fp.write(f"RETURN_CODE: {proc.returncode}\n")
                rc = max(rc, proc.returncode)
    print(f"Logs written under: {out_dir}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
