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
ATTENTION_BASELINES = ("se", "cbam", "eca", "mesc")
MESC_VARIANTS = ("avg_only", "avg_max", "avg_max_median", "spatial_only", "channel_only", "full")


def parse_csv(raw: str) -> List[str]:
    return [x.strip().lower() for x in raw.split(",") if x.strip()]


def parse_ints(raw: str) -> List[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def command(args: argparse.Namespace, dataset: str, attention: str, seed: int) -> List[str]:
    base = ["conda", "run", "-n", args.conda_env, "python"] if args.conda_env else [args.python]
    tag = f"{args.suite}_{dataset}_{attention}_{args.loss}_seed{seed}"
    cmd = base + [str(TRAIN_SCRIPT), "--dataset", dataset, "--attention", attention, "--loss", args.loss, "--seed", str(seed), "--run-tag", tag]
    if args.epochs is not None:
        cmd += ["--epochs", str(args.epochs)]
    if args.batch_size is not None:
        cmd += ["--batch-size", str(args.batch_size)]
    return cmd


def write_header(fp, dataset: str, attention: str, seed: int, cmd: List[str]) -> None:
    fp.write(f"DATASET: {dataset}\n")
    fp.write(f"METHOD: ResNet50 + layer3 {attention}\n")
    fp.write(f"SEED: {seed}\n")
    fp.write(f"ATTENTION: {attention}\n")
    fp.write("COMMAND: " + " ".join(cmd) + "\n")
    fp.write("=" * 90 + "\n")
    fp.flush()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("attention", "mesc-internal"), default="attention")
    parser.add_argument("--datasets", default="ham10000,adni,brain,chest")
    parser.add_argument("--seeds", default="1234")
    parser.add_argument("--loss", choices=("ce", "dast"), default="ce")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--conda-env", default="Paper")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    attentions = ATTENTION_BASELINES if args.suite == "attention" else MESC_VARIANTS
    out_dir = PROJECT_ROOT / "batch_logs" / f"{args.suite}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    rc = 0
    for dataset in parse_csv(args.datasets):
        for seed in parse_ints(args.seeds):
            for attention in attentions:
                cmd = command(args, dataset, attention, seed)
                log_path = out_dir / f"{args.suite}_{dataset}_{attention}_{args.loss}_seed{seed}.log"
                print(f"[RUN] {dataset} | {attention} | seed={seed} -> {log_path}")
                if args.dry_run:
                    print("      " + " ".join(cmd))
                    continue
                with log_path.open("w", encoding="utf-8", newline="") as fp:
                    write_header(fp, dataset, attention, seed, cmd)
                    proc = subprocess.run(cmd, cwd=PROJECT_ROOT, stdout=fp, stderr=subprocess.STDOUT, env=os.environ.copy())
                    fp.write("\n" + "=" * 90 + "\n")
                    fp.write(f"RETURN_CODE: {proc.returncode}\n")
                rc = max(rc, proc.returncode)
    print(f"Logs written under: {out_dir}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
