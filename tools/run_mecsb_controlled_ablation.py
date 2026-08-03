#!/usr/bin/env python
"""Run the complete KOA/ADNI controlled ablation with MECS VersionB.

For each seed and dataset this launches four arms:

1. ResNet50 + CE
2. ResNet50 + fixed DAST
3. ResNet50 + MECS VersionB + CE
4. ResNet50 + MECS VersionB + fixed DAST

Baseline arms use the untouched original scripts.  MESC arms are executed via
an import-time adapter that substitutes MECS VersionB without editing the
original training sources.  Unique timestamped run tags keep prior checkpoints
and results intact.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

import run_multiseed_experiments as base


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VERSION_B_ADAPTER = PROJECT_ROOT / "tools" / "run_training_with_mecsb.py"
ALLOWED_DATASETS = ("koa", "adni")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", default="koa,adni", help="koa, adni, or koa,adni")
    parser.add_argument("--seeds", default=",".join(map(str, base.DEFAULT_SEEDS)))
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
    parser.add_argument(
        "--conda-env",
        default="none",
        help="Conda environment name, or 'none' to use the currently active Python.",
    )
    parser.add_argument(
        "--tag-prefix",
        default=None,
        help="Checkpoint/run-tag prefix. Default includes a timestamp to avoid overwrites.",
    )
    parser.add_argument("--skip-runs", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    args.dataset_names = base.parse_csv_list(args.datasets)
    if not args.dataset_names or any(name not in ALLOWED_DATASETS for name in args.dataset_names):
        parser.error("--datasets only supports koa, adni, or koa,adni.")
    if len(set(args.dataset_names)) != len(args.dataset_names):
        parser.error("--datasets contains duplicates.")
    args.seed_values = base.parse_ints(args.seeds)
    if not args.seed_values:
        parser.error("--seeds must contain at least one integer.")
    if args.epochs <= 0 or args.patience <= 0:
        parser.error("--epochs and --patience must be positive.")
    if args.skip_runs < 0:
        parser.error("--skip-runs must be non-negative.")
    if args.dast_tau <= 0 or args.dast_gamma < 0:
        parser.error("--dast-tau must be > 0 and --dast-gamma must be >= 0.")

    if args.koa_data_root is None and "koa" in args.dataset_names:
        raw = os.getenv("KNEE_DATA_ROOT", "").strip()
        args.koa_data_root = Path(raw) if raw else None
    if args.adni_data_root is None and "adni" in args.dataset_names:
        raw = os.getenv("ALZHEIMER_DATA_ROOT", "").strip()
        args.adni_data_root = Path(raw) if raw else None
    if "koa" in args.dataset_names and args.koa_data_root is None:
        parser.error("KOA requires --koa-data-root or KNEE_DATA_ROOT.")
    if "adni" in args.dataset_names and args.adni_data_root is None:
        parser.error("ADNI requires --adni-data-root or ALZHEIMER_DATA_ROOT.")

    for attr in ("koa_data_root", "adni_data_root"):
        value = getattr(args, attr)
        if value is not None:
            setattr(args, attr, value.expanduser().resolve())
    return args


def validate_data_roots(args: argparse.Namespace) -> None:
    if "koa" in args.dataset_names:
        missing = [name for name in ("train", "val", "test") if not (args.koa_data_root / name).is_dir()]
        if missing:
            raise FileNotFoundError(
                f"KOA data root {args.koa_data_root} is missing split directories: {missing}"
            )
    if "adni" in args.dataset_names and not args.adni_data_root.is_dir():
        raise FileNotFoundError(f"ADNI data root not found: {args.adni_data_root}")


def is_mesc(spec: base.RunSpec) -> bool:
    return "MESC" in spec.method


def version_b_spec(spec: base.RunSpec) -> base.RunSpec:
    if not is_mesc(spec):
        return spec
    method = spec.method.replace("MESC", "MESC VersionB")
    return base.RunSpec(
        dataset=spec.dataset,
        method=method,
        script=spec.script,
        env_prefix=spec.env_prefix,
        losses_env=spec.losses_env,
        losses_value=spec.losses_value,
    )


def training_command(args: argparse.Namespace, spec: base.RunSpec) -> list[str]:
    if not is_mesc(spec):
        return base.python_command(args, spec.script)
    return base.python_command(args, VERSION_B_ADAPTER) + ["--script", str(spec.script)]


def build_env(spec: base.RunSpec, seed: int, args: argparse.Namespace, tag_prefix: str) -> dict[str, str]:
    tau = args.dast_tau if "DAST" in spec.method else None
    gamma = args.dast_gamma if "DAST" in spec.method else None
    env = base.build_env(spec, seed, args, tag_prefix, tau=tau, gamma=gamma)
    if spec.dataset == "KOA":
        env["KNEE_DATA_ROOT"] = str(args.koa_data_root)
    elif spec.dataset == "ADNI":
        env["ALZHEIMER_DATA_ROOT"] = str(args.adni_data_root)
    if is_mesc(spec):
        env["MESC_IMPLEMENTATION"] = "MECS_VersionB"
    return env


def run_one(
    original_spec: base.RunSpec,
    seed: int,
    args: argparse.Namespace,
    out_dir: Path,
    tag_prefix: str,
) -> int:
    record_spec = version_b_spec(original_spec)
    env = build_env(record_spec, seed, args, tag_prefix)
    # Injection detection must use the original method name and script.
    cmd = training_command(args, original_spec)
    log_name = (
        f"{tag_prefix}_{record_spec.dataset.lower()}_"
        f"{base.method_slug(record_spec.method)}_seed{seed}.log"
    )
    log_path = out_dir / log_name
    print(f"[RUN] {record_spec.dataset} | {record_spec.method} | seed={seed} -> {log_path}", flush=True)
    if args.dry_run:
        print("      " + " ".join(cmd), flush=True)
        return 0

    with log_path.open("w", encoding="utf-8", newline="") as fp:
        base.write_header(fp, record_spec, seed, cmd, env)
        if is_mesc(original_spec):
            fp.write(f"MESC_IMPLEMENTATION: {env['MESC_IMPLEMENTATION']}\n")
            fp.write(f"ORIGINAL_SCRIPT: {original_spec.script}\n")
            fp.flush()
        proc = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            env=env,
            stdout=fp,
            stderr=subprocess.STDOUT,
        )
        fp.write("\n" + "=" * 90 + "\n")
        fp.write(f"RETURN_CODE: {proc.returncode}\n")

    if proc.returncode != 0:
        reason = base.classify_log_error(log_path) or f"return code {proc.returncode}"
        print(f"[STOP] {record_spec.dataset} | {record_spec.method} failed: {reason}", flush=True)
        print(f"       See log: {log_path}", flush=True)
        return proc.returncode
    reason = base.classify_log_error(log_path)
    if reason:
        print(f"[STOP] completed log contains error marker: {reason}", flush=True)
        print(f"       See log: {log_path}", flush=True)
        return 1

    tau = args.dast_tau if "DAST" in record_spec.method else None
    gamma = args.dast_gamma if "DAST" in record_spec.method else None
    record = base.parse_test_record(log_path, record_spec, seed, env, tau=tau, gamma=gamma)
    if not record:
        print("[STOP] run finished but no Test metrics were parsed.", flush=True)
        print(f"       See log: {log_path}", flush=True)
        return 1
    base.append_experiment_record(record)
    print(f"[REC] appended to {base.EXPERIMENT_RECORDS_PATH}", flush=True)
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if not args.dry_run:
        validate_data_roots(args)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag_prefix = args.tag_prefix or f"controlled_mecsb_vb_{timestamp}"
    out_dir = PROJECT_ROOT / "batch_logs" / tag_prefix
    out_dir.mkdir(parents=True, exist_ok=args.skip_runs > 0)

    specs = list(base.iter_controlled_specs(args.dataset_names))
    print(f"Datasets   : {','.join(args.dataset_names)}", flush=True)
    print(f"Seeds      : {args.seed_values}", flush=True)
    print(f"Runs       : {len(specs) * len(args.seed_values)}", flush=True)
    print(f"Epochs     : {args.epochs}", flush=True)
    print(f"Patience   : {args.patience}", flush=True)
    print(f"DAST       : tau={args.dast_tau}, gamma={args.dast_gamma}", flush=True)
    print(f"Tag prefix : {tag_prefix}", flush=True)
    print(f"Logs       : {out_dir}", flush=True)

    scheduled = 0
    for seed in args.seed_values:
        for spec in specs:
            scheduled += 1
            if scheduled <= args.skip_runs:
                print(f"[SKIP] {spec.dataset} | {version_b_spec(spec).method} | seed={seed}", flush=True)
                continue
            rc = run_one(spec, seed, args, out_dir, tag_prefix)
            if rc != 0:
                print(f"Logs written under: {out_dir}", flush=True)
                return rc
    print(f"Logs written under: {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
