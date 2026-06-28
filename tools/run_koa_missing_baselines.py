#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Run single-seed KOA comparison baselines that are missing from Table I."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from run_multiseed_experiments import (
    EXPERIMENT_RECORDS_PATH,
    append_experiment_record,
    parse_test_metrics_line,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = PROJECT_ROOT / "Knee" / "train_baseline_zoo.py"

DEFAULT_MODELS = (
    "inception_v3",
    "densenet169",
    "densenet201",
    "vit_b_16",
    "xception",
    "resnet50v2",
)

DISPLAY_NAMES = {
    "xception": "Xception",
    "inception_v3": "Inception V3",
    "densenet169": "DenseNet169",
    "densenet201": "DenseNet201",
    "vit_b_16": "ViT",
    "resnet50v2": "ResNet50V2",
    "inception_resnet_v2": "Inception-ResNetV2",
}

ALIASES = {
    "inception-v3": "inception_v3",
    "inceptionv3": "inception_v3",
    "densenet-169": "densenet169",
    "densenet-201": "densenet201",
    "vit": "vit_b_16",
    "vit-b-16": "vit_b_16",
    "resnet50-v2": "resnet50v2",
    "resnet50_v2": "resnet50v2",
    "inception-resnet-v2": "inception_resnet_v2",
    "inceptionresnetv2": "inception_resnet_v2",
}

ERROR_MARKERS = (
    "Traceback",
    "FileNotFoundError",
    "No such file",
    "not found",
    "CUDA out of memory",
    "CUDA error",
    "RuntimeError: CUDA",
)


def normalize_model_name(raw: str) -> str:
    key = raw.strip().lower().replace(" ", "_")
    return ALIASES.get(key, key)


def parse_model_list(raw: str) -> List[str]:
    return [normalize_model_name(item) for item in raw.split(",") if item.strip()]


def method_slug(method: str) -> str:
    return method.lower().replace("+", "plus").replace(" ", "_").replace("-", "_")


def python_command(args: argparse.Namespace, model_key: str) -> List[str]:
    base = ["python", "-u", str(TRAIN_SCRIPT), "--model", model_key]
    if args.no_pretrained:
        base.append("--no-pretrained")
    if args.conda_env:
        return ["conda", "run", "-n", args.conda_env] + base
    return [args.python, "-u", str(TRAIN_SCRIPT), "--model", model_key] + (["--no-pretrained"] if args.no_pretrained else [])


def build_env(args: argparse.Namespace, model_key: str, run_tag: str) -> Dict[str, str]:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["GLOBAL_EXPERIMENT_SEED"] = str(args.seed)
    env["KNEE_SEED"] = str(args.seed)
    env["KNEE_MODEL"] = model_key
    env["KNEE_RUN_TAG"] = run_tag
    if args.epochs is not None:
        env["KNEE_EPOCHS"] = str(args.epochs)
    if args.batch_size is not None:
        env["KNEE_BATCH_SIZE"] = str(args.batch_size)
    if args.image_size is not None:
        env["KNEE_IMAGE_SIZE"] = str(args.image_size)
    if args.num_workers is not None:
        env["KNEE_NUM_WORKERS"] = str(args.num_workers)
    env["KNEE_PRETRAINED"] = "0" if args.no_pretrained else "1"
    return env


def write_header(fp, model_key: str, method: str, seed: int, cmd: Sequence[str], env: Dict[str, str]) -> None:
    fp.write("DATASET: KOA\n")
    fp.write(f"METHOD: {method}\n")
    fp.write(f"MODEL_KEY: {model_key}\n")
    fp.write(f"SEED: {seed}\n")
    fp.write(f"SCRIPT: {TRAIN_SCRIPT}\n")
    fp.write(f"RUN_TAG: {env.get('KNEE_RUN_TAG', '')}\n")
    fp.write(f"KNEE_PRETRAINED: {env.get('KNEE_PRETRAINED', '')}\n")
    fp.write("COMMAND: " + " ".join(cmd) + "\n")
    fp.write("=" * 90 + "\n")
    fp.flush()


def contains_error_marker(log_path: Path) -> Optional[str]:
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    for marker in ERROR_MARKERS:
        if marker.lower() in text.lower():
            return marker
    return None


def parse_record(log_path: Path, model_key: str, method: str, seed: int) -> Optional[Dict[str, object]]:
    last_test = None
    for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parsed = parse_test_metrics_line(line)
        if parsed:
            last_test = parsed
    if not last_test:
        return None
    return {
        "dataset": "KOA",
        "method": method,
        "seed": str(seed),
        "acc": f"{float(last_test['acc']) / 100.0:.8f}",
        "macro_f1": f"{float(last_test['macro_f1']):.8f}",
        "qwk": f"{float(last_test['qwk']):.8f}",
        "mae": f"{float(last_test['mae']):.8f}",
        "tau": "",
        "gamma": "",
        "source": str(log_path),
        "run_tag": f"koa_missing_{method_slug(method)}_seed{seed}",
        "completed_at": datetime.now().isoformat(timespec="seconds"),
        "model_key": model_key,
    }


def run_one(args: argparse.Namespace, out_dir: Path, model_key: str) -> int:
    method = DISPLAY_NAMES.get(model_key, model_key)
    run_tag = f"koa_missing_{method_slug(method)}_seed{args.seed}"
    env = build_env(args, model_key, run_tag)
    cmd = python_command(args, model_key)
    log_path = out_dir / f"{run_tag}.log"
    print(f"[RUN] KOA | {method} | seed={args.seed} -> {log_path}", flush=True)
    if args.dry_run:
        print("      " + " ".join(cmd), flush=True)
        return 0

    with log_path.open("w", encoding="utf-8", newline="") as fp:
        write_header(fp, model_key, method, args.seed, cmd, env)
        proc = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, stdout=fp, stderr=subprocess.STDOUT)
        fp.write("\n" + "=" * 90 + "\n")
        fp.write(f"RETURN_CODE: {proc.returncode}\n")

    if proc.returncode != 0:
        reason = contains_error_marker(log_path) or f"return code {proc.returncode}"
        print(f"[STOP] KOA | {method} failed: {reason}", flush=True)
        print(f"       See log: {log_path}", flush=True)
        return proc.returncode

    reason = contains_error_marker(log_path)
    if reason:
        print(f"[STOP] KOA | {method} log contains error marker: {reason}", flush=True)
        print(f"       See log: {log_path}", flush=True)
        return 1

    record = parse_record(log_path, model_key, method, args.seed)
    if not record:
        print(f"[STOP] KOA | {method} finished but no Test metrics were parsed.", flush=True)
        print(f"       See log: {log_path}", flush=True)
        return 1
    append_experiment_record(record)
    print(f"[REC] appended to {EXPERIMENT_RECORDS_PATH}", flush=True)
    return 0


def estimate_runs(models_to_run: Iterable[str]) -> None:
    model_list = list(models_to_run)
    print(f"Scheduled KOA missing-baseline runs: {len(model_list)}")
    print("Models: " + ", ".join(DISPLAY_NAMES.get(model, model) for model in model_list))
    print("Rough time: about 25-60 minutes per model on the current KOA protocol, depending on backbone size.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS), help="Comma-separated model keys.")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--conda-env", default="Paper", help="Set empty string to use --python directly.")
    parser.add_argument("--no-pretrained", action="store_true", help="Disable ImageNet pretrained weights.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-runs", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    models_to_run = parse_model_list(args.models)
    estimate_runs(models_to_run)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = PROJECT_ROOT / "batch_logs" / f"koa_missing_baselines_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    for index, model_key in enumerate(models_to_run, start=1):
        if index <= args.skip_runs:
            print(f"[SKIP] KOA | {DISPLAY_NAMES.get(model_key, model_key)} | seed={args.seed}", flush=True)
            continue
        rc = run_one(args, out_dir, model_key)
        if rc != 0:
            print(f"Logs written under: {out_dir}", flush=True)
            return rc

    print(f"Logs written under: {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
