#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Run controlled MESC/DAST experiments with reproducible seeds.

The script launches existing dataset runners and records stdout/stderr logs.
It does not alter reported numbers; use summarize_experiment_logs.py to convert
finished logs or summary CSV files into tables.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEEDS = (42, 777, 1234, 2024, 3407)
EXPERIMENT_RECORDS_PATH = PROJECT_ROOT / "analysis_tables" / "experiment_records.csv"
TEST_LINE_RE = re.compile(
    r"Test\s*\|.*?acc=(?P<acc>[0-9.]+)%.*?macro_f1=(?P<f1>[0-9.]+)"
    r".*?qwk=(?P<qwk>-?[0-9.]+).*?mae=(?P<mae>[0-9.]+)",
    re.IGNORECASE,
)
METRIC_VALUE_RE = re.compile(r"([A-Za-z0-9_]+)=(-?[0-9]+(?:\.[0-9]+)?)%?")
STOP_ERROR_RE = re.compile(
    r"(Traceback|FileNotFoundError|No such file|not found|CUDA out of memory|CUDA error|RuntimeError: CUDA)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class RunSpec:
    dataset: str
    method: str
    script: Path
    env_prefix: str
    losses_env: Optional[str] = None
    losses_value: Optional[str] = None


CONTROLLED_SPECS: Dict[str, Sequence[RunSpec]] = {
    "koa": (
        RunSpec("KOA", "ResNet50", PROJECT_ROOT / "Knee" / "ResNet_baseline.py", "KNEE"),
        RunSpec("KOA", "ResNet50 + DAST", PROJECT_ROOT / "Knee" / "ResNet_baseline+Loss4.py", "KNEE"),
        RunSpec("KOA", "ResNet50 + MESC", PROJECT_ROOT / "Knee" / "ResNet_layer3+MECS+CE.py", "KNEE"),
        RunSpec("KOA", "ResNet50 + MESC + DAST", PROJECT_ROOT / "Knee" / "ResNet_layer3+MECS+Loss4.py", "KNEE"),
    ),
    "ham10000": (
        RunSpec("HAM10000", "ResNet50", PROJECT_ROOT / "HAM10000_Loss" / "ResNet_baseline.py", "HAM10000", "HAM10000_LOSSES", "ce"),
        RunSpec("HAM10000", "ResNet50 + DAST", PROJECT_ROOT / "HAM10000_Loss" / "ResNet_baseline.py", "HAM10000", "HAM10000_LOSSES", "dast"),
        RunSpec("HAM10000", "ResNet50 + MESC", PROJECT_ROOT / "HAM10000_Loss" / "ResNet_layer3+MECS.py", "HAM10000", "HAM10000_LOSSES", "ce"),
        RunSpec("HAM10000", "ResNet50 + MESC + DAST", PROJECT_ROOT / "HAM10000_Loss" / "ResNet_layer3+MECS.py", "HAM10000", "HAM10000_LOSSES", "dast"),
    ),
    "adni": (
        RunSpec("ADNI", "ResNet50", PROJECT_ROOT / "Alzheimer_MRI_Loss" / "ResNet_baseline.py", "ALZHEIMER", "ALZHEIMER_LOSSES", "ce"),
        RunSpec("ADNI", "ResNet50 + DAST", PROJECT_ROOT / "Alzheimer_MRI_Loss" / "ResNet_baseline.py", "ALZHEIMER", "ALZHEIMER_LOSSES", "dast"),
        RunSpec("ADNI", "ResNet50 + MESC", PROJECT_ROOT / "Alzheimer_MRI_Loss" / "ResNet_layer3+MECS.py", "ALZHEIMER", "ALZHEIMER_LOSSES", "ce"),
        RunSpec("ADNI", "ResNet50 + MESC + DAST", PROJECT_ROOT / "Alzheimer_MRI_Loss" / "ResNet_layer3+MECS.py", "ALZHEIMER", "ALZHEIMER_LOSSES", "dast"),
    ),
    "brain": (
        RunSpec("Brain Tumor MRI", "ResNet50", PROJECT_ROOT / "Brain_Tumor_MRI_Loss" / "ResNet_baseline.py", "BRAIN_MRI", "BRAIN_MRI_LOSSES", "ce"),
        RunSpec("Brain Tumor MRI", "ResNet50 + DAST", PROJECT_ROOT / "Brain_Tumor_MRI_Loss" / "ResNet_baseline.py", "BRAIN_MRI", "BRAIN_MRI_LOSSES", "dast"),
        RunSpec("Brain Tumor MRI", "ResNet50 + MESC", PROJECT_ROOT / "Brain_Tumor_MRI_Loss" / "ResNet_layer3+MECS.py", "BRAIN_MRI", "BRAIN_MRI_LOSSES", "ce"),
        RunSpec("Brain Tumor MRI", "ResNet50 + MESC + DAST", PROJECT_ROOT / "Brain_Tumor_MRI_Loss" / "ResNet_layer3+MECS.py", "BRAIN_MRI", "BRAIN_MRI_LOSSES", "dast"),
    ),
    "chest": (
        RunSpec("Chest X-ray Image", "ResNet50", PROJECT_ROOT / "chest-x-ray-image_Loss" / "ResNet_baseline.py", "CHESTXRAY", "CHESTXRAY_LOSSES", "ce"),
        RunSpec("Chest X-ray Image", "ResNet50 + DAST", PROJECT_ROOT / "chest-x-ray-image_Loss" / "ResNet_baseline.py", "CHESTXRAY", "CHESTXRAY_LOSSES", "dast"),
        RunSpec("Chest X-ray Image", "ResNet50 + MESC", PROJECT_ROOT / "chest-x-ray-image_Loss" / "ResNet_layer3+MECS.py", "CHESTXRAY", "CHESTXRAY_LOSSES", "ce"),
        RunSpec("Chest X-ray Image", "ResNet50 + MESC + DAST", PROJECT_ROOT / "chest-x-ray-image_Loss" / "ResNet_layer3+MECS.py", "CHESTXRAY", "CHESTXRAY_LOSSES", "dast"),
    ),
}

LOSS_COMPARISON_SPECS: Dict[str, Dict[str, RunSpec]] = {
    "koa": {
        "ce": RunSpec("KOA", "ResNet50", PROJECT_ROOT / "Knee" / "ResNet_baseline.py", "KNEE"),
        "label_smoothing_ce": RunSpec(
            "KOA",
            "ResNet50 + Label Smoothing",
            PROJECT_ROOT / "Knee" / "ResNet_baseline_loss_compare.py",
            "KNEE",
            "KNEE_LOSS",
            "label_smoothing_ce",
        ),
        "sord_ce": RunSpec(
            "KOA",
            "ResNet50 + SORD-CE",
            PROJECT_ROOT / "Knee" / "ResNet_baseline_loss_compare.py",
            "KNEE",
            "KNEE_LOSS",
            "sord_ce",
        ),
        "dast": RunSpec("KOA", "ResNet50 + DAST", PROJECT_ROOT / "Knee" / "ResNet_baseline+Loss4.py", "KNEE"),
    },
    "adni": {
        "ce": RunSpec(
            "ADNI",
            "ResNet50",
            PROJECT_ROOT / "Alzheimer_MRI_Loss" / "ResNet_baseline.py",
            "ALZHEIMER",
            "ALZHEIMER_LOSSES",
            "ce",
        ),
        "label_smoothing_ce": RunSpec(
            "ADNI",
            "ResNet50 + Label Smoothing",
            PROJECT_ROOT / "Alzheimer_MRI_Loss" / "ResNet_baseline.py",
            "ALZHEIMER",
            "ALZHEIMER_LOSSES",
            "label_smoothing_ce",
        ),
        "sord_ce": RunSpec(
            "ADNI",
            "ResNet50 + SORD-CE",
            PROJECT_ROOT / "Alzheimer_MRI_Loss" / "ResNet_baseline.py",
            "ALZHEIMER",
            "ALZHEIMER_LOSSES",
            "sord_ce",
        ),
        "dast": RunSpec(
            "ADNI",
            "ResNet50 + DAST",
            PROJECT_ROOT / "Alzheimer_MRI_Loss" / "ResNet_baseline.py",
            "ALZHEIMER",
            "ALZHEIMER_LOSSES",
            "dast",
        ),
    },
}


def parse_csv_list(raw: str) -> List[str]:
    return [x.strip().lower() for x in raw.split(",") if x.strip()]


def parse_ints(raw: str) -> List[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def method_slug(method: str) -> str:
    return method.lower().replace("+", "plus").replace("-", "_").replace(" ", "_")


LOSS_ALIASES = {
    "cross_entropy": "ce",
    "ls": "label_smoothing_ce",
    "ls_ce": "label_smoothing_ce",
    "label_smoothing": "label_smoothing_ce",
    "label-smoothing": "label_smoothing_ce",
    "label_smoothing_ce": "label_smoothing_ce",
    "sord": "sord_ce",
    "sord-ce": "sord_ce",
    "sord_ce": "sord_ce",
}


def normalize_loss_name(raw: str) -> str:
    key = raw.strip().lower()
    return LOSS_ALIASES.get(key, key)


def parse_loss_list(raw: str) -> List[str]:
    losses = [normalize_loss_name(x) for x in raw.split(",") if x.strip()]
    if len(losses) == 1 and losses[0] == "all":
        return ["ce", "label_smoothing_ce", "sord_ce", "dast"]
    return losses


def parse_test_metrics_line(line: str) -> Optional[Dict[str, str]]:
    if "Test |" not in line:
        return None
    legacy = TEST_LINE_RE.search(line)
    if legacy:
        gd = legacy.groupdict()
        return {"acc": gd["acc"], "macro_f1": gd["f1"], "qwk": gd["qwk"], "mae": gd["mae"]}
    metrics = {key.lower(): value for key, value in METRIC_VALUE_RE.findall(line)}
    required = {"acc", "macro_f1", "qwk", "mae"}
    if required.issubset(metrics):
        return {
            "acc": metrics["acc"],
            "macro_f1": metrics["macro_f1"],
            "qwk": metrics["qwk"],
            "mae": metrics["mae"],
        }
    return None


def python_command(args: argparse.Namespace, script: Path) -> List[str]:
    conda_env = (args.conda_env or "").strip()
    if conda_env and conda_env.lower() not in {"none", "null", "false", "0"}:
        return ["conda", "run", "-n", conda_env, "python", "-u", str(script)]
    return [args.python, "-u", str(script)]


def build_env(
    spec: RunSpec,
    seed: int,
    args: argparse.Namespace,
    tag_prefix: str,
    tau: Optional[float] = None,
    gamma: Optional[float] = None,
) -> Dict[str, str]:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    tag = f"{tag_prefix}_{spec.dataset.lower().replace(' ', '_')}_{method_slug(spec.method)}_seed{seed}"
    env["GLOBAL_EXPERIMENT_SEED"] = str(seed)
    env[f"{spec.env_prefix}_SEED"] = str(seed)
    env[f"{spec.env_prefix}_RUN_TAG"] = tag
    if args.epochs is not None:
        env[f"{spec.env_prefix}_EPOCHS"] = str(args.epochs)
    if args.batch_size is not None:
        env[f"{spec.env_prefix}_BATCH_SIZE"] = str(args.batch_size)
    if spec.losses_env and spec.losses_value:
        env[spec.losses_env] = spec.losses_value
    label_smoothing = getattr(args, "label_smoothing", None)
    sord_tau = getattr(args, "sord_tau", None)
    if label_smoothing is not None:
        env[f"{spec.env_prefix}_LABEL_SMOOTHING"] = str(label_smoothing)
    if sord_tau is not None:
        env[f"{spec.env_prefix}_SORD_TAU"] = str(sord_tau)
    if tau is not None:
        env[f"{spec.env_prefix}_DAST_TAU"] = str(tau)
    if gamma is not None:
        env[f"{spec.env_prefix}_DAST_GAMMA"] = str(gamma)
    return env


def write_header(fp, spec: RunSpec, seed: int, cmd: Sequence[str], env: Dict[str, str]) -> None:
    fp.write(f"DATASET: {spec.dataset}\n")
    fp.write(f"METHOD: {spec.method}\n")
    fp.write(f"SEED: {seed}\n")
    fp.write(f"SCRIPT: {spec.script}\n")
    fp.write(f"RUN_TAG: {env.get(f'{spec.env_prefix}_RUN_TAG', '')}\n")
    if spec.losses_env:
        fp.write(f"{spec.losses_env}: {env.get(spec.losses_env, '')}\n")
    if spec.losses_value:
        fp.write(f"LOSS: {spec.losses_value}\n")
    smoothing_key = f"{spec.env_prefix}_LABEL_SMOOTHING"
    sord_tau_key = f"{spec.env_prefix}_SORD_TAU"
    if smoothing_key in env:
        fp.write(f"LABEL_SMOOTHING: {env.get(smoothing_key, '')}\n")
    if sord_tau_key in env:
        fp.write(f"SORD_TAU: {env.get(sord_tau_key, '')}\n")
    tau_key = f"{spec.env_prefix}_DAST_TAU"
    gamma_key = f"{spec.env_prefix}_DAST_GAMMA"
    if tau_key in env or gamma_key in env:
        fp.write(f"DAST_TAU: {env.get(tau_key, '')}\n")
        fp.write(f"DAST_GAMMA: {env.get(gamma_key, '')}\n")
    fp.write("COMMAND: " + " ".join(cmd) + "\n")
    fp.write("=" * 90 + "\n")
    fp.flush()


def parse_test_record(log_path: Path, spec: RunSpec, seed: int, env: Dict[str, str], tau=None, gamma=None) -> Optional[Dict[str, object]]:
    last_test = None
    for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parsed = parse_test_metrics_line(line)
        if parsed:
            last_test = parsed
    if not last_test:
        return None
    return {
        "dataset": spec.dataset,
        "method": spec.method,
        "seed": str(seed),
        "acc": f"{float(last_test['acc']) / 100.0:.8f}",
        "macro_f1": f"{float(last_test['macro_f1']):.8f}",
        "qwk": f"{float(last_test['qwk']):.8f}",
        "mae": f"{float(last_test['mae']):.8f}",
        "tau": "" if tau is None else str(tau),
        "gamma": "" if gamma is None else str(gamma),
        "source": str(log_path),
        "run_tag": env.get(f"{spec.env_prefix}_RUN_TAG", ""),
        "completed_at": datetime.now().isoformat(timespec="seconds"),
    }


def append_experiment_record(record: Dict[str, object], records_path: Path = EXPERIMENT_RECORDS_PATH) -> None:
    records_path.parent.mkdir(parents=True, exist_ok=True)
    default_fieldnames = [
        "dataset",
        "method",
        "seed",
        "acc",
        "macro_f1",
        "qwk",
        "mae",
        "tau",
        "gamma",
        "source",
        "run_tag",
        "completed_at",
    ]
    write_header_row = not records_path.exists() or records_path.stat().st_size == 0
    fieldnames = default_fieldnames
    if not write_header_row:
        with records_path.open("r", encoding="utf-8-sig", newline="") as fp:
            reader = csv.reader(fp)
            existing = next(reader, None)
        if existing:
            fieldnames = existing
    with records_path.open("a", encoding="utf-8-sig", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames, extrasaction="ignore")
        if write_header_row:
            writer.writeheader()
        writer.writerow(record)


def classify_log_error(log_path: Path) -> str:
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    match = STOP_ERROR_RE.search(text)
    return match.group(0) if match else ""


def run_one(spec: RunSpec, seed: int, args: argparse.Namespace, out_dir: Path, tag_prefix: str, tau=None, gamma=None) -> int:
    env = build_env(spec, seed, args, tag_prefix, tau=tau, gamma=gamma)
    cmd = python_command(args, spec.script)
    log_name = f"{tag_prefix}_{spec.dataset.lower().replace(' ', '_')}_{method_slug(spec.method)}_seed{seed}"
    if tau is not None and gamma is not None:
        log_name += f"_tau{str(tau).replace('.', 'p')}_gamma{str(gamma).replace('.', 'p')}"
    log_path = out_dir / f"{log_name}.log"
    print(f"[RUN] {spec.dataset} | {spec.method} | seed={seed} -> {log_path}", flush=True)
    if args.dry_run:
        print("      " + " ".join(cmd), flush=True)
        return 0
    with log_path.open("w", encoding="utf-8", newline="") as fp:
        write_header(fp, spec, seed, cmd, env)
        proc = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, stdout=fp, stderr=subprocess.STDOUT)
        fp.write("\n" + "=" * 90 + "\n")
        fp.write(f"RETURN_CODE: {proc.returncode}\n")
    if proc.returncode != 0:
        reason = classify_log_error(log_path) or f"return code {proc.returncode}"
        print(f"[STOP] {spec.dataset} | {spec.method} | seed={seed} failed: {reason}", flush=True)
        print(f"       See log: {log_path}", flush=True)
        return proc.returncode
    reason = classify_log_error(log_path)
    if reason:
        print(f"[STOP] {spec.dataset} | {spec.method} | seed={seed} log contains error marker: {reason}", flush=True)
        print(f"       See log: {log_path}", flush=True)
        return 1
    record = parse_test_record(log_path, spec, seed, env, tau=tau, gamma=gamma)
    if not record:
        print(f"[STOP] {spec.dataset} | {spec.method} | seed={seed} finished but no Test metrics were parsed.", flush=True)
        print(f"       See log: {log_path}", flush=True)
        return 1
    append_experiment_record(record)
    print(f"[REC] appended to {EXPERIMENT_RECORDS_PATH}", flush=True)
    return 0


def iter_controlled_specs(datasets: Iterable[str]) -> Iterable[RunSpec]:
    for name in datasets:
        if name == "all":
            for key in ("koa", "ham10000", "adni", "brain", "chest"):
                yield from CONTROLLED_SPECS[key]
            return
        if name not in CONTROLLED_SPECS:
            raise ValueError(f"Unknown dataset '{name}'. Choose from {sorted(CONTROLLED_SPECS)} or all.")
        yield from CONTROLLED_SPECS[name]


def iter_loss_comparison_specs(datasets: Iterable[str], losses: Sequence[str]) -> Iterable[RunSpec]:
    for dataset_name in datasets:
        if dataset_name == "all":
            dataset_keys = ("koa", "adni")
        else:
            dataset_keys = (dataset_name,)
        for key in dataset_keys:
            if key not in LOSS_COMPARISON_SPECS:
                raise ValueError(
                    f"Unknown loss-comparison dataset '{key}'. Choose from {sorted(LOSS_COMPARISON_SPECS)} or all."
                )
            for loss_name in losses:
                if loss_name not in LOSS_COMPARISON_SPECS[key]:
                    raise ValueError(
                        f"Unsupported loss '{loss_name}' for dataset '{key}'. "
                        f"Choose from {sorted(LOSS_COMPARISON_SPECS[key])} or all."
                    )
                yield LOSS_COMPARISON_SPECS[key][loss_name]


def spec_loss_name(spec: RunSpec) -> str:
    if spec.losses_value:
        return normalize_loss_name(spec.losses_value)
    method = spec.method.lower()
    if "dast" in method:
        return "dast"
    if "sord" in method:
        return "sord_ce"
    if "label" in method:
        return "label_smoothing_ce"
    return "ce"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("controlled", "loss-comparison", "koa-dast-grid"), default="controlled")
    parser.add_argument("--datasets", default=None, help="Comma-separated datasets or 'all'.")
    parser.add_argument(
        "--losses",
        default="label_smoothing_ce,sord_ce",
        help="Losses for --suite loss-comparison. Use 'all' for CE, label smoothing, SORD-CE, and DAST.",
    )
    parser.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    parser.add_argument("--hparam-seed", type=int, default=1234, help="Single seed used by koa-dast-grid.")
    parser.add_argument("--taus", default="0.5,1.0,1.5,2.0")
    parser.add_argument("--gammas", default="0.0,1.0,1.5,2.0")
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--sord-tau", type=float, default=1.0)
    parser.add_argument("--dast-tau", type=float, default=1.0)
    parser.add_argument("--dast-gamma", type=float, default=1.5)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--conda-env", default="Paper", help="Use 'none' to run --python directly.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-runs", type=int, default=0, help="Skip this many scheduled runs from the beginning.")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = PROJECT_ROOT / "batch_logs" / f"{args.suite}_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    rc = 0
    scheduled_index = 0
    dataset_arg = args.datasets or ("koa,adni" if args.suite == "loss-comparison" else "koa,ham10000,adni")
    if args.suite == "controlled":
        seeds = parse_ints(args.seeds)
        specs = list(iter_controlled_specs(parse_csv_list(dataset_arg)))
        for seed in seeds:
            for spec in specs:
                scheduled_index += 1
                if scheduled_index <= args.skip_runs:
                    print(f"[SKIP] {spec.dataset} | {spec.method} | seed={seed}", flush=True)
                    continue
                rc = run_one(spec, seed, args, out_dir, "controlled")
                if rc != 0:
                    print(f"Logs written under: {out_dir}")
                    return rc
    elif args.suite == "loss-comparison":
        seeds = parse_ints(args.seeds)
        losses = parse_loss_list(args.losses)
        specs = list(iter_loss_comparison_specs(parse_csv_list(dataset_arg), losses))
        for seed in seeds:
            for spec in specs:
                scheduled_index += 1
                loss_name = spec_loss_name(spec)
                tau = None
                gamma = None
                if loss_name == "dast":
                    tau = args.dast_tau
                    gamma = args.dast_gamma
                elif loss_name == "sord_ce":
                    tau = args.sord_tau
                    gamma = 0.0
                if scheduled_index <= args.skip_runs:
                    print(f"[SKIP] {spec.dataset} | {spec.method} | seed={seed}", flush=True)
                    continue
                rc = run_one(spec, seed, args, out_dir, "loss_comparison", tau=tau, gamma=gamma)
                if rc != 0:
                    print(f"Logs written under: {out_dir}")
                    return rc
    else:
        koa_specs = [
            RunSpec("KOA", "ResNet50 + DAST", PROJECT_ROOT / "Knee" / "ResNet_baseline+Loss4.py", "KNEE"),
            RunSpec("KOA", "ResNet50 + MESC + DAST", PROJECT_ROOT / "Knee" / "ResNet_layer3+MECS+Loss4.py", "KNEE"),
        ]
        for tau in [float(x) for x in parse_csv_list(args.taus)]:
            for gamma in [float(x) for x in parse_csv_list(args.gammas)]:
                for spec in koa_specs:
                    scheduled_index += 1
                    if scheduled_index <= args.skip_runs:
                        print(f"[SKIP] {spec.dataset} | {spec.method} | seed={args.hparam_seed}", flush=True)
                        continue
                    rc = run_one(spec, args.hparam_seed, args, out_dir, "koa_dast_grid", tau=tau, gamma=gamma)
                    if rc != 0:
                        print(f"Logs written under: {out_dir}")
                        return rc
    print(f"Logs written under: {out_dir}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
