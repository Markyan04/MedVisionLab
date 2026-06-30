#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Run one-factor-at-a-time DAST tau/gamma sensitivity experiments on KOA.

This runner reuses the existing KOA training scripts and log parser from
run_multiseed_experiments.py. It schedules:

* tau sweep with gamma fixed at the default gamma;
* gamma sweep with tau fixed at the default tau;
* the default setting only once.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

from run_multiseed_experiments import DEFAULT_SEEDS, PROJECT_ROOT, RunSpec, parse_ints, run_one


DEFAULT_TAUS = (0.5, 1.0, 1.5, 2.0)
DEFAULT_GAMMAS = (0.5, 1.0, 1.5, 2.0)
DEFAULT_TAU = 1.0
DEFAULT_GAMMA = 1.5

MODEL_SPECS = {
    "baseline": RunSpec("KOA", "ResNet50 + DAST", PROJECT_ROOT / "Knee" / "ResNet_baseline+Loss4.py", "KNEE"),
    "mesc": RunSpec("KOA", "ResNet50 + MESC + DAST", PROJECT_ROOT / "Knee" / "ResNet_layer3+MECS+Loss4.py", "KNEE"),
}


@dataclass(frozen=True)
class SensitivitySetting:
    sweep: str
    tau: float
    gamma: float


def parse_csv_floats(raw: str) -> List[float]:
    values = [float(x.strip()) for x in raw.split(",") if x.strip()]
    if not values:
        raise ValueError("At least one numeric value is required.")
    return values


def parse_csv_tokens(raw: str) -> List[str]:
    values = [x.strip().lower() for x in raw.split(",") if x.strip()]
    if not values:
        raise ValueError("At least one value is required.")
    return values


def same_float(a: float, b: float) -> bool:
    return abs(float(a) - float(b)) < 1e-12


def build_settings(
    taus: Sequence[float],
    gammas: Sequence[float],
    default_tau: float,
    default_gamma: float,
) -> List[SensitivitySetting]:
    settings: List[SensitivitySetting] = []
    seen: set[Tuple[float, float]] = set()

    for tau in taus:
        key = (float(tau), float(default_gamma))
        if key in seen:
            continue
        seen.add(key)
        sweep = "default" if same_float(tau, default_tau) else "tau"
        settings.append(SensitivitySetting(sweep=sweep, tau=float(tau), gamma=float(default_gamma)))

    for gamma in gammas:
        key = (float(default_tau), float(gamma))
        if key in seen:
            continue
        seen.add(key)
        sweep = "default" if same_float(gamma, default_gamma) else "gamma"
        settings.append(SensitivitySetting(sweep=sweep, tau=float(default_tau), gamma=float(gamma)))

    return settings


def iter_specs(models: Iterable[str]) -> List[RunSpec]:
    specs: List[RunSpec] = []
    for model in models:
        if model not in MODEL_SPECS:
            raise ValueError(f"Unknown model '{model}'. Choose from {sorted(MODEL_SPECS)}.")
        specs.append(MODEL_SPECS[model])
    return specs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        default="baseline",
        help="Comma-separated models: baseline, mesc. Default: baseline.",
    )
    parser.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    parser.add_argument("--taus", default=",".join(map(str, DEFAULT_TAUS)))
    parser.add_argument("--gammas", default=",".join(map(str, DEFAULT_GAMMAS)))
    parser.add_argument("--default-tau", type=float, default=DEFAULT_TAU)
    parser.add_argument("--default-gamma", type=float, default=DEFAULT_GAMMA)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--early-delta", type=float, default=None)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--conda-env", default="Paper", help="Use 'none' to run --python directly.")
    parser.add_argument("--tag-prefix", default="dast_sensitivity")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-runs", type=int, default=0)
    args = parser.parse_args()

    taus = parse_csv_floats(args.taus)
    gammas = parse_csv_floats(args.gammas)
    if any(tau <= 0 for tau in taus):
        raise ValueError("All tau values must be > 0.")
    if any(gamma < 0 for gamma in gammas):
        raise ValueError("All gamma values must be >= 0.")

    settings = build_settings(taus, gammas, args.default_tau, args.default_gamma)
    seeds = parse_ints(args.seeds)
    specs = iter_specs(parse_csv_tokens(args.models))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_prefix = f"{args.tag_prefix}_{timestamp}"
    out_dir = PROJECT_ROOT / "batch_logs" / f"{args.tag_prefix}_{timestamp}"
    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    total = len(settings) * len(seeds) * len(specs)
    print(f"Scheduled DAST sensitivity runs: {total}")
    print(f"Models: {', '.join(spec.method for spec in specs)}")
    print(f"Seeds: {', '.join(map(str, seeds))}")
    print(f"Tau sweep: taus={taus}, gamma={args.default_gamma}")
    print(f"Gamma sweep: tau={args.default_tau}, gammas={gammas}")
    print(f"Default setting is run once: tau={args.default_tau}, gamma={args.default_gamma}")
    print(f"Logs written under: {out_dir}")

    scheduled_index = 0
    for seed in seeds:
        for setting in settings:
            for spec in specs:
                scheduled_index += 1
                sweep_prefix = f"{run_prefix}_{setting.sweep}"
                if scheduled_index <= args.skip_runs:
                    print(
                        f"[SKIP] {spec.dataset} | {spec.method} | seed={seed} | "
                        f"tau={setting.tau} | gamma={setting.gamma}"
                    )
                    continue
                rc = run_one(
                    spec,
                    seed,
                    args,
                    out_dir,
                    sweep_prefix,
                    tau=setting.tau,
                    gamma=setting.gamma,
                )
                if rc != 0:
                    print(f"Logs written under: {out_dir}")
                    return rc

    print(f"Logs written under: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
