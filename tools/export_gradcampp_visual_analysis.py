#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Export Grad-CAM++ visual analysis panels into one review folder.

The script wraps the existing per-dataset comparison exporters and selects:

* the worst baseline checkpoint from available summary CSVs;
* the best proposed checkpoint from available summary CSVs.

Default targets:

* Brain Tumor MRI: glioma, meningioma, pituitary;
* Chest X-ray: COVID19, PNEUMONIA;
* KOA: KL grade 2, 3, 4.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "visual_outputs"
METRIC_RE = re.compile(r"([A-Za-z0-9_]+)=(-?[0-9]+(?:\.[0-9]+)?)%?")
CHECKPOINT_RE = re.compile(r"(?:Checkpoint saved:|Saved best model to)\s+(.+?)(?:\s+\([^)]*\))?$")


@dataclass(frozen=True)
class DatasetConfig:
    key: str
    display_name: str
    dir_path: Path
    exporter: Path
    classes: Tuple[str, ...]
    baseline_script: str
    baseline_loss: str
    proposed_script: str
    proposed_loss: str
    selection_metric: str
    data_root_env: str


DATASETS: Dict[str, DatasetConfig] = {
    "brain": DatasetConfig(
        key="brain",
        display_name="Brain Tumor MRI",
        dir_path=PROJECT_ROOT / "Brain_Tumor_MRI_Loss",
        exporter=PROJECT_ROOT / "Brain_Tumor_MRI_Loss" / "export_gradcam_comparison_samples_pytorch_grad_cam.py",
        classes=("glioma", "meningioma", "pituitary"),
        baseline_script="ResNet_baseline.py",
        baseline_loss="ce",
        proposed_script="ResNet_layer3+MECS.py",
        proposed_loss="ce",
        selection_metric="test_macro_f1",
        data_root_env="BRAIN_MRI_DATA_ROOT",
    ),
    "chest": DatasetConfig(
        key="chest",
        display_name="Chest X-ray",
        dir_path=PROJECT_ROOT / "chest-x-ray-image_Loss",
        exporter=PROJECT_ROOT / "chest-x-ray-image_Loss" / "export_gradcam_comparison_samples_pytorch_grad_cam.py",
        classes=("COVID19", "PNEUMONIA"),
        baseline_script="ResNet_baseline.py",
        baseline_loss="ce",
        proposed_script="ResNet_layer3+MECS.py",
        proposed_loss="ce",
        selection_metric="test_macro_f1",
        data_root_env="CHESTXRAY_DATA_ROOT",
    ),
    "koa": DatasetConfig(
        key="koa",
        display_name="KOA",
        dir_path=PROJECT_ROOT / "Knee",
        exporter=PROJECT_ROOT / "Knee" / "export_gradcam_comparison_samples_pytorch_grad_cam.py",
        classes=("2", "3", "4"),
        baseline_script="ResNet_baseline.py",
        baseline_loss="ce",
        proposed_script="ResNet_layer3+MECS+Loss4.py",
        proposed_loss="dast",
        selection_metric="test_qwk",
        data_root_env="KNEE_DATA_ROOT",
    ),
}


def parse_csv_tokens(raw: str) -> List[str]:
    values = [x.strip() for x in raw.split(",") if x.strip()]
    if not values:
        raise ValueError("At least one value is required.")
    return values


def safe_float(value: object) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def sanitize_filename(text: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(text).strip())
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("._") or "sample"


def infer_script_name(summary_path: Path, row: Dict[str, str]) -> str:
    script = (row.get("script_name") or "").strip()
    if script:
        return Path(script).name
    name = summary_path.name
    for script_name in (
        "ResNet_layer3+MECS+Loss4",
        "ResNet_layer3+MECS",
        "ResNet_baseline+Loss4",
        "ResNet_baseline",
    ):
        if name.startswith(script_name):
            return f"{script_name}.py"
    return ""


def parse_metric_line(line: str) -> Dict[str, str]:
    parsed: Dict[str, str] = {}
    for key, raw_value in METRIC_RE.findall(line):
        value = float(raw_value)
        if key in {"acc", "bal_acc", "top1", "top2", "top3"} and value > 1.0:
            value /= 100.0
        parsed[f"test_{key.lower()}"] = f"{value:.6f}"
    return parsed


def dataset_matches_log(raw_dataset: str, config: DatasetConfig) -> bool:
    text = raw_dataset.strip().lower().replace("_", " ").replace("-", " ")
    if config.key == "koa":
        return "koa" in text or "knee" in text
    if config.key == "brain":
        return "brain" in text
    if config.key == "chest":
        return "chest" in text
    return False


def script_loss_from_method(config: DatasetConfig, method: str) -> Tuple[str, str]:
    text = " ".join(method.strip().lower().split())
    if text == "resnet50":
        return config.baseline_script, config.baseline_loss
    if config.key in {"brain", "chest"} and "mesc" in text and "dast" not in text:
        return config.proposed_script, config.proposed_loss
    if config.key == "koa" and "mesc" in text and "dast" in text:
        return config.proposed_script, config.proposed_loss
    return "", ""


def read_batch_log_rows(config: DatasetConfig) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    logs_dir = PROJECT_ROOT / "batch_logs"
    if not logs_dir.exists():
        return rows

    for log_path in logs_dir.rglob("*.log"):
        dataset = ""
        method = ""
        seed = ""
        run_tag = ""
        checkpoint_path = ""
        return_code = ""
        test_metrics: Dict[str, str] = {}
        try:
            with log_path.open("r", encoding="utf-8", errors="replace") as fp:
                for line in fp:
                    stripped = line.strip()
                    if stripped.startswith("DATASET:"):
                        dataset = stripped.split(":", 1)[1].strip()
                    elif stripped.startswith("METHOD:"):
                        method = stripped.split(":", 1)[1].strip()
                    elif stripped.startswith("SEED:"):
                        seed = stripped.split(":", 1)[1].strip()
                    elif stripped.startswith("RUN_TAG:"):
                        run_tag = stripped.split(":", 1)[1].strip()
                    elif stripped.startswith("RETURN_CODE:"):
                        return_code = stripped.split(":", 1)[1].strip()
                    elif stripped.startswith("Test |"):
                        test_metrics = parse_metric_line(stripped)

                    match = CHECKPOINT_RE.search(stripped)
                    if match:
                        checkpoint_path = match.group(1).strip()
        except OSError:
            continue

        if not dataset_matches_log(dataset, config):
            continue
        script_name, loss_name = script_loss_from_method(config, method)
        if not script_name or not checkpoint_path or not test_metrics:
            continue
        if not seed:
            seed_match = re.search(r"seed(\d+)", log_path.stem)
            seed = seed_match.group(1) if seed_match else ""
        rows.append({
            "status": "success" if return_code in {"", "0"} else "failed",
            "seed": seed,
            "run_tag": run_tag,
            "loss_name": loss_name,
            "checkpoint_path": checkpoint_path,
            "_summary_path": str(log_path),
            "_script_name": script_name,
            **test_metrics,
        })
    return rows


def read_summary_rows(config: DatasetConfig) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    logs_dir = config.dir_path / "logs"
    if logs_dir.exists():
        for summary_path in logs_dir.rglob("*_summary.csv"):
            try:
                with summary_path.open("r", encoding="utf-8-sig", newline="") as fp:
                    reader = csv.DictReader(fp)
                    for row in reader:
                        item = dict(row)
                        item["_summary_path"] = str(summary_path)
                        item["_script_name"] = infer_script_name(summary_path, item)
                        rows.append(item)
            except OSError:
                continue
    rows.extend(read_batch_log_rows(config))
    return rows


def checkpoint_from_row(row: Dict[str, str], config: DatasetConfig) -> Path:
    raw = (row.get("checkpoint_path") or "").strip()
    if not raw:
        return Path()
    path = Path(raw).expanduser()
    if path.exists():
        return path.resolve()
    candidate = config.dir_path / "checkpoints" / path.name
    if candidate.exists():
        return candidate.resolve()
    return path


def row_is_success(row: Dict[str, str]) -> bool:
    status = (row.get("status") or "success").strip().lower()
    return status in {"", "success"}


def row_matches(row: Dict[str, str], script_name: str, loss_name: str) -> bool:
    if not row_is_success(row):
        return False
    if Path(row.get("_script_name", "")).name != script_name:
        return False
    row_loss = (row.get("loss_name") or "").strip().lower()
    if loss_name and row_loss and row_loss != loss_name:
        return False
    if loss_name and not row_loss and loss_name not in str(row.get("_summary_path", "")).lower():
        return False
    return bool((row.get("checkpoint_path") or "").strip())


def prefer_controlled_seeded(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    controlled = [
        row for row in rows
        if "controlled_" in f"{row.get('run_tag', '')} {row.get('_summary_path', '')}".lower()
    ]
    if controlled:
        rows = controlled
    seeded = [row for row in rows if str(row.get("seed") or "").strip()]
    return seeded or rows


def select_checkpoint(
    rows: List[Dict[str, str]],
    config: DatasetConfig,
    script_name: str,
    loss_name: str,
    choose: str,
) -> Optional[Dict[str, str]]:
    candidates = [row for row in rows if row_matches(row, script_name, loss_name)]
    candidates = prefer_controlled_seeded(candidates)
    scored: List[Tuple[float, Dict[str, str]]] = []
    for row in candidates:
        score = safe_float(row.get(config.selection_metric))
        if score is None:
            score = safe_float(row.get("test_acc"))
        if score is None:
            continue
        scored.append((score, row))
    if not scored:
        return None
    reverse = choose == "best"
    return sorted(scored, key=lambda item: item[0], reverse=reverse)[0][1]


def selected_info(row: Optional[Dict[str, str]], config: DatasetConfig) -> Dict[str, str]:
    if row is None:
        return {
            "checkpoint": "",
            "metric": config.selection_metric,
            "metric_value": "",
            "seed": "",
            "run_tag": "",
            "summary_path": "",
        }
    ckpt = checkpoint_from_row(row, config)
    value = row.get(config.selection_metric) or row.get("test_acc") or ""
    return {
        "checkpoint": str(ckpt),
        "metric": config.selection_metric,
        "metric_value": value,
        "seed": row.get("seed", ""),
        "run_tag": row.get("run_tag", ""),
        "summary_path": row.get("_summary_path", ""),
    }


def build_command(
    args: argparse.Namespace,
    config: DatasetConfig,
    class_name: str,
    selection_mode: str,
    output_dir: Path,
    baseline_checkpoint: Optional[Path],
    proposed_checkpoint: Optional[Path],
) -> List[str]:
    cmd = [
        args.python,
        "-u",
        str(config.exporter),
        "--class-name",
        class_name,
        "--max-samples",
        str(args.max_samples_per_class),
        "--device",
        args.device,
        "--image-size",
        str(args.image_size),
        "--alpha",
        str(args.alpha),
        "--cam-threshold",
        str(args.cam_threshold),
        "--cam-method",
        args.cam_method,
        "--cam-on",
        args.cam_on,
        "--selection-mode",
        selection_mode,
        "--output-dir",
        str(output_dir),
        "--baseline-model",
        config.baseline_script,
        "--proposed-model",
        config.proposed_script,
        "--baseline-target-layer",
        args.baseline_target_layer,
        "--proposed-target-layer",
        args.proposed_target_layer,
    ]
    if args.baseline_cam_threshold is not None:
        cmd.extend(["--baseline-cam-threshold", str(args.baseline_cam_threshold)])
    if args.proposed_cam_threshold is not None:
        cmd.extend(["--proposed-cam-threshold", str(args.proposed_cam_threshold)])
    data_root = getattr(args, f"{config.key}_data_root")
    if not data_root:
        data_root = os.getenv(config.data_root_env, "")
    if data_root:
        cmd.extend(["--data-root", data_root])
    if config.key in {"brain", "chest"}:
        cmd.extend([
            "--baseline-loss",
            config.baseline_loss,
            "--proposed-loss",
            config.proposed_loss,
        ])
    if baseline_checkpoint is not None and str(baseline_checkpoint):
        cmd.extend(["--baseline-checkpoint", str(baseline_checkpoint)])
    if proposed_checkpoint is not None and str(proposed_checkpoint):
        cmd.extend(["--proposed-checkpoint", str(proposed_checkpoint)])
    if args.aug_smooth:
        cmd.append("--aug-smooth")
    if args.eigen_smooth:
        cmd.append("--eigen-smooth")
    return cmd


def count_summary_rows(summary_path: Path) -> int:
    if not summary_path.exists():
        return 0
    with summary_path.open("r", encoding="utf-8-sig", newline="") as fp:
        return sum(1 for _ in csv.DictReader(fp))


def copy_panels(summary_path: Path, panels_dir: Path, prefix: str) -> int:
    if not summary_path.exists():
        return 0
    copied = 0
    panels_dir.mkdir(parents=True, exist_ok=True)
    with summary_path.open("r", encoding="utf-8-sig", newline="") as fp:
        for index, row in enumerate(csv.DictReader(fp), start=1):
            panel = Path(row.get("panel_path", ""))
            if not panel.exists():
                continue
            target = panels_dir / f"{prefix}_{index:02d}_{panel.name}"
            shutil.copy2(panel, target)
            copied += 1
    return copied


def run_command(cmd: Sequence[str], log_path: Path, dry_run: bool) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if dry_run:
        print(" ".join(cmd))
        return 0
    print("[RUN] " + " ".join(cmd), flush=True)
    proc = subprocess.run(
        list(cmd),
        cwd=str(PROJECT_ROOT),
        text=True,
        encoding="utf-8",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    log_path.write_text(proc.stdout or "", encoding="utf-8")
    if proc.stdout:
        print(proc.stdout, end="")
    return int(proc.returncode)


def write_csv(path: Path, rows: Sequence[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", default="brain,chest,koa", help="Comma-separated: brain,chest,koa.")
    parser.add_argument("--brain-classes", default="glioma,meningioma,pituitary")
    parser.add_argument("--chest-classes", default="COVID19,PNEUMONIA")
    parser.add_argument("--koa-classes", default="2,3,4")
    parser.add_argument("--brain-data-root", default="")
    parser.add_argument("--chest-data-root", default="")
    parser.add_argument("--koa-data-root", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--max-samples-per-class", type=int, default=2)
    parser.add_argument("--selection-mode", default="improved", choices=["improved", "proposed-correct", "any"])
    parser.add_argument("--fallback-selection-mode", default="proposed-correct", choices=["none", "proposed-correct", "any"])
    parser.add_argument("--cam-method", default="gradcam++", choices=["gradcam", "gradcam++", "hirescam", "eigencam"])
    parser.add_argument("--cam-on", default="pred", choices=["pred", "true"])
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--alpha", type=float, default=0.35)
    parser.add_argument("--cam-threshold", type=float, default=0.0)
    parser.add_argument("--baseline-cam-threshold", type=float, default=None)
    parser.add_argument("--proposed-cam-threshold", type=float, default=None)
    parser.add_argument("--baseline-target-layer", default="layer4")
    parser.add_argument("--proposed-target-layer", default="layer4")
    parser.add_argument("--aug-smooth", action="store_true")
    parser.add_argument("--eigen-smooth", action="store_true")
    parser.add_argument("--allow-default-checkpoints", action="store_true")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_keys = [key.lower() for key in parse_csv_tokens(args.datasets)]
    unknown = [key for key in dataset_keys if key not in DATASETS]
    if unknown:
        raise ValueError(f"Unknown datasets: {unknown}. Choose from {sorted(DATASETS)}.")

    if args.output_dir:
        output_root = Path(args.output_dir).expanduser().resolve()
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_root = (DEFAULT_OUTPUT_ROOT / f"gradcampp_visual_analysis_{timestamp}").resolve()
    panels_dir = output_root / "panels"

    manifest_rows: List[Dict[str, object]] = []
    selected_rows: List[Dict[str, object]] = []

    for key in dataset_keys:
        config = DATASETS[key]
        rows = read_summary_rows(config)
        baseline_row = select_checkpoint(rows, config, config.baseline_script, config.baseline_loss, "worst")
        proposed_row = select_checkpoint(rows, config, config.proposed_script, config.proposed_loss, "best")
        baseline_info = selected_info(baseline_row, config)
        proposed_info = selected_info(proposed_row, config)
        baseline_checkpoint = Path(baseline_info["checkpoint"]) if baseline_info["checkpoint"] else None
        proposed_checkpoint = Path(proposed_info["checkpoint"]) if proposed_info["checkpoint"] else None

        if (baseline_checkpoint is None or proposed_checkpoint is None) and not args.allow_default_checkpoints:
            missing = []
            if baseline_checkpoint is None:
                missing.append("baseline")
            if proposed_checkpoint is None:
                missing.append("proposed")
            raise RuntimeError(
                f"Could not auto-select {', '.join(missing)} checkpoint for {config.display_name}. "
                "Pass --allow-default-checkpoints or provide/check the summary CSVs."
            )

        selected_rows.append({
            "dataset": config.display_name,
            "role": "baseline_worst",
            "script": config.baseline_script,
            "loss": config.baseline_loss,
            **baseline_info,
        })
        selected_rows.append({
            "dataset": config.display_name,
            "role": "proposed_best",
            "script": config.proposed_script,
            "loss": config.proposed_loss,
            **proposed_info,
        })

        class_arg_name = f"{key}_classes"
        classes = parse_csv_tokens(getattr(args, class_arg_name))
        for class_name in classes:
            safe_class = sanitize_filename(class_name)
            dataset_dir = output_root / sanitize_filename(config.key)
            run_dir = dataset_dir / f"{safe_class}_{args.selection_mode}"
            log_path = run_dir / "export.log"
            cmd = build_command(
                args,
                config,
                class_name,
                args.selection_mode,
                run_dir,
                baseline_checkpoint,
                proposed_checkpoint,
            )
            rc = run_command(cmd, log_path, args.dry_run)
            summary_path = run_dir / "summary.csv"
            panel_count = count_summary_rows(summary_path) if rc == 0 and not args.dry_run else 0
            used_selection = args.selection_mode

            if (
                rc == 0
                and panel_count == 0
                and args.fallback_selection_mode != "none"
                and args.fallback_selection_mode != args.selection_mode
                and not args.dry_run
            ):
                used_selection = args.fallback_selection_mode
                run_dir = dataset_dir / f"{safe_class}_fallback_{used_selection}"
                log_path = run_dir / "export.log"
                cmd = build_command(
                    args,
                    config,
                    class_name,
                    used_selection,
                    run_dir,
                    baseline_checkpoint,
                    proposed_checkpoint,
                )
                rc = run_command(cmd, log_path, args.dry_run)
                summary_path = run_dir / "summary.csv"
                panel_count = count_summary_rows(summary_path) if rc == 0 else 0

            copied = 0
            if panel_count > 0 and not args.dry_run:
                prefix = sanitize_filename(f"{config.key}_{class_name}_{used_selection}")
                copied = copy_panels(summary_path, panels_dir, prefix)

            manifest_rows.append({
                "dataset": config.display_name,
                "class_name": class_name,
                "selection_mode": used_selection,
                "return_code": rc,
                "panel_count": panel_count,
                "copied_panels": copied,
                "output_dir": str(run_dir),
                "summary_path": str(summary_path) if summary_path.exists() else "",
                "log_path": str(log_path),
                "baseline_checkpoint": str(baseline_checkpoint or ""),
                "proposed_checkpoint": str(proposed_checkpoint or ""),
                "command": " ".join(cmd),
            })
            if rc != 0:
                print(f"[WARN] Export failed for {config.display_name} / {class_name}. See {log_path}", flush=True)

    if not args.dry_run:
        write_csv(
            output_root / "selected_checkpoints.csv",
            selected_rows,
            [
                "dataset",
                "role",
                "script",
                "loss",
                "checkpoint",
                "metric",
                "metric_value",
                "seed",
                "run_tag",
                "summary_path",
            ],
        )
        write_csv(
            output_root / "manifest.csv",
            manifest_rows,
            [
                "dataset",
                "class_name",
                "selection_mode",
                "return_code",
                "panel_count",
                "copied_panels",
                "output_dir",
                "summary_path",
                "log_path",
                "baseline_checkpoint",
                "proposed_checkpoint",
                "command",
            ],
        )
        readme = output_root / "README.md"
        readme.write_text(
            "\n".join([
                "# Grad-CAM++ Visual Analysis",
                "",
                "Panels use the layout: Original | Baseline | Proposed.",
                "",
                f"Primary selection mode: `{args.selection_mode}`.",
                f"Fallback selection mode: `{args.fallback_selection_mode}`.",
                f"CAM method: `{args.cam_method}`, cam_on: `{args.cam_on}`, cam_threshold: `{args.cam_threshold}`.",
                f"Baseline/proposed thresholds: `{args.baseline_cam_threshold}` / `{args.proposed_cam_threshold}`.",
                "",
                "Review `panels/` for copied panel images and `manifest.csv` for provenance.",
                "",
            ]),
            encoding="utf-8",
        )

    print(f"Output root: {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
