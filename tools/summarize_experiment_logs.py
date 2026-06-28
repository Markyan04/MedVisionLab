#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Summarize produced experiment logs/CSVs into CSV and LaTeX tables."""

from __future__ import annotations

import argparse
import csv
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_LINE_RE = re.compile(
    r"Test\s*\|.*?acc=(?P<acc>[0-9.]+)%.*?macro_f1=(?P<f1>[0-9.]+)"
    r".*?qwk=(?P<qwk>-?[0-9.]+).*?mae=(?P<mae>[0-9.]+)",
    re.IGNORECASE,
)
METRIC_VALUE_RE = re.compile(r"([A-Za-z0-9_]+)=(-?[0-9]+(?:\.[0-9]+)?)%?")
HEADER_RE = re.compile(r"^(DATASET|METHOD|SEED|RUN_TAG|DAST_TAU|DAST_GAMMA):\s*(.*)$")
SEED_RE = re.compile(r"seed(\d+)", re.IGNORECASE)


def dataset_from_path(path: Path) -> str:
    parts = set(path.parts)
    if "Knee" in parts:
        return "KOA"
    if "HAM10000_Loss" in parts:
        return "HAM10000"
    if "Alzheimer_MRI_Loss" in parts:
        return "ADNI"
    if "Brain_Tumor_MRI_Loss" in parts:
        return "Brain Tumor MRI"
    if "chest-x-ray-image_Loss" in parts:
        return "Chest X-ray Image"
    return ""


def method_from_script(script_name: str, loss_name: str) -> str:
    text = script_name.lower()
    loss = (loss_name or "").lower()
    has_mesc = "mecs" in text
    has_dast = loss == "dast" or "loss4" in text or "dast" in text
    if has_mesc and has_dast:
        return "ResNet50 + MESC + DAST"
    if has_mesc:
        return "ResNet50 + MESC"
    if has_dast:
        return "ResNet50 + DAST"
    return "ResNet50"


def parse_float(value: object, percent: bool = False) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        val = float(text)
    except ValueError:
        return None
    return val / 100.0 if percent else val


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


def seed_from_row(row: Dict[str, str]) -> str:
    if row.get("seed"):
        return str(row["seed"])
    match = SEED_RE.search(row.get("run_tag", ""))
    return match.group(1) if match else ""


def read_summary_csv(path: Path) -> Iterable[Dict[str, object]]:
    dataset = dataset_from_path(path)
    script_hint = path.stem.replace("_summary", "")
    with path.open("r", encoding="utf-8-sig", newline="") as fp:
        for row in csv.DictReader(fp):
            if row.get("status", "success") != "success":
                continue
            script_name = row.get("script_name") or script_hint
            loss_name = row.get("loss_name", "")
            yield {
                "dataset": dataset,
                "method": method_from_script(script_name, loss_name),
                "seed": seed_from_row(row),
                "source": str(path),
                "acc": parse_float(row.get("test_acc")),
                "macro_f1": parse_float(row.get("test_macro_f1")),
                "qwk": parse_float(row.get("test_qwk")),
                "mae": parse_float(row.get("test_mae")),
                "tau": row.get("dast_tau", ""),
                "gamma": row.get("dast_gamma", ""),
            }


def read_batch_log(path: Path) -> Optional[Dict[str, object]]:
    header: Dict[str, str] = {}
    last_test = None
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = HEADER_RE.match(line.strip())
        if m:
            header[m.group(1).lower()] = m.group(2).strip()
        parsed = parse_test_metrics_line(line)
        if parsed:
            last_test = parsed
    if not last_test or "dataset" not in header or "method" not in header:
        return None
    return {
        "dataset": header["dataset"],
        "method": header["method"],
        "seed": header.get("seed", ""),
        "source": str(path),
        "acc": parse_float(last_test["acc"], percent=True),
        "macro_f1": parse_float(last_test["macro_f1"]),
        "qwk": parse_float(last_test["qwk"]),
        "mae": parse_float(last_test["mae"]),
        "tau": header.get("dast_tau", ""),
        "gamma": header.get("dast_gamma", ""),
    }


def one_off_accuracy_from_confusion(path: Path) -> Optional[float]:
    rows: List[List[int]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as fp:
        for row in csv.reader(fp):
            nums = []
            for cell in row:
                try:
                    nums.append(int(float(cell)))
                except ValueError:
                    pass
            if nums:
                rows.append(nums[-5:] if len(nums) >= 5 else nums)
    if not rows:
        return None
    total = 0
    ok = 0
    for i, row in enumerate(rows):
        for j, count in enumerate(row):
            total += count
            if abs(i - j) <= 1:
                ok += count
    return ok / total if total else None


def collect_records(root: Path) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    for path in root.rglob("*_summary.csv"):
        records.extend(read_summary_csv(path))
    for path in root.rglob("*.log"):
        rec = read_batch_log(path)
        if rec:
            records.append(rec)
    return [
        r for r in records
        if r.get("dataset") and r.get("acc") is not None and r.get("macro_f1") is not None
    ]


def fmt_mean_std(values: List[float]) -> Tuple[float, float]:
    if not values:
        return math.nan, math.nan
    if len(values) == 1:
        return values[0], 0.0
    return statistics.mean(values), statistics.stdev(values)


def metric_cell(values: List[float], percent: bool = False) -> str:
    mean, std = fmt_mean_std(values)
    if math.isnan(mean):
        return "--"
    scale = 100.0 if percent else 1.0
    if len(values) == 1:
        return f"{mean * scale:.2f}" if percent else f"{mean:.4f}"
    return (f"{mean * scale:.2f} $\\pm$ {std * scale:.2f}") if percent else (f"{mean:.4f} $\\pm$ {std:.4f}")


def write_outputs(records: List[Dict[str, object]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "experiment_records.csv"
    with raw_path.open("w", encoding="utf-8-sig", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=["dataset", "method", "seed", "acc", "macro_f1", "qwk", "mae", "tau", "gamma", "source"])
        writer.writeheader()
        writer.writerows(records)

    grouped: Dict[Tuple[str, str], List[Dict[str, object]]] = defaultdict(list)
    for rec in records:
        grouped[(str(rec["dataset"]), str(rec["method"]))].append(rec)

    summary_path = out_dir / "controlled_ablation_summary.csv"
    tex_path = out_dir / "controlled_ablation_summary.tex"
    with summary_path.open("w", encoding="utf-8-sig", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(["dataset", "method", "n", "acc_mean", "acc_std", "macro_f1_mean", "macro_f1_std", "qwk_mean", "qwk_std", "mae_mean", "mae_std"])
        for (dataset, method), rows in sorted(grouped.items()):
            acc = [float(r["acc"]) for r in rows if r.get("acc") is not None]
            f1 = [float(r["macro_f1"]) for r in rows if r.get("macro_f1") is not None]
            qwk = [float(r["qwk"]) for r in rows if r.get("qwk") is not None]
            mae = [float(r["mae"]) for r in rows if r.get("mae") is not None]
            writer.writerow([dataset, method, len(rows), *fmt_mean_std(acc), *fmt_mean_std(f1), *fmt_mean_std(qwk), *fmt_mean_std(mae)])

    lines = [
        "\\begin{tabular}{llccc}",
        "\\toprule",
        "Dataset & Method & ACC (\\%) & Macro-F1 & n \\\\",
        "\\midrule",
    ]
    for (dataset, method), rows in sorted(grouped.items()):
        acc = [float(r["acc"]) for r in rows if r.get("acc") is not None]
        f1 = [float(r["macro_f1"]) for r in rows if r.get("macro_f1") is not None]
        lines.append(f"{dataset} & {method} & {metric_cell(acc, percent=True)} & {metric_cell(f1)} & {len(rows)} \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    tex_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {raw_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {tex_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "analysis_tables")
    args = parser.parse_args()
    records = collect_records(args.root)
    write_outputs(records, args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
