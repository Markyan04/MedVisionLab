#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Summarize produced experiment logs/CSVs into CSV, Markdown, and LaTeX tables."""

from __future__ import annotations

import argparse
import csv
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

try:
    from scipy import stats as scipy_stats
except Exception:  # pragma: no cover - scipy is listed in requirements.
    scipy_stats = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_LINE_RE = re.compile(
    r"Test\s*\|.*?acc=(?P<acc>[0-9.]+)%.*?macro_f1=(?P<f1>[0-9.]+)"
    r".*?qwk=(?P<qwk>-?[0-9.]+).*?mae=(?P<mae>[0-9.]+)",
    re.IGNORECASE,
)
METRIC_VALUE_RE = re.compile(r"([A-Za-z0-9_]+)=(-?[0-9]+(?:\.[0-9]+)?)%?")
HEADER_RE = re.compile(
    r"^(DATASET|METHOD|SEED|RUN_TAG|LOSS|LABEL_SMOOTHING|SORD_TAU|DAST_TAU|DAST_GAMMA):\s*(.*)$"
)
SEED_RE = re.compile(r"seed(\d+)", re.IGNORECASE)

LOSS_ORDER = ("CE", "Label Smoothing", "SORD-CE", "DAST")
LOSS_COMPARISONS = (
    ("CE", "Label Smoothing"),
    ("CE", "SORD-CE"),
    ("SORD-CE", "DAST"),
    ("CE", "DAST"),
)
LOSS_METRICS = (
    ("acc", "ACC", True),
    ("macro_f1", "Macro-F1", False),
    ("mae", "MAE", False),
    ("qwk", "QWK", False),
)
LOSS_DATASETS = {"KOA", "ADNI"}


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


def canonical_loss_name(raw: object) -> str:
    key = str(raw or "").strip().lower().replace("-", "_").replace(" ", "_")
    if key in {"ce", "cross_entropy", "crossentropyloss"}:
        return "CE"
    if key in {"label_smoothing_ce", "label_smoothing", "ls", "ls_ce"}:
        return "Label Smoothing"
    if key in {"sord", "sord_ce"}:
        return "SORD-CE"
    if key in {"dast", "loss4", "distance_aware_soft_target"}:
        return "DAST"
    return ""


def loss_from_method(method: object, loss_name: object = "") -> str:
    loss = canonical_loss_name(loss_name)
    if loss:
        return loss

    text = str(method or "").lower()
    if "label" in text and "smooth" in text:
        return "Label Smoothing"
    if "sord" in text:
        return "SORD-CE"
    if "dast" in text or "loss4" in text:
        return "DAST"
    if text.strip() == "resnet50":
        return "CE"
    return ""


def method_from_script(script_name: str, loss_name: str) -> str:
    text = script_name.lower()
    loss = canonical_loss_name(loss_name)
    has_mesc = "mecs" in text
    if loss == "Label Smoothing":
        return "ResNet50 + Label Smoothing"
    if loss == "SORD-CE":
        return "ResNet50 + SORD-CE"

    has_dast = loss == "DAST" or "loss4" in text or "dast" in text
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
    if not text or text.lower() in {"none", "nan"}:
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
            method = method_from_script(script_name, loss_name)
            yield {
                "dataset": dataset,
                "method": method,
                "loss": loss_from_method(method, loss_name),
                "loss_name": loss_name,
                "seed": seed_from_row(row),
                "run_tag": row.get("run_tag", ""),
                "source": str(path),
                "acc": parse_float(row.get("test_acc")),
                "macro_f1": parse_float(row.get("test_macro_f1")),
                "qwk": parse_float(row.get("test_qwk")),
                "mae": parse_float(row.get("test_mae")),
                "tau": row.get("dast_tau", ""),
                "gamma": row.get("dast_gamma", ""),
                "label_smoothing": row.get("label_smoothing", ""),
            }


def read_experiment_records_csv(path: Path) -> Iterable[Dict[str, object]]:
    with path.open("r", encoding="utf-8-sig", newline="") as fp:
        for row in csv.DictReader(fp):
            if not row.get("dataset") or not row.get("method"):
                continue
            method = row.get("method", "")
            loss_name = row.get("loss") or row.get("loss_name", "")
            yield {
                "dataset": row.get("dataset", ""),
                "method": method,
                "loss": loss_from_method(method, loss_name),
                "loss_name": loss_name,
                "seed": seed_from_row(row),
                "run_tag": row.get("run_tag", ""),
                "source": row.get("source", str(path)),
                "acc": parse_float(row.get("acc")),
                "macro_f1": parse_float(row.get("macro_f1")),
                "qwk": parse_float(row.get("qwk")),
                "mae": parse_float(row.get("mae")),
                "tau": row.get("tau", ""),
                "gamma": row.get("gamma", ""),
                "label_smoothing": row.get("label_smoothing", ""),
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
    method = header["method"]
    loss_name = header.get("loss", "")
    tau = header.get("sord_tau", "") or header.get("dast_tau", "")
    gamma = header.get("dast_gamma", "")
    return {
        "dataset": header["dataset"],
        "method": method,
        "loss": loss_from_method(method, loss_name),
        "loss_name": loss_name,
        "seed": header.get("seed", ""),
        "run_tag": header.get("run_tag", ""),
        "source": str(path),
        "acc": parse_float(last_test["acc"], percent=True),
        "macro_f1": parse_float(last_test["macro_f1"]),
        "qwk": parse_float(last_test["qwk"]),
        "mae": parse_float(last_test["mae"]),
        "tau": tau,
        "gamma": gamma,
        "label_smoothing": header.get("label_smoothing", ""),
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


def record_identity(rec: Dict[str, object], identity: str) -> Tuple[str, str, str, str, str]:
    return (
        str(rec.get("dataset") or ""),
        str(rec.get("method") or ""),
        str(rec.get("loss") or ""),
        str(rec.get("seed") or ""),
        identity,
    )


def record_preference(rec: Dict[str, object]) -> Tuple[int, int]:
    return (
        1 if rec.get("run_tag") else 0,
        1 if str(rec.get("source", "")).lower().endswith("_summary.csv") else 0,
    )


def dedupe_records(records: Iterable[Dict[str, object]]) -> List[Dict[str, object]]:
    dedup: Dict[Tuple[str, str, str, str, str], Dict[str, object]] = {}
    source_index: Dict[Tuple[str, str, str, str, str], Tuple[str, str, str, str, str]] = {}
    tag_index: Dict[Tuple[str, str, str, str, str], Tuple[str, str, str, str, str]] = {}
    for rec in records:
        source = str(rec.get("source") or "")
        run_tag = str(rec.get("run_tag") or "")
        source_key = record_identity(rec, source)
        tag_key = record_identity(rec, run_tag) if run_tag else None
        key = source_index.get(source_key)
        if key is None and tag_key is not None:
            key = tag_index.get(tag_key)
        if key is None:
            key = tag_key or source_key
            dedup[key] = rec
            source_index[source_key] = key
            if tag_key is not None:
                tag_index[tag_key] = key
            continue
        if record_preference(rec) > record_preference(dedup[key]):
            dedup[key] = rec
        source_index[source_key] = key
        if tag_key is not None:
            tag_index[tag_key] = key
    return list(dedup.values())


def collect_records(root: Path) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    records_path = root / "analysis_tables" / "experiment_records.csv"
    if records_path.exists():
        records.extend(read_experiment_records_csv(records_path))
    for path in root.rglob("*_summary.csv"):
        records.extend(read_summary_csv(path))
    for path in root.rglob("*.log"):
        rec = read_batch_log(path)
        if rec:
            records.append(rec)
    records = [
        r for r in records
        if r.get("dataset") and r.get("acc") is not None and r.get("macro_f1") is not None
    ]
    return dedupe_records(records)


def fmt_mean_std(values: List[float]) -> Tuple[float, float]:
    if not values:
        return math.nan, math.nan
    if len(values) == 1:
        return values[0], 0.0
    return statistics.mean(values), statistics.stdev(values)


def metric_cell(values: List[float], percent: bool = False, latex: bool = True) -> str:
    mean, std = fmt_mean_std(values)
    if math.isnan(mean):
        return "--"
    scale = 100.0 if percent else 1.0
    if percent:
        value = f"{mean * scale:.2f}"
        spread = f"{std * scale:.2f}"
    else:
        value = f"{mean:.4f}"
        spread = f"{std:.4f}"
    if len(values) == 1:
        return value
    pm = " $\\pm$ " if latex else " \u00b1 "
    return f"{value}{pm}{spread}"


def write_controlled_outputs(records: List[Dict[str, object]], out_dir: Path) -> None:
    raw_path = out_dir / "experiment_records.csv"
    raw_fields = [
        "dataset",
        "method",
        "loss",
        "loss_name",
        "seed",
        "acc",
        "macro_f1",
        "qwk",
        "mae",
        "tau",
        "gamma",
        "label_smoothing",
        "run_tag",
        "source",
    ]
    with raw_path.open("w", encoding="utf-8-sig", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=raw_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)

    grouped: Dict[Tuple[str, str], List[Dict[str, object]]] = defaultdict(list)
    for rec in records:
        grouped[(str(rec["dataset"]), str(rec["method"]))].append(rec)

    summary_path = out_dir / "controlled_ablation_summary.csv"
    tex_path = out_dir / "controlled_ablation_summary.tex"
    with summary_path.open("w", encoding="utf-8-sig", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow([
            "dataset",
            "method",
            "n",
            "acc_mean",
            "acc_std",
            "macro_f1_mean",
            "macro_f1_std",
            "qwk_mean",
            "qwk_std",
            "mae_mean",
            "mae_std",
        ])
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


def is_loss_comparison_record(rec: Dict[str, object]) -> bool:
    dataset = str(rec.get("dataset") or "")
    loss = str(rec.get("loss") or "")
    seed = str(rec.get("seed") or "")
    method = str(rec.get("method") or "")
    tag_source = f"{rec.get('run_tag', '')} {rec.get('source', '')}".lower()
    if dataset not in LOSS_DATASETS or loss not in LOSS_ORDER or not seed:
        return False
    if "mesc" in method.lower():
        return False
    if "controlled_" not in tag_source and "loss_comparison_" not in tag_source:
        return False
    return method in {
        "ResNet50",
        "ResNet50 + DAST",
        "ResNet50 + Label Smoothing",
        "ResNet50 + SORD-CE",
    }


def loss_record_preference(rec: Dict[str, object]) -> Tuple[int, int]:
    tag_source = f"{rec.get('run_tag', '')} {rec.get('source', '')}".lower()
    return (
        1 if "loss_comparison_" in tag_source else 0,
        1 if str(rec.get("source", "")).lower().endswith("_summary.csv") else 0,
    )


def collect_loss_records(records: List[Dict[str, object]]) -> List[Dict[str, object]]:
    selected = [r for r in records if is_loss_comparison_record(r)]
    by_key: Dict[Tuple[str, str, str], Dict[str, object]] = {}
    for rec in selected:
        key = (str(rec["dataset"]), str(rec["loss"]), str(rec["seed"]))
        if key not in by_key or loss_record_preference(rec) > loss_record_preference(by_key[key]):
            by_key[key] = rec
    return sorted(
        by_key.values(),
        key=lambda r: (
            str(r["dataset"]),
            LOSS_ORDER.index(str(r["loss"])) if str(r["loss"]) in LOSS_ORDER else 99,
            int(str(r["seed"])) if str(r["seed"]).isdigit() else 999999,
        ),
    )


def table_values(rows: List[Dict[str, object]], metric: str) -> List[float]:
    return [float(r[metric]) for r in rows if r.get(metric) is not None]


def paired_t_test(a_values: List[float], b_values: List[float]) -> Tuple[str, float, float]:
    if len(a_values) != len(b_values) or len(a_values) < 2:
        return "paired t-test", math.nan, math.nan
    diffs = [b - a for a, b in zip(a_values, b_values)]
    if all(abs(d) < 1e-12 for d in diffs):
        return "paired t-test", 0.0, 1.0
    if scipy_stats is None:
        return "paired t-test", math.nan, math.nan
    result = scipy_stats.ttest_rel(b_values, a_values, nan_policy="omit")
    return "paired t-test", float(result.statistic), float(result.pvalue)


def write_loss_comparison_outputs(records: List[Dict[str, object]], out_dir: Path) -> None:
    loss_records = collect_loss_records(records)
    if not loss_records:
        return

    raw_path = out_dir / "loss_comparison_records.csv"
    summary_path = out_dir / "loss_comparison_summary.csv"
    md_path = out_dir / "loss_comparison_summary.md"
    tex_path = out_dir / "loss_comparison_summary.tex"
    tests_path = out_dir / "loss_comparison_paired_tests.csv"
    tests_md_path = out_dir / "loss_comparison_paired_tests.md"

    raw_fields = [
        "dataset",
        "loss",
        "method",
        "seed",
        "acc",
        "macro_f1",
        "mae",
        "qwk",
        "tau",
        "gamma",
        "label_smoothing",
        "run_tag",
        "source",
    ]
    with raw_path.open("w", encoding="utf-8-sig", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=raw_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(loss_records)

    grouped: Dict[Tuple[str, str], List[Dict[str, object]]] = defaultdict(list)
    for rec in loss_records:
        grouped[(str(rec["dataset"]), str(rec["loss"]))].append(rec)

    with summary_path.open("w", encoding="utf-8-sig", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow([
            "dataset",
            "loss",
            "n",
            "acc_mean",
            "acc_std",
            "macro_f1_mean",
            "macro_f1_std",
            "mae_mean",
            "mae_std",
            "qwk_mean",
            "qwk_std",
        ])
        for dataset in sorted({d for d, _ in grouped}):
            for loss in LOSS_ORDER:
                rows = grouped.get((dataset, loss), [])
                if not rows:
                    continue
                writer.writerow([
                    dataset,
                    loss,
                    len(rows),
                    *fmt_mean_std(table_values(rows, "acc")),
                    *fmt_mean_std(table_values(rows, "macro_f1")),
                    *fmt_mean_std(table_values(rows, "mae")),
                    *fmt_mean_std(table_values(rows, "qwk")),
                ])

    md_lines = [
        "| Dataset | Loss | ACC | Macro-F1 | MAE | QWK | n |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    tex_lines = [
        "\\begin{tabular}{llccccc}",
        "\\toprule",
        "Dataset & Loss & ACC (\\%) & Macro-F1 & MAE & QWK & n \\\\",
        "\\midrule",
    ]
    for dataset in sorted({d for d, _ in grouped}):
        for loss in LOSS_ORDER:
            rows = grouped.get((dataset, loss), [])
            if not rows:
                continue
            acc = table_values(rows, "acc")
            f1 = table_values(rows, "macro_f1")
            mae = table_values(rows, "mae")
            qwk = table_values(rows, "qwk")
            md_lines.append(
                f"| {dataset} | {loss} | {metric_cell(acc, percent=True, latex=False)} | "
                f"{metric_cell(f1, latex=False)} | {metric_cell(mae, latex=False)} | "
                f"{metric_cell(qwk, latex=False)} | {len(rows)} |"
            )
            tex_lines.append(
                f"{dataset} & {loss} & {metric_cell(acc, percent=True)} & "
                f"{metric_cell(f1)} & {metric_cell(mae)} & {metric_cell(qwk)} & {len(rows)} \\\\"
            )
    tex_lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    tex_path.write_text("\n".join(tex_lines), encoding="utf-8")

    by_dataset_loss_seed: Dict[Tuple[str, str, str], Dict[str, object]] = {
        (str(r["dataset"]), str(r["loss"]), str(r["seed"])): r for r in loss_records
    }

    test_rows: List[Dict[str, object]] = []
    for dataset in sorted({str(r["dataset"]) for r in loss_records}):
        for loss_a, loss_b in LOSS_COMPARISONS:
            seeds_a = {seed for d, loss, seed in by_dataset_loss_seed if d == dataset and loss == loss_a}
            seeds_b = {seed for d, loss, seed in by_dataset_loss_seed if d == dataset and loss == loss_b}
            common_seeds = sorted(
                seeds_a & seeds_b,
                key=lambda s: (0, int(s)) if str(s).isdigit() else (1, str(s)),
            )
            for metric, metric_label, _ in LOSS_METRICS:
                a_values = [float(by_dataset_loss_seed[(dataset, loss_a, seed)][metric]) for seed in common_seeds]
                b_values = [float(by_dataset_loss_seed[(dataset, loss_b, seed)][metric]) for seed in common_seeds]
                test_name, statistic, p_value = paired_t_test(a_values, b_values)
                mean_delta = statistics.mean([b - a for a, b in zip(a_values, b_values)]) if common_seeds else math.nan
                test_rows.append({
                    "dataset": dataset,
                    "comparison": f"{loss_a} vs {loss_b}",
                    "metric": metric_label,
                    "n": len(common_seeds),
                    "mean_delta_second_minus_first": mean_delta,
                    "test": test_name,
                    "statistic": statistic,
                    "p_value": p_value,
                    "seeds": ",".join(common_seeds),
                })

    with tests_path.open("w", encoding="utf-8-sig", newline="") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=[
                "dataset",
                "comparison",
                "metric",
                "n",
                "mean_delta_second_minus_first",
                "test",
                "statistic",
                "p_value",
                "seeds",
            ],
        )
        writer.writeheader()
        writer.writerows(test_rows)

    tests_md_lines = [
        "| Dataset | Comparison | Metric | n | Delta | p-value | Test |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    for row in test_rows:
        delta = row["mean_delta_second_minus_first"]
        p_value = row["p_value"]
        tests_md_lines.append(
            f"| {row['dataset']} | {row['comparison']} | {row['metric']} | {row['n']} | "
            f"{float(delta):.6f} | {float(p_value):.6g} | {row['test']} |"
            if not math.isnan(float(delta)) and not math.isnan(float(p_value))
            else f"| {row['dataset']} | {row['comparison']} | {row['metric']} | {row['n']} | -- | -- | {row['test']} |"
        )
    tests_md_path.write_text("\n".join(tests_md_lines) + "\n", encoding="utf-8")

    print(f"Wrote {raw_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {md_path}")
    print(f"Wrote {tex_path}")
    print(f"Wrote {tests_path}")
    print(f"Wrote {tests_md_path}")


def write_outputs(records: List[Dict[str, object]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    write_controlled_outputs(records, out_dir)
    write_loss_comparison_outputs(records, out_dir)


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
