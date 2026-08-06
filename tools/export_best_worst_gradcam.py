#!/usr/bin/env python
"""Export Grad-CAM++ panels where the best MESC+DAST is right and worst baseline is wrong.

The best proposed and worst baseline runs are selected from an experiment-records
CSV using QWK by default.  KOA uses its fixed test directory.  ADNI uses the
intersection of the two selected seeds' test sets so every exported image was
held out from both models.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

from PIL import Image, ImageDraw
import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]
KNEE_DIR = PROJECT_ROOT / "Knee"
ADNI_DIR = PROJECT_ROOT / "Alzheimer_MRI_Loss"
for path in (PROJECT_ROOT, KNEE_DIR, ADNI_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from gradcam_shared import (  # noqa: E402
    extract_logits,
    load_script_module,
    sanitize_filename,
    tensor_to_pil,
)
from MECS_VersionB import MECS_VersionB  # noqa: E402
from pytorch_grad_cam_shared import (  # noqa: E402
    CAM_METHOD_CHOICES,
    build_cam_images,
    ensure_pytorch_grad_cam,
    resolve_target_module,
)


PROPOSED_METHOD = "ResNet50 + MESC VersionB + DAST"
BASELINE_METHOD = "ResNet50"
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
KOA_CLASS_NAMES = ("0_Normal", "1_Doubtful", "2_Mild", "3_Moderate", "4_Severe")
ADNI_CLASS_NAMES = (
    "NonDemented",
    "VeryMildDemented",
    "MildDemented",
    "ModerateDemented",
)


@dataclass(frozen=True)
class SelectedRun:
    dataset: str
    role: str
    method: str
    seed: int
    metric: str
    metric_value: float
    run_tag: str
    row: dict[str, str]


@dataclass(frozen=True)
class ImageRecord:
    image_path: Path
    label_index: int
    label_name: str
    relative_path: str


class RecordDataset(data.Dataset):
    def __init__(self, records: Sequence[ImageRecord], transform):
        self.records = list(records)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        with Image.open(record.image_path) as image:
            image = image.convert("RGB")
        return self.transform(image), record.label_index, index


class LogitsOnlyModel(nn.Module):
    """Make tuple-returning ADNI models compatible with pytorch-grad-cam."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return extract_logits(self.model(x))


def comma_list(raw: str) -> list[str]:
    return [part.strip().lower() for part in raw.split(",") if part.strip()]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", default="koa,adni", help="koa, adni, or koa,adni")
    parser.add_argument("--records", type=Path, default=None, help="Experiment records CSV; auto-detected by default.")
    parser.add_argument("--metric", default="qwk", choices=("qwk", "macro_f1", "acc"))
    parser.add_argument("--koa-data-root", type=Path, default=None)
    parser.add_argument("--adni-data-root", type=Path, default=None)
    parser.add_argument("--koa-baseline-checkpoint", type=Path, default=None)
    parser.add_argument("--koa-proposed-checkpoint", type=Path, default=None)
    parser.add_argument("--adni-baseline-checkpoint", type=Path, default=None)
    parser.add_argument("--adni-proposed-checkpoint", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-samples", type=int, default=0, help="0 exports every qualifying sample.")
    parser.add_argument("--classes", default="", help="Optional class names or indices, comma-separated.")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--target-layer", default="layer4")
    parser.add_argument("--cam-method", default="gradcam++", choices=CAM_METHOD_CHOICES)
    parser.add_argument("--cam-on", default="pred", choices=("pred", "true"))
    parser.add_argument("--alpha", type=float, default=0.35)
    parser.add_argument("--cam-threshold", type=float, default=0.0)
    parser.add_argument("--aug-smooth", action="store_true")
    parser.add_argument("--eigen-smooth", action="store_true")
    parser.add_argument("--save-components", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Print selections and checkpoint paths without loading data/models.")
    args = parser.parse_args(argv)

    args.dataset_names = comma_list(args.datasets)
    if not args.dataset_names or any(name not in {"koa", "adni"} for name in args.dataset_names):
        parser.error("--datasets supports koa, adni, or koa,adni.")
    if len(args.dataset_names) != len(set(args.dataset_names)):
        parser.error("--datasets contains duplicates.")
    if args.max_samples < 0 or args.image_size <= 0 or args.batch_size <= 0 or args.num_workers < 0:
        parser.error("invalid max-samples, image-size, batch-size, or num-workers.")
    if not 0 <= args.alpha <= 1 or not 0 <= args.cam_threshold <= 1:
        parser.error("--alpha and --cam-threshold must be in [0, 1].")

    args.koa_data_root = args.koa_data_root or os.getenv("KNEE_DATA_ROOT")
    args.adni_data_root = args.adni_data_root or os.getenv("ALZHEIMER_DATA_ROOT")
    for name in ("koa", "adni"):
        value = getattr(args, f"{name}_data_root")
        if value:
            setattr(args, f"{name}_data_root", Path(value).expanduser().resolve())
        for role in ("baseline", "proposed"):
            attr = f"{name}_{role}_checkpoint"
            checkpoint = getattr(args, attr)
            if checkpoint:
                setattr(args, attr, checkpoint.expanduser().resolve())
    args.class_filters = set(comma_list(args.classes))
    return args


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def records_coverage(rows: Sequence[dict[str, str]], datasets_to_find: Sequence[str]) -> int:
    pairs = {(row.get("dataset", "").upper(), row.get("method", "")) for row in rows}
    return sum(
        (dataset.upper(), method) in pairs
        for dataset in datasets_to_find
        for method in (BASELINE_METHOD, PROPOSED_METHOD)
    )


def resolve_records_file(explicit: Path | None, datasets_to_find: Sequence[str]) -> Path:
    if explicit is not None:
        path = explicit.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Experiment records not found: {path}")
        return path

    candidates = list((PROJECT_ROOT / "analysis_tables").glob("**/experiment_records.csv"))
    ranked = []
    for path in candidates:
        try:
            rows = read_csv(path)
        except Exception:
            continue
        coverage = records_coverage(rows, datasets_to_find)
        ranked.append((coverage, len(rows), path.stat().st_mtime, path))
    if not ranked:
        raise FileNotFoundError("No experiment_records.csv found under analysis_tables/.")
    coverage, _, _, selected = max(ranked)
    required = len(datasets_to_find) * 2
    if coverage < required:
        raise RuntimeError(
            f"No records CSV covers all requested baseline/proposed pairs (best coverage {coverage}/{required})."
        )
    return selected.resolve()


def select_run(
    rows: Sequence[dict[str, str]],
    dataset_name: str,
    role: str,
    metric: str,
) -> SelectedRun:
    method = PROPOSED_METHOD if role == "proposed" else BASELINE_METHOD
    candidates = []
    for row in rows:
        if row.get("dataset", "").upper() != dataset_name.upper() or row.get("method") != method:
            continue
        try:
            value = float(row[metric])
            seed = int(row["seed"])
        except (KeyError, TypeError, ValueError):
            continue
        candidates.append((value, seed, row))
    if not candidates:
        raise RuntimeError(f"No valid {dataset_name}/{method} rows contain metric={metric}.")
    selector = max if role == "proposed" else min
    value, seed, row = selector(candidates, key=lambda item: (item[0], item[1]))
    run_tag = row.get("run_tag", "").strip()
    if not run_tag:
        raise RuntimeError(f"Selected {dataset_name}/{method}/seed={seed} has no run_tag.")
    return SelectedRun(
        dataset=dataset_name.lower(),
        role=role,
        method=method,
        seed=seed,
        metric=metric,
        metric_value=value,
        run_tag=run_tag,
        row=dict(row),
    )


def expected_checkpoint(run: SelectedRun) -> Path:
    if run.dataset == "koa" and run.role == "baseline":
        name = f"best_resnet50_knee_oa_standard_ce_{run.run_tag}.pt"
        return KNEE_DIR / "checkpoints" / name
    if run.dataset == "koa":
        name = f"best_resnet50_mecs_layer3_dast_knee_oa_new_{run.run_tag}.pt"
        return KNEE_DIR / "checkpoints" / name
    if run.role == "baseline":
        name = f"best_ResNet_baseline_ce_{run.run_tag}.pt"
        return ADNI_DIR / "checkpoints" / name
    name = f"best_ResNet_layer3+MECS_dast_{run.run_tag}.pt"
    return ADNI_DIR / "checkpoints" / name


def resolve_checkpoint(run: SelectedRun, explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit
    expected = expected_checkpoint(run).resolve()
    if expected.is_file():
        return expected
    checkpoint_dir = KNEE_DIR / "checkpoints" if run.dataset == "koa" else ADNI_DIR / "checkpoints"
    matches = sorted(checkpoint_dir.glob(f"*{run.run_tag}*.pt")) if checkpoint_dir.is_dir() else []
    if len(matches) == 1:
        return matches[0].resolve()
    return expected


def resolve_device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def koa_records(data_root: Path) -> tuple[list[ImageRecord], list[str], dict[str, object]]:
    test_dir = data_root / "test"
    if not test_dir.is_dir():
        raise FileNotFoundError(f"KOA test directory not found: {test_dir}")
    dataset = datasets.ImageFolder(test_dir)
    if len(dataset.classes) != len(KOA_CLASS_NAMES):
        raise RuntimeError(f"Expected five KOA classes, found: {dataset.classes}")
    rows = []
    for raw_path, label in dataset.samples:
        image_path = Path(raw_path).resolve()
        rows.append(
            ImageRecord(
                image_path=image_path,
                label_index=int(label),
                label_name=KOA_CLASS_NAMES[label],
                relative_path=str(image_path.relative_to(test_dir)),
            )
        )
    return rows, list(KOA_CLASS_NAMES), {"policy": "fixed_test", "test_size": len(rows)}


def adni_test_samples(data_root: Path, seed: int):
    from alzheimer_mri_loss_experiment_common import collect_ordered_samples

    samples, class_names = collect_ordered_samples(data_root, ADNI_CLASS_NAMES)
    targets = [label for _, label in samples]
    _, test_samples = train_test_split(
        samples,
        test_size=0.2,
        random_state=seed,
        stratify=targets,
    )
    return test_samples, list(class_names)


def adni_records(
    data_root: Path,
    baseline_seed: int,
    proposed_seed: int,
) -> tuple[list[ImageRecord], list[str], dict[str, object]]:
    baseline_test, class_names = adni_test_samples(data_root, baseline_seed)
    proposed_test, _ = adni_test_samples(data_root, proposed_seed)
    baseline_map = {Path(path).resolve(): int(label) for path, label in baseline_test}
    proposed_map = {Path(path).resolve(): int(label) for path, label in proposed_test}
    common_paths = sorted(set(baseline_map).intersection(proposed_map), key=lambda path: str(path))
    rows = []
    for image_path in common_paths:
        label = baseline_map[image_path]
        if proposed_map[image_path] != label:
            raise RuntimeError(f"ADNI label mismatch for: {image_path}")
        rows.append(
            ImageRecord(
                image_path=image_path,
                label_index=label,
                label_name=class_names[label],
                relative_path=str(image_path.relative_to(data_root)),
            )
        )
    if not rows:
        raise RuntimeError("The selected ADNI test splits have no common images.")
    metadata = {
        "policy": "intersection_of_model_test_sets",
        "baseline_seed": baseline_seed,
        "baseline_test_size": len(baseline_test),
        "proposed_seed": proposed_seed,
        "proposed_test_size": len(proposed_test),
        "intersection_size": len(rows),
    }
    return rows, class_names, metadata


def eval_transform(dataset_name: str, image_size: int):
    operations = []
    if dataset_name == "adni":
        operations.append(transforms.Grayscale(num_output_channels=3))
    operations.extend(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD),
        ]
    )
    return transforms.Compose(operations)


def instantiate_without_pretrained(module, builder):
    module_models = getattr(module, "models", None)
    if module_models is None or not hasattr(module_models, "resnet50"):
        return builder()
    original = module_models.resnet50

    def no_weights(*args, **kwargs):
        kwargs.pop("pretrained", None)
        kwargs["weights"] = None
        return original(*args, **kwargs)

    module_models.resnet50 = no_weights
    try:
        return builder()
    finally:
        module_models.resnet50 = original


def build_koa_model(role: str, num_classes: int) -> nn.Module:
    script = KNEE_DIR / ("ResNet_baseline.py" if role == "baseline" else "ResNet_layer3+MECS+Loss4.py")
    module = load_script_module(script, prefix=f"best_worst_gradcam_koa_{role}")
    if role == "proposed":
        module.MECS_VersionA = MECS_VersionB
        model_class = module.CustomResNet50MECS
        return instantiate_without_pretrained(module, lambda: model_class(num_classes=num_classes))
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def build_adni_model(role: str, num_classes: int) -> nn.Module:
    script = ADNI_DIR / ("ResNet_baseline.py" if role == "baseline" else "ResNet_layer3+MECS.py")
    module = load_script_module(script, prefix=f"best_worst_gradcam_adni_{role}")
    if role == "proposed":
        module.MECS_VersionA = MECS_VersionB
    return module.build_model(num_classes)


def load_checkpoint(path: Path, model: nn.Module, device: torch.device) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location=device)
    if isinstance(checkpoint, dict):
        state = checkpoint.get("model_state_dict", checkpoint.get("model_state", checkpoint))
    else:
        state = checkpoint
    model.load_state_dict(state)


@torch.no_grad()
def collect_predictions(model: nn.Module, loader, device: torch.device) -> dict[int, dict[str, object]]:
    model.eval()
    output: dict[int, dict[str, object]] = {}
    for images, labels, indices in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = extract_logits(model(images))
        probabilities = torch.softmax(logits, dim=1)
        confidence, prediction = probabilities.max(dim=1)
        true_probability = probabilities.gather(1, labels.unsqueeze(1)).squeeze(1)
        for index, pred, conf, true_prob in zip(
            indices.tolist(), prediction.tolist(), confidence.tolist(), true_probability.tolist()
        ):
            output[int(index)] = {
                "prediction": int(pred),
                "confidence": float(conf),
                "true_probability": float(true_prob),
            }
    return output


def class_is_selected(record: ImageRecord, filters: set[str]) -> bool:
    if not filters:
        return True
    return str(record.label_index) in filters or record.label_name.lower() in filters


def compose_panel(
    original: Image.Image,
    baseline_overlay: Image.Image,
    proposed_overlay: Image.Image,
    info_lines: Sequence[str],
    original_title: str,
    baseline_title: str,
    proposed_title: str,
) -> Image.Image:
    margin = 20
    header_height = 135
    width = original.width * 3 + margin * 4
    height = header_height + original.height + margin
    panel = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(panel)
    for row_index, line in enumerate(info_lines):
        draw.text((margin, 10 + row_index * 18), line, fill="black")
    positions = (margin, margin * 2 + original.width, margin * 3 + original.width * 2)
    titles = (original_title, baseline_title, proposed_title)
    images = (original.convert("RGB"), baseline_overlay.convert("RGB"), proposed_overlay.convert("RGB"))
    for x, title, image in zip(positions, titles, images):
        draw.text((x, header_height - 24), title, fill="black")
        panel.paste(image, (x, header_height))
    return panel


def write_rows(path: Path, rows: Iterable[dict[str, object]], fieldnames: Sequence[str] | None = None) -> None:
    rows = list(rows)
    if fieldnames is None:
        ordered = []
        for row in rows:
            for key in row:
                if key not in ordered:
                    ordered.append(key)
        fieldnames = ordered
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def export_dataset(
    dataset_name: str,
    baseline_run: SelectedRun,
    proposed_run: SelectedRun,
    baseline_checkpoint: Path,
    proposed_checkpoint: Path,
    output_dir: Path,
    args: argparse.Namespace,
) -> int:
    data_root = args.koa_data_root if dataset_name == "koa" else args.adni_data_root
    if data_root is None or not data_root.is_dir():
        raise FileNotFoundError(f"{dataset_name.upper()} data root not found: {data_root}")
    if dataset_name == "koa":
        records, class_names, split_metadata = koa_records(data_root)
    else:
        records, class_names, split_metadata = adni_records(
            data_root,
            baseline_seed=baseline_run.seed,
            proposed_seed=proposed_run.seed,
        )

    transform = eval_transform(dataset_name, args.image_size)
    dataset = RecordDataset(records, transform)
    loader = data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    device = resolve_device(args.device)
    build_model = build_koa_model if dataset_name == "koa" else build_adni_model
    baseline_model = build_model("baseline", len(class_names)).to(device)
    proposed_model = build_model("proposed", len(class_names)).to(device)
    load_checkpoint(baseline_checkpoint, baseline_model, device)
    load_checkpoint(proposed_checkpoint, proposed_model, device)
    baseline_model.eval()
    proposed_model.eval()

    baseline_predictions = collect_predictions(baseline_model, loader, device)
    proposed_predictions = collect_predictions(proposed_model, loader, device)
    candidates = []
    for index, record in enumerate(records):
        if not class_is_selected(record, args.class_filters):
            continue
        baseline = baseline_predictions[index]
        proposed = proposed_predictions[index]
        if baseline["prediction"] == record.label_index or proposed["prediction"] != record.label_index:
            continue
        score = float(proposed["true_probability"]) - float(baseline["true_probability"])
        candidates.append((score, index, record, baseline, proposed))
    candidates.sort(key=lambda item: (-item[0], item[2].relative_path))
    if args.max_samples > 0:
        candidates = candidates[: args.max_samples]

    output_dir.mkdir(parents=True, exist_ok=True)
    components_dir = output_dir / "components"
    if args.save_components:
        components_dir.mkdir(parents=True, exist_ok=True)
    baseline_cam_model = LogitsOnlyModel(baseline_model).to(device).eval()
    proposed_cam_model = LogitsOnlyModel(proposed_model).to(device).eval()
    baseline_target = resolve_target_module(baseline_model, args.target_layer)
    proposed_target = resolve_target_module(proposed_model, args.target_layer)

    print(f"\n[{dataset_name.upper()}] comparison pool: {len(records)}")
    print(f"[{dataset_name.upper()}] split policy: {split_metadata}")
    print(f"[{dataset_name.upper()}] qualifying proposed-correct/baseline-wrong: {len(candidates)}")
    print(f"[{dataset_name.upper()}] output: {output_dir}")

    summary_rows = []
    for rank, (score, index, record, baseline, proposed) in enumerate(candidates, start=1):
        with Image.open(record.image_path) as image:
            pil_image = image.convert("RGB")
        input_tensor = transform(pil_image).unsqueeze(0).to(device)
        baseline_pred = int(baseline["prediction"])
        proposed_pred = int(proposed["prediction"])
        baseline_cam_class = record.label_index if args.cam_on == "true" else baseline_pred
        proposed_cam_class = record.label_index if args.cam_on == "true" else proposed_pred
        def original_from_tensor(tensor):
            return tensor_to_pil(tensor, mean=MEAN, std=STD)

        original, baseline_heatmap, baseline_overlay, _ = build_cam_images(
            method_name=args.cam_method,
            model=baseline_cam_model,
            target_module=baseline_target,
            input_tensor=input_tensor,
            class_idx=baseline_cam_class,
            image_size=args.image_size,
            alpha=args.alpha,
            original_from_tensor=original_from_tensor,
            cam_threshold=args.cam_threshold,
            aug_smooth=args.aug_smooth,
            eigen_smooth=args.eigen_smooth,
        )
        _, proposed_heatmap, proposed_overlay, _ = build_cam_images(
            method_name=args.cam_method,
            model=proposed_cam_model,
            target_module=proposed_target,
            input_tensor=input_tensor,
            class_idx=proposed_cam_class,
            image_size=args.image_size,
            alpha=args.alpha,
            original_from_tensor=original_from_tensor,
            cam_threshold=args.cam_threshold,
            aug_smooth=args.aug_smooth,
            eigen_smooth=args.eigen_smooth,
        )

        true_name = class_names[record.label_index]
        baseline_name = class_names[baseline_pred]
        proposed_name = class_names[proposed_pred]
        info_lines = (
            f"dataset={dataset_name.upper()} | true={true_name} | image={record.relative_path}",
            f"baseline(seed={baseline_run.seed})={baseline_name} ({baseline['confidence']:.4f}) | WRONG",
            f"proposed(seed={proposed_run.seed})={proposed_name} ({proposed['confidence']:.4f}) | CORRECT",
            f"method={args.cam_method} | cam_on={args.cam_on} | layer={args.target_layer} | selection_score={score:.4f}",
        )
        panel = compose_panel(
            original,
            baseline_overlay,
            proposed_overlay,
            info_lines,
            original_title=f"Original (true: {true_name})",
            baseline_title=f"Baseline ({baseline_name})",
            proposed_title=f"Proposed ({proposed_name})",
        )
        stem = sanitize_filename(
            f"{rank:04d}_{dataset_name}_true-{true_name}_base-{baseline_name}_prop-{proposed_name}_{Path(record.relative_path).stem}"
        )
        panel_path = output_dir / f"{stem}_panel.png"
        panel.save(panel_path)
        component_paths = {
            "original_path": "",
            "baseline_heatmap_path": "",
            "baseline_overlay_path": "",
            "proposed_heatmap_path": "",
            "proposed_overlay_path": "",
        }
        if args.save_components:
            images_to_save = {
                "original_path": (original, "original"),
                "baseline_heatmap_path": (baseline_heatmap, "baseline_heatmap"),
                "baseline_overlay_path": (baseline_overlay, "baseline_overlay"),
                "proposed_heatmap_path": (proposed_heatmap, "proposed_heatmap"),
                "proposed_overlay_path": (proposed_overlay, "proposed_overlay"),
            }
            for key, (image, suffix) in images_to_save.items():
                path = components_dir / f"{stem}_{suffix}.png"
                image.save(path)
                component_paths[key] = str(path)

        summary_rows.append(
            {
                "rank": rank,
                "dataset": dataset_name.upper(),
                "image_path": str(record.image_path),
                "relative_path": record.relative_path,
                "true_index": record.label_index,
                "true_label": true_name,
                "baseline_seed": baseline_run.seed,
                "baseline_prediction_index": baseline_pred,
                "baseline_prediction": baseline_name,
                "baseline_confidence": baseline["confidence"],
                "baseline_true_probability": baseline["true_probability"],
                "proposed_seed": proposed_run.seed,
                "proposed_prediction_index": proposed_pred,
                "proposed_prediction": proposed_name,
                "proposed_confidence": proposed["confidence"],
                "proposed_true_probability": proposed["true_probability"],
                "selection_score": score,
                "cam_method": args.cam_method,
                "cam_on": args.cam_on,
                "target_layer": args.target_layer,
                "baseline_cam_target": class_names[baseline_cam_class],
                "proposed_cam_target": class_names[proposed_cam_class],
                "panel_path": str(panel_path),
                **component_paths,
            }
        )
        print(
            f"[{dataset_name.upper()} {rank}/{len(candidates)}] {record.relative_path} | "
            f"true={true_name} | baseline={baseline_name} | proposed={proposed_name}"
        )

    write_rows(
        output_dir / "summary.csv",
        summary_rows,
        fieldnames=(
            "rank", "dataset", "image_path", "relative_path", "true_index", "true_label",
            "baseline_seed", "baseline_prediction_index", "baseline_prediction", "baseline_confidence",
            "baseline_true_probability", "proposed_seed", "proposed_prediction_index", "proposed_prediction",
            "proposed_confidence", "proposed_true_probability", "selection_score", "cam_method", "cam_on",
            "target_layer", "baseline_cam_target", "proposed_cam_target", "panel_path", "original_path",
            "baseline_heatmap_path", "baseline_overlay_path", "proposed_heatmap_path", "proposed_overlay_path",
        ),
    )
    write_rows(
        output_dir / "split_metadata.csv",
        [{"dataset": dataset_name.upper(), **split_metadata}],
    )
    return len(summary_rows)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    records_path = resolve_records_file(args.records, args.dataset_names)
    rows = read_csv(records_path)
    selections: dict[str, dict[str, SelectedRun]] = {}
    checkpoints: dict[str, dict[str, Path]] = {}
    for dataset_name in args.dataset_names:
        selections[dataset_name] = {
            "baseline": select_run(rows, dataset_name.upper(), "baseline", args.metric),
            "proposed": select_run(rows, dataset_name.upper(), "proposed", args.metric),
        }
        checkpoints[dataset_name] = {}
        for role in ("baseline", "proposed"):
            explicit = getattr(args, f"{dataset_name}_{role}_checkpoint")
            checkpoints[dataset_name][role] = resolve_checkpoint(selections[dataset_name][role], explicit)

    print(f"Records: {records_path}")
    for dataset_name in args.dataset_names:
        for role in ("baseline", "proposed"):
            run = selections[dataset_name][role]
            checkpoint = checkpoints[dataset_name][role]
            direction = "worst" if role == "baseline" else "best"
            print(
                f"{dataset_name.upper()} {direction} {role}: seed={run.seed}, "
                f"{run.metric}={run.metric_value:.4f}, checkpoint={checkpoint} "
                f"(exists={checkpoint.is_file()})"
            )
    if args.dry_run:
        return 0

    ensure_pytorch_grad_cam()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = args.output_dir or PROJECT_ROOT / "visual_outputs" / f"best_worst_gradcam_{timestamp}"
    output_root = output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    selection_rows = []
    total_exported = 0
    for dataset_name in args.dataset_names:
        for role in ("baseline", "proposed"):
            run = selections[dataset_name][role]
            selection_rows.append(
                {
                    "dataset": dataset_name.upper(),
                    "role": role,
                    "method": run.method,
                    "seed": run.seed,
                    "metric": run.metric,
                    "metric_value": run.metric_value,
                    "run_tag": run.run_tag,
                    "checkpoint": str(checkpoints[dataset_name][role]),
                    "records": str(records_path),
                }
            )
        dataset_output = output_root / dataset_name
        total_exported += export_dataset(
            dataset_name=dataset_name,
            baseline_run=selections[dataset_name]["baseline"],
            proposed_run=selections[dataset_name]["proposed"],
            baseline_checkpoint=checkpoints[dataset_name]["baseline"],
            proposed_checkpoint=checkpoints[dataset_name]["proposed"],
            output_dir=dataset_output,
            args=args,
        )
    write_rows(output_root / "selected_checkpoints.csv", selection_rows)
    print(f"\nExported panels: {total_exported}")
    print(f"Output root: {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
