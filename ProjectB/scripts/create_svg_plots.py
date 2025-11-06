#!/usr/bin/env python3
"""Generate simple SVG plots (loss curves and confusion matrices) without external deps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

SVG_HEADER = "http://www.w3.org/2000/svg"


def _load(path: Path) -> Dict:
    return json.loads(path.read_text())


def _scale(value: float, min_val: float, max_val: float, size: float) -> float:
    if max_val == min_val:
        return size / 2
    return (value - min_val) / (max_val - min_val) * size


def _create_line_svg(
    epochs: Iterable[float],
    series: List[Tuple[str, List[float], str]],
    title: str,
    y_label: str,
    output: Path,
) -> None:
    width, height = 640, 400
    margin = 60
    content_width = width - 2 * margin
    content_height = height - 2 * margin

    all_values = [v for _, values, _ in series for v in values if v is not None]
    if not all_values:
        return
    min_y = min(all_values)
    max_y = max(all_values)
    if max_y == min_y:
        max_y += 1.0
    x_values = list(epochs)
    min_x, max_x = min(x_values), max(x_values)

    def point(epoch: float, value: float) -> Tuple[float, float]:
        x = margin + _scale(epoch, min_x, max_x, content_width)
        y = height - margin - _scale(value, min_y, max_y, content_height)
        return x, y

    lines: List[str] = []
    # Background and axes
    lines.append(f'<svg xmlns="{SVG_HEADER}" width="{width}" height="{height}" font-family="Arial">')
    lines.append(f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff" stroke="#e0e0e0"/>')
    lines.append(f'<line x1="{margin}" y1="{height-margin}" x2="{width-margin}" y2="{height-margin}" stroke="#000" stroke-width="1"/>')
    lines.append(f'<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height-margin}" stroke="#000" stroke-width="1"/>')

    # Title and labels
    lines.append(f'<text x="{width/2}" y="{margin/2}" text-anchor="middle" font-size="18">{title}</text>')
    lines.append(f'<text x="{margin/3}" y="{height/2}" text-anchor="middle" font-size="14" transform="rotate(-90 {margin/3},{height/2})">{y_label}</text>')
    lines.append(f'<text x="{width/2}" y="{height-10}" text-anchor="middle" font-size="14">Epoch</text>')

    # Axis ticks
    for epoch in x_values:
        x, _ = point(epoch, min_y)
        lines.append(f'<line x1="{x}" y1="{height-margin}" x2="{x}" y2="{height-margin+5}" stroke="#000"/>')
        lines.append(f'<text x="{x}" y="{height-margin+20}" text-anchor="middle" font-size="12">{epoch}</text>')

    for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
        value = min_y + frac * (max_y - min_y)
        y = height - margin - _scale(value, min_y, max_y, content_height)
        lines.append(f'<line x1="{margin-5}" y1="{y}" x2="{margin}" y2="{y}" stroke="#000"/>')
        lines.append(f'<text x="{margin-10}" y="{y+4}" text-anchor="end" font-size="12">{value:.2f}</text>')

    # Plot series
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    legend_items = []
    for idx, (label, values, color) in enumerate(series):
        color = color or palette[idx % len(palette)]
        points = [point(epoch, value) for epoch, value in zip(x_values, values) if value is not None]
        if len(points) < 2:
            continue
        path_data = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
        lines.append(f'<polyline points="{path_data}" fill="none" stroke="{color}" stroke-width="2"/>')
        legend_items.append((label, color))

    # Legend
    legend_x = width - margin - 150
    legend_y = margin
    for idx, (label, color) in enumerate(legend_items):
        y = legend_y + idx * 20
        lines.append(f'<rect x="{legend_x}" y="{y}" width="12" height="12" fill="{color}"/>')
        lines.append(f'<text x="{legend_x + 18}" y="{y + 10}" font-size="12">{label}</text>')

    lines.append("</svg>")
    output.write_text("\n".join(lines))


def _create_confusion_svg(cm: Dict[str, int], title: str, output: Path) -> None:
    width, height = 360, 360
    margin = 60
    cell_size = (width - 2 * margin) / 2
    labels = [["TN", "FP"], ["FN", "TP"]]
    values = [[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]]

    lines = [f'<svg xmlns="{SVG_HEADER}" width="{width}" height="{height}" font-family="Arial">']
    lines.append(f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff" stroke="#e0e0e0"/>')
    lines.append(f'<text x="{width/2}" y="{margin/2}" text-anchor="middle" font-size="18">{title}</text>')
    for i in range(2):
        for j in range(2):
            x = margin + j * cell_size
            y = margin + i * cell_size
            value = values[i][j]
            intensity = min(1.0, value / max(max(row) for row in values))
            fill = f"rgb({int(255 - 155 * intensity)}, {int(255 - 155 * intensity)}, 255)"
            lines.append(f'<rect x="{x}" y="{y}" width="{cell_size}" height="{cell_size}" fill="{fill}" stroke="#000"/>')
            lines.append(f'<text x="{x + cell_size/2}" y="{y + cell_size/2}" text-anchor="middle" font-size="20" dominant-baseline="middle">{value}</text>')
            lines.append(f'<text x="{x + 10}" y="{y + 20}" font-size="12">{labels[i][j]}</text>')

    lines.append(f'<text x="{margin/2}" y="{height/2}" text-anchor="middle" font-size="14" transform="rotate(-90 {margin/2},{height/2})">Actual</text>')
    lines.append(f'<text x="{width/2}" y="{height - margin/4}" text-anchor="middle" font-size="14">Predicted</text>')
    lines.append("</svg>")
    output.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trainer", type=Path, required=True)
    parser.add_argument("--manual", type=Path, required=True)
    parser.add_argument("--classical", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/plots"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    trainer = _load(args.trainer)
    manual = _load(args.manual)
    classical = _load(args.classical)

    trainer_epochs = [entry["epoch"] for entry in trainer.get("history", {}).get("epochs", [])]
    manual_epochs = [entry["epoch"] for entry in manual.get("history", {}).get("epochs", [])]

    _create_line_svg(
        trainer_epochs,
        [
            ("Train Loss", [entry.get("train_loss") for entry in trainer["history"]["epochs"]], "#1f77b4"),
            ("Validation Loss", [entry.get("eval_loss") for entry in trainer["history"]["epochs"]], "#ff7f0e"),
        ],
        "Trainer Fine-Tuning Loss",
        "Loss",
        args.output_dir / "trainer_loss.svg",
    )
    _create_line_svg(
        trainer_epochs,
        [("Validation Accuracy", [entry.get("eval_accuracy") for entry in trainer["history"]["epochs"]], "#2ca02c")],
        "Trainer Validation Accuracy",
        "Accuracy",
        args.output_dir / "trainer_accuracy.svg",
    )

    _create_line_svg(
        manual_epochs,
        [
            ("Train Loss", [entry.get("train_loss") for entry in manual["history"]["epochs"]], "#1f77b4"),
            ("Validation Loss", [entry.get("eval_loss") for entry in manual["history"]["epochs"]], "#ff7f0e"),
        ],
        "Manual Fine-Tuning Loss",
        "Loss",
        args.output_dir / "manual_loss.svg",
    )
    _create_line_svg(
        manual_epochs,
        [
            ("Train Accuracy", [entry.get("train_accuracy") for entry in manual["history"]["epochs"]], "#9467bd"),
            ("Validation Accuracy", [entry.get("eval_accuracy") for entry in manual["history"]["epochs"]], "#2ca02c"),
        ],
        "Manual Accuracy",
        "Accuracy",
        args.output_dir / "manual_accuracy.svg",
    )

    trainer_cm = trainer.get("test", {}).get("report", {}).get("confusion_matrix")
    if trainer_cm:
        _create_confusion_svg(trainer_cm, "Trainer Confusion Matrix", args.output_dir / "trainer_confusion.svg")

    manual_cm = manual.get("test", {}).get("report", {}).get("confusion_matrix")
    if manual_cm:
        _create_confusion_svg(manual_cm, "Manual Confusion Matrix", args.output_dir / "manual_confusion.svg")

    classical_cm = classical.get("logistic_regression", {}).get("test", {}).get("report", {}).get("confusion_matrix")
    if classical_cm:
        _create_confusion_svg(classical_cm, "Logistic Regression Confusion Matrix", args.output_dir / "classical_confusion.svg")


if __name__ == "__main__":  # pragma: no cover
    main()
