"""Evaluation helpers shared across Project B experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

__all__ = [
    "ClassificationReport",
    "compute_classification_report",
    "confusion_matrix_as_dict",
]


@dataclass
class ClassificationReport:
    accuracy: float
    precision: float
    recall: float
    f1: float
    confusion_matrix: Optional[Dict[str, int]] = None

    def as_dict(self) -> Dict[str, float]:
        data = {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
        }
        if self.confusion_matrix is not None:
            data["confusion_matrix"] = self.confusion_matrix
        return data


def confusion_matrix_as_dict(y_true: Iterable[int], y_pred: Iterable[int]) -> Dict[str, int]:
    cm = confusion_matrix(list(y_true), list(y_pred))
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
        return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
    raise ValueError("Expected binary confusion matrix with shape (2, 2)")


def compute_classification_report(
    y_true: Iterable[int], y_pred: Iterable[int], *, include_confusion: bool = True
) -> ClassificationReport:
    y_true_arr = np.asarray(list(y_true))
    y_pred_arr = np.asarray(list(y_pred))

    accuracy = accuracy_score(y_true_arr, y_pred_arr)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true_arr, y_pred_arr, average="binary"
    )

    confusion = None
    if include_confusion:
        confusion = confusion_matrix_as_dict(y_true_arr, y_pred_arr)

    return ClassificationReport(
        accuracy=float(accuracy),
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        confusion_matrix=confusion,
    )
