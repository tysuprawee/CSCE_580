"""Utilities for CSCE 580 Project B (IMDB sentiment fine-tuning)."""

from importlib import import_module
from typing import TYPE_CHECKING

from .cleaning import clean_review, clean_reviews

if TYPE_CHECKING:  # pragma: no cover - type checkers only
    from .data import (
        PreparedSplits,
        label_to_name,
        load_raw_imdb,
        load_testcases,
        prepare_dataset_splits,
        tokenize_dataset,
        tokenize_testcases,
    )
    from .metrics import ClassificationReport, compute_classification_report
    from .baselines import BaselineConfig, evaluate_base_distilbert, evaluate_gpt_classifier

__all__ = [
    "clean_review",
    "clean_reviews",
    "PreparedSplits",
    "label_to_name",
    "load_raw_imdb",
    "load_testcases",
    "prepare_dataset_splits",
    "tokenize_dataset",
    "tokenize_testcases",
    "ClassificationReport",
    "compute_classification_report",
    "BaselineConfig",
    "evaluate_base_distilbert",
    "evaluate_gpt_classifier",
]

_LAZY_EXPORTS = {
    "PreparedSplits": ("data", "PreparedSplits"),
    "label_to_name": ("data", "label_to_name"),
    "load_raw_imdb": ("data", "load_raw_imdb"),
    "load_testcases": ("data", "load_testcases"),
    "prepare_dataset_splits": ("data", "prepare_dataset_splits"),
    "tokenize_dataset": ("data", "tokenize_dataset"),
    "tokenize_testcases": ("data", "tokenize_testcases"),
    "ClassificationReport": ("metrics", "ClassificationReport"),
    "compute_classification_report": ("metrics", "compute_classification_report"),
    "BaselineConfig": ("baselines", "BaselineConfig"),
    "evaluate_base_distilbert": ("baselines", "evaluate_base_distilbert"),
    "evaluate_gpt_classifier": ("baselines", "evaluate_gpt_classifier"),
}


def __getattr__(name):
    if name in _LAZY_EXPORTS:
        module_name, attr_name = _LAZY_EXPORTS[name]
        module = import_module(f".{module_name}", __name__)
        value = getattr(module, attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
