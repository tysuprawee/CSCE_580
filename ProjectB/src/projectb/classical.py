"""Classical ML baselines for Project B."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from .data import load_raw_imdb, prepare_dataset_splits, load_testcases, label_to_name
from .metrics import compute_classification_report

__all__ = ["ClassicalConfig", "train_classical_models"]


@dataclass
class ClassicalConfig:
    max_features: int = 20000
    limit_train: Optional[int] = 5000
    limit_test: Optional[int] = 2000
    seed: int = 42


def _build_pipeline(estimator):
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=None,
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.95,
                    sublinear_tf=True,
                    lowercase=True,
                ),
            ),
            ("clf", estimator),
        ]
    )


def _score_testcases(model, testcases: Sequence) -> list[Dict[str, object]]:
    texts = [case.text for case in testcases]
    preds = model.predict(texts)
    proba = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(texts)[:, 1]

    results = []
    for idx, case in enumerate(testcases):
        probability = float(proba[idx]) if proba is not None else None
        pred_label = int(preds[idx])
        results.append(
            {
                "id": case.id,
                "text": case.text,
                "true_label": case.label,
                "true_label_name": label_to_name(case.label),
                "predicted_label": pred_label,
                "predicted_label_name": label_to_name(pred_label),
                "positive_probability": probability,
                "correct": bool(pred_label == case.label),
            }
        )
    return results


def train_classical_models(config: ClassicalConfig) -> Dict[str, Dict[str, object]]:
    raw = load_raw_imdb()
    prepared = prepare_dataset_splits(
        raw,
        seed=config.seed,
        limit_train=config.limit_train,
        limit_test=config.limit_test,
    )

    X_train = prepared.train["text"]
    y_train = np.array(prepared.train["label"])
    X_val = prepared.validation["text"]
    y_val = np.array(prepared.validation["label"])
    X_test = prepared.test["text"]
    y_test = np.array(prepared.test["label"])

    models = {
        "logistic_regression": _build_pipeline(
            LogisticRegression(max_iter=1000, class_weight="balanced", random_state=config.seed)
        ),
        "multinomial_nb": _build_pipeline(MultinomialNB(alpha=0.5)),
    }

    testcases = load_testcases()

    results: Dict[str, Dict[str, object]] = {}
    for name, pipeline in models.items():
        pipeline.fit(X_train, y_train)
        val_preds = pipeline.predict(X_val)
        test_preds = pipeline.predict(X_test)
        val_report = compute_classification_report(y_val, val_preds)
        test_report = compute_classification_report(y_test, test_preds)
        results[name] = {
            "validation": {"report": val_report.as_dict()},
            "test": {"report": test_report.as_dict()},
            "testcases": _score_testcases(pipeline, testcases),
        }
    return results
