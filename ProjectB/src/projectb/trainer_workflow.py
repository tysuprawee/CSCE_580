"""High-level wrapper around the 🤗 Trainer API for Project B."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)

from .data import (
    PreparedSplits,
    label_to_name,
    load_raw_imdb,
    prepare_dataset_splits,
    tokenize_dataset,
    tokenize_testcases,
)
from .metrics import compute_classification_report

__all__ = ["TrainerConfig", "run_trainer_pipeline"]


@dataclass
class TrainerConfig:
    model_name: str = "distilbert-base-uncased"
    output_dir: str = "artifacts/trainer"
    learning_rate: float = 2e-5
    batch_size: int = 16
    num_epochs: int = 2
    weight_decay: float = 0.01
    max_length: int = 256
    limit_train: Optional[int] = 2000
    limit_test: Optional[int] = 1000
    seed: int = 42
    evaluation_strategy: str = "epoch"


def _build_trainer_datasets(
    config: TrainerConfig,
) -> tuple[PreparedSplits, PreparedSplits, PreTrainedTokenizerBase]:
    raw = load_raw_imdb()
    prepared = prepare_dataset_splits(
        raw,
        seed=config.seed,
        limit_train=config.limit_train,
        limit_test=config.limit_test,
    )

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenised = tokenize_dataset(
        prepared,
        tokenizer,
        max_length=config.max_length,
        padding="max_length",
    )

    return prepared, tokenised, tokenizer


def _summarise_history(log_history) -> dict:
    epochs: dict[int, dict] = {}
    for record in log_history:
        epoch_float = record.get("epoch")
        if epoch_float is None:
            continue
        epoch = int(round(epoch_float))
        epoch_entry = epochs.setdefault(epoch, {"epoch": epoch})
        if "loss" in record:
            epoch_entry.setdefault("train_loss_values", []).append(record["loss"])
        if "eval_loss" in record:
            epoch_entry["eval_loss"] = record["eval_loss"]
        if "eval_accuracy" in record:
            epoch_entry["eval_accuracy"] = record["eval_accuracy"]
        if "eval_f1" in record:
            epoch_entry["eval_f1"] = record["eval_f1"]

    history = []
    for epoch in sorted(epochs):
        entry = epochs[epoch]
        loss_values = entry.pop("train_loss_values", None)
        if loss_values:
            entry["train_loss"] = float(np.mean(loss_values))
        history.append(entry)
    return {"epochs": history}


def _score_testcases(trainer: Trainer, tokenizer, max_length: int) -> list[dict]:
    testcases, tokenised = tokenize_testcases(tokenizer, max_length=max_length)
    predictions = trainer.predict(tokenised)
    logits = predictions.predictions
    results = []
    for case, logit in zip(testcases, logits):
        probs = np.exp(logit - np.max(logit))
        probs = probs / probs.sum()
        pred_label = int(np.argmax(probs))
        results.append(
            {
                "id": case.id,
                "text": case.text,
                "true_label": case.label,
                "true_label_name": label_to_name(case.label),
                "predicted_label": pred_label,
                "predicted_label_name": label_to_name(pred_label),
                "positive_probability": float(probs[1]),
                "correct": bool(pred_label == case.label),
            }
        )
    return results


def run_trainer_pipeline(config: TrainerConfig) -> dict:
    """Train DistilBERT using the 🤗 Trainer API and return evaluation metrics."""

    prepared, tokenised, tokenizer = _build_trainer_datasets(config)

    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name, num_labels=2
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        report = compute_classification_report(labels, predictions)
        metrics = report.as_dict()
        return metrics

    training_args = TrainingArguments(
        output_dir=config.output_dir,
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        num_train_epochs=config.num_epochs,
        weight_decay=config.weight_decay,
        evaluation_strategy=config.evaluation_strategy,
        logging_strategy="epoch",
        save_strategy="no",
        seed=config.seed,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenised.train,
        eval_dataset=tokenised.validation,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    val_predictions = trainer.predict(tokenised.validation)
    val_preds = np.argmax(val_predictions.predictions, axis=-1)
    val_report = compute_classification_report(val_predictions.label_ids, val_preds)
    val_loss = float(val_predictions.metrics.get("test_loss", 0.0))

    test_predictions = trainer.predict(tokenised.test)
    test_preds = np.argmax(test_predictions.predictions, axis=-1)
    test_report = compute_classification_report(test_predictions.label_ids, test_preds)
    test_loss = float(test_predictions.metrics.get("test_loss", 0.0))

    return {
        "history": _summarise_history(trainer.state.log_history),
        "validation": {"loss": val_loss, "report": val_report.as_dict()},
        "test": {"loss": test_loss, "report": test_report.as_dict()},
        "testcases": _score_testcases(trainer, tokenizer, config.max_length),
        "training_summary": {
            "train_runtime": trainer.state.train_runtime,
            "train_samples": trainer.state.global_step * config.batch_size,
        },
    }
