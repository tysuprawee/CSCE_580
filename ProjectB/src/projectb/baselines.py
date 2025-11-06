"""Baseline evaluations for non-fine-tuned transformer models."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Dict, Iterable, Optional, Sequence

import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    GPT2LMHeadModel,
    GPT2TokenizerFast,
)

from .data import (
    PreparedSplits,
    label_to_name,
    load_raw_imdb,
    load_testcases,
    prepare_dataset_splits,
    tokenize_dataset,
)
from .metrics import compute_classification_report

__all__ = [
    "BaselineConfig",
    "evaluate_base_distilbert",
    "evaluate_gpt_classifier",
]


@dataclass
class BaselineConfig:
    """Shared configuration for baseline evaluations."""

    distilbert_model_name: str = "distilbert-base-uncased"
    gpt_model_name: str = "openai-community/gpt2"
    max_length: int = 256
    limit_train: Optional[int] = 2000
    limit_test: Optional[int] = 1000
    batch_size: int = 16
    seed: int = 42
    device: Optional[str] = None


def _build_base_loaders(
    *, config: BaselineConfig
) -> tuple[PreparedSplits, DataLoader, DataLoader, DataLoader, AutoTokenizer]:
    """Prepare tokenised splits and dataloaders for evaluation."""

    raw = load_raw_imdb()
    prepared = prepare_dataset_splits(
        raw,
        seed=config.seed,
        limit_train=config.limit_train,
        limit_test=config.limit_test,
    )

    tokenizer = AutoTokenizer.from_pretrained(config.distilbert_model_name)
    tokenised = tokenize_dataset(
        prepared,
        tokenizer,
        max_length=config.max_length,
        padding="max_length",
    )

    collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest")

    def _loader(ds, *, shuffle: bool = False) -> DataLoader:
        ds = ds.remove_columns([c for c in ds.column_names if c not in {"input_ids", "attention_mask", "label"}])
        ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
        return DataLoader(ds, batch_size=config.batch_size, shuffle=shuffle, collate_fn=collator)

    train_loader = _loader(tokenised.train)
    val_loader = _loader(tokenised.validation)
    test_loader = _loader(tokenised.test)

    return tokenised, train_loader, val_loader, test_loader, tokenizer


def _evaluate_model(model, loader: DataLoader, device: torch.device) -> tuple[Dict[str, float], float]:
    """Return metrics dict and average loss for a classification model."""

    model.eval()
    preds: list[int] = []
    labels: list[int] = []
    total_loss = 0.0
    total_examples = 0

    with torch.inference_mode():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            batch_labels = batch["labels"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=batch_labels)
            logits = outputs.logits
            batch_preds = torch.argmax(logits, dim=-1)
            preds.extend(batch_preds.cpu().tolist())
            labels.extend(batch_labels.cpu().tolist())
            total_loss += outputs.loss.item() * batch_labels.size(0)
            total_examples += batch_labels.size(0)

    avg_loss = total_loss / total_examples if total_examples else 0.0
    report = compute_classification_report(labels, preds).as_dict()
    return report, float(avg_loss)


def _score_testcases_sequence_classifier(model, tokenizer, device, max_length: int) -> list[Dict[str, object]]:
    """Evaluate a sequence classifier on curated testcases."""

    testcases = load_testcases()
    results: list[Dict[str, object]] = []

    model.eval()
    with torch.inference_mode():
        for case in testcases:
            encoded = tokenizer(
                case.text,
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=max_length,
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}
            outputs = model(**encoded)
            probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
            pred_label = int(probs.argmax())
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


def evaluate_base_distilbert(config: BaselineConfig) -> Dict[str, object]:
    """Evaluate the off-the-shelf DistilBERT classifier without task-specific fine-tuning."""

    tokenised, train_loader, val_loader, test_loader, tokenizer = _build_base_loaders(config=config)

    device = torch.device(config.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = AutoModelForSequenceClassification.from_pretrained(
        config.distilbert_model_name, num_labels=2
    )
    model.to(device)

    val_report, val_loss = _evaluate_model(model, val_loader, device)
    test_report, test_loss = _evaluate_model(model, test_loader, device)

    start = perf_counter()
    _ = _evaluate_model(model, train_loader, device)
    elapsed = perf_counter() - start
    avg_latency = elapsed / len(tokenised.train) if len(tokenised.train) else 0.0

    return {
        "validation": {"loss": val_loss, "report": val_report},
        "test": {"loss": test_loss, "report": test_report},
        "testcases": _score_testcases_sequence_classifier(model, tokenizer, device, config.max_length),
        "inference": {
            "avg_latency_seconds": avg_latency,
            "device": str(device),
        },
    }


def _label_log_prob(
    model: GPT2LMHeadModel,
    tokenizer: GPT2TokenizerFast,
    prompt_ids: Sequence[int],
    label_ids: Sequence[int],
    device: torch.device,
) -> float:
    """Return log-probability of `label_ids` conditioned on `prompt_ids`."""

    combined = torch.tensor([prompt_ids + label_ids], device=device)
    attention_mask = torch.ones_like(combined, device=device)

    with torch.inference_mode():
        outputs = model(input_ids=combined, attention_mask=attention_mask)
        logits = outputs.logits
        log_probs = torch.nn.functional.log_softmax(logits[:, :-1, :], dim=-1)

    prompt_length = len(prompt_ids)
    total_log_prob = 0.0
    for offset, token_id in enumerate(label_ids):
        position = prompt_length + offset - 1
        total_log_prob += log_probs[0, position, token_id].item()
    return total_log_prob


def _score_with_gpt(
    model: GPT2LMHeadModel,
    tokenizer: GPT2TokenizerFast,
    text: str,
    *,
    max_length: int,
    device: torch.device,
) -> tuple[int, float]:
    """Return (predicted_label, positive_probability) using label log-probabilities."""

    prompt_ids = tokenizer.encode(text, add_special_tokens=False)
    max_context = model.config.n_positions
    max_prompt = min(max_length, max_context)
    label_variants = {0: " negative", 1: " positive"}

    scores = {}
    for label, suffix in label_variants.items():
        label_ids = tokenizer.encode(suffix, add_special_tokens=False)
        prompt_window = max(1, max_prompt - len(label_ids))
        usable_prompt = prompt_ids[-min(len(prompt_ids), prompt_window) :]
        log_prob = _label_log_prob(model, tokenizer, usable_prompt, label_ids, device)
        scores[label] = log_prob

    max_log = max(scores.values())
    exp_scores = {label: torch.exp(torch.tensor(score - max_log)).item() for label, score in scores.items()}
    denom = sum(exp_scores.values())
    probs = {label: value / denom for label, value in exp_scores.items()}
    predicted = max(probs, key=probs.get)
    return int(predicted), float(probs[1])


def evaluate_gpt_classifier(config: BaselineConfig) -> Dict[str, object]:
    """Approximate sentiment classification using GPT-2 via label likelihoods."""

    raw = load_raw_imdb()
    prepared = prepare_dataset_splits(
        raw,
        seed=config.seed,
        limit_train=config.limit_train,
        limit_test=config.limit_test,
    )

    tokenizer = GPT2TokenizerFast.from_pretrained(config.gpt_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = torch.device(config.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = GPT2LMHeadModel.from_pretrained(config.gpt_model_name)
    model.to(device)
    model.eval()

    def _predict(texts: Iterable[str]) -> tuple[list[int], list[float], float]:
        predictions: list[int] = []
        probabilities: list[float] = []
        start = perf_counter()
        for text in texts:
            pred, prob = _score_with_gpt(model, tokenizer, text, max_length=config.max_length, device=device)
            predictions.append(pred)
            probabilities.append(prob)
        elapsed = perf_counter() - start
        avg_latency = elapsed / len(predictions) if predictions else 0.0
        return predictions, probabilities, avg_latency

    val_texts = list(prepared.validation["text"])
    test_texts = list(prepared.test["text"])

    val_preds, val_probs, val_latency = _predict(val_texts)
    test_preds, test_probs, test_latency = _predict(test_texts)

    val_report = compute_classification_report(prepared.validation["label"], val_preds).as_dict()
    test_report = compute_classification_report(prepared.test["label"], test_preds).as_dict()

    testcase_results = []
    for case in load_testcases():
        pred, prob = _score_with_gpt(model, tokenizer, case.text, max_length=config.max_length, device=device)
        testcase_results.append(
            {
                "id": case.id,
                "text": case.text,
                "true_label": case.label,
                "true_label_name": label_to_name(case.label),
                "predicted_label": pred,
                "predicted_label_name": label_to_name(pred),
                "positive_probability": prob,
                "correct": bool(pred == case.label),
            }
        )

    return {
        "validation": {"report": val_report},
        "test": {"report": test_report},
        "testcases": testcase_results,
        "inference": {
            "avg_latency_seconds_validation": val_latency,
            "avg_latency_seconds_test": test_latency,
            "device": str(device),
        },
    }
