"""Custom PyTorch fine-tuning loop for DistilBERT."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
)

from torch.optim import AdamW

from .data import (
    PreparedSplits,
    label_to_name,
    load_raw_imdb,
    load_testcases,
    prepare_dataset_splits,
    tokenize_dataset,
)
from .metrics import ClassificationReport, compute_classification_report

__all__ = ["ManualTrainingConfig", "run_manual_training"]


@dataclass
class ManualTrainingConfig:
    model_name: str = "distilbert-base-uncased"
    max_length: int = 256
    limit_train: Optional[int] = 2000
    limit_test: Optional[int] = 1000
    batch_size: int = 8
    num_epochs: int = 2
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    seed: int = 42
    device: Optional[str] = None  # autodetect if None


def _prepare_dataloaders(
    config: ManualTrainingConfig,
) -> tuple[PreparedSplits, DataLoader, DataLoader, DataLoader, PreTrainedTokenizerBase]:
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

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest")

    def _as_loader(ds, *, shuffle: bool = False):
        ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
        return DataLoader(ds, batch_size=config.batch_size, shuffle=shuffle, collate_fn=data_collator)

    train_loader = _as_loader(tokenised.train, shuffle=True)
    val_loader = _as_loader(tokenised.validation)
    test_loader = _as_loader(tokenised.test)

    return tokenised, train_loader, val_loader, test_loader, tokenizer


def _evaluate(model, data_loader, device) -> tuple[ClassificationReport, float]:
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0.0
    total_examples = 0
    with torch.inference_mode():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            total_loss += outputs.loss.item() * labels.size(0)
            total_examples += labels.size(0)
    avg_loss = total_loss / total_examples if total_examples else 0.0
    return compute_classification_report(all_labels, all_preds), float(avg_loss)


def _score_testcases(model, tokenizer, device, max_length: int) -> list[Dict[str, object]]:
    testcases = load_testcases()
    model.eval()
    results: list[Dict[str, object]] = []
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


def run_manual_training(config: ManualTrainingConfig) -> Dict[str, Dict[str, object]]:
    tokenised, train_loader, val_loader, test_loader, tokenizer = _prepare_dataloaders(config)

    device = torch.device(config.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name, num_labels=2
    )
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    num_training_steps = len(train_loader) * config.num_epochs
    num_warmup_steps = int(num_training_steps * config.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
    )

    history = []

    for epoch in range(config.num_epochs):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_examples = 0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            running_loss += loss.item() * labels.size(0)
            preds = torch.argmax(outputs.logits, dim=-1)
            running_correct += (preds == labels).sum().item()
            running_examples += labels.size(0)

        train_loss = running_loss / running_examples if running_examples else 0.0
        train_accuracy = running_correct / running_examples if running_examples else 0.0

        val_report, val_loss = _evaluate(model, val_loader, device)
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": float(train_loss),
                "train_accuracy": float(train_accuracy),
                "eval_loss": float(val_loss),
                "eval_accuracy": val_report.accuracy,
                "eval_f1": val_report.f1,
            }
        )

    val_report, val_loss = _evaluate(model, val_loader, device)
    test_report, test_loss = _evaluate(model, test_loader, device)

    return {
        "history": {"epochs": history},
        "validation": {"loss": float(val_loss), "report": val_report.as_dict()},
        "test": {"loss": float(test_loss), "report": test_report.as_dict()},
        "testcases": _score_testcases(model, tokenizer, device, config.max_length),
    }
