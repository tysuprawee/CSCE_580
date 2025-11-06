"""Dataset loading and preprocessing utilities for Project B."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

from datasets import Dataset, DatasetDict, load_dataset
from transformers import PreTrainedTokenizerBase

from .cleaning import clean_reviews

__all__ = [
    "PreparedSplits",
    "load_raw_imdb",
    "prepare_dataset_splits",
    "tokenize_dataset",
    "TestCase",
    "load_testcases",
    "tokenize_testcases",
    "label_to_name",
]


@dataclass
class PreparedSplits:
    """Container for cleaned and (optionally) tokenised dataset splits."""

    train: Dataset
    validation: Dataset
    test: Dataset

    def select_columns(self, columns: Iterable[str]) -> "PreparedSplits":
        """Return a copy containing only the specified columns."""

        return PreparedSplits(
            train=self.train.select_columns(columns),
            validation=self.validation.select_columns(columns),
            test=self.test.select_columns(columns),
        )


@dataclass
class TestCase:
    """Represents a curated review used for qualitative evaluation."""

    id: str
    text: str
    label: int


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_TESTCASE_PATH = _PROJECT_ROOT / "data" / "testcases.json"
_LABEL_MAP = {0: "negative", 1: "positive"}


def label_to_name(label: int) -> str:
    return _LABEL_MAP.get(label, str(label))


def load_testcases(path: Optional[Path] = None) -> Sequence[TestCase]:
    """Load the curated testcases from disk."""

    testcase_path = Path(path) if path is not None else _TESTCASE_PATH
    data = json.loads(testcase_path.read_text())
    return [TestCase(**item) for item in data]


def tokenize_testcases(
    tokenizer: PreTrainedTokenizerBase,
    *,
    max_length: int = 256,
) -> Tuple[Sequence[TestCase], Dataset]:
    """Return the raw and tokenised testcase datasets."""

    testcases = load_testcases()
    dataset = Dataset.from_dict(
        {
            "id": [tc.id for tc in testcases],
            "text": [tc.text for tc in testcases],
            "label": [tc.label for tc in testcases],
        }
    )

    tokenised = dataset.map(
        lambda batch: tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        ),
        batched=True,
    )

    return testcases, tokenised


def load_raw_imdb(cache_dir: Optional[str] = None) -> DatasetDict:
    """Load the IMDB dataset using 🤗 Datasets."""

    return load_dataset("imdb", cache_dir=cache_dir)


def _clean_dataset(ds: Dataset) -> Dataset:
    return ds.map(lambda batch: {"text": clean_reviews(batch["text"])}, batched=True)


def prepare_dataset_splits(
    dataset: DatasetDict,
    *,
    validation_size: float = 0.1,
    seed: int = 42,
    limit_train: Optional[int] = None,
    limit_test: Optional[int] = None,
) -> PreparedSplits:
    """Return cleaned train/validation/test datasets.

    Args:
        dataset: Raw IMDB dataset as returned by :func:`load_raw_imdb`.
        validation_size: Fraction of the training split to allocate to validation.
        seed: Random seed for shuffling.
        limit_train: Optional cap on the number of training examples (useful for
            quick experiments during development).
        limit_test: Optional cap on the number of test examples.
    """

    train_ds = dataset["train"]
    test_ds = dataset["test"]

    if limit_train is not None:
        train_ds = train_ds.shuffle(seed=seed).select(range(limit_train))
    if limit_test is not None:
        test_ds = test_ds.shuffle(seed=seed).select(range(limit_test))

    cleaned_train = _clean_dataset(train_ds)
    cleaned_test = _clean_dataset(test_ds)

    train_validation = cleaned_train.train_test_split(test_size=validation_size, seed=seed)

    return PreparedSplits(
        train=train_validation["train"],
        validation=train_validation["test"],
        test=cleaned_test,
    )


def tokenize_dataset(
    splits: PreparedSplits,
    tokenizer: PreTrainedTokenizerBase,
    *,
    max_length: int = 256,
    padding: str = "max_length",
    truncation: bool = True,
) -> PreparedSplits:
    """Tokenise dataset splits using the provided tokenizer."""

    def _tokenize(batch):
        return tokenizer(
            batch["text"],
            padding=padding,
            truncation=truncation,
            max_length=max_length,
        )

    tokenised_train = splits.train.map(_tokenize, batched=True)
    tokenised_validation = splits.validation.map(_tokenize, batched=True)
    tokenised_test = splits.test.map(_tokenize, batched=True)

    return PreparedSplits(
        train=tokenised_train,
        validation=tokenised_validation,
        test=tokenised_test,
    )
