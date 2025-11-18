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


def load_raw_imdb(
    cache_dir: Optional[str] = None,
    *,
    dataset_name: str = "imdb",
    dataset_path: Optional[str] = None,
) -> DatasetDict:
    """Load the IMDB dataset using 🤗 Datasets or local parquet files.

    Args:
        cache_dir: Optional path to reuse 🤗 Datasets caches.
        dataset_name: Dataset builder or repository name passed to
            :func:`datasets.load_dataset` when ``dataset_path`` is ``None``.
        dataset_path: When provided, load ``train``/``test`` splits from parquet
            files stored on disk. This mirrors the file layout produced by
            ``huggingface-cli download stanfordnlp/imdb`` (for example,
            ``plain_text/train-00000-of-00001.parquet``).
    """

    if dataset_path is not None:
        return _load_local_imdb_from_parquet(Path(dataset_path))

    return load_dataset(dataset_name, cache_dir=cache_dir)


def _load_local_imdb_from_parquet(base_path: Path) -> DatasetDict:
    """Load IMDB splits from parquet files saved on disk.

    The helper searches recursively under ``base_path`` for files named
    ``train*.parquet`` and ``test*.parquet`` so it works with both flat
    directories as well as the ``plain_text`` subfolder distributed on the
    Hugging Face Hub.
    """

    if not base_path.exists():
        raise FileNotFoundError(f"Dataset path '{base_path}' does not exist")

    def _split_files(split: str) -> list[str]:
        files = sorted(str(path) for path in base_path.rglob(f"{split}*.parquet"))
        if not files:
            raise FileNotFoundError(
                f"Could not find any parquet files for split '{split}' under {base_path}"
            )
        return files

    train_files = _split_files("train")
    test_files = _split_files("test")

    datasets = {
        "train": Dataset.from_parquet(train_files),
        "test": Dataset.from_parquet(test_files),
    }

    unsupervised_files = list(base_path.rglob("unsupervised*.parquet"))
    if unsupervised_files:
        datasets["unsupervised"] = Dataset.from_parquet(
            [str(path) for path in unsupervised_files]
        )

    return DatasetDict(datasets)


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
