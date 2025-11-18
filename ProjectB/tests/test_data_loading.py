"""Tests for offline IMDB dataset loading helpers."""

from pathlib import Path

import pytest
from datasets import Dataset

from projectb.data import load_raw_imdb


def _write_parquet_split(base: Path, split: str, *, texts: list[str], labels: list[int]) -> None:
    dataset = Dataset.from_dict({"text": texts, "label": labels})
    target_dir = base / "plain_text"
    target_dir.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(str(target_dir / f"{split}-00000-of-00001.parquet"))


def test_load_raw_imdb_from_local_parquet(tmp_path):
    dataset_root = tmp_path / "stanfordnlp" / "imdb"
    _write_parquet_split(dataset_root, "train", texts=["good"], labels=[1])
    _write_parquet_split(dataset_root, "test", texts=["bad"], labels=[0])
    _write_parquet_split(dataset_root, "unsupervised", texts=["meh"], labels=[0])

    dataset = load_raw_imdb(dataset_path=str(dataset_root))

    assert dataset["train"].num_rows == 1
    assert dataset["test"].num_rows == 1
    assert dataset["unsupervised"].num_rows == 1
    assert dataset["train"][0]["text"] == "good"
    assert dataset["test"][0]["label"] == 0


def test_load_raw_imdb_missing_path(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_raw_imdb(dataset_path=str(tmp_path / "missing"))
