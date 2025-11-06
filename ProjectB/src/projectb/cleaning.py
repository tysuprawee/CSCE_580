"""Utility functions for cleaning raw IMDB movie reviews."""

from __future__ import annotations

import html
import re
from typing import Iterable, List

__all__ = ["clean_review", "clean_reviews"]

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s+")
_CONTRACTIONS = {
    "can't": "cannot",
    "won't": "will not",
    "n't": " not",
    "'re": " are",
    "'s": " is",
    "'d": " would",
    "'ll": " will",
    "'t": " not",
    "'ve": " have",
    "'m": " am",
}


def _expand_contractions(text: str) -> str:
    result = text
    for contraction, expanded in _CONTRACTIONS.items():
        result = re.sub(contraction, expanded, result, flags=re.IGNORECASE)
    return result


def clean_review(review: str) -> str:
    """Return a lightly normalised version of a raw IMDB review.

    The cleaning intentionally remains conservative – it removes HTML tags,
    decodes HTML entities, expands a small set of common contractions, and
    normalises whitespace. This keeps the semantic content intact while
    matching the expectations of both transformer and classical models.
    """

    # Remove HTML tags and decode entities.
    no_tags = _HTML_TAG_RE.sub(" ", review)
    unescaped = html.unescape(no_tags)

    # Expand common contractions and collapse whitespace.
    expanded = _expand_contractions(unescaped)
    normalised = _WHITESPACE_RE.sub(" ", expanded)

    return normalised.strip()


def clean_reviews(reviews: Iterable[str]) -> List[str]:
    """Clean an iterable of reviews in a memory-efficient fashion."""

    return [clean_review(review) for review in reviews]
