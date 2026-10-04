"""PULSE scoring. Sign convention: positive values favor group A, negative favor group B."""

from typing import Literal
from collections.abc import Sequence

import numpy as np
import pandas as pd

from pulse.domain.types import Token


def norm_prob_diff(logprobs_a: Sequence[float], logprobs_b: Sequence[float]) -> list[float]:
    """Normalized probability difference P(A) - P(B) for each A/B completion pair.

    With P(A) = e^a / (e^a + e^b), the difference simplifies to tanh((a - b) / 2),
    which avoids underflow for long completions with very negative log-likelihoods.
    """
    a, b = np.asarray(logprobs_a, dtype=float), np.asarray(logprobs_b, dtype=float)
    return np.tanh((a - b) / 2).tolist()


def ground_truth_diff(pct_a: Sequence[float], pct_b: Sequence[float]) -> list[float]:
    """Normalized difference of observed vote shares, on the same scale as `norm_prob_diff`."""
    a, b = np.asarray(pct_a, dtype=float), np.asarray(pct_b, dtype=float)
    return ((a - b) / (a + b)).tolist()


def standard_error(values: pd.DataFrame) -> pd.Series:
    """Standard error of each row's mean: sample standard deviation / sqrt(n)."""
    return values.std(axis=1) / np.sqrt(values.count(axis=1))


def predicted_group(mean: float) -> Literal["A", "B"]:
    """The group a mean P(A) - P(B) predicts; a tie counts as B."""
    return "A" if mean > 0 else "B"


def token_prefixes(tokens: Sequence[Token]) -> list[str]:
    """Text preceding each token: ["", t0, t0+t1, ...]."""
    prefixes, text = [], ""
    for token in tokens:
        prefixes.append(text)
        text += token.token
    return prefixes


def renormalize(logprobs: Sequence[float]) -> np.ndarray:
    """Probabilities of a top-k distribution, renormalized to sum to 1 over the top-k."""
    logprobs = np.asarray(logprobs, dtype=float)

    # softmax - sub max for numerical stability
    probs = np.exp(logprobs - logprobs.max())
    return probs / probs.sum()


def elbow_rank(logprobs: Sequence[float], min_p: float) -> int:
    """Smallest rank whose cumulative probability reaches `min_p`.

    `logprobs` are a sorted top-k next-token distribution; they are renormalized over the top-k.
    """
    return int(np.searchsorted(np.cumsum(renormalize(logprobs)), min_p)) + 1
