"""Summaries of saved poll runs: per-group means, uncertainty, predictions and real-world comparison."""

from dataclasses import dataclass

import pandas as pd

from pulse.domain.poll import PulseResults
from pulse.domain.scoring import standard_error, predicted_group, ground_truth_diff


@dataclass(frozen=True)
class PollSummary:
    scores: pd.DataFrame  # persona group x completion alias: P(A) - P(B)
    groups: pd.DataFrame  # per group: mean, se, prediction ("A"/"B"), actual (from vote shares, NaN if unknown)
    has_personas: bool


def summarize(results: PulseResults, aliases: list[str] | None = None) -> PollSummary:
    """Scores of the given completion aliases (all by default), summarized per persona group.

    The personas table's first column labels each group (e.g. "group" or "state"); a poll without
    personas is a single group named after the poll. Real vote shares come from "A pct"/"B pct".
    """
    docs = pd.DataFrame(results.docs) if results.docs else None
    index = pd.Index(docs.iloc[:, 0], name=docs.columns[0]) if docs is not None else pd.Index([results.task])

    scores = pd.DataFrame(results.metrics, index=index)
    if aliases is not None:
        scores = scores[aliases]

    mean = scores.mean(axis=1)
    groups = pd.DataFrame({"mean": mean, "se": standard_error(scores), "prediction": mean.map(predicted_group)})

    if docs is not None and {"A pct", "B pct"}.issubset(docs.columns):
        groups["actual"] = ground_truth_diff(pct_a=docs["A pct"], pct_b=docs["B pct"])
    else:
        groups["actual"] = float("nan")

    return PollSummary(scores=scores, groups=groups, has_personas=docs is not None)
