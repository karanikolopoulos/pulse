import math

import pytest

from pulse.domain.poll import PulseResults
from pulse.services.results import summarize


def test_summarize_per_persona_group():
    results = PulseResults(
        task="poll",
        model="m",
        metrics=[{"x": 0.2, "y": 0.6}, {"x": -0.5, "y": -0.1}],
        dataset={
            "docs": [
                {"group": "Men", "persona": "a man", "A pct": 0.4, "B pct": 0.6},
                {"group": "Women", "persona": "a woman", "A pct": 0.6, "B pct": 0.4},
            ],
            "completions": {"A": ["a", "b"], "B": ["c", "d"], "alias": ["x", "y"]},
        },
    )

    groups = summarize(results).groups

    assert list(groups.index) == ["Men", "Women"]
    assert groups["mean"].tolist() == pytest.approx([0.4, -0.3])
    assert groups["se"].tolist() == pytest.approx([0.2, 0.2])  # sample std of 2 values / sqrt(2)
    assert groups["prediction"].tolist() == ["A", "B"]
    assert groups["actual"].tolist() == pytest.approx([-0.2, 0.2])

    only_x = summarize(results, aliases=["x"])
    assert list(only_x.scores.columns) == ["x"]


def test_poll_without_personas_is_one_group():
    results = PulseResults(task="poll", model="m", metrics=[{"x": 0.0}], dataset={})

    summary = summarize(results)

    assert list(summary.groups.index) == ["poll"]
    assert summary.groups["prediction"].tolist() == ["B"]  # a tie counts as B
    assert math.isnan(summary.groups["actual"].iloc[0])
    assert not summary.has_personas
