"""Results chart: each persona group's mean P(A) - P(B), with its 95% interval and the real vote-share difference."""

import altair as alt
import pandas as pd

from pulse.application import PollSummary

GROUP_COLORS = {"A": "#2563EB", "B": "#DC2626"}  # blue / red: stays distinct under red-green colour blindness
ROW_HEIGHT = 26
# five-pointed star as a Vega SVG path, in the unit square
STAR = "M0,-1L0.29,-0.4L0.95,-0.31L0.47,0.15L0.59,0.81L0,0.5L-0.59,0.81L-0.47,0.15L-0.95,-0.31L-0.29,-0.4Z"


def prediction_chart(summary: PollSummary) -> alt.LayerChart:
    """Mean with mean ± 1.96·SE per group, coloured by the predicted group; a star marks the real difference."""
    groups = summary.groups
    data = pd.DataFrame(
        {
            "group": groups.index.astype(str),
            "mean": groups["mean"],
            # P(A) - P(B) is bounded by [-1, 1], so the interval is clamped to the axis
            "low": (groups["mean"] - 1.96 * groups["se"]).clip(-1, 1),
            "high": (groups["mean"] + 1.96 * groups["se"]).clip(-1, 1),
            "se": groups["se"],
            "prediction": groups["prediction"],
            "actual": groups["actual"],
        }
    )

    y = alt.Y("group:N", sort=list(data["group"]), title=None)
    x_scale = alt.Scale(domain=[-1, 1])
    color = alt.Color(
        "prediction:N",
        scale=alt.Scale(domain=list(GROUP_COLORS), range=list(GROUP_COLORS.values())),
        legend=alt.Legend(title="Predicted group", orient="top"),
    )
    tooltip = [
        alt.Tooltip("group:N", title="Group"),
        alt.Tooltip("prediction:N", title="Predicted"),
        alt.Tooltip("mean:Q", title="Mean P(A) − P(B)", format="+.3f"),
        alt.Tooltip("se:Q", title="SE", format=".4f"),
        alt.Tooltip("actual:Q", title="Actual difference", format="+.3f"),
    ]

    base = alt.Chart(data).encode(y=y)
    zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(strokeDash=[4, 4], opacity=0.6).encode(x="x:Q")
    interval = base.mark_rule(strokeWidth=2, clip=True).encode(
        x=alt.X("low:Q", scale=x_scale), x2="high:Q", color=color
    )
    mean = base.mark_circle(size=80, opacity=1).encode(
        x=alt.X(
            "mean:Q",
            scale=x_scale,
            axis=alt.Axis(values=[-1, 0, 1], format="+d", title="Mean P(A) − P(B)  (A > 0 > B)"),
        ),
        color=color,
        tooltip=tooltip,
    )
    layers = [zero, interval, mean]

    if data["actual"].notna().any():
        actual = base.mark_point(shape=STAR, size=120, filled=True, color="gray", opacity=0.7).encode(
            x=alt.X("actual:Q", scale=x_scale), tooltip=tooltip
        )
        layers.append(actual)

    return alt.layer(*layers).properties(height=ROW_HEIGHT * len(data))
