import seaborn as sns
import matplotlib.pyplot as plt

from matplotlib.lines import Line2D

from pulse_ui.utils.tools import Latex, register_fonts
from pulse.services.results import PollSummary

STAR = {
    "xdata": [0],
    "ydata": [0],
    "marker": "*",
    "color": "black",
    "markersize": 6,
    "linestyle": "None",
    "label": "actual difference",
}

CIRCLE = {
    "xdata": [0],
    "ydata": [0],
    "marker": "o",
    "color": "black",
    "markersize": 4,
    "linestyle": "None",
    "label": "predicted difference",
}

register_fonts()
plt.rcParams["font.family"] = "Source Sans 3"
plt.rcParams["font.weight"] = "medium"
plt.rcParams["font.size"] = 13


def lineplot(summary: PollSummary, figsize=(8, 6), group_a_color="blue", group_b_color="red") -> plt.Figure:
    """Each completion's P(A) - P(B) per persona group, coloured by the group's prediction,
    with the real vote-share difference as a star where known."""
    groups = summary.groups
    diff = summary.scores.rename_axis("Target Group").reset_index().melt(id_vars="Target Group")
    diff["mean"] = diff["Target Group"].map(groups["prediction"])
    if groups["actual"].notna().any():
        diff["pct_diff"] = diff["Target Group"].map(groups["actual"])

    fig, ax = plt.subplots(figsize=figsize, dpi=300)

    sns.pointplot(
        data=diff,
        x="value",
        y="Target Group",
        hue="mean",
        dodge=False,
        palette={"A": group_a_color, "B": group_b_color},
        linestyle="none",
        markersize=5,
        linewidth=2,
        seed=2025,
    )

    if "pct_diff" in diff:
        ax.scatter(
            data=diff,
            x="pct_diff",
            y="Target Group",
            marker="*",
            c="black",
            edgecolors="black",
            linewidths=0.8,
            s=45,
            alpha=0.5,
            zorder=2,
        )

    _, fig_height = figsize
    ax.set_ylabel("")
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.legend(
        handles=[Line2D(**CIRCLE), Line2D(**STAR)],
        loc="upper center",
        bbox_to_anchor=(0.475, 1.01 + 0.075 * (7 / fig_height)),
        ncols=2,
    )
    ax.set_xticks([-1, 0, 1])
    ax.set_xlabel(Latex.diff)
    plt.margins(y=0.02)

    return fig
