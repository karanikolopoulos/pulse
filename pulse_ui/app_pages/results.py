from dataclasses import replace

import pandas as pd
import streamlit as st

from pulse_ui.session import pulse, results_view
from pulse.application import PollSummary
from pulse_ui.utils.plot import prediction_chart
from pulse_ui.utils.tools import Latex, styler


def select(runs: pd.DataFrame) -> tuple:
    model = st.selectbox(
        "Select model",
        sorted(runs.model.unique()),
    )
    model_runs = runs[runs.model == model]
    task = st.selectbox("Select task", model_runs.task.sort_values())

    if task != results_view.task:
        results_view.task = task
        del results_view.columns  # select all completions of the new task

    return model, task


def task_summary(results) -> None:
    choices = results.choices.item()

    st.multiselect(
        label="Selected completions",
        options=choices["alias"],
        default=choices["alias"],
        key=results_view.key("columns"),
    )

    menu_df = pd.DataFrame(
        {
            "Alias": choices["alias"],
            "Group A": choices["A"],
            "Group B": choices["B"],
        }
    )

    st.popover("All completions", width="stretch").dataframe(menu_df)


def _escape_dollar(label):
    """Two `$` in a label would start LaTeX math in markdown tables."""
    if isinstance(label, str) and label.count("$") >= 2:  # noqa: PLR2004
        return label.replace("$", r"\$")
    return label


def results_section(task: str, model: str) -> None:
    chart_summary = pulse().summarize(task=task, model=model, aliases=results_view.columns)
    summary = replace(  # the markdown tables need `$` escaped; the chart does not
        chart_summary,
        scores=chart_summary.scores.rename(index=_escape_dollar),
        groups=chart_summary.groups.rename(index=_escape_dollar),
    )

    agg_tab, diff_tab = st.tabs(("Aggregated Results", "Individual Results"))

    diff_tab.table(
        data=styler(
            df=summary.scores,
            subset=list(summary.scores.columns),
            a_color="#a4c2f4",
            b_color="#ea9999",
        ),
    )

    with agg_tab:
        df_col, line_col = st.columns((0.3, 0.6), gap="large")

        groups = summary.groups
        agg_df = pd.DataFrame(
            {
                "pred": groups["prediction"],
                Latex.diff: groups["mean"].round(3).astype(str),
                "SE": groups["se"].round(4).astype(str),
            }
        ).rename_axis("Target Group")

        styled = styler(
            df=agg_df,
            subset=["pred"],
            a_color="#a4c2f4",
            b_color="#ea9999",
            cond=lambda x: x == "A",
        )
        df_col.table(data=styled)

        if summary.has_personas:
            with line_col:
                chart_section(chart_summary)


def chart_section(summary: PollSummary) -> None:
    st.altair_chart(
        prediction_chart(summary),
        width="stretch",
        alt="Mean P(A) minus P(B) per persona group with 95% intervals, and the real vote-share difference",
    )
    st.caption("Dot and line: mean and 95% interval over completions. Star: real vote-share difference.")


runs = pd.DataFrame(
    [(r.task, r.model, r.metrics, r.docs, r.completions) for r in pulse().results()],
    columns=["task", "model", "metrics", "docs", "choices"],
)

with st.sidebar:
    st.markdown("**Results**")
    model, task = select(runs=runs)

if runs.empty:  # guard
    st.warning("No experiments found.")
    st.stop()

st.subheader(f"{results_view.task} results")

run = runs[(runs.model == model) & (runs.task == task)]

task_summary(results=run)  # menu container
results_section(task=task, model=model)  # data container
st.space("large")  # bottom padding
