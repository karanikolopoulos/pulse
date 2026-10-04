from dataclasses import replace

import pandas as pd
import streamlit as st

from pulse_ui.session import pulse, results_view
from pulse.application import PollSummary
from pulse_ui.utils.plot import lineplot
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
    """Two `$` in a label would start LaTeX math in markdown tables and matplotlib."""
    if isinstance(label, str) and label.count("$") >= 2:  # noqa: PLR2004
        return label.replace("$", r"\$")
    return label


def results_section(task: str, model: str) -> None:
    summary = pulse().summarize(task=task, model=model, aliases=results_view.columns)
    summary = replace(
        summary,
        scores=summary.scores.rename(index=_escape_dollar),
        groups=summary.groups.rename(index=_escape_dollar),
    )

    agg_tab, diff_tab = st.tabs(("Aggregated Results", "Individual Results"))

    diff_tab.table(
        data=styler(
            df=summary.scores,
            subset=list(summary.scores.columns),
            a_color="#a4c2f4",
            b_color="#ea9999",
            cell_text_color="#ffffff",
        ),
    )

    with agg_tab:
        df_col, line_col = st.columns((0.3, 0.7))

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
            cell_text_color="#ffffff",
            cond=lambda x: x == "A",
        )
        df_col.table(data=styled)

        if summary.has_personas:
            with line_col:
                lineplot_section(summary)


def setup_sidebar() -> None:
    st.sidebar.divider()

    x_col, y_col = st.sidebar.columns(2)

    x_col.select_slider(
        label="Figure x",
        options=[round(x * 0.1, 1) for x in range(50, 201)],
        value=8.0,
        key=results_view.key("fig_x"),
    )

    y_col.select_slider(
        label="Figure y",
        options=[round(x * 0.1, 1) for x in range(50, 201)],
        value=8,
        key=results_view.key("fig_y"),
    )


def lineplot_section(summary: PollSummary) -> None:
    fig = lineplot(
        summary=summary,
        figsize=(results_view.fig_x, results_view.fig_y),
        group_a_color="blue",
        group_b_color="red",
    )
    pointplot_col, *_ = st.columns([0.5, 0.1, 0.1])
    with pointplot_col:
        st.pyplot(fig)


runs = pd.DataFrame(
    [(r.task, r.model, r.metrics, r.docs, r.completions) for r in pulse().results()],
    columns=["task", "model", "metrics", "docs", "choices"],
)

with st.sidebar:
    st.write("Results")
    model, task = select(runs=runs)

if runs.empty:  # guard
    st.warning("No experiments found.")
    st.stop()

st.subheader(f"{results_view.task} results")

run = runs[(runs.model == model) & (runs.task == task)]

task_summary(results=run)  # menu container
setup_sidebar()  # sidebar options
results_section(task=task, model=model)  # data container
st.empty().container(height=100, border=False)  # bottom padding
