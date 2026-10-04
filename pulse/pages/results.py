import pandas as pd
import streamlit as st

from pulse.utils.plot import lineplot
from pulse.utils.tools import Latex, styler
from pulse.pages.session import SESSION, Session
from pulse.domain.scoring import ground_truth_diff

ALIAS_COLUMN = 0


def select(runs: pd.DataFrame) -> tuple:
    model = st.selectbox(
        "Select model",
        sorted(runs.model.unique()),
    )
    model_runs = runs[runs.model == model]
    task = st.selectbox("Select task", model_runs.task.sort_values())

    if task != SESSION.results_task:
        SESSION.results_task = task
        del SESSION.results_columns  # select all completions of the new task

    return model, task


def task_summary(results) -> None:
    choices = results.choices.item()

    st.multiselect(
        label="Selected completions",
        options=choices["alias"],
        default=choices["alias"],
        key=Session.results_columns.key,
    )

    menu_df = pd.DataFrame(
        {
            "Alias": choices["alias"],
            "Group A": choices["A"],
            "Group B": choices["B"],
        }
    )

    st.popover("All completions", use_container_width=True).dataframe(menu_df)


def diff_section(results) -> pd.DataFrame:
    def _escape_dollar(x):
        if isinstance(x, str) and x.count("$") >= 2:  # noqa: PLR2004
            return x.replace("$", r"\$")

        return x

    docs = results.docs.item()
    has_docs = bool(docs)
    metrics = results.metrics.item()

    if has_docs:
        index = pd.DataFrame(docs).iloc[:, ALIAS_COLUMN]
    else:
        index = results.task.item()

    diff = pd.DataFrame(metrics)
    if has_docs:
        diff.index = index.map(_escape_dollar)
    else:
        diff.index = [index]

    diff = diff[SESSION.results_columns]

    diff["mean"] = diff.mean(axis=1)
    diff["SE"] = diff.std(axis=1) / diff.count(axis=1).apply(lambda x: x**0.5)

    agg_tab, diff_tab = st.tabs(("Aggregated Results", "Individual Results"))

    diff_tab.table(
        data=styler(
            df=diff.drop(["mean", "SE"], axis=1),
            subset=SESSION.results_columns,
            a_color="#a4c2f4",
            b_color="#ea9999",
            cell_text_color="#ffffff",
        ),
    )

    with agg_tab:
        df_col, line_col = st.columns((0.3, 0.7))

        agg_df = pd.DataFrame(diff[["mean", "SE"]])

        pred = agg_df["mean"].apply(lambda x: "A" if x > 0 else "B")
        agg_df["mean"] = agg_df["mean"].round(3).astype(str)
        agg_df["SE"] = agg_df["SE"].round(4).astype(str)
        agg_df.index.name = "Target Group"
        agg_df = pd.DataFrame({"pred": pred, Latex.diff: agg_df["mean"], "SE": agg_df["SE"]})

        styled = styler(
            df=agg_df,
            subset=["pred"],
            a_color="#a4c2f4",
            b_color="#ea9999",
            cell_text_color="#ffffff",
            cond=lambda x: x == "A",
        )
        df_col.table(data=styled)

        if has_docs:
            with line_col:
                lineplot_section(diff=diff, docs=pd.DataFrame(docs))


def get_ground_truth(docs: pd.DataFrame) -> list | None:
    if {"A pct", "B pct"}.issubset(docs.columns):
        return ground_truth_diff(pct_a=docs["A pct"], pct_b=docs["B pct"])


def setup_sidebar() -> None:
    st.sidebar.divider()

    x_col, y_col = st.sidebar.columns(2)

    x_col.select_slider(
        label="Figure x",
        options=[round(x * 0.1, 1) for x in range(50, 201)],
        value=8.0,
        key=Session.fig_x.key,
    )

    y_col.select_slider(
        label="Figure y",
        options=[round(x * 0.1, 1) for x in range(50, 201)],
        value=8,
        key=Session.fig_y.key,
    )


def lineplot_section(diff: pd.DataFrame, docs: pd.DataFrame) -> None:
    if ground_truth := get_ground_truth(docs=docs):
        diff["pct_diff"] = ground_truth
        id_vars = ["Target Group", "pct_diff", "mean"]
    else:
        id_vars = ["Target Group", "mean"]

    diff["mean"] = diff["mean"].map(lambda x: "A" if x > 0 else "B")
    diff = diff.drop("SE", axis=1).rename_axis("Target Group").reset_index().melt(id_vars=id_vars)

    fig = lineplot(
        diff=diff,
        figsize=(SESSION.fig_x, SESSION.fig_y),
        group_a_color="blue",
        group_b_color="red",
    )
    pointplot_col, *_ = st.columns([0.5, 0.1, 0.1])
    with pointplot_col:
        st.pyplot(fig)


st.header("PULSE - Polling Using LLM-based Sentiment Extraction")
runs = pd.DataFrame(
    [(r.task, r.model, r.metrics, r.docs, r.completions) for r in SESSION.storage.results()],
    columns=["task", "model", "metrics", "docs", "choices"],
)

with st.sidebar:
    st.write("Results")
    model, task = select(runs=runs)

if runs.empty:  # guard
    st.warning("No experiments found.")
    st.stop()

st.subheader(f"{SESSION.results_task} results")

run = runs[(runs.model == model) & (runs.task == task)]

task_summary(results=run)  # menu container
setup_sidebar()  # sidebar options
diff = diff_section(results=run)  # data container
st.empty().container(height=100, border=False)  # bottom padding
