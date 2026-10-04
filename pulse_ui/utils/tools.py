from enum import StrEnum
from pathlib import Path
from collections.abc import Callable

import pandas as pd
import streamlit as st

from pandas.io.formats.style import Styler

DELAY = 0.5  # seconds a toast stays readable before a rerun


@st.cache_data
def _read_css(file_path: Path) -> str:
    with open(file_path) as f:
        return f"<style>{f.read()}</style>"


def load_css(file_path: Path) -> None:
    st.markdown(_read_css(file_path), unsafe_allow_html=True)


class Placeholder(StrEnum):
    persona = "You are a citizen of the U.S."
    question = "Who will you vote for in the 2024 U.S. presidential election?"
    answer = "I will vote for"
    completion = " the Democratic"


class Latex(StrEnum):
    diff = r"$\overline{\mathrm{diff}}$"


def styler(
    df: pd.DataFrame | Styler,
    subset: list[str] | None = None,
    a_color: str = "#a4c2f4",
    b_color: str = "#ea9999",
    cond: Callable = lambda x: x >= 0,
) -> Styler:
    """Colours the cells of `subset` by group A/B; the text of coloured cells is black, so it reads on any theme."""

    def _fn(x):  # style condition for data cells
        color = a_color if cond(x) else b_color
        return f"background-color: {color}; color: black"

    return align(df.style.map(_fn, subset=pd.IndexSlice[:, subset]))


def align(styler: Styler) -> Styler:
    """Centres headers and cells and left-aligns row labels; colours come from the theme."""
    return styler.set_table_styles(
        [
            {"selector": "th.blank", "props": [("text-align", "center")]},
            {"selector": "th.row_heading", "props": [("text-align", "left")]},
            {"selector": "th.col_heading", "props": [("text-align", "center")]},
            {"selector": "td > div", "props": [("text-align", "center")]},
        ],
        overwrite=False,
    )


def get_position_table(
    rankings: pd.DataFrame,
    yes_bg: str = "lightgreen",
    no_bg: str = "lightcoral",
) -> Styler:
    assert {"ranks", "elbows", "token_strings"}.issubset(rankings.columns)

    def _cell_color(bg: str) -> str:
        color = "color: black"
        text_align = "text-align:center"
        return f"background-color: {bg}; {color}; {text_align}"

    max_len = max(len(r) for r in rankings["ranks"])

    token_data = {f"token {i}": [] for i in range(max_len)}
    style_data = {f"token {i}": [] for i in range(max_len)}

    for _, row in rankings.iterrows():
        tokens = row["token_strings"]
        ranks = row["ranks"]
        elbows = row["elbows"]

        for i in range(max_len):
            col_name = f"token {i}"
            token = tokens[i] if i < len(tokens) else ""
            token_data[col_name].append(token)

            if i < len(ranks) and i < len(elbows):
                if ranks[i] <= elbows[i]:
                    style_data[col_name].append(_cell_color(yes_bg))
                else:
                    style_data[col_name].append(_cell_color(no_bg))
            else:
                style_data[col_name].append("")

    token_data["logprob"] = rankings["logprob"].round(4).astype(str).tolist()
    style_data["logprob"] = [""] * len(rankings)

    comp_df = pd.DataFrame(token_data)
    style_df = pd.DataFrame(style_data)

    return comp_df.style.apply(lambda col: style_df[col.name], axis=0)
