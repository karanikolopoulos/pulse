from enum import StrEnum
from pathlib import Path
from collections.abc import Callable

import pandas as pd
import matplotlib.font_manager as fm

from pandas.io.formats.style import Styler

STATIC = Path(__file__).resolve().parent.parent / "static"


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
    cell_text_color: str = "white",
    cond: Callable = lambda x: x >= 0,
) -> Styler:
    def _fn(x):  # style condition for data cells
        color = a_color if cond(x) else b_color
        return f"background-color: {color}; color: black"

    styler = df.style.map(_fn, subset=pd.IndexSlice[:, subset])
    return apply_html(styler, cell_text_color=cell_text_color)


def apply_html(styler: Styler, cell_text_color: str = "white") -> Styler:
    return styler.set_table_styles(
        [
            {  # index name
                "selector": "th.blank",
                "props": [
                    ("color", cell_text_color),
                    ("text-align", "center"),
                ],
            },
            {  # index
                "selector": "th.row_heading",
                "props": [
                    ("color", cell_text_color),
                    ("text-align", "left"),
                ],
            },
            {  # column text color
                "selector": "th.col_heading",
                "props": [
                    ("color", cell_text_color),
                    ("text-align", "center"),
                ],
            },
            {  # cell text color
                "selector": "td > div",
                "props": [
                    ("text-align", "center"),
                ],
            },
        ],
        overwrite=False,
    )


def register_fonts(font_dir: Path = STATIC) -> None:
    for font in font_dir.rglob("*.ttf"):
        fm.fontManager.addfont(font)


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
