"""Completion analysis: rank the poll's completions token by token, then show Side A and Side B."""

import time

import pandas as pd
import streamlit as st

from streamlit.delta_generator import DeltaGenerator

from pulse_ui.session import pulse, analysis
from pulse.application import Ranker
from pulse_ui.utils.tools import apply_html, get_position_table
from pulse_ui.components.poll_form import draft_poll


def show(container: DeltaGenerator) -> None:
    """Ranks when requested (on page load or Ctrl+R), then shows the latest rankings."""
    if analysis.rank_flag:
        analysis.rank_flag = False
        _rank(container)

    if rankings := analysis.rankings:
        A_df, B_df = rankings
        _side(container, title="Side A", rankings=A_df)
        _side(container, title="Side B", rankings=B_df)


def request() -> None:
    """Rank again on the next run."""
    analysis.rank_flag = True
    st.rerun()


def _rank(container: DeltaGenerator) -> None:
    poll = draft_poll()
    if clause := pulse().check_ranking(poll):
        analysis.rankings = None
        container.warning(clause.msg)
        return

    ranker = pulse().ranker(poll)
    if ranker != analysis.ranker:  # same poll and model: reuse the computed ranking
        analysis.ranker = ranker

    analysis.rankings = _with_progress(container=container, ranker=analysis.ranker)


def _with_progress(container: DeltaGenerator, ranker: Ranker, delay: float = 0.5) -> tuple[pd.DataFrame, pd.DataFrame]:
    with container:
        pbar = st.progress(0, text="")
        _step(pbar, value=0.0, text="Ranking completions.", delay=delay)
        _step(pbar, value=0.2, text="Scoring completions.", delay=delay)
        _ = ranker.sequences  # trigger cached property
        _step(pbar, value=0.5, text="Calculating elbows.", delay=delay)
        _ = ranker.elbows  # trigger cached property
        _step(pbar, value=0.8, text="Gathering metrics.", delay=delay)
        rankings = ranker.rankings  # trigger cached property

    _step(pbar, value=1.0, text="Rankings complete.", delay=delay)
    pbar.empty()
    return rankings


def _step(p_bar, value: float, text: str, delay: float = 0.0) -> None:
    p_bar.progress(value=value, text=text)
    time.sleep(delay)


def _side(container: DeltaGenerator, title: str, rankings: pd.DataFrame) -> None:
    container.markdown(f"**{title}**", text_alignment="center")
    with container.container(border=False, height=350):
        st.table(apply_html(styler=get_position_table(rankings=rankings), cell_text_color="white"))
