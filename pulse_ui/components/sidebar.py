"""Sidebar: conference logo, page links and the connection form."""

from pathlib import Path

import streamlit as st

from pulse_ui.components import connection
from pulse_ui.navigation import PAGES

LOGO = Path(__file__).parents[1] / "static" / "icdm2025logo-sidebar.png"
LOGO_LINK = "https://www3.cs.stonybrook.edu/~icdm2025/"


def init_sidebar() -> None:
    with st.sidebar:
        st.image(LOGO, alt="ICDM 2025", width="stretch", link=LOGO_LINK)
        for page in PAGES:
            st.page_link(page)
        st.divider()
        connection.show()
