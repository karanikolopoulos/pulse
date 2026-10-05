"""The app's pages; Streamlit's own page list is hidden and the sidebar draws the links."""

import streamlit as st

PAGES = [
    st.Page("app_pages/referendum.py", title="Polling", icon=":material/how_to_vote:"),
    st.Page("app_pages/results.py", title="Results", icon=":material/bar_chart:"),
    st.Page("app_pages/explorer.py", title="Explorer", icon=":material/manage_search:"),
]


def init_pages() -> st.Page:
    """Register the pages and return the one to run."""
    return st.navigation(PAGES, position="hidden")
