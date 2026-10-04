from pathlib import Path

import streamlit as st

from pulse_ui.components import sidebar
from pulse_ui.utils.login import hf_login
from pulse_ui.utils.tools import load_css

hf_login()

conf_path = Path(".streamlit")

st.set_page_config(layout="wide")

load_css(conf_path / "styles.css")
load_css(conf_path / "logo.css")

page = st.navigation(
    {
        "PULSE": [
            st.Page("app_pages/referendum.py", title="Polling", icon=":material/how_to_vote:"),
            st.Page("app_pages/results.py", title="Results", icon=":material/bar_chart:"),
            st.Page("app_pages/explorer.py", title="Explorer", icon=":material/manage_search:"),
        ]
    }
)
st.header("PULSE - Polling Using LLM-based Sentiment Extraction")
with st.sidebar:
    sidebar.show()

page.run()
