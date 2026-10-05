from pathlib import Path

import streamlit as st

from pulse_ui.navigation import init_pages
from pulse_ui.utils.login import hf_login
from pulse_ui.utils.tools import load_css
from pulse_ui.utils.notify import show_pending
from pulse_ui.components.sidebar import init_sidebar

hf_login()

st.set_page_config(layout="wide")
load_css(Path(__file__).parent / ".streamlit" / "styles.css")

show_pending()
page = init_pages()
st.header("PULSE - Polling Using LLM-based Sentiment Extraction")
init_sidebar()

page.run()
