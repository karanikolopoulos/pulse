import os

import streamlit as st

from huggingface_hub import login


@st.cache_resource
def hf_login() -> None:
    """Logs in to the Hugging Face Hub (for gated tokenizers) once per process, not on every rerun."""
    if hf_token := os.environ.get("HF_TOKEN"):
        login(token=hf_token)
