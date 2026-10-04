import time

from pathlib import Path

import streamlit as st

from streamlit.delta_generator import DeltaGenerator

from pulse.domain.poll import PulseConfig
from pulse_ui.pages.session import form, pulse, connection

DELAY = 0.5


@st.cache_data
def _read_css(file_path: Path) -> str:
    with open(file_path) as f:
        css = f"<style>{f.read()}</style>"
    return css


def load_css(file_path: Path) -> None:
    css_string = _read_css(file_path)
    st.markdown(css_string, unsafe_allow_html=True)


def st_md(
    text: str,
    container: DeltaGenerator | None = None,
    font_size: str = "16px",
    **styles: str,
) -> None:
    styles = {"font-size": font_size, **styles}
    style_str = "; ".join(f"{k}: {v}" for k, v in styles.items() if v is not None)

    target = container if container is not None else st
    target.markdown(
        f"<div style='{style_str}'>{text}</div>",
        unsafe_allow_html=True,
    )


def connect(url: str, api_key: str) -> None:
    try:
        pulse().connect(url, api_key or None)
    except ConnectionError as error:
        st.toast(str(error))
        return

    connection.credentials = {"base_url": url, "token": api_key}
    st.toast(f"Connected to {url}")


def assign_model() -> None:
    model = pulse().use_model(connection.model_choice)
    if not model.max_logprobs_known:
        st.toast(f"Could not read max_logprobs for '{model.model}', using {model.max_logprobs}.")

    st.toast(f"Assigned model: {model.model}")
    time.sleep(DELAY)


def sidebar_connection() -> None:
    with st.form("connection_form"):
        url = st.text_input(
            label="URL",
            value=(connection.credentials or {}).get("base_url"),
            placeholder="http://localhost:8000",
        )
        api_key = st.text_input(
            label="API Key",
            value=(connection.credentials or {}).get("token"),
            placeholder="EMPTY",
            type="password",
        )

        if st.form_submit_button("Connect") and url:
            connect(url=url, api_key=api_key)

    app = pulse()
    if app.connection.is_connected.ok:
        models = app.models()
        index = models.index(app.model.model) if app.model else None

        st.selectbox(
            label="Select a model",
            options=models,
            index=index,
            on_change=assign_model,
            key=connection.key("model_choice"),
        )


def draft_poll() -> PulseConfig:
    """The poll as currently entered in the form."""
    return PulseConfig(
        name=form.poll_name,
        persona=form.persona,
        docs=form.selected_persona,
        question=form.question,
        answer=form.answer,
        completions=form.selected_completions,
    )


def is_update() -> bool:
    """The form edits the selected, saved poll rather than a new one."""
    return bool(form.selected_task) and form.poll_name == form.selected_task
