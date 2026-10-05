"""Connection form: connect to a model server and choose a model."""

import streamlit as st

from pulse_ui.session import pulse, connection


def show() -> None:
    with st.form("connection_form"):
        credentials = connection.credentials or {}
        url = st.text_input(label="URL", value=credentials.get("base_url"), placeholder="http://localhost:8000")
        api_key = st.text_input(label="API key", value=credentials.get("token"), placeholder="EMPTY", type="password")

        if st.form_submit_button("Connect") and url:
            _connect(url=url, api_key=api_key)

    app = pulse()
    if app.connection.is_connected.ok:
        models = app.models()
        st.selectbox(
            label="Select a model",
            options=models,
            index=models.index(app.model.model) if app.model else None,
            on_change=_use_model,
            key=connection.key("model_choice"),
        )


def _connect(url: str, api_key: str) -> None:
    try:
        pulse().connect(url, api_key or None)
    except ConnectionError as error:
        st.toast(str(error))
        return

    connection.credentials = {"base_url": url, "token": api_key}
    st.toast(f"Connected to {url}")


def _use_model() -> None:
    model = pulse().use_model(connection.model_choice)
    if not model.max_logprobs_known:
        st.toast(f"Could not read max_logprobs for '{model.model}', using {model.max_logprobs}.")

    st.toast(f"Assigned model: {model.model}")
