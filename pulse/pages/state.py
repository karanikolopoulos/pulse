import time

from http import HTTPStatus
from pathlib import Path

import streamlit as st

from streamlit.delta_generator import DeltaGenerator

from pulse.ports import LanguageModel
from pulse.domain.poll import PulseConfig, poll_chat
from pulse.domain.types import Chat
from pulse.domain.guards import PollGuards, ConnectionGuards
from pulse.pages.session import SESSION, Session
from pulse.services.ranking import Ranker
from pulse.adapters.vllm.client import DEFAULT_MAX_LOGPROBS, VLLMConnection

DELAY = 0.5

# Caching


@st.cache_data
def _read_css(file_path: Path) -> str:
    with open(file_path) as f:
        css = f"<style>{f.read()}</style>"
    return css


@st.cache_resource
def get_ranker(
    client: LanguageModel,
    chat: Chat,
    group_a: list[str],
    group_b: list[str],
    v_pct: float,
    min_p: float,
) -> Ranker:
    return Ranker(
        model=client,
        chat=chat,
        group_a=group_a,
        group_b=group_b,
        v_pct=v_pct,
        min_p=min_p,
    )


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
    code = VLLMConnection.is_alive(url=url)

    if code == HTTPStatus.OK:
        credentials = {"base_url": url, "token": api_key}

        SESSION.server = VLLMConnection(**credentials)
        SESSION.credentials = credentials
        SESSION.selected_model = None
        st.toast(f"vLLM connection - {url} - {code}")
    else:
        st.toast(f"{url} - {code}")


def assign_model() -> None:
    selected_model = SESSION.model_choice
    client = SESSION.server.open(selected_model)
    if not client.max_logprobs_known:
        st.toast(f"Could not read max_logprobs for '{selected_model}', using {DEFAULT_MAX_LOGPROBS}.")

    SESSION.client = client
    SESSION.selected_model = selected_model

    st.toast(f"Assigned model: {selected_model}")
    time.sleep(DELAY)


def sidebar_connection() -> None:
    with st.form("connection_form"):
        url = st.text_input(
            label="URL",
            value=SESSION.credentials.get("base_url"),
            placeholder="http://localhost:8000",
        )
        api_key = st.text_input(
            label="API Key",
            value=SESSION.credentials.get("token"),
            placeholder="EMPTY",
            type="password",
        )

        if st.form_submit_button("Connect") and url:
            connect(url=url, api_key=api_key)

    if connection_guards().is_connected.ok:
        models = SESSION.server.models()
        index = models.index(model) if (model := SESSION.selected_model) else None

        st.selectbox(
            label="Select a model",
            options=models,
            index=index,
            on_change=assign_model,
            key=Session.model_choice.key,
        )


def draft_poll() -> PulseConfig:
    """The poll as currently entered in the form."""
    return PulseConfig(
        name=SESSION.poll_name,
        persona=SESSION.persona,
        docs=SESSION.selected_persona,
        question=SESSION.question,
        answer=SESSION.answer,
        completions=SESSION.selected_completions,
    )


def is_update() -> bool:
    """The form edits the selected, saved poll rather than a new one."""
    return bool(SESSION.selected_task) and SESSION.poll_name == SESSION.selected_task


def connection_guards() -> ConnectionGuards:
    """This session's connection, as the domain guards see it."""
    client = SESSION.client
    return ConnectionGuards(
        connected=bool(SESSION.server),
        model=SESSION.selected_model,
        chat_template=client is not None and client.has_chat_template,
    )


def get_chat() -> Chat | None:
    poll = draft_poll()
    if not (clause := PollGuards(poll).has_prompts).ok:
        st.toast(clause.msg)
        return None

    return poll_chat(poll)
