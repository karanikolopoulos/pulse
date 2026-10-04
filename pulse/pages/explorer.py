import pandas as pd
import streamlit as st

from streamlit.logger import get_logger

from pulse.pages.state import (
    st_md,
    get_chat,
    connection_guards,
    sidebar_connection,
)
from pulse.utils.tools import Placeholder as ph
from pulse.domain.types import Chat
from pulse.domain.guards import first_error
from pulse.pages.session import SESSION, Session
from pulse.services.explorer import next_token_table

st.header("PULSE - Polling Using LLM-based Sentiment Extraction")

logger = get_logger(__name__)


with st.sidebar:
    sidebar_connection()


def prompt_container():
    max_logprobs = SESSION.client.max_logprobs

    with st.form("prompt_form"):
        st.text_input(
            label="Persona",
            placeholder=ph.persona,
            **Session.persona.widget(),
        )
        st.text_input(
            label="Question",
            placeholder=ph.question,
            **Session.question.widget(),
        )
        answer_col, comp_col = st.columns(2)

        answer_col.text_input(
            label="Answer",
            placeholder=ph.answer,
            **Session.answer.widget(),
        )
        comp_col.text_input(
            label="completion",
            placeholder=ph.completion,
            key=Session.completion.key,
            help="Mind the leading whitespace!",
        )

        inp_col, btn_col = st.columns(2, vertical_alignment="bottom")

        logprobs = inp_col.number_input(
            label=f"Num. of tokens (max: {max_logprobs})",
            min_value=5,
            max_value=max_logprobs,
            value=min(128, max_logprobs),
            step=1,
        )

        btn_col.form_submit_button(
            label="Sample next token",  # 🕵️‍♂️
            on_click=sample,
            args=(logprobs,),
            use_container_width=True,
        )


@st.cache_data
def get_next_tokens(model_id: str, context: Chat, continuation: str, k: int) -> pd.DataFrame:
    """`model_id` keys the cache, so switching models doesn't return another model's tokens."""
    return next_token_table(model=SESSION.client, chat=context, prefix=continuation, k=k)


def sample(logprobs: int) -> None:
    if chat := get_chat():
        logger.info(chat)
        SESSION.sample_df = get_next_tokens(
            model_id=SESSION.client.model,
            context=chat,
            continuation=SESSION.completion,
            k=logprobs,
        )
    else:
        SESSION.sample_df = None


_, column, _ = st.columns((0.2, 0.2, 0.2))
column.subheader("Explorer")

if failure := first_error(connection_guards()):
    st.error(failure.msg)
    st.stop()

st_md(text="Prompt", container=column)
with column:
    prompt_container()

st_md("Next token", container=column)
next_container = column.container(border=True, height=165)
sample_df = SESSION.sample_df
if sample_df is not None:
    next_container.dataframe(sample_df, height=415)

st.write("")
