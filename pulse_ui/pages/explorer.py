import streamlit as st

from streamlit.logger import get_logger

from pulse_ui.pages.state import st_md, draft_poll, sidebar_connection
from pulse_ui.utils.tools import Placeholder as ph
from pulse_ui.pages.session import form, pulse, explorer

st.header("PULSE - Polling Using LLM-based Sentiment Extraction")

logger = get_logger(__name__)


with st.sidebar:
    sidebar_connection()


def prompt_container():
    max_logprobs = pulse().model.max_logprobs

    with st.form("prompt_form"):
        st.text_input(
            label="Persona",
            placeholder=ph.persona,
            **form.bind("persona"),
        )
        st.text_input(
            label="Question",
            placeholder=ph.question,
            **form.bind("question"),
        )
        answer_col, comp_col = st.columns(2)

        answer_col.text_input(
            label="Answer",
            placeholder=ph.answer,
            **form.bind("answer"),
        )
        comp_col.text_input(
            label="completion",
            placeholder=ph.completion,
            key=explorer.key("completion"),
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


def sample(logprobs: int) -> None:
    poll = draft_poll()
    if failure := pulse().check_explore(poll):
        st.toast(failure.msg)
        explorer.sample_df = None
        return

    logger.info(poll)
    explorer.sample_df = pulse().next_tokens(poll, prefix=explorer.completion or "", k=logprobs)


_, column, _ = st.columns((0.2, 0.2, 0.2))
column.subheader("Explorer")

if failure := pulse().check_connection():
    st.error(failure.msg)
    st.stop()

st_md(text="Prompt", container=column)
with column:
    prompt_container()

st_md("Next token", container=column)
next_container = column.container(border=True, height=165)
sample_df = explorer.sample_df
if sample_df is not None:
    next_container.dataframe(sample_df, height=415)

st.write("")
