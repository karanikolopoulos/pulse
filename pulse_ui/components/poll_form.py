"""The poll being edited: choose a saved poll, edit its prompts, save, run or delete it."""

from typing import Literal
from dataclasses import replace

import streamlit as st

from pulse_ui.session import form, pulse
from pulse.domain.poll import Status, PulseConfig
from pulse_ui.components import tables
from pulse_ui.utils.tools import Placeholder as ph
from pulse_ui.utils.notify import notify


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


def selector() -> None:
    """Saved polls, and the save / run / delete buttons."""
    t_col, btn_col = st.columns((0.4, 0.6), vertical_alignment="bottom")
    t_col.selectbox(
        label="Polls",
        options=pulse().polls(),
        index=None,
        placeholder="Select a poll",
        **form.bind("selected_task"),
        on_change=_load,
    )

    disabled = not form.selected_task
    with btn_col.container(horizontal=True):
        if st.button("Save", icon=":material/save:", key="save_task", width="stretch"):
            save()
        if st.button("Run", icon=":material/play_arrow:", disabled=disabled, key="run_task", width="stretch"):
            run()
        if st.button("Delete", icon=":material/delete:", disabled=disabled, key="delete_task", width="stretch"):
            pulse().delete_poll(form.selected_task)
            st.rerun()


@st.fragment
def prompts() -> None:
    """Persona (optionally a template filled from a personas table), question and answer."""
    st.markdown("**Prompts**")
    with st.container(border=True):
        st.text_input(label="Persona", placeholder="You are {{ persona }}.", **form.bind("persona"))
        with st.expander("Batch personas"):
            tables.picker("personas")
        st.text_input(label="Question", placeholder=ph.question, **form.bind("question"))
        st.text_input(label="Answer", placeholder=ph.answer, **form.bind("answer"))


def save() -> None:
    """Update the selected poll, or save the form as a new poll."""
    poll = draft_poll()
    if failure := pulse().check_save(poll):
        st.toast(failure.msg)
        return

    if not is_update():
        _save_as()
    elif _store(poll=poll, op="updated"):
        st.rerun()


def run() -> None:
    """Run the selected saved poll with the chosen model."""
    if clause := pulse().check_run(form.selected_task):
        st.toast(clause.msg)
        return

    _run(name=form.selected_task)


def _reset() -> None:
    for field in ("poll_name", "persona", "question", "answer", "selected_persona", "selected_completions"):
        setattr(form, field, None)


def _load() -> None:
    """Fill the form with the selected saved poll."""
    if not (task_name := form.selected_task):
        _reset()
        return

    poll, problems = pulse().load_poll(task_name)
    form.poll_name = poll.name
    form.persona = poll.persona
    form.question = poll.question
    form.answer = poll.answer
    form.selected_persona = poll.docs
    form.selected_completions = poll.completions

    for clause in problems:
        st.toast(clause.msg)


def _store(poll: PulseConfig, op: Literal["saved", "updated"]) -> bool:
    """Add or update a poll in storage; a new poll becomes the selected one."""
    status = pulse().save_poll(poll)
    if status != Status.OK:
        st.toast(f"Save failed: {status}")
        return False

    notify(f"Poll {op}.")
    if op == "saved":
        form.poll_name = poll.name
        form.selected_task = poll.name  # a widget key: settable here because this runs as a callback
    return True


@st.dialog("Save poll", width="small")
def _save_as() -> None:
    name = st.text_input(label="Poll name", key=form.key("new_poll_name"))
    poll = replace(draft_poll(), name=name)

    if st.button("Save", disabled=not name.strip(), on_click=_store, args=(poll, "saved")):
        if form.poll_name == name:  # saved by the callback
            st.rerun()


@st.dialog("Run poll", width="large")
def _run(name: str) -> None:
    with st.spinner(f"Running {name} poll"):
        st.info(form.question)
        results = pulse().run_poll(name)

    notify(f"Run completed for poll '{name}' with model '{results.model}'")
    st.rerun()
