import time

from typing import Literal
from dataclasses import replace

import pandas as pd
import streamlit as st

from streamlit.logger import get_logger
from streamlit.delta_generator import DeltaGenerator
from streamlit.runtime.uploaded_file_manager import UploadedFile

from pulse.domain.poll import Status, TableKind, PulseConfig
from pulse.pages.state import (
    DELAY,
    st_md,
    get_chat,
    is_update,
    draft_poll,
    get_ranker,
    connection_guards,
    sidebar_connection,
)
from pulse.utils.tools import Placeholder as ph, apply_html, get_position_table
from pulse.domain.guards import RunGuards, PollGuards, TableGuards, RankingGuards, first_error
from pulse.pages.session import SESSION, Session, SessionValue
from pulse.pages.shortcuts import activate_shortcuts
from pulse.services.polling import run_poll
from pulse.services.ranking import Ranker

# region CONFIGURATION
logger = get_logger(__name__)
HEADER = "PULSE - Polling Using LLM-based Sentiment Extraction"
HEIGHT = 800
# endregion


# region HELPERS
def _load_dataframe_from_file(uploaded_file: UploadedFile) -> pd.DataFrame:
    """Loads a DataFrame from an uploaded CSV or JSON file."""
    if uploaded_file.name.endswith(".json"):
        return pd.read_json(uploaded_file)
    if uploaded_file.name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    raise ValueError("Unsupported file type. Use CSV or JSON.")


def _validate_entity_creation(
    name: str,
    kind: TableKind,
    changed_df: pd.DataFrame,
    columns: list[str] | None,
    has_default_columns: bool,
) -> None:
    """Validate entity creation inputs and raise errors if invalid."""
    if not name:
        st.error(f"Please provide a name for the {kind}.")
        st.stop()

    if name in SESSION.storage.table_names(kind=kind):
        st.error(f"{kind.capitalize()} '{name}' already exists.")
        st.stop()

    if not has_default_columns and columns and not any(columns):
        st.error("Please provide at least one column name.")
        st.stop()

    if changed_df.empty:
        st.error("Empty dataframe. Add some rows.")
        st.stop()


# endregion


# region CRUD
def create_entity(
    kind: TableKind,
    default_columns: list[str] | None = None,
    column_placeholder: str = "",
) -> None:
    """Generic function to create new personas or completions."""
    name = st.text_input("Name")
    uploaded_file = st.file_uploader(f"Upload {kind} file", type=("csv", "json"))

    # Load or create DataFrame
    if uploaded_file:
        try:
            df = _load_dataframe_from_file(uploaded_file)
        except Exception as e:
            st.error(f"Failed to read file: {e}")
            st.stop()
    else:
        if default_columns:
            columns = default_columns
        else:
            columns_input = st.text_input("Columns (comma-separated)", placeholder=column_placeholder)
            columns = list(map(str.strip, columns_input.split(",")))
        df = pd.DataFrame(columns=columns)

    changed = st.data_editor(df, num_rows="dynamic")

    if st.button("Save"):
        _validate_entity_creation(
            name,
            kind,
            changed,
            columns if not uploaded_file else None,
            bool(default_columns),
        )

        status = SESSION.storage.add_table(kind=kind, name=name, df=changed.reset_index(drop=True))
        if status == Status.OK:
            st.toast(f"Created {kind} '{name}'")
            time.sleep(DELAY)
            st.rerun()
        else:
            st.error(f"Failed with status: {status.value}")


def edit_entity(kind: TableKind, selected_name: str) -> None:
    """Generic function to edit existing personas or completions."""
    st.write(selected_name.capitalize())
    changed = st.data_editor(SESSION.storage.get_table(kind=kind, name=selected_name), num_rows="dynamic")

    if st.button("Save"):
        if (status := SESSION.storage.update_table(kind=kind, name=selected_name, df=changed)) != Status.OK:
            st.error(f"Failed with status: {status.value}")
            st.stop()

        st.toast(f"Updated {kind} '{selected_name}'")
        time.sleep(DELAY)
        st.rerun()


def delete_entity(kind: TableKind, selected_name: str, selection: SessionValue[str | None]) -> None:
    """Generic function to delete personas or completions."""
    st.error(f"Confirm: Delete {kind} '{selected_name}'?")
    if st.button("Confirm"):
        SESSION.storage.delete_table(kind=kind, name=selected_name)
        selection.set(None)
        st.toast(f"Deleted {kind} '{selected_name}'")
        time.sleep(DELAY)
        st.rerun()


@st.dialog("Create personas", width="large")
def new_persona() -> None:
    create_entity(kind="personas", default_columns=None, column_placeholder="demographic, group, persona")


@st.dialog("Edit Persona", width="large")
def edit_persona(selected_persona: str) -> None:
    edit_entity(kind="personas", selected_name=selected_persona)


@st.dialog("Delete Persona", width="small")
def delete_persona(selected_persona: str) -> None:
    delete_entity(kind="personas", selected_name=selected_persona, selection=Session.selected_persona)


@st.dialog("Create completions", width="large")
def create_completions() -> None:
    create_entity(kind="completions", default_columns=["A", "B", "alias"], column_placeholder="A, B, alias")


@st.dialog("Edit Completions File", width="large")
def edit_completions(selected_completions: str) -> None:
    edit_entity(kind="completions", selected_name=selected_completions)


@st.dialog("Delete Completions", width="small")
def delete_completions(selected_completions: str) -> None:
    delete_entity(kind="completions", selected_name=selected_completions, selection=Session.selected_completions)


# endregion


# region TASK OPERATIONS
def reset_config() -> None:
    """Clear the form for a new poll."""
    for value in (
        Session.poll_name,
        Session.persona,
        Session.question,
        Session.answer,
        Session.selected_persona,
        Session.selected_completions,
    ):
        value.set(None)


def load_selected_task() -> None:
    """Fill the form with the selected saved poll."""
    if not (task_name := SESSION.selected_task):
        reset_config()
        return

    poll = SESSION.storage.get_poll(task_name)
    tables = TableGuards(
        poll=poll,
        personas=SESSION.storage.table_names(kind="personas"),
        completions=SESSION.storage.table_names(kind="completions"),
    )

    SESSION.poll_name = poll.name
    SESSION.persona = poll.persona
    SESSION.question = poll.question
    SESSION.answer = poll.answer
    SESSION.selected_persona = poll.docs if tables.personas_exist.ok else None
    SESSION.selected_completions = poll.completions if tables.completions_exist.ok else None

    for clause in tables.errors:
        st.toast(clause.msg)


def store_poll(poll: PulseConfig, op: Literal["saved", "updated"]) -> bool:
    """Add or update a poll in storage; a new poll becomes the selected one."""
    status = SESSION.storage.save_poll(poll)
    if status != Status.OK:
        st.toast(f"Save failed: {status}")
        return False

    st.toast(f"Task {op} successfully.")
    if op == "saved":
        SESSION.poll_name = poll.name
        SESSION.selected_task = poll.name  # widget key: only settable before the selectbox renders, i.e. in a callback
    return True


def pre_save() -> None:
    """Validate the poll before update/save."""
    poll = draft_poll()
    if failure := first_error(PollGuards(poll)):
        st.toast(failure.msg)
        return

    if not is_update():
        save()
    elif store_poll(poll=poll, op="updated"):
        time.sleep(DELAY)
        st.rerun()


@st.dialog("Save Poll", width="small")
def save() -> None:
    """Save the poll under a new name."""
    name = st.text_input(label="Poll Name", key=Session.new_poll_name.key)
    poll = replace(draft_poll(), name=name)

    if st.button("Save", disabled=not name.strip(), on_click=store_poll, args=(poll, "saved")):
        if SESSION.poll_name == name:  # saved by the callback
            time.sleep(DELAY)
            st.rerun()


def pre_run() -> None:
    """Validate and run the selected poll."""
    if clause := first_error(connection_guards(), RunGuards(draft_poll(), selected_poll=SESSION.selected_task)):
        st.toast(clause.msg)
        return

    run_task(name=SESSION.selected_task)


@st.dialog("Evaluation", width="large")
def run_task(name: str) -> None:
    """Execute evaluation task with the selected model."""
    with st.spinner(f"Running {name} poll"):
        st.info(SESSION.question)
        results = run_poll(runner=SESSION.client, storage=SESSION.storage, poll_name=name)

    st.success(f"Run completed for Poll '{name}' with model '{results.model}'")
    time.sleep(2)
    st.rerun()


# endregion


# region RANKING ANALYSIS
def step(p_bar, value: int | float, text: str, delay: float = 0.0) -> None:
    """Update progress bar with value and text."""
    p_bar.progress(value=value, text=text)
    time.sleep(delay)


def pre_rank(container: DeltaGenerator) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """Validate and prepare data for ranking completions."""
    if clause := first_error(connection_guards(), RankingGuards(draft_poll())):
        SESSION.rankings = None
        container.warning(clause.msg)
        return None

    chat, completions = get_chat(), SESSION.selected_completions
    completions = SESSION.storage.get_table(kind="completions", name=completions).to_dict(orient="list")

    # setup ranker for vllm client and parameters - cached
    ranker = get_ranker(
        client=SESSION.client,
        chat=chat,
        group_a=completions["A"],
        group_b=completions["B"],
        v_pct=st.secrets.V_PCT,
        min_p=st.secrets.MIN_P,
    )

    if rankings := rank(container=container, ranker=ranker):
        SESSION.rankings = rankings
        # st.rerun()
    else:
        SESSION.rankings = None


def rank(container: DeltaGenerator, ranker: Ranker, delay: float = 0.5) -> tuple[pd.DataFrame, pd.DataFrame]:
    with container:
        pbar = st.progress(0, text="")
        step(p_bar=pbar, value=0.0, text="Ranking completions.", delay=delay)

        step(p_bar=pbar, value=0.2, text="Scoring completions.", delay=delay)
        _ = ranker.sequences  # trigger cached property
        step(p_bar=pbar, value=0.5, text="Calculating elbows.", delay=delay)
        _ = ranker.elbows  # trigger cached property
        step(p_bar=pbar, value=0.8, text="Gathering metrics.", delay=delay)
        rankings = ranker.rankings  # trigger cached property

    step(p_bar=pbar, value=1.0, text="Rankings complete.", delay=delay)
    pbar.empty()

    return rankings


# endregion


# region UI CONTAINERS
def select_container() -> None:
    """Render poll selection and action buttons (save/run/delete)."""
    t_col, btn_col = st.columns((0.4, 0.6), vertical_alignment="bottom")
    t_col.selectbox(
        label="Polls",
        options=SESSION.storage.poll_names(),
        index=None,
        placeholder="Select a Poll",
        **Session.selected_task.widget(),
        on_change=load_selected_task,
    )

    disabled = not SESSION.selected_task
    save_col, run_col, del_col = btn_col.columns(3)
    if save_col.button(label="Save", use_container_width=True, key="save_task"):
        pre_save()
    if run_col.button(label="Run", use_container_width=True, disabled=disabled, key="run_task"):
        pre_run()
    if del_col.button(label="Delete", use_container_width=True, disabled=disabled, key="delete_task"):
        SESSION.storage.delete_poll(SESSION.selected_task)
        st.rerun()


@st.fragment
def prompt_container() -> None:
    """Render prompt configuration UI (persona, question, answer)."""
    st_md(text="Prompts")
    with st.container(border=True):
        st.text_input(
            label="Persona",
            placeholder="You are {{ persona }}.",
            **Session.persona.widget(),
        )
        with st.expander("Batch personas"):
            batch_container()
        st.text_input(
            label="Question",
            placeholder=ph.question,
            **Session.question.widget(),
        )
        st.text_input(
            label="Answer",
            placeholder=ph.answer,
            **Session.answer.widget(),
        )


def batch_container() -> None:
    """Render batch personas selection and management UI."""
    st.selectbox(
        label="Select personas",
        options=SESSION.storage.table_names(kind="personas"),
        index=None,
        **Session.selected_persona.widget(),
    )
    new_col, edit_col, del_col = st.columns(3)
    new_col.button(
        label="Create",
        on_click=new_persona,
        key="new_persona",
        use_container_width=True,
    )

    selected_persona = SESSION.selected_persona
    edit_col.button(
        label="View/Edit",
        on_click=edit_persona,
        args=(selected_persona,),
        disabled=not selected_persona,
        key="edit_persona",
        use_container_width=True,
    )
    del_col.button(
        label="Delete",
        on_click=delete_persona,
        args=(selected_persona,),
        disabled=not selected_persona,
        key="delete_persona",
        use_container_width=True,
    )


@st.fragment
def completions_container() -> None:
    """Render completions selection and management UI."""
    st.selectbox(
        label="Select completions",
        options=SESSION.storage.table_names(kind="completions"),
        index=None,
        **Session.selected_completions.widget(),
    )

    new_col, edit_col, del_col = st.columns(3)
    new_col.button("Create", on_click=create_completions, use_container_width=True)

    selected_completions = SESSION.selected_completions
    edit_col.button(
        label="View/Edit",
        on_click=edit_completions,
        args=(selected_completions,),
        disabled=not selected_completions,
        use_container_width=True,
    )
    del_col.button(
        label="Delete",
        on_click=delete_completions,
        args=(selected_completions,),
        disabled=not selected_completions,
        use_container_width=True,
    )


def analysis_container(parent: DeltaGenerator) -> None:
    """Render analysis tables for ranked completions (Side A and Side B)."""
    if rankings := SESSION.rankings:
        A_df, B_df = rankings
        # Group A
        A_pos = get_position_table(rankings=A_df)
        st_md(text="Side A", container=parent, font_size="18px", **{"text-align": "center"})
        with parent.container(border=False, height=350):
            st.table(apply_html(styler=A_pos, cell_text_color="white"))
        # Group B
        B_pos = get_position_table(rankings=B_df)
        st_md(text="Side B", container=parent, font_size="18px", **{"text-align": "center"})
        with parent.container(border=False, height=350):
            st.table(apply_html(styler=B_pos, cell_text_color="white"))


# endregion


def raise_rank_flag():
    SESSION.rank_flag = True
    st.rerun()


with st.sidebar:
    sidebar_connection()
    query_params = st.query_params
    if query_params.get("debug") == "True":
        st.write(draft_poll())

st.header(HEADER)

task_col, comp_col = st.columns(2)
task_col.markdown("#### Create a Poll")
comp_col.markdown("#### Completion Analysis")
task_cont = task_col.container(border=True, height=HEIGHT)
comp_cont = comp_col.container(border=True, height=HEIGHT)

with task_cont:
    select_container()
    prompt_container()  # fragment
with task_cont.container(border=True):
    completions_container()  # fragment

if SESSION.rank_flag:
    SESSION.rank_flag = False
    pre_rank(container=comp_cont)

analysis_container(parent=comp_cont)

activate_shortcuts(
    fn_map={
        "save": pre_save,
        "rank": raise_rank_flag,
        "run": pre_run,
    }
)
_, cap = comp_col.columns((2.75, 1))
cap.caption("Press Ctrl+K to open the legend")
