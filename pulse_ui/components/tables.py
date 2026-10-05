"""Personas and completions tables: a picker with create, view/edit and delete dialogs."""

import pandas as pd
import streamlit as st

from streamlit.runtime.uploaded_file_manager import UploadedFile

from pulse_ui.session import form, pulse
from pulse.domain.poll import Status, TableKind
from pulse_ui.utils.notify import notify

# the form field holding each kind's selected table
FIELD: dict[TableKind, str] = {"personas": "selected_persona", "completions": "selected_completions"}


def picker(kind: TableKind) -> None:
    """Select a table of this kind for the poll, or create, edit or delete one."""
    create, edit, delete = _DIALOGS[kind]
    field = FIELD[kind]

    st.selectbox(label=f"Select {kind}", options=pulse().tables(kind=kind), index=None, **form.bind(field))

    selected = getattr(form, field)
    with st.container(horizontal=True):
        st.button("Create", icon=":material/add:", on_click=create, key=f"{kind}_create", width="stretch")
        st.button(
            "View/edit",
            icon=":material/edit:",
            on_click=edit,
            args=(selected,),
            disabled=not selected,
            key=f"{kind}_edit",
            width="stretch",
        )
        st.button(
            "Delete",
            icon=":material/delete:",
            on_click=delete,
            args=(selected,),
            disabled=not selected,
            key=f"{kind}_delete",
            width="stretch",
        )


@st.fragment
def completions() -> None:
    picker("completions")


def _create(kind: TableKind, default_columns: list[str] | None = None, column_placeholder: str = "") -> None:
    name = st.text_input("Name")
    uploaded_file = st.file_uploader(f"Upload {kind} file", type=("csv", "json"))

    columns = None
    if uploaded_file:
        try:
            df = _read_upload(uploaded_file)
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
        _validate_new(name=name, kind=kind, df=changed, columns=columns, has_default_columns=bool(default_columns))

        status = pulse().add_table(kind=kind, name=name, df=changed.reset_index(drop=True))
        if status == Status.OK:
            notify(f"Created {kind} '{name}'")
            st.rerun()
        else:
            st.error(f"Failed with status: {status.value}")


def _edit(kind: TableKind, name: str) -> None:
    st.markdown(f"**{name.capitalize()}**")
    changed = st.data_editor(pulse().table(kind=kind, name=name), num_rows="dynamic")

    if st.button("Save"):
        if (status := pulse().update_table(kind=kind, name=name, df=changed)) != Status.OK:
            st.error(f"Failed with status: {status.value}")
            st.stop()

        notify(f"Updated {kind} '{name}'")
        st.rerun()


def _delete(kind: TableKind, name: str) -> None:
    st.error(f"Confirm: Delete {kind} '{name}'?")
    if st.button("Confirm"):
        pulse().delete_table(kind=kind, name=name)
        setattr(form, FIELD[kind], None)
        notify(f"Deleted {kind} '{name}'")
        st.rerun()


def _read_upload(uploaded_file: UploadedFile) -> pd.DataFrame:
    if uploaded_file.name.endswith(".json"):
        return pd.read_json(uploaded_file)
    if uploaded_file.name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    raise ValueError("Unsupported file type. Use CSV or JSON.")


def _validate_new(
    name: str,
    kind: TableKind,
    df: pd.DataFrame,
    columns: list[str] | None,
    has_default_columns: bool,
) -> None:
    """Stops the dialog with an error if the new table can't be created."""
    if not name:
        st.error(f"Please provide a name for the {kind}.")
        st.stop()

    if name in pulse().tables(kind=kind):
        st.error(f"{kind.capitalize()} '{name}' already exists.")
        st.stop()

    if not has_default_columns and columns and not any(columns):
        st.error("Please provide at least one column name.")
        st.stop()

    if df.empty:
        st.error("Empty dataframe. Add some rows.")
        st.stop()


@st.dialog("Create persona table", width="large")
def _create_personas() -> None:
    _create(kind="personas", column_placeholder="demographic, group, persona")


@st.dialog("Edit persona table", width="large")
def _edit_personas(name: str) -> None:
    _edit(kind="personas", name=name)


@st.dialog("Delete persona table", width="small")
def _delete_personas(name: str) -> None:
    _delete(kind="personas", name=name)


@st.dialog("Create completion table", width="large")
def _create_completions() -> None:
    _create(kind="completions", default_columns=["A", "B", "alias"], column_placeholder="A, B, alias")


@st.dialog("Edit completion table", width="large")
def _edit_completions(name: str) -> None:
    _edit(kind="completions", name=name)


@st.dialog("Delete completion table", width="small")
def _delete_completions(name: str) -> None:
    _delete(kind="completions", name=name)


_DIALOGS = {
    "personas": (_create_personas, _edit_personas, _delete_personas),
    "completions": (_create_completions, _edit_completions, _delete_completions),
}
