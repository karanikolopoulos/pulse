"""Typed access to `st.session_state`: everything PULSE keeps between reruns, in one place."""

from collections.abc import Callable

import pandas as pd
import streamlit as st

from pulse.ports import ChatModel, ModelServer
from pulse.utils.paths import DATA
from pulse.adapters.file_storage import FileStorage


class SessionValue[T]:
    """One `st.session_state` entry, keyed by its attribute name on `Session`.

    Reads return `default` (or a value built once by `factory`) until the entry is set.
    `Session.<name>.key` is the key to give the widget that edits it, or `**Session.<name>.widget()`
    to also keep its value across pages. A widget's entry can only be set before the widget renders
    in the current run, e.g. in a callback.
    """

    def __init__(self, default: T | None = None, factory: Callable[[], T] | None = None):
        self.default = default
        self.factory = factory

    def __set_name__(self, owner: type, name: str) -> None:
        self.key = name

    def get(self) -> T:
        if self.key not in st.session_state and self.factory is not None:
            st.session_state[self.key] = self.factory()
        return st.session_state.get(self.key, self.default)

    def widget(self) -> dict[str, str]:
        """Widget kwargs: bind the widget to this entry and keep its value when switching pages."""
        return {"key": self.key, "persist_state": "session"}

    def set(self, value: T) -> None:
        st.session_state[self.key] = value

    def clear(self) -> None:
        st.session_state.pop(self.key, None)

    def __get__(self, obj: object, objtype: type | None = None) -> T:
        return self if obj is None else self.get()

    def __set__(self, obj: object, value: T) -> None:
        self.set(value)

    def __delete__(self, obj: object) -> None:
        self.clear()


class Session:
    # storage and model server
    storage = SessionValue[FileStorage](factory=lambda: FileStorage(root=DATA))
    credentials = SessionValue[dict[str, str]](factory=dict)  # last connection form values
    server = SessionValue[ModelServer | None]()
    model_choice = SessionValue[str | None]()  # model selectbox
    selected_model = SessionValue[str | None]()
    client = SessionValue[ChatModel | None]()

    # poll form; its fields are the widgets, shared by the polling and explorer pages
    selected_task = SessionValue[str | None]()  # polls selectbox
    poll_name = SessionValue[str | None]()  # saved poll the form edits
    new_poll_name = SessionValue[str | None]()  # save dialog input
    persona = SessionValue[str | None]()
    question = SessionValue[str | None]()
    answer = SessionValue[str | None]()
    selected_persona = SessionValue[str | None]()
    selected_completions = SessionValue[str | None]()

    # completion analysis
    rank_flag = SessionValue[bool](default=True)
    rankings = SessionValue[tuple[pd.DataFrame, pd.DataFrame] | None]()

    # explorer
    completion = SessionValue[str | None]()
    sample_df = SessionValue[pd.DataFrame | None]()

    # results
    results_task = SessionValue[str | None]()
    results_columns = SessionValue[list[str] | None]()  # completions multiselect
    fig_x = SessionValue[float | None]()
    fig_y = SessionValue[float | None]()


SESSION = Session()
