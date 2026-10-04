"""What the UI keeps between Streamlit reruns: this session's `Pulse`, and screen state grouped by concern.

    form.persona = "You are a voter."          # st.session_state["PollForm.persona"]
    st.text_input("Persona", **form.bind("persona"))
    pulse().check_run(form.selected_task)

State only: pages decide what to do with it, the facade does it.
"""

import inspect

from typing import Any, ClassVar

import pandas as pd
import streamlit as st

from pulse.bootstrap import bootstrap
from pulse.application import Pulse, Ranker

_PULSE = "pulse"


def pulse() -> Pulse:
    """This browser session's `Pulse`, built by `bootstrap()` on first use."""
    if _PULSE not in st.session_state:
        st.session_state[_PULSE] = bootstrap()
    return st.session_state[_PULSE]


class SessionGroup:
    """Fields declared like a dataclass; each lives in `st.session_state` under "<Group>.<field>".

    Reading a field that was never set returns its declared default. A widget's field can only be
    set before the widget renders in the current run, e.g. in a callback.
    """

    _defaults: ClassVar[dict[str, Any]]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        cls._defaults = {}
        for name in inspect.get_annotations(cls):
            cls._defaults[name] = cls.__dict__.get(name)
            if name in cls.__dict__:
                delattr(cls, name)  # reads fall through to __getattr__

    def key(self, field: str) -> str:
        if field not in self._defaults:
            raise AttributeError(f"{type(self).__name__} has no field {field!r}")
        return f"{type(self).__name__}.{field}"

    def bind(self, field: str) -> dict[str, str]:
        """Widget kwargs: the widget edits this field and keeps its value when switching pages."""
        return {"key": self.key(field), "persist_state": "session"}

    def __getattr__(self, field: str) -> Any:
        return st.session_state.get(self.key(field), self._defaults[field])

    def __setattr__(self, field: str, value: Any) -> None:
        st.session_state[self.key(field)] = value

    def __delattr__(self, field: str) -> None:
        st.session_state.pop(self.key(field), None)


class ConnectionForm(SessionGroup):
    credentials: dict[str, str] | None = None  # last connection form values
    model_choice: str | None = None  # model selectbox


class PollForm(SessionGroup):
    """The poll being edited; its fields are widgets shared by the polling and explorer pages."""

    selected_task: str | None = None  # polls selectbox
    poll_name: str | None = None  # saved poll the form edits
    new_poll_name: str | None = None  # save dialog input
    persona: str | None = None
    question: str | None = None
    answer: str | None = None
    selected_persona: str | None = None
    selected_completions: str | None = None


class Analysis(SessionGroup):
    rank_flag: bool = True
    ranker: Ranker | None = None  # kept so ranking the same poll again is instant
    rankings: tuple[pd.DataFrame, pd.DataFrame] | None = None


class Explorer(SessionGroup):
    completion: str | None = None
    sample_df: pd.DataFrame | None = None


class ResultsView(SessionGroup):
    task: str | None = None
    columns: list[str] | None = None  # completions multiselect
    fig_x: float | None = None
    fig_y: float | None = None


connection = ConnectionForm()
form = PollForm()
analysis = Analysis()
explorer = Explorer()
results_view = ResultsView()
