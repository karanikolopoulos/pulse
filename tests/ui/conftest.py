from pathlib import Path

import pandas as pd
import pytest

from fakes import FakeServer, InMemoryStorage
from streamlit.testing.v1 import AppTest

from pulse.bootstrap import bootstrap
from pulse.application import Pulse
from pulse.domain.poll import PulseConfig

ENTRYPOINT = str(Path(__file__).resolve().parents[2] / "pulse_ui" / "app.py")

QUESTION = "Who will you vote for?"
ANSWER = "I will vote for"


@pytest.fixture
def pulse() -> Pulse:
    """A Pulse connected to the fake server, with two polls, their tables and one saved run."""
    storage = InMemoryStorage()
    storage.add_table(
        kind="completions",
        name="elections",
        df=pd.DataFrame({"A": ["the Democrat", "Biden"], "B": ["the Republican", "Trump"], "alias": ["party", "name"]}),
    )
    storage.add_table(
        kind="personas",
        name="groups",
        df=pd.DataFrame(
            {"group": ["Men", "Women"], "persona": ["a man", "a woman"], "A pct": [0.4, 0.6], "B pct": [0.6, 0.4]}
        ),
    )
    storage.save_poll(
        PulseConfig(
            name="single", persona="You are a voter.", question=QUESTION, answer=ANSWER, completions="elections"
        )
    )
    storage.save_poll(
        PulseConfig(
            name="batch",
            persona="You are {{ persona }}.",
            docs="groups",
            question=QUESTION,
            answer=ANSWER,
            completions="elections",
        )
    )

    app = bootstrap(storage=storage, connect=lambda url, token: FakeServer())
    app.connect("http://fake")
    app.use_model("fake")
    app.run_poll("batch")
    return app


@pytest.fixture
def at(pulse) -> AppTest:
    """The app, opened on the polling page, running on `pulse`."""
    at = AppTest.from_file(ENTRYPOINT, default_timeout=30)
    at.session_state["pulse"] = pulse
    return at.run()
