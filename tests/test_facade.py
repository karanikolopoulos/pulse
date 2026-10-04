import pandas as pd
import pytest

from fakes import FakeServer, InMemoryStorage

from pulse.bootstrap import bootstrap
from pulse.application import Blocked
from pulse.domain.poll import PulseConfig

POLL = PulseConfig(name="p", persona="You are a voter.", question="Who?", answer="I vote", completions="c")


def test_use_cases_are_guarded():
    pulse = bootstrap(storage=InMemoryStorage(), connect=lambda url, token: FakeServer())
    pulse.add_table(kind="completions", name="c", df=pd.DataFrame({"A": ["x"], "B": ["y"], "alias": ["x/y"]}))
    pulse.save_poll(POLL)
    pulse.save_poll(PulseConfig(**{**vars(POLL), "name": "orphan", "answer": "I pick", "completions": "gone"}))

    orphan, problems = pulse.load_poll("orphan")
    assert orphan.completions is None and [c.code for c in problems] == ["COMPLETIONS_NOT_FOUND"]

    assert pulse.check_run("p").code == "NOT_CONNECTED"
    with pytest.raises(Blocked):
        pulse.run_poll("p")

    pulse.connect("http://server")
    pulse.use_model(pulse.models()[0])

    assert pulse.check_run("missing").code == "NO_POLL_SELECTED"
    assert pulse.run_poll("p").metrics == [{"x/y": 0.5}]
    assert len(pulse.results()) == 1

    a, b = pulse.ranker(POLL, v_pct=0.5, min_p=0.99).rankings
    assert list(a.index) == ["x"] and list(b.index) == ["y"]

    assert pulse.check_ranking(PulseConfig(**{**vars(POLL), "persona": "You are {{ persona }}."})).code == (
        "PERSONA_TEMPLATE"
    )
