import math

import pandas as pd
import pytest

from pulse.ports import ChatModel, ModelServer
from pulse.bootstrap import bootstrap
from pulse.domain.poll import PulseConfig
from pulse.domain.types import Token, Sequence
from pulse.services.facade import Blocked
from pulse.adapters.file_storage import FileStorage


class FakeModel(ChatModel):
    model = "fake"
    max_logprobs = 10
    max_logprobs_known = True
    has_chat_template = True

    def score(self, chat, continuations):
        return [Sequence(tokens=[Token(token=f" {w}", logprob=-1.0, rank=1) for w in c.split()]) for c in continuations]

    def next_tokens(self, chat, prefixes, k):
        return [[Token(token=str(i), logprob=math.log(p), rank=i) for i, p in enumerate((0.5, 0.3, 0.2), 1)]] * len(
            prefixes
        )

    def run_poll(self, poll, docs, completions):
        return [dict.fromkeys(completions["alias"], 0.5)]


class FakeServer(ModelServer):
    def models(self):
        return ["fake"]

    def open(self, model):
        return FakeModel()


POLL = PulseConfig(name="p", persona="You are a voter.", question="Who?", answer="I vote", completions="c")


def test_use_cases_are_guarded(tmp_path):
    pulse = bootstrap(storage=FileStorage(root=tmp_path), connect=lambda url, token: FakeServer())
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
