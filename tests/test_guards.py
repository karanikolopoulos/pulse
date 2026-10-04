import inspect

from pulse.domain.poll import PulseConfig
from pulse.domain.guards import (
    Clause,
    Guards,
    RunGuards,
    PollGuards,
    TableGuards,
    RankingGuards,
    ConnectionGuards,
)

POLL = PulseConfig(persona="You are a voter.", question="Who?", answer="I will vote for", completions="elections")
BATCH = PulseConfig(**{**vars(POLL), "persona": "You are {{ persona }}.", "docs": "demographics"})

# one passing instance of every guard group
GROUPS = [
    ConnectionGuards(connected=True, model="Qwen/Qwen3.8-27B", chat_template=True),
    PollGuards(POLL),
    RunGuards(POLL, selected_poll="poll"),
    RankingGuards(POLL),
    TableGuards(BATCH, personas=["demographics"], completions=["elections"]),
]


def codes(clauses: list[Clause]) -> list[str]:
    return [clause.code for clause in clauses]


def test_every_clause_is_checked():
    assert {type(group) for group in GROUPS} == set(Guards.__subclasses__())

    for group in GROUPS:
        assert group.passes
        properties = [
            name
            for name, attr in inspect.getmembers(type(group))
            if isinstance(attr, property) and inspect.signature(attr.fget).return_annotation is Clause
        ]
        assert {getattr(group, name).code for name in properties} <= set(codes(list(group))), type(group)


def test_poll_guards():
    assert PollGuards(BATCH).passes
    assert codes(PollGuards(PulseConfig(**{**vars(POLL), "docs": "demographics"})).errors) == [
        "PERSONAS_WITHOUT_PLACEHOLDER"
    ]
    assert codes(PollGuards(PulseConfig(**{**vars(BATCH), "docs": None})).errors) == ["PLACEHOLDER_WITHOUT_PERSONAS"]


def test_ranking_needs_a_concrete_persona():
    assert codes(RankingGuards(BATCH).errors) == ["PERSONA_TEMPLATE"]


def test_table_guards():
    assert codes(TableGuards(BATCH, personas=[], completions=["elections"]).errors) == ["PERSONAS_NOT_FOUND"]


def test_connection_guards():
    assert codes(ConnectionGuards(connected=True, model=None, chat_template=False).errors) == [
        "NO_MODEL",
        "NO_CHAT_TEMPLATE",
    ]
