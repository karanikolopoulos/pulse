"""Fast stand-ins for the ports: a model server and model that answer instantly, and storage in memory."""

import math

from copy import copy

import pandas as pd

from pulse.ports import Storage, ChatModel, ModelServer
from pulse.domain.poll import Status, TableKind, PulseConfig, PulseResults, check_table
from pulse.domain.types import Token, Sequence


class FakeModel(ChatModel):
    """Every word is one token with rank 1; every next-token distribution is (0.5, 0.3, 0.2)."""

    model = "fake"
    max_logprobs = 10
    max_logprobs_known = True
    has_chat_template = True

    def score(self, chat, continuations):
        return [Sequence(tokens=[Token(token=f" {w}", logprob=-1.0, rank=1) for w in c.split()]) for c in continuations]

    def next_tokens(self, chat, prefixes, k):
        tokens = [Token(token=str(i), logprob=math.log(p), rank=i) for i, p in enumerate((0.5, 0.3, 0.2), 1)]
        return [tokens] * len(prefixes)

    def run_poll(self, poll, docs, completions):
        return [dict.fromkeys(completions["alias"], 0.5) for _ in docs or [None]]


class FakeServer(ModelServer):
    def models(self):
        return ["fake"]

    def open(self, model):
        return FakeModel()


class InMemoryStorage(Storage):
    """`Storage` kept in dicts: the same rules as `FileStorage`, nothing on disk."""

    def __init__(self) -> None:
        self._polls: dict[str, PulseConfig] = {}
        self._tables: dict[TableKind, dict[str, pd.DataFrame]] = {"personas": {}, "completions": {}}
        self._results: dict[tuple[str, str], PulseResults] = {}

    def poll_names(self) -> list[str]:
        return list(self._polls)

    def get_poll(self, name: str) -> PulseConfig:
        return copy(self._polls[name])

    def save_poll(self, poll: PulseConfig) -> Status:
        if any(poll == saved for saved in self._polls.values()):
            return Status.DUPLICATE_POLL
        self._polls[poll.name] = copy(poll)
        return Status.OK

    def delete_poll(self, name: str) -> Status:
        return Status.OK if self._polls.pop(name, None) else Status.NOT_FOUND

    def table_names(self, kind: TableKind) -> list[str]:
        return list(self._tables[kind])

    def get_table(self, kind: TableKind, name: str) -> pd.DataFrame:
        return self._tables[kind][name]

    def add_table(self, kind: TableKind, name: str, df: pd.DataFrame) -> Status:
        if name in self._tables[kind]:
            return Status.DUPLICATE
        return self._put(kind, name, df)

    def update_table(self, kind: TableKind, name: str, df: pd.DataFrame) -> Status:
        if name not in self._tables[kind]:
            return Status.NOT_FOUND
        return self._put(kind, name, df)

    def delete_table(self, kind: TableKind, name: str) -> Status:
        return Status.OK if self._tables[kind].pop(name, None) is not None else Status.NOT_FOUND

    def results(self) -> list[PulseResults]:
        return list(self._results.values())

    def save_results(self, results: PulseResults) -> None:
        self._results[(results.task, results.model)] = results

    def _put(self, kind: TableKind, name: str, df: pd.DataFrame) -> Status:
        if (status := check_table(kind=kind, df=df)) != Status.OK:
            return status
        self._tables[kind][name] = df.copy()
        return Status.OK
