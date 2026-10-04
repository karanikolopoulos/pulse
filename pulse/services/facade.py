"""Everything the UI can do with PULSE; pages only present it.

Each use case has a check, which returns the first failing guard for the UI to show, and an
action, which runs the same check and raises `Blocked` if it fails.
"""

from dataclasses import replace
from collections.abc import Callable

import pandas as pd

from pulse.ports import Storage, ChatModel, ModelServer
from pulse.domain.poll import Status, TableKind, PulseConfig, PulseResults, poll_chat
from pulse.domain.guards import (
    Clause,
    RunGuards,
    PollGuards,
    TableGuards,
    RankingGuards,
    ConnectionGuards,
    first_error,
)
from pulse.services.polling import run_poll
from pulse.services.ranking import Ranker
from pulse.services.results import PollSummary, summarize
from pulse.services.explorer import next_token_table


class Blocked(Exception):
    """A use case was called although its guards fail."""

    def __init__(self, clause: Clause):
        super().__init__(clause.msg)
        self.clause = clause


class Pulse:
    def __init__(self, storage: Storage, connect: Callable[[str, str | None], ModelServer]):
        """`connect(url, token)` opens a model server, e.g. `VLLMConnection.connect`."""
        self._storage = storage
        self._connect = connect
        self.server: ModelServer | None = None
        self.model: ChatModel | None = None

    # connection
    def connect(self, url: str, token: str | None = None) -> None:
        """Raises ConnectionError if the server doesn't answer."""
        self.server = self._connect(url, token)
        self.model = None

    def models(self) -> list[str]:
        return self.server.models() if self.server else []

    def use_model(self, model: str) -> ChatModel:
        self.model = self.server.open(model)
        return self.model

    @property
    def connection(self) -> ConnectionGuards:
        return ConnectionGuards(
            connected=self.server is not None,
            model=self.model.model if self.model else None,
            chat_template=self.model is not None and self.model.has_chat_template,
        )

    def check_connection(self) -> Clause | None:
        return first_error(self.connection)

    # catalogue: polls, personas and completions tables, results
    def polls(self) -> list[str]:
        return self._storage.poll_names()

    def load_poll(self, name: str) -> tuple[PulseConfig, list[Clause]]:
        """The saved poll, with references to missing tables cleared, and what was missing."""
        poll = self._storage.get_poll(name)
        tables = TableGuards(poll=poll, personas=self.tables("personas"), completions=self.tables("completions"))
        poll = replace(
            poll,
            docs=poll.docs if tables.personas_exist.ok else None,
            completions=poll.completions if tables.completions_exist.ok else None,
        )
        return poll, tables.errors

    def delete_poll(self, name: str) -> Status:
        return self._storage.delete_poll(name)

    def tables(self, kind: TableKind) -> list[str]:
        return self._storage.table_names(kind=kind)

    def table(self, kind: TableKind, name: str) -> pd.DataFrame:
        return self._storage.get_table(kind=kind, name=name)

    def add_table(self, kind: TableKind, name: str, df: pd.DataFrame) -> Status:
        return self._storage.add_table(kind=kind, name=name, df=df)

    def update_table(self, kind: TableKind, name: str, df: pd.DataFrame) -> Status:
        return self._storage.update_table(kind=kind, name=name, df=df)

    def delete_table(self, kind: TableKind, name: str) -> Status:
        return self._storage.delete_table(kind=kind, name=name)

    def results(self) -> list[PulseResults]:
        return self._storage.results()

    def summarize(self, task: str, model: str, aliases: list[str] | None = None) -> PollSummary:
        """The saved run of `task` with `model`, summarized per persona group."""
        results = next(r for r in self.results() if r.task == task and r.model == model)
        return summarize(results, aliases=aliases)

    # saving
    def check_save(self, poll: PulseConfig) -> Clause | None:
        return first_error(PollGuards(poll))

    def save_poll(self, poll: PulseConfig) -> Status:
        self._require(self.check_save(poll))
        return self._storage.save_poll(poll)

    # running a saved poll
    def check_run(self, name: str | None) -> Clause | None:
        saved = name in self._storage.poll_names()
        poll = self._storage.get_poll(name) if saved else PulseConfig()
        return first_error(self.connection, RunGuards(poll, selected_poll=name if saved else None))

    def run_poll(self, name: str) -> PulseResults:
        self._require(self.check_run(name))
        return run_poll(runner=self.model, storage=self._storage, poll_name=name)

    # completion analysis of one persona
    def check_ranking(self, poll: PulseConfig) -> Clause | None:
        return first_error(self.connection, RankingGuards(poll))

    def ranker(self, poll: PulseConfig, v_pct: float, min_p: float) -> Ranker:
        self._require(self.check_ranking(poll))
        completions = self._storage.get_table(kind="completions", name=poll.completions).to_dict(orient="list")
        return Ranker(
            model=self.model,
            chat=poll_chat(poll),
            group_a=completions["A"],
            group_b=completions["B"],
            v_pct=v_pct,
            min_p=min_p,
        )

    # next-token explorer
    def check_explore(self, poll: PulseConfig) -> Clause | None:
        return first_error(self.connection, [PollGuards(poll).has_prompts])

    def next_tokens(self, poll: PulseConfig, prefix: str, k: int) -> pd.DataFrame:
        self._require(self.check_explore(poll))
        return next_token_table(model=self.model, chat=poll_chat(poll), prefix=prefix, k=k)

    @staticmethod
    def _require(failure: Clause | None) -> None:
        if failure is not None:
            raise Blocked(failure)
