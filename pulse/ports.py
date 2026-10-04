"""Interfaces PULSE depends on; adapters implement them."""

from typing import Any, Protocol

import pandas as pd

from pulse.domain.poll import Status, TableKind, PulseConfig, PulseResults
from pulse.domain.types import Chat, Token, Sequence


class LanguageModel(Protocol):
    """A chat model that exposes token-level log-probabilities.

    If the chat ends with an assistant message, it is continued rather than closed.
    """

    @property
    def max_logprobs(self) -> int: ...

    @property
    def has_chat_template(self) -> bool: ...

    @property
    def max_logprobs_known(self) -> bool:
        """False if the server's limit could not be read and `max_logprobs` is a fallback."""
        ...

    def score(self, chat: Chat, continuations: list[str]) -> list[Sequence]:
        """Logprob and rank of every token of each continuation, appended verbatim to the chat."""
        ...

    def next_tokens(self, chat: Chat, prefixes: list[str], k: int) -> list[list[Token]]:
        """Top-k next tokens after the chat followed by each prefix."""
        ...


class PollRunner(Protocol):
    """Runs a whole poll: every persona, every A/B completion pair."""

    @property
    def model(self) -> str: ...

    def run_poll(
        self,
        poll: PulseConfig,
        docs: list[dict[str, Any]] | None,
        completions: dict[str, list[str]],
    ) -> list[dict[str, float]]:
        """P(A) - P(B) per completion alias, one dict per persona (a single one without personas)."""
        ...


class ChatModel(LanguageModel, PollRunner, Protocol):
    """A model that can be explored, ranked and polled."""


class ModelServer(Protocol):
    """A server hosting chat models, e.g. one vLLM instance or a router in front of several."""

    def models(self) -> list[str]:
        """Ids of the served models; asks the server on every call."""
        ...

    def open(self, model: str) -> ChatModel: ...


class Storage(Protocol):
    """Polls, their personas and completions tables, and poll results."""

    def poll_names(self) -> list[str]: ...

    def get_poll(self, name: str) -> PulseConfig: ...

    def save_poll(self, poll: PulseConfig) -> Status:
        """Creates or overwrites the poll named `poll.name`; refused if a saved poll has the same content."""
        ...

    def delete_poll(self, name: str) -> Status: ...

    def table_names(self, kind: TableKind) -> list[str]: ...

    def get_table(self, kind: TableKind, name: str) -> pd.DataFrame: ...

    def add_table(self, kind: TableKind, name: str, df: pd.DataFrame) -> Status: ...

    def update_table(self, kind: TableKind, name: str, df: pd.DataFrame) -> Status: ...

    def delete_table(self, kind: TableKind, name: str) -> Status: ...

    def results(self) -> list[PulseResults]: ...

    def save_results(self, results: PulseResults) -> None: ...
