"""Deterministic checks on a poll, before it is saved, run or ranked."""

from abc import ABC, abstractmethod
from typing import Literal, NamedTuple
from itertools import chain
from dataclasses import dataclass
from collections.abc import Iterable, Iterator

from pulse.domain.poll import PulseConfig, has_placeholder

Severity = Literal["error", "warning"]
ClauseCode = Literal[
    "NOT_CONNECTED",
    "NO_MODEL",
    "NO_CHAT_TEMPLATE",
    "INCOMPLETE_PROMPTS",
    "PLACEHOLDER_WITHOUT_PERSONAS",
    "PERSONAS_WITHOUT_PLACEHOLDER",
    "NO_COMPLETIONS",
    "NO_POLL_SELECTED",
    "PERSONA_TEMPLATE",
    "PERSONAS_NOT_FOUND",
    "COMPLETIONS_NOT_FOUND",
]


class Clause(NamedTuple):
    ok: bool
    code: ClauseCode
    msg: str
    severity: Severity = "error"


class Guards(ABC):
    """A group of clauses about one thing.

    `__iter__` yields every clause property, in priority order: the UI reports the first failing error.
    """

    @abstractmethod
    def __iter__(self) -> Iterator[Clause]: ...

    @property
    def passes(self) -> bool:
        """No error fails; warnings don't block."""
        return not self.errors

    @property
    def errors(self) -> list[Clause]:
        return [clause for clause in self if not clause.ok and clause.severity == "error"]

    @property
    def warnings(self) -> list[Clause]:
        return [clause for clause in self if not clause.ok and clause.severity == "warning"]


def first_error(*groups: Iterable[Clause]) -> Clause | None:
    """The first failing error across groups, in order."""
    return next((clause for clause in chain(*groups) if not clause.ok and clause.severity == "error"), None)


@dataclass(frozen=True)
class ConnectionGuards(Guards):
    """A server is connected and a chat model is selected."""

    connected: bool
    model: str | None
    chat_template: bool

    @property
    def is_connected(self) -> Clause:
        return Clause(
            ok=self.connected,
            code="NOT_CONNECTED",
            msg="Enter OpenAI compatible server credentials.",
        )

    @property
    def has_selected_model(self) -> Clause:
        return Clause(
            ok=bool(self.model),
            code="NO_MODEL",
            msg="Select one of the available models.",
        )

    @property
    def has_chat_template(self) -> Clause:
        return Clause(
            ok=self.chat_template,
            code="NO_CHAT_TEMPLATE",
            msg="Selected model has no chat template.",
        )

    def __iter__(self) -> Iterator[Clause]:
        yield self.is_connected
        yield self.has_selected_model
        yield self.has_chat_template


@dataclass(frozen=True)
class PollGuards(Guards):
    """A poll that can be saved and run."""

    poll: PulseConfig

    @property
    def has_prompts(self) -> Clause:
        return Clause(
            ok=all((self.poll.persona, self.poll.question, self.poll.answer)),
            code="INCOMPLETE_PROMPTS",
            msg="Incomplete prompt configuration.",
        )

    @property
    def placeholder_has_personas(self) -> Clause:
        """A persona template needs a personas table to fill it."""
        return Clause(
            ok=not has_placeholder(self.poll.persona) or bool(self.poll.docs),
            code="PLACEHOLDER_WITHOUT_PERSONAS",
            msg="Persona contains placeholder but no file selected.",
        )

    @property
    def personas_have_placeholder(self) -> Clause:
        """A personas table needs a persona template to fill."""
        return Clause(
            ok=not self.poll.docs or has_placeholder(self.poll.persona),
            code="PERSONAS_WITHOUT_PLACEHOLDER",
            msg="Batch personas selected but no placeholder in prompt.",
        )

    @property
    def has_completions(self) -> Clause:
        return Clause(
            ok=bool(self.poll.completions),
            code="NO_COMPLETIONS",
            msg="No completions selected.",
        )

    def __iter__(self) -> Iterator[Clause]:
        yield self.has_prompts
        yield self.placeholder_has_personas
        yield self.personas_have_placeholder
        yield self.has_completions


@dataclass(frozen=True)
class RunGuards(Guards):
    """A saved poll that can be run."""

    poll: PulseConfig
    selected_poll: str | None

    @property
    def has_selected_poll(self) -> Clause:
        return Clause(
            ok=bool(self.selected_poll),
            code="NO_POLL_SELECTED",
            msg="No task selected.",
        )

    def __iter__(self) -> Iterator[Clause]:
        yield self.has_selected_poll
        yield from PollGuards(self.poll)


@dataclass(frozen=True)
class RankingGuards(Guards):
    """A poll whose completions can be ranked: ranking needs one concrete persona."""

    poll: PulseConfig

    @property
    def single_persona(self) -> Clause:
        return Clause(
            ok=not has_placeholder(self.poll.persona),
            code="PERSONA_TEMPLATE",
            msg="Persona prompt contains placeholder.",
        )

    def __iter__(self) -> Iterator[Clause]:
        poll = PollGuards(self.poll)
        yield poll.has_prompts
        yield self.single_persona
        yield poll.has_completions


@dataclass(frozen=True)
class TableGuards(Guards):
    """The personas and completions tables a poll refers to exist."""

    poll: PulseConfig
    personas: Iterable[str]
    completions: Iterable[str]

    @property
    def personas_exist(self) -> Clause:
        return Clause(
            ok=not self.poll.docs or self.poll.docs in self.personas,
            code="PERSONAS_NOT_FOUND",
            msg=f"Persona file '{self.poll.docs}' not found.",
        )

    @property
    def completions_exist(self) -> Clause:
        return Clause(
            ok=self.poll.completions in self.completions,
            code="COMPLETIONS_NOT_FOUND",
            msg=f"Completions file '{self.poll.completions}' not found.",
        )

    def __iter__(self) -> Iterator[Clause]:
        yield self.personas_exist
        yield self.completions_exist
