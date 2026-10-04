"""Polls, their inputs and results, independent of where they are stored."""

from enum import StrEnum
from typing import Any, Literal
from dataclasses import dataclass

import pandas as pd

from jinja2 import Environment, meta

from pulse.domain.types import Chat

TableKind = Literal["personas", "completions"]

REQUIRED_COLUMNS: dict[TableKind, set[str]] = {
    "personas": set(),
    "completions": {"A", "B", "alias"},
}


class Status(StrEnum):
    OK = "ok"
    NOT_FOUND = "Not found"
    DUPLICATE = "Name already exists"
    DUPLICATE_POLL = "A poll with the same configuration already exists"
    INVALID_SCHEMA = "Invalid schema (required fields: 'A', 'B', 'alias')"
    MISSING_VALUES = "File contains missing values"


def check_table(kind: TableKind, df: pd.DataFrame) -> Status:
    """Validation every storage applies before saving a personas or completions table."""
    if not REQUIRED_COLUMNS[kind].issubset(df.columns):
        return Status.INVALID_SCHEMA
    if df.isnull().any().any():
        return Status.MISSING_VALUES
    return Status.OK


@dataclass
class PulseConfig:
    """Poll configuration, updated by the UI."""

    name: str | None = None
    persona: str | None = None
    docs: str | None = None  # name of the personas table
    question: str | None = None
    answer: str | None = None
    completions: str | None = None  # name of the completions table

    __hash__ = None

    def __eq__(self, other: object) -> bool:
        """Same poll content, regardless of its name."""
        attrs = ("persona", "question", "answer", "docs", "completions")
        if not isinstance(other, PulseConfig):
            return False

        return all(getattr(self, attr) == getattr(other, attr) for attr in attrs)


def has_placeholder(template: str | None) -> bool:
    """Whether a prompt is a Jinja template, e.g. "You are {{ persona }}."."""
    return any(meta.find_undeclared_variables(Environment().parse(template or "")))


def poll_chat(poll: PulseConfig) -> Chat:
    """The conversation a poll asks: persona, question, and the answer the model continues."""
    return [
        {"role": "system", "content": poll.persona},
        {"role": "user", "content": poll.question},
        {"role": "assistant", "content": poll.answer},
    ]


@dataclass
class PulseResults:
    task: str
    model: str
    metrics: list[dict[str, float]]  # P(A) - P(B) per completion alias, one dict per persona
    dataset: dict[str, Any]

    def __post_init__(self) -> None:
        self.has_docs = bool(self.docs)

    def __repr__(self) -> str:
        return f"PulseResults(model={self.model}, task={self.task})"

    def __str__(self) -> str:
        return repr(self)

    @property
    def docs(self) -> dict[str, Any]:
        return self.dataset.get("docs", {})

    @property
    def completions(self) -> dict[str, Any]:
        return self.dataset.get("completions", {})
