"""Core PULSE types, independent of the inference server."""

import json
import math

from typing import TypedDict
from dataclasses import dataclass

Message = dict[str, str]  # {"role": ..., "content": ...}
Chat = list[Message]


class SequenceData(TypedDict):
    text: str
    logprob: float | None
    avg_logprob: float | None
    perplexity: float | None
    ranks: list[int]
    token_strings: list[str]


@dataclass
class Token:
    token: str
    logprob: float | None
    rank: int | None

    def __str__(self):
        return f"Token(rank={self.rank}, token={self.token}, logprob={self.logprob})"

    def __repr__(self):
        return self.__str__()


@dataclass
class Sequence:
    tokens: list[Token]

    def __post_init__(self):
        n = len([t for t in self.tokens if t.rank])

        self.text = "".join([t.token for t in self.tokens])
        self.logprob = sum(token.logprob for token in self.tokens) if n else None
        self.avg_logprob = self.logprob / n if n else None
        self.ppl = math.exp(-self.avg_logprob) if n else None
        self.ranks = [token.rank for token in self.tokens]

    @property
    def data(self) -> SequenceData:
        return {
            "text": self.text.strip(),
            "logprob": self.logprob,
            "avg_logprob": self.avg_logprob,
            "perplexity": self.ppl,
            "ranks": self.ranks,
            "token_strings": [t.token for t in self.tokens],
        }

    def __str__(self):
        return json.dumps(self.data, indent=2)

    def __repr__(self):
        return self.__str__()
