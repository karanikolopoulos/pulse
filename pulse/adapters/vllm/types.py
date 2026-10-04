"""vLLM response types and parsers."""

import json

from typing import Self
from operator import itemgetter
from functools import cached_property
from dataclasses import dataclass

from pulse.domain.types import Chat, Token, Sequence


@dataclass(frozen=True)
class ModelCard:
    """vLLM /v1/models response"""

    id: str
    root: str

    def __str__(self):
        return self.id

    def __eq__(self, other):
        if isinstance(other, ModelCard):
            return self.root == other.root
        return False

    def __hash__(self):
        return hash(self.root)

    def __getstate__(self):
        return {"id": self.id, "root": self.root}

    def __setstate__(self, state):
        object.__setattr__(self, "id", state["id"])
        object.__setattr__(self, "root", state["root"])

    @property
    def payload(self) -> dict:
        return {"model": self.id}

    @classmethod
    def from_dict(cls, data: dict) -> Self:
        return cls(
            **{
                "id": data.get("id"),
                "root": data.get("root"),
            }
        )


class SampleRequest:
    def __init__(self, context: Chat, continuation: str):
        self.args = (context, continuation)


def token_from_prompt_logprob(prompt_logprob: dict) -> Token:
    """One entry of vLLM's `prompt_logprobs`: {token_id: {logprob, rank, decoded_token}}."""
    token_data: dict = next(iter(prompt_logprob.values()))

    return Token(
        token=token_data.get("decoded_token"),
        logprob=token_data.get("logprob"),
        rank=token_data.get("rank"),
    )


def tokens_from_top_logprobs(top_logprobs: dict[str, float]) -> list[Token]:
    """OpenAI-style `top_logprobs` ({token: logprob}), ranked by logprob."""
    sorted_logprobs = sorted(top_logprobs.items(), key=itemgetter(1), reverse=True)

    return [Token(token=token, logprob=logprob, rank=rank) for rank, (token, logprob) in enumerate(sorted_logprobs, 1)]


def sequence_from_prompt_logprobs(prompt_logprobs: list[dict]) -> Sequence:
    return Sequence(tokens=[token_from_prompt_logprob(prompt) for prompt in prompt_logprobs])


@dataclass
class Prompt:
    choice: dict
    ctxlen: int

    def __post_init__(self):
        *_, self.top_logprobs = self.choice["logprobs"]["top_logprobs"]
        self.prompt_logprobs = self.choice["prompt_logprobs"]

    @cached_property
    def context(self) -> Sequence | None:
        if self.ctxlen <= 1:
            return None
        return sequence_from_prompt_logprobs(prompt_logprobs=self.prompt_logprobs[1 : self.ctxlen])

    @cached_property
    def continuation(self) -> Sequence | None:
        if self.ctxlen >= len(self.prompt_logprobs):
            return None
        return sequence_from_prompt_logprobs(prompt_logprobs=self.prompt_logprobs[self.ctxlen :])

    @cached_property
    def next_tokens(self) -> list[Token]:
        return tokens_from_top_logprobs(top_logprobs=self.top_logprobs)

    def __str__(self):
        return json.dumps(
            {
                "Context": self.context,
                "Continuation": self.continuation,
                "Next Tokens": self.next_tokens,
            },
            indent=2,
        )

    def __repr__(self):
        return self.__str__()
