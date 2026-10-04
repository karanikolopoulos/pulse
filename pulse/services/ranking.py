"""Completion analysis: how each completion token ranks against the model's elbow at that position."""

from functools import cached_property
from dataclasses import dataclass

import pandas as pd

from pulse.ports import LanguageModel
from pulse.domain.types import Chat, Sequence
from pulse.domain.scoring import elbow_rank, token_prefixes

V_PCT = 0.2  # share of max_logprobs sampled at each position
MIN_P = 0.99  # cumulative probability that defines the elbow


@dataclass
class Ranker:
    """Each step is a cached property, so the UI can report progress between them."""

    model: LanguageModel
    chat: Chat
    group_a: list[str]
    group_b: list[str]

    v_pct: float = V_PCT
    min_p: float = MIN_P

    @property
    def completions(self) -> list[str]:
        return self.group_a + self.group_b

    @cached_property
    def sequences(self) -> list[Sequence]:
        """Each completion's tokens with their logprob and rank."""
        return self.model.score(chat=self.chat, continuations=[f" {c}" for c in self.completions])

    @cached_property
    def elbows(self) -> list[list[int]]:
        """Elbow rank at every token position of each completion."""
        prefixes = [token_prefixes(sequence.tokens) for sequence in self.sequences]
        unique = list(dict.fromkeys(p for completion_prefixes in prefixes for p in completion_prefixes))

        k = int(self.model.max_logprobs * self.v_pct)
        distributions = self.model.next_tokens(chat=self.chat, prefixes=unique, k=k)
        elbow_at = {
            prefix: elbow_rank(logprobs=[t.logprob for t in tokens], min_p=self.min_p)
            for prefix, tokens in zip(unique, distributions)
        }

        return [[elbow_at[p] for p in completion_prefixes] for completion_prefixes in prefixes]

    @cached_property
    def rankings(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Metrics, ranks and elbows per completion, for group A and group B."""
        df = pd.DataFrame(
            [sequence.data for sequence in self.sequences],
            index=pd.Index(self.completions, name="completion"),
        ).drop(columns="text")
        df["elbows"] = self.elbows

        n_a = len(self.group_a)
        return df.iloc[:n_a], df.iloc[n_a:]
