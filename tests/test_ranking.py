import math

from pulse.domain.types import Token, Sequence
from pulse.services.ranking import Ranker


class FakeModel:
    """Every word is one token with rank 1; every next-token distribution is (0.5, 0.3, 0.2)."""

    max_logprobs = 10

    def score(self, chat, continuations):
        return [
            Sequence(tokens=[Token(token=f" {word}", logprob=-1.0, rank=1) for word in text.split()])
            for text in continuations
        ]

    def next_tokens(self, chat, prefixes, k):
        probs = (0.5, 0.3, 0.2)
        return [[Token(token=str(i), logprob=math.log(p), rank=i) for i, p in enumerate(probs, 1)] for _ in prefixes]


def test_rankings_split_groups_with_elbows():
    ranker = Ranker(model=FakeModel(), chat=[], group_a=["the Democratic nominee"], group_b=["Trump", "Donald Trump"])

    a, b = ranker.rankings

    assert list(a.index) == ["the Democratic nominee"]
    assert list(b.index) == ["Trump", "Donald Trump"]
    assert a.loc["the Democratic nominee", "ranks"] == [1, 1, 1]
    # elbow for min_p=0.99 on (0.5, 0.3, 0.2) is rank 3, at every token position
    assert b.loc["Donald Trump", "elbows"] == [3, 3]
