import numpy as np
import pandas as pd

from pulse.ports import LanguageModel
from pulse.domain.types import Chat
from pulse.domain.scoring import renormalize


def next_token_table(model: LanguageModel, chat: Chat, prefix: str, k: int) -> pd.DataFrame:
    """Top-k next tokens after the chat and prefix, with probabilities renormalized over the top-k."""
    (tokens,) = model.next_tokens(chat=chat, prefixes=[prefix], k=k)
    probs = renormalize([token.logprob for token in tokens])

    return pd.DataFrame(
        {
            "Token": [token.token for token in tokens],
            "Probability": probs,
            "Cumulative Probability": np.cumsum(probs),
        },
        index=pd.Index([token.rank for token in tokens], name="Rank"),
    )
