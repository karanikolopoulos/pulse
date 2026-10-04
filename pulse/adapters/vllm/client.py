import re
import logging

from http import HTTPStatus
from typing import Any, Final
from functools import cached_property
from dataclasses import field, dataclass

import requests

from pulse.ports import ChatModel, ModelServer
from pulse.domain.poll import PulseConfig
from pulse.domain.types import Chat, Token, Sequence
from pulse.adapters.vllm.types import Prompt, ModelCard, SampleRequest
from pulse.adapters.vllm.polling import run_poll
from pulse.adapters.vllm.completions import VLLMCompletions

_LOGGER: Final = logging.getLogger(__name__)

DEFAULT_MAX_LOGPROBS: Final = 20  # vLLM's default --max-logprobs

# larger than any vocabulary, so vLLM always rejects it and reports its limit
_PROBE_LOGPROBS: Final = 10_000_000
# e.g. "Requested sample logprobs of 10000000, which is greater than max allowed: 248320"
_MAX_ALLOWED: Final = re.compile(r"max allowed: (\d+)")


@dataclass(frozen=True)
class VLLMModel(ChatModel):
    """`LanguageModel` backed by a vLLM server's /v1/completions endpoint."""

    base_url: str
    model_card: ModelCard
    token: str = field(default="EMPTY", compare=False, hash=False)
    max_logprobs: int = DEFAULT_MAX_LOGPROBS
    seed: int = 2025
    max_logprobs_known: bool = field(default=True, compare=False)  # False: read failed, using the default

    @cached_property
    def model(self) -> str:
        return self.model_card.id

    @cached_property
    def completions_endpoint(self) -> str:
        return f"{self.base_url}/v1/completions"

    @cached_property
    def lm(self) -> VLLMCompletions:
        """lm-eval model, also used directly by `lm_eval.evaluate`."""
        _LOGGER.info("Accessing vLLM model: %s", self.model_card.id)
        return VLLMCompletions(
            base_url=self.completions_endpoint,
            api_key=self.token,
            model_card=self.model_card,
            seed=self.seed,
        )

    @property
    def has_chat_template(self) -> bool:
        return bool(self.lm.tokenizer.chat_template)

    def score(self, chat: Chat, continuations: list[str]) -> list[Sequence]:
        prompts = self._sample(chat=chat, continuations=continuations, logprobs=1)
        return [prompt.continuation for prompt in prompts]

    def next_tokens(self, chat: Chat, prefixes: list[str], k: int) -> list[list[Token]]:
        prompts = self._sample(chat=chat, continuations=prefixes, logprobs=k)
        return [prompt.next_tokens for prompt in prompts]

    def run_poll(
        self,
        poll: PulseConfig,
        docs: list[dict[str, Any]] | None,
        completions: dict[str, list[str]],
    ) -> list[dict[str, float]]:
        return run_poll(lm=self.lm, poll=poll, docs=docs, completions=completions)

    def _sample(self, chat: Chat, continuations: list[str], logprobs: int) -> list[Prompt]:
        requests = [SampleRequest(context=chat, continuation=continuation) for continuation in continuations]
        return self.lm.sample(requests=requests, extra_body={"logprobs": logprobs, "echo": False})


@dataclass(frozen=True)
class VLLMConnection(ModelServer):
    base_url: str
    token: str | None = "EMPTY"
    seed: int = 2025

    @property
    def headers(self) -> dict:
        return {"Authorization": f"Bearer {self.token}"} if self.token else {}

    def get_models(self) -> tuple[ModelCard]:
        resp = requests.get(
            f"{self.base_url}/v1/models",
            headers=self.headers,
        )
        resp.raise_for_status()

        data = resp.json().get("data", {})
        return tuple(ModelCard.from_dict(m) for m in data)

    def models(self) -> list[str]:
        return [card.id for card in self.get_models()]

    def open(self, model: str) -> VLLMModel:
        """The served model with this id, with the server's max_logprobs."""
        card = next(card for card in self.get_models() if card.id == model)
        max_logprobs = self.resolve_max_logprobs(model_card=card)

        return VLLMModel(
            base_url=self.base_url,
            token=self.token,
            model_card=card,
            max_logprobs=max_logprobs or DEFAULT_MAX_LOGPROBS,
            seed=self.seed,
            max_logprobs_known=max_logprobs is not None,
        )

    def resolve_max_logprobs(self, model_card: ModelCard) -> int | None:
        """Reads the server's --max-logprobs from vLLM's validation error on an over-limit request.

        vLLM resolves `--max-logprobs -1` to the vocabulary size before validating. Returns None if
        the limit could not be read (e.g. not a vLLM server, or the error message changed).
        """
        payload = {**model_card.payload, "prompt": " ", "max_tokens": 1, "logprobs": _PROBE_LOGPROBS}
        try:
            resp = requests.post(
                f"{self.base_url}/v1/completions",
                json=payload,
                headers=self.headers,
                timeout=10,
            )
            message = resp.json()["error"]["message"]
        except (requests.RequestException, ValueError, KeyError, TypeError):
            message = ""

        if match := _MAX_ALLOWED.search(message):
            return int(match.group(1))

        _LOGGER.warning("Could not read max_logprobs for %s", model_card.id)
        return None

    @classmethod
    def connect(cls, url: str, token: str | None = None) -> "VLLMConnection":
        """A connection to a running server; raises ConnectionError if /health doesn't answer OK."""
        if (code := cls.is_alive(url=url)) != HTTPStatus.OK:
            raise ConnectionError(f"{url} - {code}")
        return cls(base_url=url, token=token)

    @staticmethod
    def is_alive(url: str) -> int:
        health_endpoint = url.rstrip("/") + "/health"

        try:
            resp = requests.get(url=health_endpoint, timeout=5)
            return resp.status_code
        except requests.RequestException:
            return HTTPStatus.NOT_FOUND
