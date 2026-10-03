import re

from http import HTTPStatus
from typing import Final
from functools import cached_property
from dataclasses import field, dataclass

import requests

from streamlit import logger

from pulse.connection.types import (
    Prompt,
    ModelCard,
    SampleRequest,
)
from pulse.connection.vllm_completions import VLLMCompletions

_LOGGER: Final = logger.get_logger(__name__)

DEFAULT_MAX_LOGPROBS: Final = 20  # vLLM's default --max-logprobs

# larger than any vocabulary, so vLLM always rejects it and reports its limit
_PROBE_LOGPROBS: Final = 10_000_000
# e.g. "Requested sample logprobs of 10000000, which is greater than max allowed: 248320"
_MAX_ALLOWED: Final = re.compile(r"max allowed: (\d+)")


@dataclass(frozen=True)
class VLLMInstance:
    base_url: str
    model_card: ModelCard
    token: str = field(default="EMPTY", compare=False, hash=False)
    max_logprobs: int = DEFAULT_MAX_LOGPROBS
    seed: int = 2025

    @cached_property
    def model(self) -> str:
        return self.model_card.id

    @cached_property
    def completions_endpoint(self) -> str:
        return f"{self.base_url}/v1/completions"

    @cached_property
    def lm(self) -> VLLMCompletions:
        _LOGGER.info(f"Accessing vLLM model: {self.model_card.id}")
        return VLLMCompletions(
            base_url=self.completions_endpoint,
            api_key=self.token,
            model_card=self.model_card,
            seed=self.seed,
        )

    def sample(self, requests: list[SampleRequest], **kwargs) -> list[Prompt]:
        return self.lm.sample(requests=requests, **kwargs)


@dataclass(frozen=True)
class VLLMConnection:
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

    def get_vllm_client(self, model_card: ModelCard, max_logprobs: int) -> VLLMInstance:
        return VLLMInstance(
            base_url=self.base_url,
            token=self.token,
            model_card=model_card,
            max_logprobs=max_logprobs,
            seed=self.seed,
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

    @staticmethod
    def is_alive(url: str) -> int:
        health_endpoint = url.rstrip("/") + "/health"

        try:
            resp = requests.get(url=health_endpoint, timeout=5)
            return resp.status_code
        except requests.RequestException:
            return HTTPStatus.NOT_FOUND
