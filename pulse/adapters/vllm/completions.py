import logging

from typing import Final
from operator import itemgetter
from functools import cached_property

from jinja2 import TemplateError
from lm_eval.models.api_models import TemplateAPI

from pulse.domain.types import Chat, Token
from pulse.adapters.vllm.types import Prompt, ModelCard, SampleRequest

_LOGGER: Final = logging.getLogger(__name__)


def merge_system_message(chat: Chat) -> Chat:
    """Prepends a leading system message to the first user message, without modifying `chat`."""
    if not chat or chat[0]["role"] != "system":
        return chat

    system, *rest = chat
    if rest and rest[0]["role"] == "user":
        user, *rest = rest
        rest = [{**user, "content": f"{system['content']}\n{user['content']}"}, *rest]

    return rest


class VLLMCompletions(TemplateAPI):
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model_card: ModelCard,
        **kwargs,
    ):
        self.model_card = model_card

        super().__init__(
            base_url=base_url,  # /v1/completions
            model=model_card.root,  # tokenizer
            **kwargs,
        )

        self.model = model_card.id
        self.api_key = api_key

    @property
    def model_id(self) -> str:
        return self.model_card.id

    def sample(self, requests: list[SampleRequest], **kwargs) -> list[Prompt]:
        assert self.tokenized_requests

        sample_requests = []
        for chat, continuation in [req.args for req in requests]:
            context = self.apply_chat_template(
                chat_history=chat,
                # continue a final assistant message (e.g. "I will vote for") instead of closing it
                add_generation_prompt=chat[-1]["role"] != "assistant",
            )

            context_enc = self.tok_encode(context)
            continuation_enc = self.tok_encode(continuation, add_special_tokens=False)
            sample_requests.append((None, context_enc, continuation_enc))

        inputs, ctxlens, _ = self.batch_loglikelihood_requests([sample_requests])
        outputs = self.model_call(messages=inputs, generate=False, **kwargs)
        parsed = self.parse_logprobs(
            outputs=outputs,
            tokens=inputs,
            ctxlens=ctxlens,
            **kwargs,
        )

        return parsed

    def _create_payload(
        self,
        messages: list[list[int]] | list[dict] | list[str] | str,
        generate=False,
        gen_kwargs: dict | None = None,
        seed: int = 2025,
        eos=None,
        **kwargs,
    ) -> dict:
        if generate:
            raise NotImplementedError

        extra_body = kwargs.pop("extra_body", {})

        to_ret = {
            "model": self.model_id,
            "prompt": messages,
            "temperature": 1,
            "max_tokens": 1,
            "logprobs": 20,
            "seed": seed,
            "echo": True,
            "prompt_logprobs": 1,
            **extra_body,  # will overwrite
        }

        _LOGGER.debug("payload: %s", to_ret)
        return to_ret

    @staticmethod
    def parse_logprobs(
        outputs: dict | list[dict],
        tokens: list[list[int]] = None,
        ctxlens: list[int] = None,
        **kwargs,
    ) -> list[list[Token]]:
        results = []
        if not isinstance(outputs, list):
            outputs = [outputs]

        for out in outputs:
            choice_ctxlen = zip(sorted(out["choices"], key=itemgetter("index")), ctxlens)
            for choice, ctxlen in choice_ctxlen:
                results.append(Prompt(choice=choice, ctxlen=ctxlen))

        return results

    @staticmethod
    def parse_generations(outputs: dict | list[dict], **kwargs) -> Prompt:
        raise NotImplementedError

    def apply_chat_template(self, chat_history, add_generation_prompt=True):
        if not self.supports_system_role:
            chat_history = merge_system_message(chat_history)
        return super().apply_chat_template(chat_history, add_generation_prompt)

    @cached_property
    def supports_system_role(self) -> bool:
        """Whether the chat template accepts a system message (e.g. Gemma's raises on one)."""
        probe = [{"role": "system", "content": "system"}, {"role": "user", "content": "user"}]
        try:
            self.tokenizer.apply_chat_template(probe, tokenize=False)
        except TemplateError:
            _LOGGER.info("%s has no system role, merging it into the user message", self.model)
            return False
        return True
