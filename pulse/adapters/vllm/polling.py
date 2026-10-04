"""Runs PULSE polls with lm-eval's evaluator on a vLLM model."""

from typing import Any
from collections.abc import Callable

import datasets

from lm_eval import evaluate
from lm_eval.api.task import ConfigurableTask
from lm_eval.api.instance import Instance

from pulse.domain.poll import PulseConfig
from pulse.domain.scoring import norm_prob_diff
from pulse.adapters.vllm.types import Prompt
from pulse.adapters.vllm.completions import VLLMCompletions

METRIC = "norm_prob_diff"


def run_poll(
    lm: VLLMCompletions,
    poll: PulseConfig,
    docs: list[dict[str, Any]] | None,
    completions: dict[str, list[str]],
) -> list[dict[str, float]]:
    """P(A) - P(B) per completion alias, one dict per persona (a single one without personas)."""
    task = PulseTask(
        config={
            "task": poll.name,
            "output_type": "loglikelihood",
            "doc_to_target": -1,
            "description": poll.persona,
            "doc_to_text": poll.question,
            "gen_prefix": poll.answer,
            "dataset_kwargs": {"docs": docs, "completions": completions},
        }
    )

    results = evaluate(
        lm=lm,
        task_dict={poll.name: task},
        write_out=True,
        log_samples=True,
        apply_chat_template=True,
        verbosity="INFO",
        confirm_run_unsafe_code=False,
    )

    return results["results"][poll.name][f"{METRIC},none"]


class PulseTask(ConfigurableTask):
    TEST_SPLIT: str = "test"

    def __init__(self, config: dict) -> None:
        super().__init__(config=config)

    def has_test_docs(self):
        return bool(self.test_docs)

    def test_docs(self):
        return self.dataset[self.TEST_SPLIT]

    def construct_requests(self, doc: dict, ctx: str, **kwargs) -> list[Instance] | Instance:
        kwargs.pop("apply_chat_template")
        kwargs.pop("chat_template")

        choices = self.doc_to_choice(doc)
        target_delimiter = self.config.target_delimiter
        arguments = [(ctx, f"{target_delimiter}{cont}") for cont in choices]

        request_list = [
            Instance(
                request_type="loglikelihood",
                doc=doc,
                arguments=arg,
                idx=i,
                **kwargs,
            )
            for i, arg in enumerate(arguments)
        ]

        return request_list

    def custom_dataset(self, **kwargs) -> dict[str, datasets.Dataset]:
        docs = kwargs.get("docs")
        completions = kwargs.get("completions")

        if docs:
            dataset = datasets.Dataset.from_list(docs)
            dataset = dataset.add_column("choices", [completions] * len(dataset))
            dataset.choices = completions
        else:
            dataset = datasets.Dataset.from_list([{"choices": completions}])
            dataset.choices = completions

        return {"test": dataset}

    def download(self, dataset_kwargs: dict[str, Any] | None = None, **kwargs) -> None:
        self.dataset = self.custom_dataset(**(self.config.metadata or {}), **(self.config.dataset_kwargs or {}))

    def process_results(self, doc: dict[str, Any], results: list[Prompt]):
        alias = doc["choices"]["alias"]
        num_choices = len(alias)

        # log likelihoods of each completion, group A first
        lls = [prompt.continuation.logprob for prompt in results]
        diff = norm_prob_diff(logprobs_a=lls[:num_choices], logprobs_b=lls[num_choices:])

        return {METRIC: dict(zip(alias, diff))}

    def doc_to_choice(self, doc: Any, doc_to_choice=None) -> list[str]:
        choices = doc["choices"]
        return choices["A"] + choices["B"]

    def aggregation(self) -> dict[str, Callable[[list], Any]]:
        def _pass(arr: list) -> list:
            return arr

        return {METRIC: _pass}

    def higher_is_better(self) -> dict[str, bool]:
        return {METRIC: True}
