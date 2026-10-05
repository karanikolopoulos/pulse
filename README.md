### PULSE – Polling Using LLM-based Sentiment Extraction (Demo at ICDM 2025)

Available at [elidek-themis-pulse.hf.space](https://elidek-themis-pulse.hf.space)

PULSE is built with [vLLM](https://github.com/vllm-project/vllm), [lm-eval](https://github.com/EleutherAI/lm-evaluation-harness) and [Streamlit](https://github.com/streamlit/streamlit).

```bash
# install
uv sync

# serve a model (separate environment with vLLM installed)
# max_logprobs in the config sets how many tokens PULSE can rank
vllm serve --config data/model/llama_3_1_8b_it.yaml

# open PULSE at http://localhost:8501 and enter the vLLM server URL (http://localhost:8000) in the sidebar
uv run pulse
```

```bash
# development
uv run pytest
uv run ruff check
uv run lint-imports
```
