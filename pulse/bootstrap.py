"""Composition root: the one place that decides which adapters PULSE runs on.

Streamlit calls `bootstrap()` once per browser session; tests call it with fakes in place of the defaults.
"""

from collections.abc import Callable

from pulse.ports import Storage, ModelServer
from pulse.application import Pulse
from pulse.utils.paths import DATA
from pulse.adapters.vllm.client import VLLMConnection
from pulse.adapters.file_storage import FileStorage


def bootstrap(
    storage: Storage | None = None,
    connect: Callable[[str, str | None], ModelServer] = VLLMConnection.connect,
) -> Pulse:
    return Pulse(storage=storage or FileStorage(root=DATA), connect=connect)
