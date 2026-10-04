"""Composition root: the one place that decides which adapters PULSE runs on.

Every front end calls `bootstrap()`: Streamlit once per browser session, a CLI once per command,
tests with fakes in place of the defaults.
"""

from collections.abc import Callable

from pulse.ports import Storage, ModelServer
from pulse.utils.paths import DATA
from pulse.services.facade import Pulse
from pulse.adapters.vllm.client import VLLMConnection
from pulse.adapters.file_storage import FileStorage


def bootstrap(
    storage: Storage | None = None,
    connect: Callable[[str, str | None], ModelServer] = VLLMConnection.connect,
) -> Pulse:
    return Pulse(storage=storage or FileStorage(root=DATA), connect=connect)
