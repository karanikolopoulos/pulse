import pandas as pd
import pytest

from fakes import InMemoryStorage

from pulse.domain.poll import Status, PulseConfig, PulseResults
from pulse.adapters.file_storage import FileStorage

COMPLETIONS = pd.DataFrame({"A": ["Biden"], "B": ["Trump"], "alias": ["lastname"]})
POLL = PulseConfig(name="poll", persona="You are a voter.", question="Who?", answer="I vote", completions="c")


@pytest.fixture(params=["file", "memory"])
def storage(request, tmp_path):
    return FileStorage(root=tmp_path) if request.param == "file" else InMemoryStorage()


def test_storage_contract(storage):
    """Every Storage adapter follows the same rules."""
    assert storage.add_table(kind="completions", name="c", df=COMPLETIONS) == Status.OK
    assert storage.add_table(kind="completions", name="c", df=COMPLETIONS) == Status.DUPLICATE
    assert storage.add_table(kind="completions", name="bad", df=COMPLETIONS[["A"]]) == Status.INVALID_SCHEMA
    assert storage.update_table(kind="completions", name="missing", df=COMPLETIONS) == Status.NOT_FOUND
    assert storage.table_names(kind="completions") == ["c"]

    assert storage.save_poll(POLL) == Status.OK
    assert storage.save_poll(PulseConfig(**{**vars(POLL), "name": "copy"})) == Status.DUPLICATE_POLL
    assert storage.save_poll(PulseConfig(**{**vars(POLL), "answer": "I pick"})) == Status.OK  # overwrite by name
    assert storage.get_poll("poll").answer == "I pick"

    storage.save_results(PulseResults(task="poll", model="org/model", metrics=[{"lastname": 0.5}], dataset={}))
    assert [r.metrics for r in storage.results()] == [[{"lastname": 0.5}]]

    assert storage.delete_poll("poll") == Status.OK
    assert storage.delete_poll("poll") == Status.NOT_FOUND
    assert storage.delete_table(kind="completions", name="c") == Status.OK
    assert storage.table_names(kind="completions") == []


def test_file_storage_survives_reload(tmp_path):
    storage = FileStorage(root=tmp_path)
    storage.add_table(kind="completions", name="c", df=COMPLETIONS)
    storage.save_poll(POLL)
    storage.save_results(PulseResults(task="poll", model="org/model", metrics=[{"lastname": 0.5}], dataset={}))

    reloaded = FileStorage(root=tmp_path)
    assert reloaded.poll_names() == ["poll"]
    assert reloaded.get_poll("poll") == POLL
    assert reloaded.get_table(kind="completions", name="c").equals(COMPLETIONS)
    assert [r.metrics for r in reloaded.results()] == [[{"lastname": 0.5}]]
