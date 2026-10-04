import pandas as pd

from pulse.domain.poll import Status, PulseConfig, PulseResults
from pulse.adapters.file_storage import FileStorage


def test_file_storage_round_trip(tmp_path):
    storage = FileStorage(root=tmp_path)
    completions = pd.DataFrame({"A": ["Biden"], "B": ["Trump"], "alias": ["lastname"]})
    poll = PulseConfig(name="poll", persona="You are a voter.", question="Who?", answer="I vote", completions="c")

    assert storage.add_table(kind="completions", name="c", df=completions) == Status.OK
    assert storage.add_table(kind="completions", name="bad", df=completions[["A"]]) == Status.INVALID_SCHEMA
    assert storage.save_poll(poll) == Status.OK
    assert storage.save_poll(PulseConfig(**{**vars(poll), "name": "copy"})) == Status.DUPLICATE_POLL
    storage.save_results(PulseResults(task="poll", model="org/model", metrics=[{"lastname": 0.5}], dataset={}))

    reloaded = FileStorage(root=tmp_path)
    assert reloaded.poll_names() == ["poll"]
    assert reloaded.get_poll("poll") == poll
    assert reloaded.get_table(kind="completions", name="c").equals(completions)
    assert [r.metrics for r in reloaded.results()] == [[{"lastname": 0.5}]]
