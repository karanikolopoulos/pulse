from pulse.ports import Storage, PollRunner
from pulse.domain.poll import PulseResults


def run_poll(runner: PollRunner, storage: Storage, poll_name: str) -> PulseResults:
    """Runs a saved poll and stores its results, together with the inputs it was run on."""
    poll = storage.get_poll(poll_name)
    docs = storage.get_table("personas", poll.docs).to_dict(orient="records") if poll.docs else None
    completions = storage.get_table("completions", poll.completions).to_dict(orient="list")

    results = PulseResults(
        task=poll.name,
        model=runner.model,
        metrics=runner.run_poll(poll=poll, docs=docs, completions=completions),
        dataset={"docs": docs, "completions": completions},
    )
    storage.save_results(results)

    return results
