from fakes import InMemoryStorage
from conftest import ENTRYPOINT
from streamlit.testing.v1 import AppTest

from pulse.bootstrap import bootstrap


def load(at, poll: str) -> None:
    at.selectbox(key="PollForm.selected_task").select(poll).run()


def rank(at) -> None:
    at.session_state["Analysis.rank_flag"] = True  # what Ctrl+R does
    at.run()


def test_load_and_rank_a_poll(at):
    load(at, "single")
    assert at.text_input(key="PollForm.persona").value == "You are a voter."
    assert at.selectbox(key="PollForm.selected_completions").value == "elections"

    rank(at)
    side_a, side_b = at.session_state["Analysis.rankings"]
    assert list(side_a.index) == ["the Democrat", "Biden"]
    assert list(side_b.index) == ["the Republican", "Trump"]
    assert not at.exception


def test_batch_poll_cannot_be_ranked(at):
    load(at, "batch")
    rank(at)

    assert [warning.value for warning in at.warning] == ["Persona prompt contains placeholder."]


def test_update_a_saved_poll(at, pulse):
    load(at, "single")
    at.text_input(key="PollForm.question").input("Who will you vote for in 2028?").run()
    at.button(key="save_task").click().run()

    poll, _ = pulse.load_poll("single")
    assert poll.question == "Who will you vote for in 2028?"


def test_save_as_new_then_delete(at, pulse):
    at.text_input(key="PollForm.persona").input("You are a farmer.").run()
    at.text_input(key="PollForm.question").input("Who?").run()
    at.text_input(key="PollForm.answer").input("I pick").run()
    at.selectbox(key="PollForm.selected_completions").select("elections").run()

    # Save opens the name dialog; a dialog's own button only exists in the run that shows it
    at.session_state["PollForm.new_poll_name"] = "farmers"
    at.button(key="save_task").click().run()
    dialog_save = next(b for b in at.button if b.label == "Save" and b.key != "save_task")
    at.button(key="save_task").click()
    dialog_save.click().run()

    assert "farmers" in pulse.polls()
    assert at.session_state["PollForm.selected_task"] == "farmers"

    at.button(key="delete_task").click().run()
    assert "farmers" not in pulse.polls()


def test_form_carries_across_pages(at):
    load(at, "single")

    at.switch_page("app_pages/explorer.py").run()
    assert at.text_input(key="PollForm.persona").value == "You are a voter."
    at.text_input(key="PollForm.persona").input("You are a farmer.")
    next(b for b in at.button if b.label == "Sample next token").click().run()
    assert at.session_state["Explorer.sample_df"] is not None

    at.switch_page("app_pages/results.py").run()
    at.switch_page("app_pages/referendum.py").run()
    assert at.text_input(key="PollForm.persona").value == "You are a farmer."
    assert not at.exception


def test_results_page_summarizes_a_run(at):
    at.switch_page("app_pages/results.py").run()

    assert [subheader.value for subheader in at.subheader] == ["batch results"]
    assert len(at.table) == 2  # aggregated and per-completion tables
    assert not at.exception


def test_explorer_asks_to_connect_first():
    at = AppTest.from_file(ENTRYPOINT, default_timeout=30)
    at.session_state["pulse"] = bootstrap(storage=InMemoryStorage(), connect=lambda url, token: None)
    at.run()
    at.switch_page("app_pages/explorer.py").run()

    assert [error.value for error in at.error] == ["Enter OpenAI compatible server credentials."]
