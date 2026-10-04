from pulse.adapters.vllm.completions import merge_system_message


def test_merge_system_message_does_not_modify_input():
    chat = [
        {"role": "system", "content": "You are a citizen of the U.S."},
        {"role": "user", "content": "Who will you vote for?"},
        {"role": "assistant", "content": "I will vote for"},
    ]
    original = [dict(message) for message in chat]

    for _ in range(3):  # the ranker reuses one chat for every request
        merged = merge_system_message(chat)

    assert chat == original
    assert merged == [
        {"role": "user", "content": "You are a citizen of the U.S.\nWho will you vote for?"},
        {"role": "assistant", "content": "I will vote for"},
    ]
