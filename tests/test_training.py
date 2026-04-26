import json

from cybersoc_openenv.client import InProcessCyberSOCEnvClient
from cybersoc_openenv.training import DEFAULT_TASKS, SYSTEM_PROMPT, generate_sft_examples
from server.app import app


def test_generate_sft_examples_builds_chat_rows():
    client = InProcessCyberSOCEnvClient(app)
    try:
        examples = generate_sft_examples(
            env_client=client,
            tasks=[DEFAULT_TASKS[0]],
            episodes_per_task=1,
            seed_start=7,
        )
    finally:
        client.close()

    assert examples
    first = examples[0]
    assert first["task_id"] == "alert-triage-easy"
    assert first["messages"][0]["role"] == "system"
    assert first["messages"][0]["content"] == SYSTEM_PROMPT
    assert first["messages"][1]["role"] == "user"
    assert "current_observation" in json.loads(first["messages"][1]["content"])
    assert first["messages"][2]["role"] == "assistant"
    assert json.loads(first["messages"][2]["content"])["action_type"]
