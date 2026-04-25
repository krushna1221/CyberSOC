from cybersoc_openenv.datasets import (
    load_curated_alert_dataset,
    prompt_reference_examples,
    select_reference_examples,
)
from cybersoc_openenv.environment import CyberSOCEnvironment
from cybersoc_openenv.models import ActionType, TriageLabel


def test_curated_dataset_covers_all_tasks_and_labels() -> None:
    dataset = load_curated_alert_dataset()

    assert dataset.name == "cybersoc-curated-alerts-v1"
    assert len(dataset.records) >= 18
    assert {record.task_id for record in dataset.records} == {
        "alert-triage-easy",
        "incident-containment-medium",
        "soc-optimization-hard",
    }
    assert {record.expected_label for record in dataset.records} == {
        TriageLabel.TRUE_POSITIVE,
        TriageLabel.FALSE_POSITIVE,
    }
    assert {
        record.recommended_response.primary_action for record in dataset.records
    } >= {
        ActionType.TRIAGE_ALERT,
        ActionType.IGNORE_ALERT,
        ActionType.REQUEST_FORENSICS,
        ActionType.ISOLATE_NODE,
        ActionType.PATCH_SYSTEM,
        ActionType.BLOCK_INDICATOR,
        ActionType.ESCALATE_INCIDENT,
    }


def test_reference_examples_prefer_current_task_family() -> None:
    env = CyberSOCEnvironment()
    observation = env.reset(task_id="incident-containment-medium", seed=7)

    examples = select_reference_examples(observation, limit=2)
    prompt_examples = prompt_reference_examples(observation, limit=2)

    assert len(examples) == 2
    assert all(example.task_id == "incident-containment-medium" for example in examples)
    assert len(prompt_examples) == 2
    assert all("playbook_hint" in example for example in prompt_examples)
    assert all("reference_response" not in example for example in prompt_examples)
    assert all("expected_label" not in example for example in prompt_examples)
