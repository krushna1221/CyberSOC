from pathlib import Path

import yaml

from cybersoc_openenv.scenarios import SCENARIOS


def test_openenv_yaml_matches_registered_scenarios() -> None:
    payload = yaml.safe_load((Path(__file__).resolve().parents[1] / "openenv.yaml").read_text())
    task_entries = payload["tasks"]
    assert [task["id"] for task in task_entries] == list(SCENARIOS.keys())
    assert {task["id"]: task["difficulty"] for task in task_entries} == {
        task_id: scenario.difficulty.value for task_id, scenario in SCENARIOS.items()
    }
    assert payload["app"] == "server.app:app"
