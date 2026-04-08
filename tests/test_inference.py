import subprocess
import sys
import re

from inference import _build_openai_client, run


def test_heuristic_inference_runs_against_api_surface() -> None:
    result = run(policy="heuristic")
    assert result["environment_target"] == "in-process-fastapi"
    assert result["average_score"] >= 0.8
    assert len(result["task_results"]) == 3


def test_llm_policy_falls_back_to_heuristic_when_env_missing(monkeypatch) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("API_KEY", raising=False)
    monkeypatch.delenv("ENV_BASE_URL", raising=False)

    result = run(policy="llm")
    assert result["policy"] == "llm"
    assert result["effective_policy"] == "heuristic"
    assert result["average_score"] >= 0.8
    assert result["warnings"]


def test_openai_client_uses_default_model_name(monkeypatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "test-token")
    monkeypatch.delenv("MODEL_NAME", raising=False)

    _, model_name = _build_openai_client()
    assert model_name == "Qwen/Qwen3.5-9B:together"


def test_inference_stdout_contains_structured_blocks() -> None:
    completed = subprocess.run(
        [sys.executable, "inference.py", "--policy", "heuristic"],
        capture_output=True,
        text=True,
        check=True,
    )
    stdout = completed.stdout
    assert re.search(r"^\[START\] task=\S+$", stdout, re.MULTILINE)
    assert re.search(r"^\[STEP\] step=\d+ reward=-?\d+\.\d{4}$", stdout, re.MULTILINE)
    assert re.search(r"^\[END\] task=\S+ score=\d+\.\d{4} steps=\d+$", stdout, re.MULTILINE)
