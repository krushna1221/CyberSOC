import subprocess
import sys

from inference import run


def test_heuristic_inference_runs_against_api_surface() -> None:
    result = run(policy="heuristic")
    assert result["environment_target"] == "in-process-fastapi"
    assert result["average_score"] >= 0.8
    assert len(result["task_results"]) == 3


def test_llm_policy_falls_back_to_heuristic_when_env_missing(monkeypatch) -> None:
    monkeypatch.delenv("MODEL_NAME", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("API_KEY", raising=False)
    monkeypatch.delenv("ENV_BASE_URL", raising=False)

    result = run(policy="llm")
    assert result["policy"] == "llm"
    assert result["effective_policy"] == "heuristic"
    assert result["average_score"] >= 0.8
    assert result["warnings"]


def test_inference_stdout_contains_structured_blocks() -> None:
    completed = subprocess.run(
        [sys.executable, "inference.py", "--policy", "heuristic"],
        capture_output=True,
        text=True,
        check=True,
    )
    stdout = completed.stdout
    assert "[START]" in stdout
    assert "[STEP]" in stdout
    assert "[END]" in stdout
