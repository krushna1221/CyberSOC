from inference import run


def test_heuristic_inference_runs_against_api_surface() -> None:
    result = run(policy="heuristic")
    assert result["environment_target"] == "in-process-fastapi"
    assert result["average_score"] >= 0.8
    assert len(result["task_results"]) == 3
