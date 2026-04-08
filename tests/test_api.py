from fastapi.testclient import TestClient

from server.app import app


def test_api_health_and_reset() -> None:
    client = TestClient(app)

    root = client.get("/")
    assert root.status_code == 200
    assert "text/html" in root.headers["content-type"]
    assert "CyberSOC OpenEnv" in root.text

    js = client.get("/assets/soc.js")
    assert js.status_code == 200
    assert "text/javascript" in js.headers["content-type"] or "application/javascript" in js.headers["content-type"]

    status = client.get("/api/status")
    assert status.status_code == 200
    assert status.json()["status"] == "ok"
    assert status.json()["session_id"]

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"

    reset = client.post("/reset", json={"task_id": "alert-triage-easy", "seed": 7})
    assert reset.status_code == 200
    payload = reset.json()
    assert payload["session_id"]
    assert payload["task"]["task_id"] == "alert-triage-easy"
    assert payload["observation"]["task_id"] == "alert-triage-easy"

    state = client.get("/state")
    assert state.status_code == 200
    assert state.json()["task_id"] == "alert-triage-easy"


def test_api_sessions_are_isolated() -> None:
    client_one = TestClient(app)
    client_two = TestClient(app)

    reset_one = client_one.post("/reset", json={"task_id": "alert-triage-easy", "seed": 7})
    reset_two = client_two.post("/reset", json={"task_id": "incident-containment-medium", "seed": 7})

    assert reset_one.status_code == 200
    assert reset_two.status_code == 200
    assert reset_one.json()["session_id"] != reset_two.json()["session_id"]

    step_one = client_one.post(
        "/step",
        json={"action_type": "triage_alert", "alert_id": "ALT-E1", "classification": "true_positive"},
    )
    assert step_one.status_code == 200

    state_one = client_one.get("/state")
    state_two = client_two.get("/state")
    assert state_one.status_code == 200
    assert state_two.status_code == 200
    assert state_one.json()["task_id"] == "alert-triage-easy"
    assert state_two.json()["task_id"] == "incident-containment-medium"
    assert state_one.json()["triage_decisions"]["ALT-E1"] == "true_positive"
    assert "ALT-E1" not in state_two.json()["triage_decisions"]


def test_state_requires_known_session() -> None:
    client = TestClient(app)
    response = client.get("/state?session_id=missing-session")
    assert response.status_code == 404
