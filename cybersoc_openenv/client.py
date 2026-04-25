"""Small synchronous HTTP client for the CyberSOC environment server."""

from __future__ import annotations

import httpx
from fastapi.testclient import TestClient

from .models import (
    CyberSOCAction,
    CyberSOCObservation,
    CyberSOCState,
    MetricsResponse,
    ResetResponse,
    StepResponse,
    TaskCatalog,
)


class CyberSOCEnvClient:
    """Synchronous client for `/reset`, `/step`, `/state`, and `/tasks`."""

    def __init__(self, base_url: str, timeout: float = 10.0) -> None:
        self._client = httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout)
        self._session_id: str | None = None

    def close(self) -> None:
        self._client.close()

    def tasks(self) -> TaskCatalog:
        response = self._client.get("/tasks")
        response.raise_for_status()
        return TaskCatalog.model_validate(response.json())

    def reset(self, task_id: str | None = None, seed: int | None = None) -> ResetResponse:
        response = self._client.post(
            "/reset",
            json={"task_id": task_id, "seed": seed},
        )
        response.raise_for_status()
        payload = ResetResponse.model_validate(response.json())
        self._session_id = payload.session_id
        return payload

    def step(self, action: CyberSOCAction) -> StepResponse:
        response = self._client.post(
            "/step",
            json=action.model_dump(mode="json"),
        )
        response.raise_for_status()
        return StepResponse.model_validate(response.json())

    def state(self) -> CyberSOCState:
        response = self._client.get("/state")
        response.raise_for_status()
        return CyberSOCState.model_validate(response.json())

    def observation(self) -> CyberSOCObservation:
        response = self._client.get("/observation")
        response.raise_for_status()
        return CyberSOCObservation.model_validate(response.json())

    def metrics(self) -> MetricsResponse:
        response = self._client.get("/metrics")
        response.raise_for_status()
        return MetricsResponse.model_validate(response.json())


class InProcessCyberSOCEnvClient:
    """Synchronous client that exercises the FastAPI API in-process."""

    def __init__(self, app: object) -> None:
        self._client = TestClient(app)
        self._session_id: str | None = None

    def close(self) -> None:
        self._client.close()

    def tasks(self) -> TaskCatalog:
        response = self._client.get("/tasks")
        response.raise_for_status()
        return TaskCatalog.model_validate(response.json())

    def reset(self, task_id: str | None = None, seed: int | None = None) -> ResetResponse:
        response = self._client.post(
            "/reset",
            json={"task_id": task_id, "seed": seed},
        )
        response.raise_for_status()
        payload = ResetResponse.model_validate(response.json())
        self._session_id = payload.session_id
        return payload

    def step(self, action: CyberSOCAction) -> StepResponse:
        response = self._client.post(
            "/step",
            json=action.model_dump(mode="json"),
        )
        response.raise_for_status()
        return StepResponse.model_validate(response.json())

    def state(self) -> CyberSOCState:
        response = self._client.get("/state")
        response.raise_for_status()
        return CyberSOCState.model_validate(response.json())

    def observation(self) -> CyberSOCObservation:
        response = self._client.get("/observation")
        response.raise_for_status()
        return CyberSOCObservation.model_validate(response.json())

    def metrics(self) -> MetricsResponse:
        response = self._client.get("/metrics")
        response.raise_for_status()
        return MetricsResponse.model_validate(response.json())
