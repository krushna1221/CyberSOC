"""FastAPI app exposing the CyberSOC OpenEnv endpoints."""

from __future__ import annotations

import argparse
from collections import OrderedDict
import os
from pathlib import Path
from threading import Lock
from uuid import uuid4

import uvicorn
from fastapi import Body, FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from cybersoc_openenv.environment import CyberSOCEnvironment
from cybersoc_openenv.models import (
    CyberSOCAction,
    MetricsResponse,
    CyberSOCObservation,
    CyberSOCState,
    ResetRequest,
    ResetResponse,
    RootStatus,
    SessionMetrics,
    StepResponse,
    TaskCatalog,
    TaskDefinitionView,
)
from cybersoc_openenv.scenarios import SCENARIOS


SESSION_COOKIE_NAME = "cybersoc_session"
SESSION_MAX_AGE_SECONDS = 60 * 60 * 24


class SessionStore:
    """Bounded in-memory session store for per-user environments."""

    def __init__(self, max_sessions: int = 256) -> None:
        self._lock = Lock()
        self._max_sessions = max_sessions
        self._sessions: OrderedDict[str, CyberSOCEnvironment] = OrderedDict()

    def get_or_create(self, session_id: str | None) -> tuple[str, CyberSOCEnvironment]:
        with self._lock:
            resolved_id = session_id or uuid4().hex
            env = self._sessions.get(resolved_id)
            if env is None:
                env = CyberSOCEnvironment()
                self._sessions[resolved_id] = env
            else:
                self._sessions.move_to_end(resolved_id)
            while len(self._sessions) > self._max_sessions:
                self._sessions.popitem(last=False)
            return resolved_id, env

    def get(self, session_id: str | None) -> tuple[str, CyberSOCEnvironment] | None:
        if not session_id:
            return None
        with self._lock:
            env = self._sessions.get(session_id)
            if env is None:
                return None
            self._sessions.move_to_end(session_id)
            return session_id, env

    def snapshot(self) -> list[tuple[str, CyberSOCEnvironment]]:
        with self._lock:
            return list(self._sessions.items())


SESSION_STORE = SessionStore()
STATIC_DIR = Path(__file__).with_name("static")
app = FastAPI(
    title="Autonomous CyberSOC OpenEnv++",
    version="0.2.0",
    description="Deterministic SOC workflow simulation with OpenEnv-style reset/step/state endpoints.",
)
app.mount("/assets", StaticFiles(directory=STATIC_DIR), name="assets")


def _requested_session_id(request: Request, session_id: str | None) -> str | None:
    return request.cookies.get(SESSION_COOKIE_NAME) or session_id


def _bind_session(response: Response, session_id: str) -> None:
    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=session_id,
        httponly=True,
        max_age=SESSION_MAX_AGE_SECONDS,
        samesite="lax",
    )


def _session_env(request: Request, session_id: str | None, create: bool = False) -> tuple[str, CyberSOCEnvironment]:
    requested_id = _requested_session_id(request, session_id)
    if create:
        return SESSION_STORE.get_or_create(requested_id)

    session = SESSION_STORE.get(requested_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found. Call /reset first.")
    return session


def _reset_response(session_id: str, env: CyberSOCEnvironment, task_id: str | None = None, seed: int | None = None) -> ResetResponse:
    try:
        observation = env.reset(task_id=task_id, seed=seed)
    except KeyError as exc:
        message = exc.args[0] if exc.args else "Unknown task id."
        raise HTTPException(status_code=404, detail=str(message)) from exc
    return ResetResponse(
        session_id=session_id,
        observation=observation,
        state=env.state(),
        task=env.task_definition(env.current_task_id or "alert-triage-easy"),
        available_tasks=env.available_tasks(),
    )


def _task_catalog() -> list[TaskDefinitionView]:
    env = CyberSOCEnvironment()
    return [env.task_definition(task_id) for task_id in SCENARIOS]


def _aggregate_metrics(
    sessions: list[SessionMetrics],
    scope: str,
    *,
    include_sessions: bool,
) -> MetricsResponse:
    total_actions = sum(item.total_actions for item in sessions)
    alerts_expected = sum(item.alerts_expected for item in sessions)
    return MetricsResponse(
        scope=scope,
        active_sessions=len(sessions),
        episodes_completed=sum(1 for item in sessions if item.done),
        total_actions=total_actions,
        successful_actions=sum(item.successful_actions for item in sessions),
        failed_actions=sum(item.failed_actions for item in sessions),
        total_alerts_processed=sum(item.total_alerts_processed for item in sessions),
        alerts_expected=alerts_expected,
        correct_triage=sum(item.correct_triage for item in sessions),
        false_positives=sum(item.false_positives for item in sessions),
        false_negatives=sum(item.false_negatives for item in sessions),
        triage_accuracy=round(
            (sum(item.correct_triage for item in sessions) / alerts_expected) if alerts_expected else 0.0,
            4,
        ),
        average_response_time=round(
            (sum(item.average_response_time * item.total_actions for item in sessions) / total_actions) if total_actions else 0.0,
            4,
        ),
        average_reward=round(
            (sum(item.average_reward * item.total_actions for item in sessions) / total_actions) if total_actions else 0.0,
            4,
        ),
        average_task_score=round(
            (sum(item.task_score for item in sessions) / len(sessions)) if sessions else 0.0,
            4,
        ),
        sessions=sessions if include_sessions else [],
    )


@app.get("/", include_in_schema=False)
def root(
) -> FileResponse:
    return FileResponse(STATIC_DIR / "cybersoc.html")


@app.get("/api/status", response_model=RootStatus)
def api_status(
    request: Request,
    response: Response,
    session_id: str | None = Query(default=None),
) -> RootStatus:
    resolved_session_id = _requested_session_id(request, session_id)
    session = SESSION_STORE.get(resolved_session_id)
    env = session[1] if session is not None else None
    if session is not None:
        _bind_session(response, session[0])
    return RootStatus(
        name="Autonomous CyberSOC OpenEnv++",
        version="0.2.0",
        status="ok",
        session_id=session[0] if session is not None else None,
        current_task=env.current_task_id if env is not None else None,
        tasks=_task_catalog(),
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/tasks", response_model=TaskCatalog)
def tasks(
) -> TaskCatalog:
    return TaskCatalog(tasks=_task_catalog())


@app.get("/metrics", response_model=MetricsResponse)
def metrics(
    request: Request,
    response: Response,
    session_id: str | None = Query(default=None),
) -> MetricsResponse:
    requested_id = _requested_session_id(request, session_id)
    if requested_id:
        resolved_session_id, env = _session_env(request, session_id, create=False)
        _bind_session(response, resolved_session_id)
        return _aggregate_metrics(
            [env.metrics(session_id=resolved_session_id)],
            scope="session",
            include_sessions=True,
        )

    sessions = [env.metrics(session_id=stored_session_id) for stored_session_id, env in SESSION_STORE.snapshot()]
    return _aggregate_metrics(sessions, scope="global", include_sessions=False)


@app.get("/reset", response_model=ResetResponse)
def reset_get(
    request: Request,
    response: Response,
    task_id: str | None = Query(default=None),
    seed: int | None = Query(default=None),
    session_id: str | None = Query(default=None),
) -> ResetResponse:
    resolved_session_id, env = _session_env(request, session_id, create=True)
    _bind_session(response, resolved_session_id)
    return _reset_response(resolved_session_id, env, task_id=task_id, seed=seed)


@app.post("/reset", response_model=ResetResponse)
def reset_post(
    request: Request,
    response: Response,
    payload: ResetRequest = Body(default_factory=ResetRequest),
    session_id: str | None = Query(default=None),
) -> ResetResponse:
    resolved_session_id, env = _session_env(request, session_id, create=True)
    _bind_session(response, resolved_session_id)
    return _reset_response(resolved_session_id, env, task_id=payload.task_id, seed=payload.seed)


@app.post("/step", response_model=StepResponse)
def step(
    action: CyberSOCAction,
    request: Request,
    response: Response,
    session_id: str | None = Query(default=None),
) -> StepResponse:
    resolved_session_id, env = _session_env(request, session_id, create=False)
    _bind_session(response, resolved_session_id)
    try:
        return env.step_response(action)
    except (KeyError, ValueError) as exc:
        message = exc.args[0] if exc.args else "Invalid action."
        raise HTTPException(status_code=400, detail=str(message)) from exc


@app.get("/observation", response_model=CyberSOCObservation)
def observation(
    request: Request,
    response: Response,
    session_id: str | None = Query(default=None),
) -> CyberSOCObservation:
    resolved_session_id, env = _session_env(request, session_id, create=False)
    _bind_session(response, resolved_session_id)
    return env.observation()


@app.get("/state", response_model=CyberSOCState)
def state(
    request: Request,
    response: Response,
    session_id: str | None = Query(default=None),
) -> CyberSOCState:
    resolved_session_id, env = _session_env(request, session_id, create=False)
    _bind_session(response, resolved_session_id)
    return env.state()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Autonomous CyberSOC OpenEnv++ server.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "7860")))
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()
    uvicorn.run("server.app:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
