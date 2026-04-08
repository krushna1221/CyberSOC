"""FastAPI app exposing the CyberSOC OpenEnv endpoints."""

from __future__ import annotations

import argparse
from collections import OrderedDict
from pathlib import Path
from threading import Lock
from uuid import uuid4

import uvicorn
from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from cybersoc_openenv.environment import CyberSOCEnvironment
from cybersoc_openenv.models import (
    CyberSOCAction,
    CyberSOCObservation,
    CyberSOCState,
    ResetRequest,
    ResetResponse,
    RootStatus,
    StepResponse,
    TaskCatalog,
)


SESSION_COOKIE_NAME = "cybersoc_session"
SESSION_MAX_AGE_SECONDS = 60 * 60 * 24


class SessionStore:
    """Bounded in-memory session store for per-user environments."""

    def __init__(self, max_sessions: int = 64) -> None:
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


SESSION_STORE = SessionStore()
STATIC_DIR = Path(__file__).with_name("static")
app = FastAPI(
    title="Autonomous CyberSOC OpenEnv++",
    version="0.2.0",
    description="Deterministic SOC workflow simulation with OpenEnv-style reset/step/state endpoints.",
)
app.mount("/assets", StaticFiles(directory=STATIC_DIR), name="assets")


def _requested_session_id(request: Request, session_id: str | None) -> str | None:
    return session_id or request.cookies.get(SESSION_COOKIE_NAME)


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
    observation = env.reset(task_id=task_id, seed=seed)
    return ResetResponse(
        session_id=session_id,
        observation=observation,
        state=env.state(),
        task=env.task_definition(env.current_task_id or "alert-triage-easy"),
        available_tasks=env.available_tasks(),
    )


@app.get("/", include_in_schema=False)
def root(
    request: Request,
    session_id: str | None = Query(default=None),
) -> FileResponse:
    resolved_session_id, _ = _session_env(request, session_id, create=True)
    response = FileResponse(STATIC_DIR / "cybersoc.html")
    _bind_session(response, resolved_session_id)
    return response


@app.get("/api/status", response_model=RootStatus)
def api_status(
    request: Request,
    response: Response,
    session_id: str | None = Query(default=None),
) -> RootStatus:
    resolved_session_id, env = _session_env(request, session_id, create=True)
    _bind_session(response, resolved_session_id)
    return RootStatus(
        name="Autonomous CyberSOC OpenEnv++",
        version="0.2.0",
        status="ok",
        session_id=resolved_session_id,
        current_task=env.current_task_id,
        tasks=env.available_tasks(),
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/tasks", response_model=TaskCatalog)
def tasks(
    request: Request,
    response: Response,
    session_id: str | None = Query(default=None),
) -> TaskCatalog:
    resolved_session_id, env = _session_env(request, session_id, create=True)
    _bind_session(response, resolved_session_id)
    return TaskCatalog(tasks=env.available_tasks())


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
    payload: ResetRequest,
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
    return env.step(action)


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
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()
    uvicorn.run("server.app:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
