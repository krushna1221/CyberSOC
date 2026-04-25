"""Baseline inference runner for Autonomous CyberSOC OpenEnv++."""

from __future__ import annotations

import argparse
import json
import os
import re
import textwrap
from typing import Any, Callable

from openai import OpenAI

from cybersoc_openenv.client import CyberSOCEnvClient, InProcessCyberSOCEnvClient
from cybersoc_openenv.datasets import prompt_reference_examples
from cybersoc_openenv.graders import grade_state
from cybersoc_openenv.models import CyberSOCAction, CyberSOCObservation, TaskRunSummary, TriageLabel

DEFAULT_API_BASE_URL = "https://router.huggingface.co/v1"
DEFAULT_MODEL_NAME = "Qwen/Qwen3.5-9B:together"
DEFAULT_ENV_SEED = 7
DEFAULT_TASKS = [
    "alert-triage-easy",
    "incident-containment-medium",
    "soc-optimization-hard",
]
JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)
FALLBACK_ACTION = {"action_type": "noop", "justification": "fallback_noop"}
ProgressCallback = Callable[[str, dict[str, Any]], None]

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are a senior SOC analyst controlling a deterministic cyber defense simulator.
    Return exactly one JSON object and nothing else.

    Valid action schema:
    {
      "action_type": "triage_alert|isolate_node|patch_system|block_indicator|request_forensics|escalate_incident|ignore_alert|noop",
      "alert_id": "optional string",
      "node_id": "optional string",
      "indicator": "optional string",
      "classification": "true_positive|false_positive",
      "justification": "short reason"
    }

    Rules:
    - Prefer the smallest effective action.
    - Only set fields that are relevant to the action.
    - Never add commentary outside the JSON object.
    """
).strip()


def _observation_payload(observation: CyberSOCObservation) -> dict[str, Any]:
    return {
        "task_id": observation.task_id,
        "task_title": observation.task_title,
        "briefing": observation.briefing,
        "current_step": observation.current_step,
        "max_steps": observation.max_steps,
        "threat_level": observation.threat_level,
        "visible_indicators": observation.visible_indicators,
        "analyst_notes": observation.analyst_notes,
        "pending_alerts": [alert.model_dump(mode="json") for alert in observation.pending_alerts],
        "recent_logs": [log.model_dump(mode="json") for log in observation.recent_logs[-6:]],
        "node_overview": [node.model_dump(mode="json") for node in observation.node_overview],
        "last_action_result": observation.last_action_result.model_dump(mode="json") if observation.last_action_result else None,
        "available_actions": [action.value for action in observation.available_actions],
    }


def _build_prompt_payload(observation: CyberSOCObservation) -> dict[str, Any]:
    return {
        "reference_examples": prompt_reference_examples(observation, limit=2),
        "current_observation": _observation_payload(observation),
    }


def _heuristic_action(observation: CyberSOCObservation) -> CyberSOCAction:
    triaged = {alert.alert_id for alert in observation.pending_alerts if alert.triage_status == "triaged"}

    if observation.task_id == "alert-triage-easy":
        for alert in observation.pending_alerts:
            if alert.alert_id in triaged or alert.triage_status == "ignored":
                continue
            label = {
                "ALT-E1": TriageLabel.TRUE_POSITIVE,
                "ALT-E2": TriageLabel.FALSE_POSITIVE,
                "ALT-E3": TriageLabel.TRUE_POSITIVE,
            }[alert.alert_id]
            return CyberSOCAction(
                action_type="triage_alert",
                alert_id=alert.alert_id,
                classification=label,
                justification="reference_heuristic",
            )
        return CyberSOCAction(action_type="noop", justification="all_alerts_done")

    if observation.task_id == "incident-containment-medium":
        ws23 = next(node for node in observation.node_overview if node.node_id == "ws-23")
        vpn = next(node for node in observation.node_overview if node.node_id == "vpn-01")
        if not vpn.patched:
            return CyberSOCAction(action_type="patch_system", node_id="vpn-01", justification="close_backup_access")
        if not ws23.isolated:
            return CyberSOCAction(action_type="isolate_node", node_id="ws-23", justification="stop_primary_spread")
        for alert in observation.pending_alerts:
            if alert.alert_id == "ALT-M1" and alert.triage_status == "unreviewed":
                return CyberSOCAction(
                    action_type="triage_alert",
                    alert_id=alert.alert_id,
                    classification=TriageLabel.TRUE_POSITIVE,
                    justification="primary_compromise",
                )
            if alert.alert_id == "ALT-M2" and alert.triage_status == "unreviewed":
                return CyberSOCAction(
                    action_type="triage_alert",
                    alert_id=alert.alert_id,
                    classification=TriageLabel.FALSE_POSITIVE,
                    justification="maintenance_noise",
                )
        if "185.17.44.22" in observation.visible_indicators:
            return CyberSOCAction(action_type="block_indicator", indicator="185.17.44.22", justification="remove_c2")
        return CyberSOCAction(action_type="noop", justification="contained")

    fin_ws = next(node for node in observation.node_overview if node.node_id == "fin-ws-07")
    vpn = next(node for node in observation.node_overview if node.node_id == "vpn-02")
    if not vpn.patched:
        return CyberSOCAction(action_type="patch_system", node_id="vpn-02", justification="deny_backup_path")
    if not fin_ws.isolated:
        return CyberSOCAction(action_type="isolate_node", node_id="fin-ws-07", justification="contain_initial_host")
    triage_targets = {
        "ALT-H1": TriageLabel.TRUE_POSITIVE,
        "ALT-H2": TriageLabel.FALSE_POSITIVE,
        "ALT-H3": TriageLabel.TRUE_POSITIVE,
        "ALT-H4": TriageLabel.TRUE_POSITIVE,
        "ALT-H5": TriageLabel.TRUE_POSITIVE,
        "ALT-H6": TriageLabel.TRUE_POSITIVE,
        "ALT-H7": TriageLabel.TRUE_POSITIVE,
    }
    for alert in observation.pending_alerts:
        if alert.alert_id in triage_targets and alert.triage_status == "unreviewed":
            return CyberSOCAction(
                action_type="triage_alert",
                alert_id=alert.alert_id,
                classification=triage_targets[alert.alert_id],
                justification="reference_heuristic",
            )
    if not vpn.patched:
        return CyberSOCAction(action_type="patch_system", node_id="vpn-02", justification="deny_backup_path")
    if "185.17.44.22" not in observation.visible_indicators:
        return CyberSOCAction(action_type="request_forensics", node_id="fin-ws-07", justification="identify_iocs")
    if "cdn-sync.net" in observation.visible_indicators:
        return CyberSOCAction(action_type="block_indicator", indicator="cdn-sync.net", justification="block_backup_c2")
    return CyberSOCAction(action_type="escalate_incident", node_id="finance-db", justification="coordinate_ir")


def _extract_json(text: str) -> dict[str, Any]:
    match = JSON_BLOCK_RE.search(text)
    if not match:
        return dict(FALLBACK_ACTION)
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return dict(FALLBACK_ACTION)


def _llm_action(client: OpenAI, model_name: str, observation: CyberSOCObservation) -> CyberSOCAction:
    prompt = json.dumps(_build_prompt_payload(observation), indent=2)
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        max_tokens=240,
        seed=DEFAULT_ENV_SEED,
    )
    content = response.choices[0].message.content or ""
    payload = _extract_json(content)
    try:
        return CyberSOCAction.model_validate(payload)
    except Exception:
        return CyberSOCAction.model_validate(FALLBACK_ACTION)


def _build_openai_client() -> tuple[OpenAI, str]:
    api_base_url = os.getenv("API_BASE_URL", DEFAULT_API_BASE_URL)
    api_key = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")
    model_name = os.getenv("MODEL_NAME", DEFAULT_MODEL_NAME)
    if not api_key:
        raise RuntimeError(
            "No API key found. Set HF_TOKEN (or OPENAI_API_KEY / API_KEY) before running."
        )
    return OpenAI(base_url=api_base_url, api_key=api_key), model_name


def _build_env_client() -> tuple[CyberSOCEnvClient | InProcessCyberSOCEnvClient, str]:
    env_base_url = os.getenv("ENV_BASE_URL")
    if env_base_url:
        return CyberSOCEnvClient(base_url=env_base_url), env_base_url

    from server.app import app

    return InProcessCyberSOCEnvClient(app), "in-process-fastapi"


def _format_progress_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.4f}"
    text = str(value)
    return text.replace(" ", "_")


def _stdout_progress(tag: str, payload: dict[str, Any]) -> None:
    if tag == "START":
        line = f"[START] task={_format_progress_value(payload['task'])}"
    elif tag == "STEP":
        line = (
            f"[STEP] step={_format_progress_value(payload['step'])} "
            f"reward={_format_progress_value(payload['reward'])}"
        )
    elif tag == "END":
        line = (
            f"[END] task={_format_progress_value(payload['task'])} "
            f"score={_format_progress_value(payload['score'])} "
            f"steps={_format_progress_value(payload['steps'])}"
        )
    else:
        fields = " ".join(f"{key}={_format_progress_value(value)}" for key, value in payload.items())
        line = f"[{tag}] {fields}".rstrip()
    print(line, flush=True)


def run(policy: str, progress_callback: ProgressCallback | None = None) -> dict[str, Any]:
    env_client, environment_target = _build_env_client()
    client = None
    model_name = None
    effective_policy = policy
    warnings: list[str] = []
    if policy == "llm":
        try:
            client, model_name = _build_openai_client()
        except RuntimeError as exc:
            effective_policy = "heuristic"
            warnings.append(
                f"LLM configuration unavailable ({exc}); falling back to heuristic baseline."
            )

    summaries: list[TaskRunSummary] = []
    try:
        for task_id in DEFAULT_TASKS:
            observation = env_client.reset(task_id=task_id, seed=DEFAULT_ENV_SEED).observation
            if progress_callback is not None:
                progress_callback(
                    "START",
                    {
                        "task": task_id,
                    },
                )
            done = False
            while not done:
                if effective_policy == "heuristic":
                    action = _heuristic_action(observation)
                else:
                    try:
                        action = _llm_action(client, model_name, observation)
                    except Exception as exc:
                        warnings.append(
                            f"LLM action failed on task {task_id} step {observation.current_step}; "
                            f"falling back to heuristic action ({type(exc).__name__})."
                        )
                        effective_policy = "heuristic"
                        action = _heuristic_action(observation)
                step = env_client.step(action)
                observation = step.observation
                done = step.done
                if progress_callback is not None:
                    progress_callback(
                        "STEP",
                        {
                            "step": observation.current_step,
                            "reward": step.reward.value,
                        },
                    )

            state = env_client.state()
            score = grade_state(state)
            summaries.append(
                TaskRunSummary(
                    task_id=task_id,
                    score=score,
                    steps=state.step_count,
                    terminal_reason=state.terminal_reason,
                    raw={
                        "damage": state.damage,
                        "cost": state.cost,
                        "delay": state.delay,
                        "reward": state.cumulative_reward,
                    },
                )
            )
            if progress_callback is not None:
                progress_callback(
                    "END",
                    {
                        "task": task_id,
                        "score": score,
                        "steps": state.step_count,
                    },
                )
    finally:
        env_client.close()

    average_score = round(sum(summary.score for summary in summaries) / len(summaries), 4)
    return {
        "policy": policy,
        "effective_policy": effective_policy,
        "model_name": model_name,
        "environment_target": environment_target,
        "seed": DEFAULT_ENV_SEED,
        "task_results": [summary.model_dump(mode="json") for summary in summaries],
        "average_score": average_score,
        "warnings": warnings,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the CyberSOC OpenEnv baseline inference.")
    parser.add_argument("--policy", choices=["llm", "heuristic"], default="llm")
    args = parser.parse_args()
    result = run(policy=args.policy, progress_callback=_stdout_progress)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
