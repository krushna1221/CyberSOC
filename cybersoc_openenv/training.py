"""Training helpers for collecting CyberSOC rollouts and building prompts."""

from __future__ import annotations

import json
import textwrap
from typing import Any, Iterable

from .datasets import prompt_reference_examples
from .models import CyberSOCAction, CyberSOCObservation, TriageLabel

DEFAULT_ENV_SEED = 7
DEFAULT_TASKS = [
    "alert-triage-easy",
    "incident-containment-medium",
    "soc-optimization-hard",
]

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


def observation_payload(observation: CyberSOCObservation) -> dict[str, Any]:
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


def build_prompt_payload(observation: CyberSOCObservation) -> dict[str, Any]:
    return {
        "reference_examples": prompt_reference_examples(observation, limit=2),
        "current_observation": observation_payload(observation),
    }


def heuristic_action(observation: CyberSOCObservation) -> CyberSOCAction:
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


def build_chat_messages(observation: CyberSOCObservation, action: CyberSOCAction) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(build_prompt_payload(observation), indent=2)},
        {
            "role": "assistant",
            "content": json.dumps(
                action.model_dump(mode="json", exclude_none=True),
                sort_keys=True,
            ),
        },
    ]


def generate_sft_examples(
    env_client: Any,
    tasks: Iterable[str] = DEFAULT_TASKS,
    episodes_per_task: int = 4,
    seed_start: int = DEFAULT_ENV_SEED,
) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []

    for task_offset, task_id in enumerate(tasks):
        for episode_index in range(episodes_per_task):
            seed = seed_start + task_offset * 100 + episode_index
            observation = env_client.reset(task_id=task_id, seed=seed).observation
            done = False

            while not done:
                action = heuristic_action(observation)
                row: dict[str, Any] = {
                    "task_id": task_id,
                    "seed": seed,
                    "step": observation.current_step,
                    "messages": build_chat_messages(observation, action),
                    "target_action": action.model_dump(mode="json", exclude_none=True),
                }
                step = env_client.step(action)
                row["reward"] = step.reward.value
                row["done_after_action"] = step.done
                row["terminal_reason"] = step.info.terminal_reason
                examples.append(row)
                observation = step.observation
                done = step.done

    return examples
