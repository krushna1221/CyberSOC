"""Curated alert reference data for evaluation, demos, and few-shot prompting."""

from __future__ import annotations

import json
from functools import lru_cache
from importlib.resources import files
from typing import Any, Literal

from pydantic import BaseModel, Field

from .models import (
    ActionType,
    AlertKind,
    AlertSeverity,
    CyberSOCObservation,
    Difficulty,
    TriageLabel,
)


class ReferenceResponse(BaseModel):
    primary_action: ActionType
    target_type: Literal["alert", "node", "indicator"]
    classification: TriageLabel | None = None
    justification: str


class CuratedAlertExample(BaseModel):
    example_id: str
    task_id: str
    difficulty: Difficulty
    scenario_family: str
    kind: AlertKind
    severity: AlertSeverity
    source: str
    node_id: str
    node_role: str
    headline: str
    summary: str
    indicator: str | None = None
    related_events: list[str] = Field(default_factory=list)
    expected_label: TriageLabel
    recommended_response: ReferenceResponse
    reasoning_reference: list[str] = Field(default_factory=list)


class CuratedAlertDataset(BaseModel):
    name: str
    version: str
    description: str
    records: list[CuratedAlertExample]


@lru_cache(maxsize=1)
def load_curated_alert_dataset() -> CuratedAlertDataset:
    payload = json.loads(
        files("cybersoc_openenv").joinpath("data", "curated_alerts.json").read_text(encoding="utf-8")
    )
    return CuratedAlertDataset.model_validate(payload)


def select_reference_examples(
    observation: CyberSOCObservation,
    limit: int = 2,
) -> list[CuratedAlertExample]:
    dataset = load_curated_alert_dataset()
    pending_kinds = {alert.kind for alert in observation.pending_alerts}
    visible_nodes = {node.node_id for node in observation.node_overview}
    visible_indicators = set(observation.visible_indicators)

    scored: list[tuple[int, str, CuratedAlertExample]] = []
    for record in dataset.records:
        score = 0
        if record.task_id == observation.task_id:
            score += 6
        if record.kind in pending_kinds:
            score += 3
        if record.node_id in visible_nodes:
            score += 1
        if record.indicator and record.indicator in visible_indicators:
            score += 1
        scored.append((score, record.example_id, record))

    scored.sort(key=lambda item: (-item[0], item[1]))
    return [record for _, _, record in scored[:limit]]


def prompt_reference_examples(
    observation: CyberSOCObservation,
    limit: int = 2,
) -> list[dict[str, Any]]:
    prompt_examples: list[dict[str, Any]] = []
    for record in select_reference_examples(observation, limit=limit):
        prompt_examples.append(
            {
                "kind": record.kind.value,
                "severity": record.severity.value,
                "source": record.source,
                "node_role": record.node_role,
                "headline": record.headline,
                "summary": record.summary,
                "indicator": record.indicator,
                "related_events": record.related_events[:2],
                "playbook_hint": _playbook_hint(record),
                "reasoning_reference": record.reasoning_reference[:3],
            }
        )
    return prompt_examples


def _playbook_hint(record: CuratedAlertExample) -> str:
    action = record.recommended_response.primary_action
    if action == ActionType.TRIAGE_ALERT:
        if record.expected_label == TriageLabel.FALSE_POSITIVE:
            return "verify whether benign business or travel context explains the signal before escalating."
        return "confirm the malicious signal and queue a defensive response without overreacting."
    if action == ActionType.IGNORE_ALERT:
        return "only suppress the alert after validating the benign operational context."
    if action == ActionType.ISOLATE_NODE:
        return "prioritize containment of the likely compromised host to prevent spread."
    if action == ActionType.PATCH_SYSTEM:
        return "close the exposed backup or access path that could enable attacker reuse."
    if action == ActionType.REQUEST_FORENSICS:
        return "gather deeper host or identity evidence before taking a costlier action."
    if action == ActionType.BLOCK_INDICATOR:
        return "consider blocking the visible IOC if other evidence supports active malicious infrastructure."
    if action == ActionType.ESCALATE_INCIDENT:
        return "treat the activity as major-incident material and coordinate broader response."
    return "apply the smallest effective defensive action."
