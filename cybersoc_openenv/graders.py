"""Standalone deterministic graders for the CyberSOC tasks."""

from __future__ import annotations

from .models import CyberSOCState
from .scenarios import SCENARIOS

STRICT_SCORE_EPSILON = 0.0001


def _clamp(value: float) -> float:
    return max(STRICT_SCORE_EPSILON, min(1.0 - STRICT_SCORE_EPSILON, value))


def grade_state(state: CyberSOCState) -> float:
    """Return a validator-safe task score strictly inside the open interval (0, 1)."""
    scenario = SCENARIOS[state.task_id]
    if scenario.score_mode == "triage":
        expected = {alert_id: label.value for alert_id, label in scenario.expected_triage.items()}
        correct = sum(1 for alert_id, label in expected.items() if state.triage_decisions.get(alert_id) == label)
        return round(_clamp(correct / len(expected)), 4)

    if scenario.score_mode == "containment":
        total_nodes = len(scenario.nodes)
        unresolved = len(set(state.compromised_nodes) - set(state.contained_nodes))
        return round(_clamp(1 - (unresolved / total_nodes)), 4)

    if scenario.score_mode == "optimization":
        damage_norm = min(state.damage / scenario.damage_budget, 1.0)
        cost_norm = min(state.cost / scenario.cost_budget, 1.0)
        delay_norm = min(state.delay / scenario.max_steps, 1.0)
        return round(_clamp(1 - (0.5 * damage_norm + 0.3 * cost_norm + 0.2 * delay_norm)), 4)

    raise KeyError(f"Unknown task id: {state.task_id}")
