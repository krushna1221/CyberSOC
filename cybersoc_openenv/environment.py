"""Core environment logic for Autonomous CyberSOC OpenEnv++."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any
from uuid import uuid4

from .models import (
    ActionAuditRecord,
    ActionFeedback,
    ActionType,
    AlertView,
    CyberSOCAction,
    CyberSOCObservation,
    CyberSOCReward,
    CyberSOCState,
    LogEntryView,
    NodeView,
    StepInfo,
    StepResponse,
    TaskDefinitionView,
)
from .scenarios import AlertSpec, LogSpec, NodeSpec, SCENARIOS


class CyberSOCEnvironment:
    STRICT_SCORE_EPSILON = 0.0001

    """Deterministic, partially observable SOC workflow environment."""

    ACTION_COSTS: dict[ActionType, float] = {
        ActionType.TRIAGE_ALERT: 0.20,
        ActionType.ISOLATE_NODE: 2.00,
        ActionType.PATCH_SYSTEM: 1.40,
        ActionType.BLOCK_INDICATOR: 0.60,
        ActionType.REQUEST_FORENSICS: 0.35,
        ActionType.ESCALATE_INCIDENT: 0.80,
        ActionType.IGNORE_ALERT: 0.05,
        ActionType.NOOP: 0.00,
    }

    def __init__(self, default_task_id: str = "alert-triage-easy") -> None:
        self._scenario = SCENARIOS[default_task_id]
        self._last_action_result: ActionFeedback | None = None
        self._last_action: CyberSOCAction | None = None
        self.reset(task_id=default_task_id)

    @property
    def current_task_id(self) -> str | None:
        return getattr(self, "_scenario", None).task_id if getattr(self, "_scenario", None) else None

    def available_tasks(self) -> list[TaskDefinitionView]:
        return [self.task_definition(task_id) for task_id in SCENARIOS]

    def task_definition(self, task_id: str) -> TaskDefinitionView:
        scenario = SCENARIOS[task_id]
        return TaskDefinitionView(
            task_id=scenario.task_id,
            title=scenario.title,
            difficulty=scenario.difficulty,
            goal=scenario.goal,
            success_metric=scenario.success_metric,
            max_steps=scenario.max_steps,
        )

    def reset(self, task_id: str | None = None, seed: int | None = None) -> CyberSOCObservation:
        if task_id is not None:
            if task_id not in SCENARIOS:
                raise KeyError(f"Unknown task_id: {task_id}")
            self._scenario = SCENARIOS[task_id]
        self._seed = 0 if seed is None else seed
        self._episode_id = f"{self._scenario.task_id}-{self._seed}-{uuid4().hex}"
        self._step_count = 0
        self._done = False
        self._terminal_reason: str | None = None
        self._damage = 0.0
        self._cost = 0.0
        self._delay = 0.0
        self._cumulative_reward = 0.0
        self._compromised_nodes = set(self._scenario.initial_compromised)
        self._compromised_at: dict[str, int] = {node_id: 0 for node_id in self._compromised_nodes}
        self._contained_nodes: set[str] = set()
        self._isolated_nodes: set[str] = set()
        self._patched_nodes: set[str] = set()
        self._blocked_indicators: set[str] = set()
        self._triage_decisions: dict[str, str] = {}
        self._ignored_alerts: set[str] = set()
        self._history: list[ActionAuditRecord] = []
        self._visible_alert_ids: list[str] = []
        self._visible_log_ids: list[str] = []
        self._known_alerts: dict[str, AlertSpec] = {}
        self._known_logs: dict[str, LogSpec] = {}
        self._discovered_indicators: set[str] = set()
        self._incident_escalated = False
        self._escalation_step: int | None = None
        self._exfiltration_triggered = False
        self._sensitive_compromised_at: dict[str, int] = {}
        self._backup_entry_used = False
        self._last_action_result = None
        self._last_action = None
        self._attacker_strategy = "monitoring"
        self._bad_action_count = 0

        for alert in self._scenario.initial_alerts:
            self._known_alerts[alert.alert_id] = alert
            self._add_alert_visible(alert)
        for alert_group in self._scenario.compromise_alerts.values():
            for alert in alert_group:
                self._known_alerts[alert.alert_id] = alert

        for log in self._scenario.initial_logs:
            self._known_logs[log.event_id] = log
            self._add_log_visible(log)
        for log_group in self._scenario.forensics_logs.values():
            for log in log_group:
                self._known_logs[log.event_id] = log
        for log_group in self._scenario.compromise_logs.values():
            for log in log_group:
                self._known_logs[log.event_id] = log

        for node_id in self._compromised_nodes:
            if node_id in self._scenario.sensitive_assets:
                self._sensitive_compromised_at[node_id] = self._step_count

        return self._build_observation()

    def state(self) -> CyberSOCState:
        return CyberSOCState(
            episode_id=self._episode_id,
            task_id=self._scenario.task_id,
            task_title=self._scenario.title,
            difficulty=self._scenario.difficulty,
            step_count=self._step_count,
            max_steps=self._scenario.max_steps,
            done=self._done,
            terminal_reason=self._terminal_reason,
            attacker_strategy=self._attacker_strategy,
            threat_level=round(self._threat_level(), 3),
            compromised_nodes=sorted(self._compromised_nodes),
            contained_nodes=sorted(self._contained_nodes),
            isolated_nodes=sorted(self._isolated_nodes),
            patched_nodes=sorted(self._patched_nodes),
            blocked_indicators=sorted(self._blocked_indicators),
            triage_decisions=dict(sorted(self._triage_decisions.items())),
            ignored_alerts=sorted(self._ignored_alerts),
            damage=round(self._damage, 4),
            cost=round(self._cost, 4),
            delay=round(self._delay, 4),
            cumulative_reward=round(self._cumulative_reward, 4),
            task_score=round(self._grader_score(), 4),
            visible_alert_ids=list(self._visible_alert_ids),
            visible_log_ids=list(self._visible_log_ids),
            incident_escalated=self._incident_escalated,
            exfiltration_triggered=self._exfiltration_triggered,
            history=list(self._history),
        )

    def observation(self) -> CyberSOCObservation:
        return self._build_observation()

    def step(self, action: CyberSOCAction | dict[str, Any]) -> tuple[CyberSOCObservation, CyberSOCReward, bool, StepInfo]:
        return self.step_response(action).as_tuple()

    def step_response(self, action: CyberSOCAction | dict[str, Any]) -> StepResponse:
        typed_action = CyberSOCAction.model_validate(action)
        if self._done:
            reward = CyberSOCReward(
                value=-0.02,
                cumulative=round(self._cumulative_reward - 0.02, 4),
                components={"terminal_noop": -0.02},
                explanation="Episode already ended; reset() to begin a new task.",
            )
            self._cumulative_reward = reward.cumulative
            return StepResponse(
                observation=self._build_observation(),
                reward=reward,
                done=True,
                info=StepInfo(
                    progress_score=round(self._progress_score(), 4),
                    estimated_final_score=round(self._progress_score(), 4),
                    grader_score=round(self._grader_score(), 4),
                    terminal_reason=self._terminal_reason,
                    notes=["No new state transition happened because the episode is already finished."],
                ),
            )

        pre_progress = self._progress_score()
        self._step_count += 1
        self._delay = float(self._step_count)

        components: dict[str, float] = {}
        penalties: dict[str, float] = {}
        notes: list[str] = []

        feedback, action_components, action_penalties, action_notes = self._apply_action(typed_action)
        components.update(action_components)
        penalties.update(action_penalties)
        notes.extend(action_notes)

        if self._is_repeated_action(typed_action):
            components["repeat_action"] = -0.08
            penalties["repeat_action"] = -0.08
            notes.append("Repeated identical action reduced analyst efficiency.")

        damage_delta = self._apply_background_damage()
        if damage_delta:
            penalty = round(-0.18 * damage_delta, 4)
            components["damage_pressure"] = penalty
            penalties["damage_pressure"] = penalty

        attacker_notes, attacker_penalties = self._advance_attacker(typed_action)
        notes.extend(attacker_notes)
        penalties.update(attacker_penalties)
        for key, value in attacker_penalties.items():
            components[key] = components.get(key, 0.0) + value

        progress_delta = self._progress_score() - pre_progress
        if progress_delta:
            components["progress_delta"] = round(progress_delta * 0.55, 4)

        components["time_pressure"] = -0.02
        penalties["time_pressure"] = -0.02
        reward_value = round(sum(components.values()), 4)
        self._cumulative_reward = round(self._cumulative_reward + reward_value, 4)

        self._last_action = typed_action
        self._last_action_result = feedback
        self._check_terminal()

        self._history.append(
            ActionAuditRecord(
                step=self._step_count,
                action_type=typed_action.action_type,
                target=self._action_target(typed_action),
                success=feedback.success,
                reward=reward_value,
                note=feedback.summary,
            )
        )

        reward = CyberSOCReward(
            value=reward_value,
            cumulative=self._cumulative_reward,
            components={name: round(value, 4) for name, value in sorted(components.items())},
            explanation=self._reward_explanation(components, notes),
        )
        return StepResponse(
            observation=self._build_observation(),
            reward=reward,
            done=self._done,
            info=StepInfo(
                progress_score=round(self._progress_score(), 4),
                estimated_final_score=round(self._progress_score(), 4),
                grader_score=round(self._grader_score(), 4),
                penalties={name: round(value, 4) for name, value in sorted(penalties.items())},
                terminal_reason=self._terminal_reason,
                notes=notes,
            ),
        )

    def _apply_action(
        self, action: CyberSOCAction
    ) -> tuple[ActionFeedback, dict[str, float], dict[str, float], list[str]]:
        components: dict[str, float] = {}
        penalties: dict[str, float] = {}
        notes: list[str] = []
        cost = self.ACTION_COSTS[action.action_type]
        if cost:
            self._cost = round(self._cost + cost, 4)
            penalties["action_cost"] = round(-0.05 * cost, 4)
            components["action_cost"] = penalties["action_cost"]

        if action.action_type == ActionType.TRIAGE_ALERT:
            alert = self._known_alerts.get(action.alert_id or "")
            if alert is None:
                return self._invalid_action(
                    action,
                    components,
                    penalties,
                    notes,
                    f"Unknown alert id: {action.alert_id}",
                )
            expected = self._scenario.expected_triage.get(alert.alert_id)
            self._triage_decisions[alert.alert_id] = action.classification.value
            if expected == action.classification:
                components["correct_triage"] = 0.20
                notes.append(f"{alert.alert_id} was classified correctly.")
                success = True
                impact = "Correct triage improved queue quality without operational disruption."
            else:
                self._bad_action_count += 1
                components["incorrect_triage"] = -0.18
                penalties["incorrect_triage"] = -0.18
                notes.append(f"{alert.alert_id} was misclassified.")
                success = False
                impact = "Incorrect triage increases analyst rework and may hide a threat."
            return (
                ActionFeedback(
                    action_type=action.action_type,
                    summary=f"Triage decision recorded for {alert.alert_id}.",
                    success=success,
                    impact=impact,
                    visible_changes=[f"{alert.alert_id} marked as {action.classification.value}."],
                ),
                components,
                penalties,
                notes,
            )

        if action.action_type == ActionType.IGNORE_ALERT:
            alert = self._known_alerts.get(action.alert_id or "")
            if alert is None:
                return self._invalid_action(
                    action,
                    components,
                    penalties,
                    notes,
                    f"Unknown alert id: {action.alert_id}",
                )
            self._ignored_alerts.add(alert.alert_id)
            expected_label = self._scenario.expected_triage.get(alert.alert_id)
            if expected_label is not None and expected_label.value == "true_positive":
                self._bad_action_count += 1
                components["ignored_true_positive"] = -0.20
                penalties["ignored_true_positive"] = -0.20
                notes.append(f"Ignoring {alert.alert_id} leaves a genuine threat unhandled.")
                success = False
                impact = "A real malicious signal was left unattended."
            else:
                components["ignored_false_positive"] = 0.02
                notes.append(f"{alert.alert_id} was deprioritized as likely benign noise.")
                success = True
                impact = "Queue noise was reduced with minimal analyst time."
            return (
                ActionFeedback(
                    action_type=action.action_type,
                    summary=f"Alert {alert.alert_id} was ignored.",
                    success=success,
                    impact=impact,
                    visible_changes=[f"{alert.alert_id} status is now ignored."],
                ),
                components,
                penalties,
                notes,
            )

        if action.action_type == ActionType.REQUEST_FORENSICS:
            try:
                node_id = self._resolve_node(action)
            except (KeyError, ValueError) as exc:
                return self._invalid_action(action, components, penalties, notes, str(exc))
            revealed = self._reveal_logs(self._scenario.forensics_logs.get(node_id, ()))
            if revealed:
                components["forensics_value"] = 0.08
                notes.append(f"Forensics revealed {len(revealed)} new log artifacts on {node_id}.")
                success = True
                impact = "Analyst confidence improved through deeper host evidence."
            else:
                components["forensics_noise"] = -0.03
                penalties["forensics_noise"] = -0.03
                notes.append(f"No new evidence was discovered on {node_id}.")
                success = False
                impact = "Analyst time was spent without material new intelligence."
            return (
                ActionFeedback(
                    action_type=action.action_type,
                    summary=f"Forensics request executed for {node_id}.",
                    success=success,
                    impact=impact,
                    visible_changes=[f"{len(revealed)} new logs added for {node_id}."] if revealed else [],
                ),
                components,
                penalties,
                notes,
            )

        if action.action_type == ActionType.ISOLATE_NODE:
            try:
                node_id = self._resolve_node(action)
            except (KeyError, ValueError) as exc:
                return self._invalid_action(action, components, penalties, notes, str(exc))
            self._isolated_nodes.add(node_id)
            disruption_penalty = round(-0.03 * self._node(node_id).criticality, 4)
            components["business_disruption"] = disruption_penalty
            penalties["business_disruption"] = disruption_penalty
            if node_id in self._compromised_nodes:
                self._contained_nodes.add(node_id)
                components["containment_bonus"] = 0.24
                notes.append(f"{node_id} was isolated and the active foothold is now contained.")
                success = True
                impact = "Network spread from the compromised host was cut off."
            else:
                self._bad_action_count += 1
                components["false_isolation"] = -0.14
                penalties["false_isolation"] = -0.14
                notes.append(f"{node_id} was isolated without evidence of compromise.")
                success = False
                impact = "A healthy system was disrupted, increasing business cost."
            return (
                ActionFeedback(
                    action_type=action.action_type,
                    summary=f"Isolation command executed for {node_id}.",
                    success=success,
                    impact=impact,
                    visible_changes=[f"{node_id} network state set to isolated."],
                ),
                components,
                penalties,
                notes,
            )

        if action.action_type == ActionType.PATCH_SYSTEM:
            try:
                node_id = self._resolve_node(action)
            except (KeyError, ValueError) as exc:
                return self._invalid_action(action, components, penalties, notes, str(exc))
            self._patched_nodes.add(node_id)
            if node_id in self._compromised_nodes and node_id in self._isolated_nodes:
                self._contained_nodes.add(node_id)
                components["patch_recovery"] = 0.16
                notes.append(f"{node_id} was patched after isolation, hardening the cleaned asset.")
                success = True
                impact = "The asset was recovered with lower reinfection risk."
            elif node_id not in self._compromised_nodes:
                components["preventive_patch"] = 0.14
                notes.append(f"{node_id} was patched before compromise, reducing attack surface.")
                success = True
                impact = "Preventive patching reduced a likely lateral movement target."
            else:
                components["partial_patch"] = 0.03
                notes.append(f"{node_id} was patched while still active; containment remains incomplete.")
                success = True
                impact = "The patch reduces future risk but does not remove the live foothold."
            return (
                ActionFeedback(
                    action_type=action.action_type,
                    summary=f"Patch operation executed on {node_id}.",
                    success=success,
                    impact=impact,
                    visible_changes=[f"{node_id} marked as patched."],
                ),
                components,
                penalties,
                notes,
            )

        if action.action_type == ActionType.BLOCK_INDICATOR:
            indicator = action.indicator or ""
            self._blocked_indicators.add(indicator)
            if indicator in self._scenario.malicious_indicators:
                components["indicator_block"] = 0.22
                notes.append(f"{indicator} was blocked across the estate.")
                success = True
                impact = "Attacker command-and-control options narrowed immediately."
            else:
                self._bad_action_count += 1
                components["useless_block"] = -0.05
                penalties["useless_block"] = -0.05
                notes.append(f"{indicator} is not part of the known intrusion set.")
                success = False
                impact = "Time was spent on an indicator that does not affect this intrusion."
            return (
                ActionFeedback(
                    action_type=action.action_type,
                    summary=f"Indicator block submitted for {indicator}.",
                    success=success,
                    impact=impact,
                    visible_changes=[f"{indicator} added to network block list."],
                ),
                components,
                penalties,
                notes,
            )

        if action.action_type == ActionType.ESCALATE_INCIDENT:
            try:
                node_id = self._resolve_node(action)
            except (KeyError, ValueError) as exc:
                return self._invalid_action(action, components, penalties, notes, str(exc))
            if not self._incident_escalated:
                self._incident_escalated = True
                self._escalation_step = self._step_count
                components["escalation_value"] = 0.10
                notes.append("Incident was escalated to the response lead.")
                success = True
                impact = "Additional responders are now coordinated, reducing unmanaged damage."
            else:
                components["repeat_escalation"] = -0.02
                penalties["repeat_escalation"] = -0.02
                notes.append("Incident was already escalated earlier in the episode.")
                success = False
                impact = "Repeated escalation adds overhead without changing response posture."
            return (
                ActionFeedback(
                    action_type=action.action_type,
                    summary=f"Incident escalation recorded for {node_id}.",
                    success=success,
                    impact=impact,
                    visible_changes=["External incident-response coordination flag enabled."],
                ),
                components,
                penalties,
                notes,
            )

        self._bad_action_count += 1
        noop_penalty = -0.06 if action.action_type == ActionType.NOOP else -0.04
        components["idle_time"] = noop_penalty
        penalties["idle_time"] = noop_penalty
        notes.append("No meaningful defensive action was taken.")
        return (
            ActionFeedback(
                action_type=action.action_type,
                summary="No-op recorded.",
                success=False,
                impact="Attacker freedom of movement remains unchanged.",
                visible_changes=[],
            ),
            components,
            penalties,
            notes,
        )

    def _invalid_action(
        self,
        action: CyberSOCAction,
        components: dict[str, float],
        penalties: dict[str, float],
        notes: list[str],
        reason: str,
    ) -> tuple[ActionFeedback, dict[str, float], dict[str, float], list[str]]:
        self._bad_action_count += 1
        components["invalid_target"] = -0.22
        penalties["invalid_target"] = -0.22
        notes.append(reason)
        return (
            ActionFeedback(
                action_type=action.action_type,
                summary="Action rejected due to an invalid target.",
                success=False,
                impact="The command had no defensive effect because the referenced target does not exist in this episode.",
                visible_changes=[],
            ),
            components,
            penalties,
            notes,
        )

    def _apply_background_damage(self) -> float:
        unresolved = self._uncontained_compromised_nodes()
        if not unresolved:
            return 0.0
        multiplier = 0.8 if self._incident_escalated else 1.0
        damage_delta = round(sum(self._node(node_id).criticality for node_id in unresolved) * 0.14 * multiplier, 4)
        self._damage = round(self._damage + damage_delta, 4)
        return damage_delta

    def _advance_attacker(self, action: CyberSOCAction) -> tuple[list[str], dict[str, float]]:
        if self._scenario.score_mode == "triage":
            self._attacker_strategy = "triage_only"
            return [], {}

        penalties: dict[str, float] = {}
        notes: list[str] = []
        if self._exfiltration_triggered:
            return notes, penalties

        if self._should_trigger_exfiltration():
            self._damage = round(self._damage + 2.5, 4)
            self._exfiltration_triggered = True
            self._terminal_reason = "data_exfiltration"
            self._done = True
            penalties["data_exfiltration"] = -0.60
            notes.append("The attacker completed data exfiltration from a critical asset.")
            self._attacker_strategy = "successful_exfiltration"
            return notes, penalties

        if self._should_use_backup_entry(action):
            target = self._first_viable(self._scenario.backup_entry_nodes)
            if target is not None:
                self._compromise_node(target)
                self._backup_entry_used = True
                self._attacker_strategy = "adaptive_backup_path"
                penalties["backup_path"] = -0.16
                notes.append(f"Attacker adapted and re-entered through backup access on {target}.")
                return notes, penalties

        active_sources = [
            node_id
            for node_id in sorted(self._compromised_nodes)
            if node_id not in self._contained_nodes and node_id not in self._isolated_nodes
            and (self._step_count - self._compromised_at.get(node_id, self._step_count)) >= 2
        ]
        for source in active_sources:
            for target in self._scenario.attack_graph.get(source, ()):
                if not self._is_viable_target(target):
                    continue
                self._compromise_node(target)
                penalties["attacker_progress"] = -0.18
                notes.append(f"Attacker moved laterally from {source} to {target}.")
                self._attacker_strategy = "lateral_movement"
                return notes, penalties

        if active_sources:
            self._attacker_strategy = "contained_pressure"
        else:
            self._attacker_strategy = "contained"
        return notes, penalties

    def _check_terminal(self) -> None:
        if self._done:
            return

        if self._scenario.score_mode == "triage":
            decisions = len(set(self._triage_decisions) | self._ignored_alerts)
            if decisions >= len(self._scenario.initial_alerts):
                self._done = True
                self._terminal_reason = "all_alerts_handled"
                return
        else:
            if (
                not self._uncontained_compromised_nodes()
                and not self._future_risk_nodes()
                and self._required_alerts_handled()
            ):
                self._done = True
                self._terminal_reason = "attack_contained"
                return

        if self._step_count >= self._scenario.max_steps:
            self._done = True
            self._terminal_reason = "max_steps_reached"

    def _future_risk_nodes(self) -> set[str]:
        if set(self._scenario.malicious_indicators).issubset(self._blocked_indicators) and self._scenario.malicious_indicators:
            return set()

        risk_nodes: set[str] = set()
        for source in self._compromised_nodes:
            if source in self._contained_nodes or source in self._isolated_nodes:
                continue
            for target in self._scenario.attack_graph.get(source, ()):
                if self._is_viable_target(target):
                    risk_nodes.add(target)
        for backup in self._scenario.backup_entry_nodes:
            if self._is_viable_target(backup):
                risk_nodes.add(backup)
        return risk_nodes

    def _progress_score(self) -> float:
        grader = self._grader_score()
        triage_ratio = self._triage_ratio()
        handled_ratio = self._handled_initial_alert_ratio()
        future_risk = len(self._future_risk_nodes()) / max(len(self._scenario.nodes), 1)
        if self._scenario.score_mode == "triage":
            return round(grader, 4)
        if self._scenario.score_mode == "containment":
            contained_ratio = len(self._contained_nodes) / max(len(self._scenario.initial_compromised), 1)
            indicator_ratio = (
                len(set(self._scenario.malicious_indicators) & self._blocked_indicators)
                / max(len(self._scenario.malicious_indicators), 1)
            )
            score = 0.40 * grader + 0.15 * triage_ratio + 0.15 * handled_ratio + 0.20 * contained_ratio + 0.10 * indicator_ratio
            score -= 0.20 * future_risk
            return round(self._clamp(score), 4)
        damage_norm = min(self._damage / self._scenario.damage_budget, 1.0)
        cost_norm = min(self._cost / self._scenario.cost_budget, 1.0)
        delay_norm = min(self._delay / self._scenario.max_steps, 1.0)
        score = 1.0 - (0.35 * damage_norm + 0.20 * cost_norm + 0.15 * delay_norm + 0.20 * future_risk)
        score += 0.06 * triage_ratio + 0.04 * handled_ratio
        return round(self._clamp(score), 4)

    def _grader_score(self) -> float:
        if self._scenario.score_mode == "triage":
            return round(self._clamp(self._triage_ratio()), 4)
        if self._scenario.score_mode == "containment":
            unresolved = len(self._uncontained_compromised_nodes())
            total_nodes = len(self._scenario.nodes)
            return round(self._clamp(1 - (unresolved / total_nodes)), 4)

        damage_norm = min(self._damage / self._scenario.damage_budget, 1.0)
        cost_norm = min(self._cost / self._scenario.cost_budget, 1.0)
        delay_norm = min(self._delay / self._scenario.max_steps, 1.0)
        score = 1.0 - (0.5 * damage_norm + 0.3 * cost_norm + 0.2 * delay_norm)
        return round(self._clamp(score), 4)

    def _triage_ratio(self) -> float:
        total = len(self._scenario.expected_triage)
        if total == 0:
            return 1.0
        correct = 0
        for alert_id, label in self._scenario.expected_triage.items():
            if self._triage_decisions.get(alert_id) == label.value:
                correct += 1
        return correct / total

    def _threat_level(self) -> float:
        unresolved = len(self._uncontained_compromised_nodes())
        visible_tp_alerts = sum(
            1
            for alert_id in self._visible_alert_ids
            if self._scenario.expected_triage.get(alert_id) and self._scenario.expected_triage[alert_id].value == "true_positive"
        )
        risk = (
            0.22 * unresolved
            + 0.15 * len(self._future_risk_nodes())
            + 0.08 * visible_tp_alerts
            + (0.18 if self._incident_escalated else 0.0)
            + (0.25 if self._exfiltration_triggered else 0.0)
        )
        return self._clamp(risk)

    def _build_observation(self) -> CyberSOCObservation:
        alerts = []
        for alert_id in self._visible_alert_ids:
            alert = self._known_alerts[alert_id]
            status = "unreviewed"
            if alert_id in self._ignored_alerts:
                status = "ignored"
            elif alert_id in self._triage_decisions:
                status = "triaged"
            alerts.append(
                AlertView(
                    alert_id=alert.alert_id,
                    kind=alert.kind,
                    severity=alert.severity,
                    node_id=alert.node_id,
                    headline=alert.headline,
                    summary=alert.summary,
                    source=alert.source,
                    triage_status=status,
                )
            )

        logs = [
            LogEntryView(
                event_id=self._known_logs[log_id].event_id,
                time_offset=self._known_logs[log_id].time_offset,
                node_id=self._known_logs[log_id].node_id,
                category=self._known_logs[log_id].category,
                message=self._known_logs[log_id].message,
                suspicious=self._known_logs[log_id].suspicious,
            )
            for log_id in self._visible_log_ids[-8:]
        ]

        node_views = []
        for node in self._scenario.nodes:
            node_views.append(
                NodeView(
                    node_id=node.node_id,
                    role=node.role,
                    business_unit=node.business_unit,
                    criticality=node.criticality,
                    status_hint=self._node_status_hint(node.node_id),
                    isolated=node.node_id in self._isolated_nodes,
                    patched=node.node_id in self._patched_nodes,
                )
            )

        notes = []
        if self._scenario.score_mode == "triage":
            remaining = len(self._scenario.initial_alerts) - len(set(self._triage_decisions) | self._ignored_alerts)
            notes.append(f"{remaining} alerts still require analyst handling.")
        else:
            unresolved = sorted(self._uncontained_compromised_nodes())
            if unresolved:
                notes.append(f"Active compromise remains on: {', '.join(unresolved)}.")
            risk_nodes = sorted(self._future_risk_nodes())
            if risk_nodes:
                notes.append(f"Likely next attacker targets: {', '.join(risk_nodes)}.")
            if self._scenario.require_full_alert_handling:
                remaining_alerts = [
                    alert.alert_id
                    for alert in self._scenario.initial_alerts
                    if alert.alert_id not in self._triage_decisions and alert.alert_id not in self._ignored_alerts
                ]
                if remaining_alerts:
                    notes.append(f"Initial queue still needs handling for: {', '.join(remaining_alerts)}.")
            if self._incident_escalated:
                notes.append("Incident response lead has been engaged.")
            if self._exfiltration_triggered:
                notes.append("Data exfiltration has already occurred.")

        return CyberSOCObservation(
            episode_id=self._episode_id,
            task_id=self._scenario.task_id,
            task_title=self._scenario.title,
            briefing=self._scenario.briefing,
            current_step=self._step_count,
            max_steps=self._scenario.max_steps,
            pending_alerts=alerts,
            recent_logs=logs,
            node_overview=node_views,
            threat_level=round(self._threat_level(), 3),
            visible_indicators=sorted(self._discovered_indicators),
            available_actions=list(ActionType),
            last_action_result=self._last_action_result,
            analyst_notes=notes,
        )

    def _resolve_node(self, action: CyberSOCAction) -> str:
        if action.node_id:
            self._node(action.node_id)
            return action.node_id
        if action.alert_id:
            alert = self._known_alerts.get(action.alert_id)
            if alert is None:
                raise KeyError(f"Unknown alert id: {action.alert_id}")
            return alert.node_id
        raise ValueError("Action does not resolve to a node.")

    def _action_target(self, action: CyberSOCAction) -> str | None:
        return action.node_id or action.alert_id or action.indicator

    def _is_repeated_action(self, action: CyberSOCAction) -> bool:
        previous = self._last_action
        if previous is None:
            return False
        return (
            previous.action_type == action.action_type
            and previous.alert_id == action.alert_id
            and previous.node_id == action.node_id
            and previous.indicator == action.indicator
            and previous.classification == action.classification
        )

    def _node(self, node_id: str) -> NodeSpec:
        for node in self._scenario.nodes:
            if node.node_id == node_id:
                return node
        raise KeyError(f"Unknown node: {node_id}")

    def _node_status_hint(self, node_id: str) -> str:
        if node_id in self._contained_nodes:
            return "contained"
        if node_id in self._isolated_nodes:
            return "isolated"
        if node_id in self._patched_nodes:
            return "patched"
        if self._node_has_visible_signal(node_id):
            return "suspicious"
        return "healthy"

    def _node_has_visible_signal(self, node_id: str) -> bool:
        if any(self._known_alerts[alert_id].node_id == node_id for alert_id in self._visible_alert_ids):
            return True
        return any(
            self._known_logs[log_id].node_id == node_id and self._known_logs[log_id].suspicious
            for log_id in self._visible_log_ids
        )

    def _uncontained_compromised_nodes(self) -> set[str]:
        return {node_id for node_id in self._compromised_nodes if node_id not in self._contained_nodes}

    def _required_alerts_handled(self) -> bool:
        if not self._scenario.require_full_alert_handling:
            return True
        handled = set(self._triage_decisions) | self._ignored_alerts
        return all(alert.alert_id in handled for alert in self._scenario.initial_alerts)

    def _handled_initial_alert_ratio(self) -> float:
        total = len(self._scenario.initial_alerts)
        if total == 0:
            return 1.0
        handled = sum(
            1
            for alert in self._scenario.initial_alerts
            if alert.alert_id in self._triage_decisions or alert.alert_id in self._ignored_alerts
        )
        return handled / total

    def _should_trigger_exfiltration(self) -> bool:
        if set(self._scenario.malicious_indicators).issubset(self._blocked_indicators) and self._scenario.malicious_indicators:
            return False
        for node_id in self._scenario.sensitive_assets:
            if node_id in self._uncontained_compromised_nodes():
                compromised_at = self._sensitive_compromised_at.get(node_id, self._step_count)
                if self._step_count - compromised_at >= 1:
                    return True
        return False

    def _should_use_backup_entry(self, action: CyberSOCAction) -> bool:
        if self._backup_entry_used or not self._scenario.backup_entry_nodes:
            return False
        if set(self._scenario.malicious_indicators).issubset(self._blocked_indicators) and self._scenario.malicious_indicators:
            return False
        disrupted_primary = (
            action.action_type == ActionType.ISOLATE_NODE and self._resolve_node(action) in self._scenario.initial_compromised
        ) or (
            action.action_type == ActionType.BLOCK_INDICATOR
            and bool(self._scenario.malicious_indicators)
            and (action.indicator or "") == self._scenario.malicious_indicators[0]
        )
        return disrupted_primary

    def _is_viable_target(self, node_id: str) -> bool:
        return (
            node_id not in self._compromised_nodes
            and node_id not in self._contained_nodes
            and node_id not in self._isolated_nodes
            and node_id not in self._patched_nodes
        )

    def _first_viable(self, node_ids: Iterable[str]) -> str | None:
        for node_id in node_ids:
            if self._is_viable_target(node_id):
                return node_id
        return None

    def _compromise_node(self, node_id: str) -> None:
        self._compromised_nodes.add(node_id)
        self._compromised_at[node_id] = self._step_count
        criticality = self._node(node_id).criticality
        self._damage = round(self._damage + criticality * 0.65, 4)
        for alert in self._scenario.compromise_alerts.get(node_id, ()):
            self._add_alert_visible(alert)
        self._reveal_logs(self._scenario.compromise_logs.get(node_id, ()))
        if node_id in self._scenario.sensitive_assets:
            self._sensitive_compromised_at[node_id] = self._step_count

    def _add_alert_visible(self, alert: AlertSpec) -> None:
        if alert.alert_id not in self._visible_alert_ids:
            self._visible_alert_ids.append(alert.alert_id)
        self._discover_indicators([alert.headline, alert.summary])

    def _add_log_visible(self, log: LogSpec) -> None:
        if log.event_id not in self._visible_log_ids:
            self._visible_log_ids.append(log.event_id)
        self._discover_indicators([log.message])

    def _reveal_logs(self, logs: Iterable[LogSpec]) -> list[str]:
        revealed: list[str] = []
        for log in logs:
            if log.event_id not in self._visible_log_ids:
                self._add_log_visible(log)
                revealed.append(log.event_id)
        return revealed

    def _discover_indicators(self, texts: Iterable[str]) -> None:
        for text in texts:
            lowered = text.lower()
            for indicator in self._scenario.malicious_indicators:
                if indicator.lower() in lowered:
                    self._discovered_indicators.add(indicator)

    @staticmethod
    def _clamp(value: float) -> float:
        return max(CyberSOCEnvironment.STRICT_SCORE_EPSILON, min(1.0 - CyberSOCEnvironment.STRICT_SCORE_EPSILON, value))

    def _reward_explanation(self, components: dict[str, float], notes: list[str]) -> str:
        dominant = sorted(components.items(), key=lambda item: abs(item[1]), reverse=True)[:3]
        fragments = [f"{name}={value:+.2f}" for name, value in dominant]
        if notes:
            fragments.append(notes[0])
        return "; ".join(fragments) if fragments else "Neutral transition."
