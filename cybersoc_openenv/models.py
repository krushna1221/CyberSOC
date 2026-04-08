"""Typed models for the Autonomous CyberSOC OpenEnv++ environment."""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class Difficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class AlertSeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertKind(str, Enum):
    PHISHING = "phishing"
    MALWARE = "malware"
    TOKEN_REPLAY = "token_replay"
    IMPOSSIBLE_TRAVEL = "impossible_travel"
    LATERAL_MOVEMENT = "lateral_movement"
    CREDENTIAL_THEFT = "credential_theft"
    DATA_EXFILTRATION = "data_exfiltration"
    POWERSHELL = "powershell_abuse"
    ANOMALY = "anomaly"


class TriageLabel(str, Enum):
    TRUE_POSITIVE = "true_positive"
    FALSE_POSITIVE = "false_positive"


class ActionType(str, Enum):
    TRIAGE_ALERT = "triage_alert"
    ISOLATE_NODE = "isolate_node"
    PATCH_SYSTEM = "patch_system"
    BLOCK_INDICATOR = "block_indicator"
    REQUEST_FORENSICS = "request_forensics"
    ESCALATE_INCIDENT = "escalate_incident"
    IGNORE_ALERT = "ignore_alert"
    NOOP = "noop"


class AlertView(BaseModel):
    alert_id: str
    kind: AlertKind
    severity: AlertSeverity
    node_id: str
    headline: str
    summary: str
    source: str
    triage_status: Literal["unreviewed", "triaged", "ignored"] = "unreviewed"
    available_labels: list[TriageLabel] = Field(
        default_factory=lambda: [TriageLabel.TRUE_POSITIVE, TriageLabel.FALSE_POSITIVE]
    )


class LogEntryView(BaseModel):
    event_id: str
    time_offset: str
    node_id: str
    category: str
    message: str
    suspicious: bool


class NodeView(BaseModel):
    node_id: str
    role: str
    business_unit: str
    criticality: float
    status_hint: str
    isolated: bool
    patched: bool


class ActionFeedback(BaseModel):
    action_type: ActionType
    summary: str
    success: bool
    impact: str
    visible_changes: list[str] = Field(default_factory=list)


class TaskDefinitionView(BaseModel):
    task_id: str
    title: str
    difficulty: Difficulty
    goal: str
    success_metric: str
    max_steps: int


class TaskCatalog(BaseModel):
    tasks: list[TaskDefinitionView]


class CyberSOCAction(BaseModel):
    action_type: ActionType
    alert_id: str | None = None
    node_id: str | None = None
    indicator: str | None = None
    classification: TriageLabel | None = None
    justification: str | None = None

    @model_validator(mode="after")
    def validate_fields(self) -> "CyberSOCAction":
        if self.action_type == ActionType.TRIAGE_ALERT:
            if not self.alert_id or self.classification is None:
                raise ValueError("triage_alert requires alert_id and classification")
        if self.action_type == ActionType.IGNORE_ALERT and not self.alert_id:
            raise ValueError("ignore_alert requires alert_id")
        if self.action_type in {
            ActionType.ISOLATE_NODE,
            ActionType.PATCH_SYSTEM,
            ActionType.REQUEST_FORENSICS,
            ActionType.ESCALATE_INCIDENT,
        } and not self.node_id and not self.alert_id:
            raise ValueError(f"{self.action_type} requires node_id or alert_id")
        if self.action_type == ActionType.BLOCK_INDICATOR and not self.indicator:
            raise ValueError("block_indicator requires indicator")
        return self


class CyberSOCObservation(BaseModel):
    episode_id: str
    task_id: str
    task_title: str
    briefing: str
    current_step: int
    max_steps: int
    pending_alerts: list[AlertView]
    recent_logs: list[LogEntryView]
    node_overview: list[NodeView]
    threat_level: float
    visible_indicators: list[str]
    available_actions: list[ActionType]
    last_action_result: ActionFeedback | None = None
    analyst_notes: list[str] = Field(default_factory=list)


class CyberSOCReward(BaseModel):
    value: float
    cumulative: float
    components: dict[str, float] = Field(default_factory=dict)
    explanation: str


class ActionAuditRecord(BaseModel):
    step: int
    action_type: ActionType
    target: str | None = None
    success: bool
    reward: float
    note: str


class CyberSOCState(BaseModel):
    episode_id: str
    task_id: str
    task_title: str
    difficulty: Difficulty
    step_count: int
    max_steps: int
    done: bool
    terminal_reason: str | None = None
    attacker_strategy: str
    threat_level: float
    compromised_nodes: list[str]
    contained_nodes: list[str]
    isolated_nodes: list[str]
    patched_nodes: list[str]
    blocked_indicators: list[str]
    triage_decisions: dict[str, str]
    ignored_alerts: list[str]
    damage: float
    cost: float
    delay: float
    cumulative_reward: float
    task_score: float
    visible_alert_ids: list[str]
    visible_log_ids: list[str]
    incident_escalated: bool
    exfiltration_triggered: bool
    history: list[ActionAuditRecord] = Field(default_factory=list)


class ResetRequest(BaseModel):
    task_id: str | None = None
    seed: int | None = None


class ResetResponse(BaseModel):
    session_id: str
    observation: CyberSOCObservation
    state: CyberSOCState
    task: TaskDefinitionView
    available_tasks: list[TaskDefinitionView]


class StepInfo(BaseModel):
    progress_score: float
    estimated_final_score: float
    grader_score: float
    penalties: dict[str, float] = Field(default_factory=dict)
    terminal_reason: str | None = None
    notes: list[str] = Field(default_factory=list)


class StepResponse(BaseModel):
    observation: CyberSOCObservation
    reward: CyberSOCReward
    done: bool
    info: StepInfo


class RootStatus(BaseModel):
    name: str
    version: str
    status: str
    session_id: str
    current_task: str | None
    tasks: list[TaskDefinitionView]


class TaskRunSummary(BaseModel):
    task_id: str
    score: float
    steps: int
    terminal_reason: str | None
    raw: dict[str, Any] = Field(default_factory=dict)
