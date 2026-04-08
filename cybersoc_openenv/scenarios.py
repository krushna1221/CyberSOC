"""Deterministic task definitions for the CyberSOC environment."""

from __future__ import annotations

from dataclasses import dataclass, field

from .models import AlertKind, AlertSeverity, Difficulty, TriageLabel


@dataclass(frozen=True)
class NodeSpec:
    node_id: str
    role: str
    business_unit: str
    criticality: float
    vulnerabilities: tuple[str, ...] = ()


@dataclass(frozen=True)
class AlertSpec:
    alert_id: str
    kind: AlertKind
    severity: AlertSeverity
    node_id: str
    headline: str
    summary: str
    source: str
    expected_label: TriageLabel


@dataclass(frozen=True)
class LogSpec:
    event_id: str
    time_offset: str
    node_id: str
    category: str
    message: str
    suspicious: bool


@dataclass(frozen=True)
class TaskScenario:
    task_id: str
    title: str
    difficulty: Difficulty
    goal: str
    success_metric: str
    briefing: str
    max_steps: int
    score_mode: str
    nodes: tuple[NodeSpec, ...]
    initial_alerts: tuple[AlertSpec, ...]
    initial_logs: tuple[LogSpec, ...]
    expected_triage: dict[str, TriageLabel]
    require_full_alert_handling: bool = False
    initial_compromised: tuple[str, ...] = ()
    attack_graph: dict[str, tuple[str, ...]] = field(default_factory=dict)
    backup_entry_nodes: tuple[str, ...] = ()
    sensitive_assets: tuple[str, ...] = ()
    malicious_indicators: tuple[str, ...] = ()
    forensics_logs: dict[str, tuple[LogSpec, ...]] = field(default_factory=dict)
    compromise_alerts: dict[str, tuple[AlertSpec, ...]] = field(default_factory=dict)
    compromise_logs: dict[str, tuple[LogSpec, ...]] = field(default_factory=dict)
    damage_budget: float = 8.0
    cost_budget: float = 8.0


ALERT_TRIAGE_EASY = TaskScenario(
    task_id="alert-triage-easy",
    title="Task 1 - Alert Triage",
    difficulty=Difficulty.EASY,
    goal="Classify each SOC alert as a true positive or false positive.",
    success_metric="Score = correct classifications / total alerts.",
    briefing=(
        "You are the first-line analyst on a Monday morning queue. "
        "Three alerts are waiting for triage. No containment is required for this task, "
        "but every alert must be classified correctly under limited time."
    ),
    max_steps=5,
    score_mode="triage",
    nodes=(
        NodeSpec("alice-laptop", "workstation", "sales", 0.40, ("office-macro-policy-gap",)),
        NodeSpec("vpn-01", "vpn", "infra", 0.70, ()),
        NodeSpec("mail-gw", "mail_gateway", "infra", 0.60, ()),
        NodeSpec("sales-crm", "application_server", "sales", 0.80, ()),
    ),
    initial_alerts=(
        AlertSpec(
            "ALT-E1",
            AlertKind.PHISHING,
            AlertSeverity.HIGH,
            "alice-laptop",
            "Suspicious invoice macro execution",
            "Email attachment launched powershell.exe after the user opened invoice_Q2.xlsm.",
            "EDR",
            TriageLabel.TRUE_POSITIVE,
        ),
        AlertSpec(
            "ALT-E2",
            AlertKind.IMPOSSIBLE_TRAVEL,
            AlertSeverity.MEDIUM,
            "vpn-01",
            "Impossible travel for sales manager",
            "Login from Bangalore occurred 17 minutes after a New York sign-in, but the user is on approved travel.",
            "IdP",
            TriageLabel.FALSE_POSITIVE,
        ),
        AlertSpec(
            "ALT-E3",
            AlertKind.MALWARE,
            AlertSeverity.MEDIUM,
            "mail-gw",
            "Known bad sender domain hit mail gateway",
            "Mail gateway sandbox matched the sender domain to a current phishing campaign IOC.",
            "Mail Sandbox",
            TriageLabel.TRUE_POSITIVE,
        ),
    ),
    initial_logs=(
        LogSpec(
            "LOG-E1",
            "T+00m",
            "alice-laptop",
            "process",
            "WINWORD.EXE spawned powershell.exe -EncodedCommand JAB...",
            True,
        ),
        LogSpec(
            "LOG-E2",
            "T+01m",
            "vpn-01",
            "auth",
            "User account smalik satisfied MFA from approved travel destination IN-BLR.",
            False,
        ),
        LogSpec(
            "LOG-E3",
            "T+02m",
            "mail-gw",
            "email",
            "Sender domain invoice-apac-payments.co matched threat intel confidence=0.97.",
            True,
        ),
    ),
    expected_triage={
        "ALT-E1": TriageLabel.TRUE_POSITIVE,
        "ALT-E2": TriageLabel.FALSE_POSITIVE,
        "ALT-E3": TriageLabel.TRUE_POSITIVE,
    },
    forensics_logs={
        "vpn-01": (
            LogSpec(
                "LOG-E4",
                "T+03m",
                "vpn-01",
                "travel",
                "Travel desk ticket #4431 confirms employee is in Bangalore for customer meetings.",
                False,
            ),
        ),
        "alice-laptop": (
            LogSpec(
                "LOG-E5",
                "T+04m",
                "alice-laptop",
                "network",
                "powershell.exe reached hxxp://185.17.44.22/bootstrap.ps1",
                True,
            ),
        ),
    },
    damage_budget=4.0,
    cost_budget=3.0,
)


INCIDENT_CONTAINMENT_MEDIUM = TaskScenario(
    task_id="incident-containment-medium",
    title="Task 2 - Incident Containment",
    difficulty=Difficulty.MEDIUM,
    goal="Stop ransomware-style lateral movement before it reaches critical infrastructure and close the active analyst queue cleanly.",
    success_metric="Score = 1 - (uncontained infected nodes / total nodes); the episode closes only after all opening alerts are handled.",
    briefing=(
        "An EDR beacon on workstation ws-23 likely indicates a hands-on-keyboard intrusion. "
        "Your job is to contain the spread before the attacker reaches file and identity systems. "
        "A low-confidence VPN alert is also present as a distractor, and the attacker may pivot to VPN access if the primary foothold is disrupted."
    ),
    max_steps=7,
    score_mode="containment",
    nodes=(
        NodeSpec("ws-23", "workstation", "finance", 0.40, ("smb-signing-disabled",)),
        NodeSpec("fs-02", "file_server", "finance", 0.70, ("legacy-smb",)),
        NodeSpec("backup-01", "backup_server", "infra", 0.90, ("outdated-agent",)),
        NodeSpec("dc-01", "domain_controller", "infra", 1.00, ("stale-admin-token",)),
        NodeSpec("vpn-01", "vpn", "infra", 0.60, ()),
    ),
    initial_alerts=(
        AlertSpec(
            "ALT-M1",
            AlertKind.MALWARE,
            AlertSeverity.HIGH,
            "ws-23",
            "Beaconing to known ransomware C2",
            "EDR saw rundll32.exe contacting 185.17.44.22 over TLS every 30 seconds.",
            "EDR",
            TriageLabel.TRUE_POSITIVE,
        ),
        AlertSpec(
            "ALT-M2",
            AlertKind.IMPOSSIBLE_TRAVEL,
            AlertSeverity.LOW,
            "vpn-01",
            "Possible impossible travel on VPN admin",
            "The admin's mobile IP rotated across two nearby regions during an active maintenance window.",
            "IdP",
            TriageLabel.FALSE_POSITIVE,
        ),
    ),
    initial_logs=(
        LogSpec(
            "LOG-M1",
            "T+00m",
            "ws-23",
            "process",
            "rundll32.exe loaded temp\\printerdriver.dll from user profile.",
            True,
        ),
        LogSpec(
            "LOG-M2",
            "T+01m",
            "ws-23",
            "network",
            "TLS session established to 185.17.44.22:443 with JA3 hash matching BlackSleet.",
            True,
        ),
        LogSpec(
            "LOG-M3",
            "T+02m",
            "vpn-01",
            "auth",
            "Admin account completed MFA from known maintenance ASN.",
            False,
        ),
    ),
    expected_triage={
        "ALT-M1": TriageLabel.TRUE_POSITIVE,
        "ALT-M2": TriageLabel.FALSE_POSITIVE,
    },
    require_full_alert_handling=True,
    initial_compromised=("ws-23",),
    attack_graph={
        "ws-23": ("fs-02", "backup-01"),
        "fs-02": ("dc-01",),
        "backup-01": ("dc-01",),
    },
    backup_entry_nodes=("vpn-01",),
    sensitive_assets=("dc-01",),
    malicious_indicators=("185.17.44.22",),
    forensics_logs={
        "ws-23": (
            LogSpec(
                "LOG-M4",
                "T+03m",
                "ws-23",
                "credential_access",
                "lsass.exe memory access followed by SMB session setup to \\\\fs-02\\finance.",
                True,
            ),
        ),
        "fs-02": (
            LogSpec(
                "LOG-M5",
                "T+04m",
                "fs-02",
                "fileshare",
                "Service account svc-backup touched 1,432 finance files in 90 seconds.",
                True,
            ),
        ),
    },
    compromise_alerts={
        "fs-02": (
            AlertSpec(
                "ALT-M3",
                AlertKind.LATERAL_MOVEMENT,
                AlertSeverity.HIGH,
                "fs-02",
                "Unusual SMB lateral movement into finance file server",
                "A compromised workstation reused finance credentials against fs-02.",
                "NDR",
                TriageLabel.TRUE_POSITIVE,
            ),
        ),
        "backup-01": (
            AlertSpec(
                "ALT-M4",
                AlertKind.CREDENTIAL_THEFT,
                AlertSeverity.MEDIUM,
                "backup-01",
                "Backup server service account anomaly",
                "svc-backup authenticated from a workstation subnet outside its normal schedule.",
                "SIEM",
                TriageLabel.TRUE_POSITIVE,
            ),
        ),
        "dc-01": (
            AlertSpec(
                "ALT-M5",
                AlertKind.DATA_EXFILTRATION,
                AlertSeverity.CRITICAL,
                "dc-01",
                "Directory replication request from compromised host",
                "Suspicious DCSync-like behavior detected against dc-01.",
                "Directory Audit",
                TriageLabel.TRUE_POSITIVE,
            ),
        ),
    },
    compromise_logs={
        "fs-02": (
            LogSpec(
                "LOG-M6",
                "T+05m",
                "fs-02",
                "smb",
                "New remote service created from ws-23 using reused finance credentials.",
                True,
            ),
        ),
        "backup-01": (
            LogSpec(
                "LOG-M7",
                "T+06m",
                "backup-01",
                "auth",
                "svc-backup logon type 3 from ws-23 outside maintenance window.",
                True,
            ),
        ),
        "dc-01": (
            LogSpec(
                "LOG-M8",
                "T+07m",
                "dc-01",
                "directory",
                "Replicating directory changes requested by account svc-backup.",
                True,
            ),
        ),
    },
    damage_budget=8.0,
    cost_budget=7.0,
)


SOC_OPTIMIZATION_HARD = TaskScenario(
    task_id="soc-optimization-hard",
    title="Task 3 - SOC Optimization",
    difficulty=Difficulty.HARD,
    goal="Balance containment speed, business disruption, and attacker adaptation.",
    success_metric="Score = 1 - (0.5 x damage + 0.3 x cost + 0.2 x delay).",
    briefing=(
        "Finance has reported a suspicious invoice execution on fin-ws-07 while the identity team "
        "also sees a token replay attempt on vpn-02. One HR laptop alert is probably noise. "
        "The attacker is adaptive: if you stop the first foothold without blocking backup access, "
        "they may pivot to the VPN path and continue toward the finance database."
    ),
    max_steps=8,
    score_mode="optimization",
    nodes=(
        NodeSpec("fin-ws-07", "workstation", "finance", 0.50, ("macro-hardening-gap",)),
        NodeSpec("hr-laptop-03", "workstation", "hr", 0.30, ()),
        NodeSpec("vpn-02", "vpn", "infra", 0.80, ("token-replay-gap",)),
        NodeSpec("idp-01", "identity_server", "infra", 0.90, ("legacy-oauth-app",)),
        NodeSpec("finance-db", "database", "finance", 1.00, ("stale-service-principal",)),
        NodeSpec("edr-collector", "security_server", "security", 0.70, ()),
    ),
    initial_alerts=(
        AlertSpec(
            "ALT-H1",
            AlertKind.PHISHING,
            AlertSeverity.HIGH,
            "fin-ws-07",
            "Suspicious invoice execution on finance workstation",
            "Attachment invoice_Q2m.xlsm spawned mshta.exe and outbound traffic to 185.17.44.22.",
            "EDR",
            TriageLabel.TRUE_POSITIVE,
        ),
        AlertSpec(
            "ALT-H2",
            AlertKind.POWERSHELL,
            AlertSeverity.MEDIUM,
            "hr-laptop-03",
            "Encoded PowerShell on HR laptop",
            "The process tree matches the monthly HR onboarding automation script.",
            "EDR",
            TriageLabel.FALSE_POSITIVE,
        ),
        AlertSpec(
            "ALT-H3",
            AlertKind.TOKEN_REPLAY,
            AlertSeverity.MEDIUM,
            "vpn-02",
            "VPN token replay attempt",
            "A refresh token tied to finance user jpatel was replayed from a new ASN.",
            "IdP",
            TriageLabel.TRUE_POSITIVE,
        ),
    ),
    initial_logs=(
        LogSpec(
            "LOG-H1",
            "T+00m",
            "fin-ws-07",
            "process",
            "mshta.exe launched from Excel child process and opened hidden browser control.",
            True,
        ),
        LogSpec(
            "LOG-H2",
            "T+01m",
            "hr-laptop-03",
            "script",
            "Onboarding.ps1 signed by corp automation certificate executed at scheduled time.",
            False,
        ),
        LogSpec(
            "LOG-H3",
            "T+02m",
            "vpn-02",
            "auth",
            "Refresh token replay from ASN 45102 with mismatched user agent fingerprint.",
            True,
        ),
    ),
    expected_triage={
        "ALT-H1": TriageLabel.TRUE_POSITIVE,
        "ALT-H2": TriageLabel.FALSE_POSITIVE,
        "ALT-H3": TriageLabel.TRUE_POSITIVE,
    },
    require_full_alert_handling=True,
    initial_compromised=("fin-ws-07",),
    attack_graph={
        "fin-ws-07": ("idp-01", "finance-db"),
        "vpn-02": ("idp-01",),
        "idp-01": ("finance-db", "edr-collector"),
    },
    backup_entry_nodes=("vpn-02",),
    sensitive_assets=("finance-db",),
    malicious_indicators=("185.17.44.22", "cdn-sync.net"),
    forensics_logs={
        "fin-ws-07": (
            LogSpec(
                "LOG-H4",
                "T+03m",
                "fin-ws-07",
                "network",
                "Beacon to 185.17.44.22 advertised fallback domain cdn-sync.net in tasking blob.",
                True,
            ),
            LogSpec(
                "LOG-H5",
                "T+04m",
                "fin-ws-07",
                "credential_access",
                "Browser token cache accessed immediately before SSO cookie export.",
                True,
            ),
        ),
        "vpn-02": (
            LogSpec(
                "LOG-H6",
                "T+05m",
                "vpn-02",
                "network",
                "cdn-sync.net resolved to a newly registered host with finance user token replay.",
                True,
            ),
        ),
        "idp-01": (
            LogSpec(
                "LOG-H7",
                "T+06m",
                "idp-01",
                "oauth",
                "Legacy OAuth app issued refresh token to unexpected confidential client.",
                True,
            ),
        ),
    },
    compromise_alerts={
        "vpn-02": (
            AlertSpec(
                "ALT-H4",
                AlertKind.TOKEN_REPLAY,
                AlertSeverity.HIGH,
                "vpn-02",
                "Backup VPN foothold established",
                "The attacker re-entered through the token replay path after the workstation was disrupted.",
                "IdP",
                TriageLabel.TRUE_POSITIVE,
            ),
        ),
        "idp-01": (
            AlertSpec(
                "ALT-H5",
                AlertKind.CREDENTIAL_THEFT,
                AlertSeverity.HIGH,
                "idp-01",
                "Identity provider admin consent abuse",
                "A suspicious confidential client requested long-lived finance scope tokens.",
                "IdP Audit",
                TriageLabel.TRUE_POSITIVE,
            ),
        ),
        "finance-db": (
            AlertSpec(
                "ALT-H6",
                AlertKind.DATA_EXFILTRATION,
                AlertSeverity.CRITICAL,
                "finance-db",
                "Large egress staging from finance database",
                "Compressed finance exports prepared for outbound transfer.",
                "DLP",
                TriageLabel.TRUE_POSITIVE,
            ),
        ),
        "edr-collector": (
            AlertSpec(
                "ALT-H7",
                AlertKind.ANOMALY,
                AlertSeverity.MEDIUM,
                "edr-collector",
                "Security telemetry suppression attempt",
                "An unusual service restart tried to degrade endpoint telemetry forwarding.",
                "EDR Backend",
                TriageLabel.TRUE_POSITIVE,
            ),
        ),
    },
    compromise_logs={
        "vpn-02": (
            LogSpec(
                "LOG-H8",
                "T+07m",
                "vpn-02",
                "auth",
                "Replay token accepted and converted into a fresh session for jpatel.",
                True,
            ),
        ),
        "idp-01": (
            LogSpec(
                "LOG-H9",
                "T+08m",
                "idp-01",
                "oauth",
                "App registration 'SyncUpdater' granted finance.read.all scope.",
                True,
            ),
        ),
        "finance-db": (
            LogSpec(
                "LOG-H10",
                "T+09m",
                "finance-db",
                "database",
                "2.1 GB export job created from service principal syncupdater@app.",
                True,
            ),
        ),
        "edr-collector": (
            LogSpec(
                "LOG-H11",
                "T+10m",
                "edr-collector",
                "service",
                "Telemetry forwarding service restarted with new unsigned plugin.",
                True,
            ),
        ),
    },
    damage_budget=10.0,
    cost_budget=9.0,
)


SCENARIOS: dict[str, TaskScenario] = {
    ALERT_TRIAGE_EASY.task_id: ALERT_TRIAGE_EASY,
    INCIDENT_CONTAINMENT_MEDIUM.task_id: INCIDENT_CONTAINMENT_MEDIUM,
    SOC_OPTIMIZATION_HARD.task_id: SOC_OPTIMIZATION_HARD,
}
