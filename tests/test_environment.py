from cybersoc_openenv.environment import CyberSOCEnvironment
from cybersoc_openenv.graders import grade_state
from cybersoc_openenv.models import CyberSOCAction, TriageLabel


def test_task_catalog_has_three_tasks() -> None:
    env = CyberSOCEnvironment()
    assert len(env.available_tasks()) == 3


def test_easy_task_full_score_with_correct_triage() -> None:
    env = CyberSOCEnvironment()
    env.reset(task_id="alert-triage-easy", seed=7)
    env.step(CyberSOCAction(action_type="triage_alert", alert_id="ALT-E1", classification=TriageLabel.TRUE_POSITIVE))
    env.step(CyberSOCAction(action_type="triage_alert", alert_id="ALT-E2", classification=TriageLabel.FALSE_POSITIVE))
    _, _, done, _ = env.step(
        CyberSOCAction(action_type="triage_alert", alert_id="ALT-E3", classification=TriageLabel.TRUE_POSITIVE)
    )
    assert done is True
    assert 0.99 < grade_state(env.state()) < 1.0


def test_reset_keeps_seed_but_generates_unique_episode_ids() -> None:
    env = CyberSOCEnvironment()
    first = env.reset(task_id="alert-triage-easy", seed=7)
    second = env.reset(task_id="alert-triage-easy", seed=7)
    assert first.episode_id != second.episode_id


def test_medium_task_containment_reaches_full_score() -> None:
    env = CyberSOCEnvironment()
    env.reset(task_id="incident-containment-medium", seed=7)
    env.step(CyberSOCAction(action_type="patch_system", node_id="vpn-01"))
    env.step(CyberSOCAction(action_type="isolate_node", node_id="ws-23"))
    env.step(CyberSOCAction(action_type="triage_alert", alert_id="ALT-M1", classification=TriageLabel.TRUE_POSITIVE))
    _, _, done, _ = env.step(
        CyberSOCAction(action_type="triage_alert", alert_id="ALT-M2", classification=TriageLabel.FALSE_POSITIVE)
    )
    state = env.state()
    assert done is True
    assert "ws-23" in state.contained_nodes
    assert 0.99 < grade_state(state) < 1.0


def test_medium_task_isolation_alone_triggers_backup_pressure() -> None:
    env = CyberSOCEnvironment()
    env.reset(task_id="incident-containment-medium", seed=7)
    _, _, done, _ = env.step(CyberSOCAction(action_type="isolate_node", node_id="ws-23"))
    state = env.state()
    assert done is False
    assert "vpn-01" in state.compromised_nodes
    assert grade_state(state) < 1.0


def test_hard_task_attacker_uses_backup_entry_when_primary_isolated() -> None:
    env = CyberSOCEnvironment()
    env.reset(task_id="soc-optimization-hard", seed=7)
    env.step(CyberSOCAction(action_type="request_forensics", node_id="fin-ws-07"))
    env.step(CyberSOCAction(action_type="isolate_node", node_id="fin-ws-07"))
    state = env.state()
    assert "vpn-02" in state.compromised_nodes


def test_observation_keeps_compromise_status_partial() -> None:
    env = CyberSOCEnvironment()
    observation = env.reset(task_id="soc-optimization-hard", seed=7)
    node_status = {node.node_id: node.status_hint for node in observation.node_overview}
    assert node_status["fin-ws-07"] == "suspicious"
    assert "compromised" not in node_status.values()


def test_invalid_alert_action_does_not_crash_environment() -> None:
    env = CyberSOCEnvironment()
    env.reset(task_id="alert-triage-easy", seed=7)
    observation, reward, done, info = env.step(
        CyberSOCAction(
            action_type="triage_alert",
            alert_id="DOES-NOT-EXIST",
            classification=TriageLabel.TRUE_POSITIVE,
        )
    )
    assert done is False
    assert reward.value < 0
    assert info.penalties["invalid_target"] < 0
    assert observation.last_action_result is not None
    assert observation.last_action_result.success is False


def test_repeated_identical_action_gets_repeat_penalty() -> None:
    env = CyberSOCEnvironment()
    env.reset(task_id="alert-triage-easy", seed=7)
    env.step(CyberSOCAction(action_type="noop", justification="first_noop"))
    _, reward, _, info = env.step(CyberSOCAction(action_type="noop", justification="repeat_noop"))
    assert reward.value < 0
    assert info.penalties["repeat_action"] < 0
