"""Strategy observer trigger, lifecycle, and manager integration contracts."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from coral.agent.manager import AgentManager
from coral.agent.observer import StrategyObserver
from coral.config import CoralConfig, ObserverConfig, SandboxConfig
from coral.sandbox.srt import build_srt_settings
from coral.types import Attempt
from coral.workspace import ProjectPaths


class FakeHandle:
    def __init__(self, agent_id="observer"):
        self.agent_id = agent_id
        self.alive = True
        self.process = SimpleNamespace(returncode=None, pid=987654)
        self.log_path = Path("unused.log")
        self.stopped = False

    def stop(self):
        self.stopped = True
        self.alive = False

    def interrupt(self):
        self.stop()


def attempt(i, score=None, status="regressed", budget="real"):
    return Attempt(
        str(i),
        "captain-nemo",
        "test",
        score,
        status,
        None,
        f"{i:06d}",
        metadata={"budget_class": budget},
    )


@pytest.fixture
def env(tmp_path, monkeypatch):
    now = [0.0]
    monkeypatch.setattr("coral.agent.observer.time.monotonic", lambda: now[0])
    monkeypatch.setattr("coral.agent.observer._process_identity", lambda pid: "birth")
    config = CoralConfig.from_dict(
        {
            "task": {"name": "test", "description": "Optimize"},
            "agents": {"observer": {"enabled": True}},
        }
    )
    manager = AgentManager(config)
    manager.paths = ProjectPaths(
        results_dir=tmp_path,
        task_dir=tmp_path,
        run_dir=tmp_path,
        coral_dir=tmp_path / ".coral",
        agents_dir=tmp_path / "agents",
        repo_dir=tmp_path / "repo",
    )
    history = []
    manager._read_all_run_attempts = lambda: history.copy()
    started = []

    def start(workspace, reason):
        handle = FakeHandle()
        started.append((workspace, reason, handle))
        return handle

    monkeypatch.setattr(
        StrategyObserver, "_start", lambda self, workspace, reason: start(workspace, reason)
    )
    observer = StrategyObserver(manager)
    manager._observer = observer
    return SimpleNamespace(
        manager=manager, observer=observer, history=history, now=now, started=started
    )


def test_opt_in_and_config_validation():
    assert not ObserverConfig().enabled
    for field in ("plateau_evals", "improvement_fraction", "idle_minutes", "timeout_minutes"):
        with pytest.raises(ValueError):
            ObserverConfig(**{field: 0})
    with pytest.raises(ValueError):
        ObserverConfig(cooldown_minutes=-1)
    with pytest.raises(ValueError, match="single-island"):
        CoralConfig.from_dict(
            {
                "task": {"name": "t", "description": "d"},
                "agents": {"observer": {"enabled": True}},
                "islands": {"count": 2},
            }
        )


def test_evaluation_trigger_counts_entire_batch_and_failed_real(env):
    env.history.extend([attempt(0, 100, "baseline")])
    env.observer.tick()
    env.history.extend(attempt(i, 10000, "crashed") for i in range(1, 11))
    env.observer.tick()
    assert len(env.started) == 1
    assert "10 completed" in env.started[0][1]
    assert env.observer.state["anchor"] == 100
    assert env.observer.state["stall_evals"] == 0
    assert env.manager.handles == []


def test_tune_pending_and_grader_error_do_not_count(env):
    env.history.extend(
        [
            attempt(1, 100, "baseline"),
            attempt(2, 999, budget="tune"),
            attempt(3, 999, budget="grader_error"),
            attempt(4, 999, "pending"),
        ]
    )
    env.observer.tick()
    assert env.observer.state["anchor"] == 100
    assert env.observer.state["stall_evals"] == 0
    env.history[-1] = attempt(4, None, "timeout")
    env.observer.tick()
    assert env.observer.state["stall_evals"] == 1


@pytest.mark.parametrize(
    "direction,scores",
    [
        ("maximize", [100, 102, 104, 105]),
        ("minimize", [100, 98, 96, 95]),
        ("maximize", [-100, -98, -96, -95]),
        ("minimize", [0, -1, -2, -3]),
    ],
)
def test_cumulative_small_improvements_and_direction(env, direction, scores):
    env.manager.config.grader.direction = direction
    for i, score in enumerate(scores):
        env.now[0] += 10
        env.history.append(attempt(i, score))
        env.observer.tick()
    assert env.observer.state["anchor"] == scores[-1]
    assert env.observer.state["stall_evals"] == 0
    assert env.observer.state["window_start"] == 40


def test_idle_without_evaluations_and_one_at_a_time(env):
    env.now[0] = 3600
    env.observer.tick()
    assert "active minutes" in env.started[0][1]
    env.history.extend(attempt(i) for i in range(20))
    env.observer.tick()
    assert len(env.started) == 1


def test_cooldown_and_timeout_do_not_immediately_retry(env):
    env.now[0] = 3600
    env.observer.tick()
    env.now[0] += 1200
    env.observer.tick()
    assert env.started[0][2].stopped
    assert env.observer.state["review"]["status"] == "timeout"
    assert not env.observer.state["notes"]
    env.history.extend(attempt(i) for i in range(10))
    env.observer.tick()
    assert len(env.started) == 1
    env.now[0] = 7200
    env.observer.tick()
    assert len(env.started) == 2
    assert env.started[0][0] != env.started[1][0]


def test_warmstart_history_excluded_and_resume_clock(env):
    env.history.extend(attempt(i, 100) for i in range(15))
    env.observer.state_path.unlink()
    observer = StrategyObserver(env.manager)
    assert observer.state["stall_evals"] == 0
    assert len(observer.state["seen"]) == 15
    env.now[0] = 1800
    observer.stop()
    env.now[0] = 100000
    observer = StrategyObserver(env.manager)
    observer.tick()
    assert not env.started
    env.now[0] += 1800
    observer.tick()
    assert len(env.started) == 1


@pytest.mark.parametrize(
    "status,body,expected",
    [
        (0, "# Advice\nInspect [code](../agents/a.py).", "completed"),
        (1, "Bad result", "failed"),
        (0, "  \n", "empty"),
    ],
)
def test_publication_and_diagnostics(env, status, body, expected):
    env.now[0] = 3600
    env.observer.tick()
    workspace, _, handle = env.started[0]
    (workspace / "report.md").write_text(body)
    handle.alive = False
    handle.process.returncode = status
    env.observer.tick()
    assert env.observer.state["review"]["status"] == expected
    assert (workspace / "review.json").is_file()
    assert handle.stopped
    notes = env.observer.state["notes"]
    assert bool(notes) == (expected == "completed")
    if notes:
        text, pending = env.observer.advice("agent")
        assert "creator: strategy-" in text
        assert body in text
        assert notes[0]["path"] in text
        env.observer.acknowledge("agent", pending)
        assert not env.observer.pending("agent")
        assert StrategyObserver(env.manager).pending("agent") == []
        assert StrategyObserver(env.manager).pending("paused-agent")


def test_active_delivery_and_paused_queue(env):
    active, paused = FakeHandle("active"), FakeHandle("paused")
    env.manager.handles = [active, paused]
    env.manager._is_paused = lambda aid: aid == "paused"
    calls = []

    def resume(idx, prompt, prompt_source):
        aid = env.manager.handles[idx].agent_id
        advice, notes = env.observer.advice(aid)
        calls.append((aid, advice, prompt_source))
        env.observer.acknowledge(aid, notes)
        return env.manager.handles[idx]

    env.manager._interrupt_and_resume = resume
    env.now[0] = 3600
    env.observer.tick()
    workspace, _, handle = env.started[0]
    (workspace / "report.md").write_text("Read the measurements")
    handle.alive = False
    handle.process.returncode = 0
    env.observer.tick()
    env.observer.tick()
    assert len(calls) == 1 and calls[0][0] == "active"
    assert "Read the measurements" in calls[0][1]
    assert env.observer.pending("paused")
    env.manager._is_paused = lambda aid: False
    env.observer.tick()
    assert [c[0] for c in calls] == ["active", "paused"]


def test_shutdown_and_reconcile(env, monkeypatch):
    env.now[0] = 3600
    env.observer.tick()
    kill = Mock()
    monkeypatch.setattr("coral.agent.observer.os.killpg", kill)
    resumed = StrategyObserver(env.manager)
    kill.assert_called_once()
    assert resumed.state["review"]["status"] == "interrupted"
    assert not resumed.state["notes"]
    env.observer.stop()
    assert env.started[0][2].stopped


def test_recycled_pid_is_not_killed(env, monkeypatch):
    env.now[0] = 3600
    env.observer.tick()
    monkeypatch.setattr("coral.agent.observer._process_identity", lambda pid: "different birth")
    kill = Mock()
    monkeypatch.setattr("coral.agent.observer.os.killpg", kill)
    StrategyObserver(env.manager)
    kill.assert_not_called()


def test_real_start_uses_fresh_session_and_existing_isolation(env, monkeypatch):
    # Restore the production launch path, using a fake runtime only.
    monkeypatch.undo()
    from coral.agent.observer import StrategyObserver

    runtime = SimpleNamespace(
        instruction_filename="AGENTS.md",
        shared_dir_name=".codex",
        start=Mock(return_value=FakeHandle()),
    )
    monkeypatch.setattr("coral.agent.observer.get_runtime", lambda name: runtime)
    env.manager.config.agents.observer.runtime = "codex"
    env.manager.config.agents.observer.model = "gpt-6-astra"
    env.manager._sandbox_spec_for = Mock(return_value="sandbox")
    env.manager._apply_user_isolation = Mock(return_value="credentials")
    workspace = env.manager.paths.run_dir / "observers" / "unique-review"
    workspace.mkdir(parents=True)
    StrategyObserver._start(env.observer, workspace, "idle")
    kwargs = runtime.start.call_args.kwargs
    assert kwargs["resume_session_id"] is None
    assert kwargs["model"] == "gpt-6-astra"
    assert kwargs["sandbox"] == "sandbox"
    assert kwargs["run_as_user"] == "credentials"
    assert "do not edit candidates" in kwargs["prompt"]
    assert "private/" in kwargs["prompt"]
    assert kwargs["log_dir"] == workspace / "logs"
    assert not env.manager.handles
    env.manager._sandbox_spec_for.assert_called_once_with(
        "unique-review", workspace, ".codex", read_only_team=True
    )


def test_observer_sandbox_team_is_read_only(tmp_path):
    coral = tmp_path / ".coral"
    repo = tmp_path / "repo"
    agent = tmp_path / "agents" / "a"
    observer = tmp_path / "observers" / "review"
    settings = build_srt_settings(
        SandboxConfig(),
        worktree_path=observer,
        coral_dir=coral,
        repo_dir=repo,
        shared_dir_name=".codex",
        proxy_port=1234,
        sibling_worktrees=[agent],
        read_only_team=True,
    )
    fs = settings["filesystem"]
    assert str(agent) in fs["allowRead"]
    assert str(coral / "private") in fs["denyRead"]
    assert str(coral) not in fs["allowWrite"]
    assert str(repo / ".git") not in fs["allowWrite"]
    assert {str(coral), str(repo), str(agent)} <= set(fs["denyWrite"])


def test_manager_interrupt_preserves_session_and_delivers_pending_once(env):
    """Exercise the real interrupt -> setup -> runtime start -> acknowledgement path."""
    manager = env.manager
    agent_id = manager.specs[0].agent_id
    workspace = manager.paths.agents_dir / agent_id
    workspace.mkdir(parents=True)
    (workspace / ".coral_agent_id").write_text(agent_id)
    manager.paths.repo_dir.mkdir()
    note = manager.paths.coral_dir / "public/notes/strategy-reviews/test.md"
    note.parent.mkdir(parents=True)
    note.write_text("Strategy advice and evidence")
    env.observer.state["notes"].append({"path": str(note), "delivered": []})
    runtime = SimpleNamespace(
        instruction_filename="AGENTS.md",
        shared_dir_name=".codex",
        start=Mock(return_value=FakeHandle(agent_id)),
        extract_session_id=Mock(return_value="optimizer-session"),
    )
    manager.runtimes[agent_id] = runtime
    old = FakeHandle(agent_id)
    manager.handles = [old]
    manager._interrupt_and_resume(0, "Continue", "observer:advice")
    assert old.stopped
    kwargs = runtime.start.call_args.kwargs
    assert kwargs["resume_session_id"] == "optimizer-session"
    assert str(note) in kwargs["prompt"]
    assert "Strategy advice and evidence" in kwargs["prompt"]
    assert not env.observer.pending(agent_id)
    manager._setup_and_start_agent(agent_id, resume_session_id="optimizer-session", prompt="Again")
    assert runtime.start.call_args.kwargs["prompt"] == "Again"


def test_failed_optimizer_launch_keeps_delivery_pending(env):
    manager = env.manager
    agent_id = manager.specs[0].agent_id
    workspace = manager.paths.agents_dir / agent_id
    workspace.mkdir(parents=True)
    (workspace / ".coral_agent_id").write_text(agent_id)
    manager.paths.repo_dir.mkdir()
    note = manager.paths.coral_dir / "note.md"
    note.write_text("Consider this")
    env.observer.state["notes"].append({"path": str(note), "delivered": []})
    manager.runtimes[agent_id] = SimpleNamespace(
        instruction_filename="AGENTS.md",
        shared_dir_name=".codex",
        start=Mock(side_effect=RuntimeError("launch failed")),
    )
    with pytest.raises(RuntimeError, match="launch failed"):
        manager._setup_and_start_agent(agent_id)
    assert env.observer.pending(agent_id)


def test_observer_launch_failure_respects_cooldown(env, monkeypatch):
    monkeypatch.setattr(StrategyObserver, "_start", Mock(side_effect=RuntimeError("bad runtime")))
    env.now[0] = 3600
    env.observer.tick()
    assert env.observer.state["review"]["status"] == "failed"
    assert not env.observer.state["notes"]
    env.history.extend(attempt(i) for i in range(10))
    env.observer.tick()
    assert StrategyObserver._start.call_count == 1


def test_shutdown_is_idempotent_and_excludes_stopped_time(env):
    env.now[0] = 120
    env.observer.stop()
    env.now[0] = 10000
    env.observer.stop()
    assert env.observer.state["active_seconds"] == 120


def test_invalid_report_does_not_stop_optimization(env):
    env.now[0] = 3600
    env.observer.tick()
    workspace, _, handle = env.started[0]
    (workspace / "report.md").write_bytes(b"\xff\xfe")
    handle.alive = False
    handle.process.returncode = 0
    env.observer.tick()
    assert env.observer.state["review"]["status"] == "failed"
    assert not env.observer.state["notes"]


def test_reconcile_reaps_children_after_wrapper_exit(env, monkeypatch):
    env.now[0] = 3600
    env.observer.tick()
    review = env.observer.state["review"]
    monkeypatch.setattr("coral.agent.observer._process_identity", lambda pid: "")
    monkeypatch.setattr(
        "coral.agent.observer.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(
            stdout=f"{review['pid']} codex exec review {review['workspace']}\n"
        ),
    )
    kill = Mock()
    monkeypatch.setattr("coral.agent.observer.os.killpg", kill)
    StrategyObserver(env.manager)
    kill.assert_called_once()


def test_shutdown_during_launch_does_not_leave_observer(env, monkeypatch):
    handle = FakeHandle()

    def start(*args):
        env.observer.stop()
        return handle

    monkeypatch.setattr(StrategyObserver, "_start", start)
    env.now[0] = 3600
    env.observer.tick()
    assert handle.stopped
    assert env.observer.handle is None
    assert env.observer.state["review"]["status"] == "interrupted"


def test_disabled_resume_reconciles_but_never_launches(env, monkeypatch):
    env.now[0] = 3600
    env.observer.tick()
    kill = Mock()
    monkeypatch.setattr("coral.agent.observer.os.killpg", kill)
    env.manager.config.agents.observer.enabled = False
    env.manager._observer = None
    env.manager._init_observer()
    kill.assert_called_once()
    resumed = env.manager._observer
    assert resumed.state["review"]["status"] == "interrupted"
    env.now[0] += 100000
    resumed.tick()
    assert len(env.started) == 1
    assert resumed.state["active_seconds"] == 3600


def test_initial_anchor_is_best_historical_score(env):
    env.history.extend([attempt(1, 100), attempt(2, 104)])
    env.observer.state_path.unlink()
    observer = StrategyObserver(env.manager)
    assert observer.state["anchor"] == 104
    env.history.append(attempt(3, 105))
    observer.tick()
    assert observer.state["anchor"] == 104
    assert observer.state["stall_evals"] == 1


def test_exact_fraction_boundary_tolerates_float_roundoff(env):
    env.history.append(attempt(1, 1.1))
    env.observer.tick()
    env.history.append(attempt(2, 1.1549))
    env.observer.tick()
    assert env.observer.state["anchor"] == 1.1
    env.history.append(attempt(3, 1.155))
    env.observer.tick()
    assert env.observer.state["anchor"] == 1.155
