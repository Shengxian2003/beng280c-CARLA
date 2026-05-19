"""Tests for the Specialist + Coordinator multi-agent layer."""
from __future__ import annotations

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agents.audit import AuditLog, read_log, filter_log
from agents.coordinator import Coordinator
from agents.llm import MockLLM
from agents.project_context import (
    PROJECT_CONTEXT,
    KSPACE_PATH,
    EXISTING_RECON_5ITER,
)
from agents.specialist import (
    Specialist,
    build_default_specialists,
    VERIFIER_PROMPT,
)
from agents.tools import Workspace


# ============================================================================
# Specialist construction
# ============================================================================

class TestSpecialistConfig:
    def test_default_set_has_four(self):
        specs = build_default_specialists(MockLLM(responses=[]))
        assert set(specs) == {"reconstruction", "segmentation", "verifier", "hemodynamic"}

    def test_tool_allowlists_disjoint(self):
        """Each tool should belong to exactly one specialist (the routing in
        the viewer windows assumes this; if two specialists shared a tool, the
        windows would double-count tool calls)."""
        specs = build_default_specialists(MockLLM(responses=[]))
        seen: dict[str, str] = {}
        for name, spec in specs.items():
            for t in spec.tool_names:
                assert t not in seen, \
                    f"tool {t!r} shared between {seen[t]} and {name}"
                seen[t] = name

    def test_verifier_only_has_verify(self):
        specs = build_default_specialists(MockLLM(responses=[]))
        assert specs["verifier"].tool_names == ["verify"]

    def test_demo_mode_removes_reconstruct_tool(self):
        """In demo mode the Reconstruction specialist must not be able to
        trigger MATLAB (which would block the demo for ~12 min)."""
        normal = build_default_specialists(MockLLM(responses=[]))
        assert "reconstruct" in normal["reconstruction"].tool_names

        demo = build_default_specialists(MockLLM(responses=[]), demo_mode=True)
        assert "reconstruct" not in demo["reconstruction"].tool_names
        assert demo["reconstruction"].tool_names == ["load_reconstruction"]
        # Other specialists unaffected
        assert demo["segmentation"].tool_names == normal["segmentation"].tool_names

    def test_demo_mode_updates_reconstruction_prompt(self):
        demo = build_default_specialists(MockLLM(responses=[]), demo_mode=True)
        assert "DEMO MODE" in demo["reconstruction"].system_prompt


# ============================================================================
# Project context injection — prevents path hallucination
# ============================================================================

class TestProjectContextInjection:
    """The project context (canonical paths, acquisition defaults, conventions)
    must appear in every specialist's system prompt and the coordinator's, so
    Qwen doesn't have to guess where files live."""

    def _captured_system(self, llm, messages_recorded_in):
        """Find the system message from the recorded calls."""
        msgs = messages_recorded_in[0][0]   # first call's messages
        return next(m["content"] for m in msgs if m["role"] == "system")

    def test_specialist_system_prompt_includes_kspace_path(self, tmp_path):
        llm = MockLLM(responses=[json.dumps({"done": True, "report": "ok"})])
        s = Specialist(name="verifier",
                       system_prompt="role-specific text",
                       tool_names=["verify"], llm=llm)
        log = AuditLog(tmp_path / "log.jsonl")
        s.handle("task", workspace=Workspace(), audit=log)
        log.close()
        sysmsg = self._captured_system(llm, llm.calls)
        assert KSPACE_PATH in sysmsg
        assert EXISTING_RECON_5ITER in sysmsg

    def test_specialist_system_prompt_includes_role_and_protocol(self, tmp_path):
        llm = MockLLM(responses=[json.dumps({"done": True, "report": "ok"})])
        s = Specialist(name="verifier",
                       system_prompt="ROLE_MARKER_XYZ",
                       tool_names=["verify"], llm=llm)
        log = AuditLog(tmp_path / "log.jsonl")
        s.handle("task", workspace=Workspace(), audit=log)
        log.close()
        sysmsg = self._captured_system(llm, llm.calls)
        # Project context + role + tools + protocol all present
        assert "MEDICT" in sysmsg
        assert "ROLE_MARKER_XYZ" in sysmsg
        assert "Response protocol" in sysmsg

    def test_coordinator_system_prompt_includes_kspace_path(self, tmp_path):
        coord_llm = MockLLM(responses=[json.dumps({"done": True, "summary": "x"})])
        log = AuditLog(tmp_path / "log.jsonl")
        coord = Coordinator(coord_llm, {}, Workspace(), log)
        coord.run("test")
        log.close()
        sysmsg = self._captured_system(coord_llm, coord_llm.calls)
        assert KSPACE_PATH in sysmsg

    def test_specialist_custom_context_overrides_default(self, tmp_path):
        llm = MockLLM(responses=[json.dumps({"done": True, "report": "ok"})])
        s = Specialist(name="verifier",
                       system_prompt="role",
                       tool_names=["verify"], llm=llm,
                       project_context="CUSTOM_CONTEXT_MARKER")
        log = AuditLog(tmp_path / "log.jsonl")
        s.handle("task", workspace=Workspace(), audit=log)
        log.close()
        sysmsg = self._captured_system(llm, llm.calls)
        assert "CUSTOM_CONTEXT_MARKER" in sysmsg
        assert KSPACE_PATH not in sysmsg


# ============================================================================
# Specialist internal loop
# ============================================================================

class TestSpecialistHandle:
    def _make(self, responses, tools, name="verifier"):
        llm = MockLLM(responses=responses)
        return Specialist(
            name=name,
            system_prompt="You are a test specialist.",
            tool_names=tools,
            llm=llm,
            max_steps=4,
        )

    def test_done_in_one_step(self, tmp_path):
        s = self._make(
            responses=[json.dumps({"done": True, "report": "all good"})],
            tools=["verify"],
        )
        log = AuditLog(tmp_path / "log.jsonl")
        ws = Workspace()
        report = s.handle("test task", workspace=ws, audit=log)
        log.close()
        assert report["done"] is True
        assert "all good" in report["report"]
        assert report["tools_called"] == []
        assert report["n_steps"] == 1

    def test_tool_then_done(self, tmp_path):
        # Pre-seed the workspace with a mask so verify() can run
        import numpy as np
        ws = Workspace()
        # Synthetic minimal recon + mask
        Z, Y, X, T = 6, 6, 6, 2
        ws.recon = {
            "xHat":   np.ones((Z, Y, X, T), dtype=np.complex128),
            "thetaX": np.zeros((Z, Y, X, T)),
            "thetaY": np.zeros((Z, Y, X, T)),
            "thetaZ": np.zeros((Z, Y, X, T)),
        }
        ws.masks["test_mask"] = np.ones((Z, Y, X), dtype=bool)

        s = self._make(
            responses=[
                json.dumps({"tool": "verify", "args": {"mask_name": "test_mask"}}),
                json.dumps({"done": True, "report": "verified"}),
            ],
            tools=["verify"],
        )
        log = AuditLog(tmp_path / "log.jsonl")
        report = s.handle("check it", workspace=ws, audit=log)
        log.close()
        assert report["done"] is True
        assert report["tools_called"] == ["verify"]
        assert report["n_steps"] == 2

    def test_rejects_unallowed_tool(self, tmp_path):
        # Specialist tries to call a tool it isn't allowed → should get error feedback
        # and recover by emitting done
        s = self._make(
            responses=[
                json.dumps({"tool": "analyze", "args": {"mask_name": "x"}}),  # not allowed
                json.dumps({"done": True, "report": "I picked the wrong tool, giving up"}),
            ],
            tools=["verify"],  # only verify is allowed
        )
        log = AuditLog(tmp_path / "log.jsonl")
        ws = Workspace()
        report = s.handle("task", workspace=ws, audit=log)
        log.close()
        assert report["done"] is True
        # No tools were actually invoked (analyze was rejected before dispatch)
        assert report["tools_called"] == []

    def test_invalid_json_recovers(self, tmp_path):
        # First reply is not valid JSON; the specialist should re-prompt itself
        s = self._make(
            responses=[
                "this is not json",
                json.dumps({"done": True, "report": "recovered"}),
            ],
            tools=["verify"],
        )
        log = AuditLog(tmp_path / "log.jsonl")
        ws = Workspace()
        report = s.handle("task", workspace=ws, audit=log)
        log.close()
        assert report["done"] is True
        assert "recovered" in report["report"]

    def test_max_steps_cap(self, tmp_path):
        # Never says done → should bail after max_steps with a partial report
        s = self._make(
            responses=[json.dumps({"tool": "verify", "args": {"mask_name": "nope"}})] * 10,
            tools=["verify"],
        )
        log = AuditLog(tmp_path / "log.jsonl")
        report = s.handle("task", workspace=Workspace(), audit=log)
        log.close()
        assert report["done"] is False
        assert "did not produce a report" in report["report"]

    def test_llm_calls_tagged_with_specialist_purpose(self, tmp_path):
        # The audit purpose must be "specialist.<name>" so the viewer windows can filter
        s = self._make(
            responses=[json.dumps({"done": True, "report": "ok"})],
            tools=["verify"],
            name="verifier",
        )
        log_path = tmp_path / "log.jsonl"
        log = AuditLog(log_path)
        s.handle("task", workspace=Workspace(), audit=log)
        log.close()
        entries = read_log(log_path)
        llm_calls = filter_log(entries, kind="llm_call")
        assert any(e["data"].get("purpose") == "specialist.verifier" for e in llm_calls)


# ============================================================================
# Coordinator delegation
# ============================================================================

class TestCoordinator:
    def _make_coord(self, responses, specialists_resps):
        """Build a Coordinator + specialists, all using MockLLM with their own
        scripted responses."""
        coord_llm = MockLLM(responses=responses)
        specialists = {}
        for name, resps in specialists_resps.items():
            spec_llm = MockLLM(responses=resps)
            specialists[name] = Specialist(
                name=name,
                system_prompt=f"You are the test {name} specialist.",
                tool_names=[],   # don't run any tools — just emit done immediately
                llm=spec_llm,
                max_steps=3,
            )
        return coord_llm, specialists

    def test_done_immediately(self, tmp_path):
        coord_llm, specs = self._make_coord(
            responses=[json.dumps({"done": True, "summary": "nothing to do"})],
            specialists_resps={},
        )
        log = AuditLog(tmp_path / "log.jsonl")
        coord = Coordinator(coord_llm, specs, Workspace(), log, max_delegations=5)
        result = coord.run("test goal")
        log.close()
        assert result.status == "success"
        assert "nothing to do" in result.summary
        assert result.n_delegations == 0

    def test_delegates_to_one_specialist_then_done(self, tmp_path):
        coord_llm, specs = self._make_coord(
            responses=[
                json.dumps({"delegate_to": "verifier", "task": "check it",
                            "why": "test"}),
                json.dumps({"done": True, "summary": "verifier reported back"}),
            ],
            specialists_resps={
                "verifier": [json.dumps({"done": True, "report": "checks out"})],
            },
        )
        log = AuditLog(tmp_path / "log.jsonl")
        coord = Coordinator(coord_llm, specs, Workspace(), log, max_delegations=5)
        result = coord.run("test goal")
        log.close()
        assert result.status == "success"
        assert result.specialists_used == ["verifier"]
        assert result.n_delegations == 1

    def test_unknown_specialist_rejected(self, tmp_path):
        coord_llm, specs = self._make_coord(
            responses=[
                json.dumps({"delegate_to": "ghostbusters", "task": "x", "why": "x"}),
                json.dumps({"done": True, "summary": "gave up on bad name"}),
            ],
            specialists_resps={
                "verifier": [json.dumps({"done": True, "report": "ok"})],
            },
        )
        log = AuditLog(tmp_path / "log.jsonl")
        coord = Coordinator(coord_llm, specs, Workspace(), log, max_delegations=5)
        result = coord.run("test goal")
        log.close()
        assert result.status == "success"
        assert result.specialists_used == []  # bad delegation didn't run

    def test_max_delegations_returns_partial_results(self, tmp_path):
        # Coordinator keeps delegating forever; should bail at max_delegations
        # and return status='partial' with a summary of what was learned, not
        # bare 'max_steps'.
        coord_llm, specs = self._make_coord(
            responses=[json.dumps({"delegate_to": "verifier", "task": "x",
                                    "why": "loop"})] * 20,
            specialists_resps={
                "verifier": [json.dumps({"done": True, "report": "ok"})] * 20,
            },
        )
        log = AuditLog(tmp_path / "log.jsonl")
        coord = Coordinator(coord_llm, specs, Workspace(), log, max_delegations=3)
        result = coord.run("test")
        log.close()
        assert result.status == "partial"
        assert result.n_delegations == 3
        # Summary should explain what happened
        assert "max_delegations" in result.summary
        assert "verifier" in result.summary  # specialists used should be listed

    def test_audit_log_records_delegations(self, tmp_path):
        coord_llm, specs = self._make_coord(
            responses=[
                json.dumps({"delegate_to": "verifier", "task": "go", "why": "y"}),
                json.dumps({"done": True, "summary": "done"}),
            ],
            specialists_resps={
                "verifier": [json.dumps({"done": True, "report": "verified"})],
            },
        )
        log_path = tmp_path / "log.jsonl"
        log = AuditLog(log_path)
        Coordinator(coord_llm, specs, Workspace(), log).run("test")
        log.close()
        entries = read_log(log_path)
        events = filter_log(entries, kind="event")
        # Should see delegation + coordinator_done events
        names = {e["data"]["name"] for e in events}
        assert "delegation" in names
        assert "coordinator_done" in names