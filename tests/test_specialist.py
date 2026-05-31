"""Tests for the Specialist + Coordinator multi-agent layer."""
from __future__ import annotations

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utility.audit import AuditLog, read_log, filter_log
from agents.coordinator import Coordinator
from utility.llm import MockLLM
from utility.project_context import (
    PROJECT_CONTEXT,
    KSPACE_PATH,
    EXISTING_RECON_5ITER,
)
from agents.specialist import (
    Specialist,
    build_default_specialists,
    VERIFIER_PROMPT,
)
from utility.tools import Workspace


# ============================================================================
# Specialist construction
# ============================================================================

class TestSpecialistConfig:
    def test_default_set_has_four(self):
        specs = build_default_specialists(MockLLM(responses=[]))
        assert set(specs) == {"reconstruction", "segmentation", "verifier", "hemodynamic"}

    # `read_file` and `list_dir` are intentionally shared across all
    # specialists — they are the read-side of the filesystem-grounded
    # artifact store (per-path access control is enforced at the tool
    # layer via READ_SCOPES, not by which specialist holds the tool name).
    SHARED_READ_TOOLS = {"read_file", "list_dir"}

    def test_action_tool_allowlists_disjoint(self):
        """Each action tool (recon / segment / verify / analyze) should be
        owned by exactly one specialist. read_file / list_dir are exempt —
        they are shared utilities scoped at the tool layer."""
        specs = build_default_specialists(MockLLM(responses=[]))
        seen: dict[str, str] = {}
        for name, spec in specs.items():
            for t in spec.tool_names:
                if t in self.SHARED_READ_TOOLS:
                    continue
                assert t not in seen, \
                    f"tool {t!r} shared between {seen[t]} and {name}"
                seen[t] = name

    def test_verifier_action_tool_is_verify_only(self):
        specs = build_default_specialists(MockLLM(responses=[]))
        action = [t for t in specs["verifier"].tool_names
                  if t not in self.SHARED_READ_TOOLS]
        assert action == ["verify"]

    def test_demo_mode_removes_reconstruct_tool(self):
        """In demo mode the Reconstruction specialist must not be able to
        trigger MATLAB (which would block the demo for ~12 min)."""
        normal = build_default_specialists(MockLLM(responses=[]))
        assert "reconstruct" in normal["reconstruction"].tool_names

        demo = build_default_specialists(MockLLM(responses=[]), demo_mode=True)
        assert "reconstruct" not in demo["reconstruction"].tool_names
        action_tools = [t for t in demo["reconstruction"].tool_names
                        if t not in self.SHARED_READ_TOOLS]
        assert action_tools == ["load_reconstruction"]
        # Other specialists unaffected (compare full lists, shared tools included)
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
            max_rounds=4,
        )

    def test_grounding_rejects_hallucinated_number(self, tmp_path):
        """A report citing a decimal that never appeared in any tool result
        nor in the conversation history must be rejected. The specialist
        gets a second round to recover. This is the architectural guard
        against the AS4DF verifier-hallucination incident."""
        from utility.audit import read_log, filter_log

        s = self._make(
            responses=[
                # First attempt: fabricated number "37.35"
                json.dumps({"done": True,
                            "report": "Divergence is 37.35 s^-1, mask is bad."}),
                # Second attempt: removed the number
                json.dumps({"done": True,
                            "report": "Divergence is elevated; mask is bad."}),
            ],
            tools=["verify"],
        )
        log_path = tmp_path / "log.jsonl"
        log = AuditLog(log_path)
        report = s.handle("verify the mask", workspace=Workspace(), audit=log)
        log.close()
        assert report["done"] is True
        assert "37.35" not in report["report"]
        # Audit log must record the rejection event
        events = filter_log(read_log(log_path), kind="event")
        kinds = [e["data"].get("name") for e in events]
        assert any("grounding_rejection" in k for k in kinds), \
            f"expected a grounding_rejection event in {kinds}"

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

    def test_max_rounds_cap(self, tmp_path):
        # Never says done → should bail after max_rounds with a partial report
        s = self._make(
            responses=[json.dumps({"tool": "verify", "args": {"mask_name": "nope"}})] * 10,
            tools=["verify"],
        )
        log = AuditLog(tmp_path / "log.jsonl")
        report = s.handle("task", workspace=Workspace(), audit=log)
        log.close()
        assert report["done"] is False
        assert "exhausted" in report["report"]

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
                max_rounds=3,
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
        # bare 'max_rounds'.
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

    def test_missing_mask_name_blocks_verifier_delegation(self, tmp_path):
        """When at least one mask exists in the workspace, a delegation to
        verifier without a mask_name field must be rejected with an error
        the Coordinator can react to. Architectural fix for the observed
        AS4DF session where the Verifier specialist invented mask names
        because the Coordinator never said which one to verify."""
        import numpy as np
        coord_llm, specs = self._make_coord(
            responses=[
                # First attempt: delegate to verifier with NO mask_name
                json.dumps({"delegate_to": "verifier", "task": "verify the mask",
                            "why": "test"}),
                # After the rejection, Coordinator retries with mask_name
                json.dumps({"delegate_to": "verifier", "task": "verify the mask",
                            "mask_name": "aorta_v1", "why": "test"}),
                json.dumps({"done": True, "summary": "done after fixing handoff"}),
            ],
            specialists_resps={
                "verifier": [json.dumps({"done": True, "report": "ok"})],
            },
        )
        ws = Workspace()
        ws.masks["aorta_v1"] = np.ones((4, 4, 4), dtype=bool)
        log_path = tmp_path / "log.jsonl"
        log = AuditLog(log_path)
        result = Coordinator(coord_llm, specs, ws, log,
                             max_delegations=5).run("test")
        log.close()
        # The first delegation must have been rejected, second went through
        assert result.specialists_used == ["verifier"]
        entries = read_log(log_path)
        events = filter_log(entries, kind="event")
        names = [e["data"]["name"] for e in events]
        assert "missing_mask_name_in_delegation" in names

    def test_unknown_mask_name_blocks_delegation(self, tmp_path):
        """If the Coordinator names a mask that does not exist, reject the
        delegation rather than letting the specialist call verify() with a
        ToolError-bound argument."""
        import numpy as np
        coord_llm, specs = self._make_coord(
            responses=[
                json.dumps({"delegate_to": "verifier", "task": "verify",
                            "mask_name": "aorta_v99",  # not in workspace
                            "why": "wrong-name test"}),
                json.dumps({"done": True, "summary": "bail"}),
            ],
            specialists_resps={
                "verifier": [json.dumps({"done": True, "report": "ok"})],
            },
        )
        ws = Workspace()
        ws.masks["aorta_v1"] = np.ones((4, 4, 4), dtype=bool)
        log_path = tmp_path / "log.jsonl"
        log = AuditLog(log_path)
        result = Coordinator(coord_llm, specs, ws, log,
                             max_delegations=5).run("test")
        log.close()
        assert result.specialists_used == []  # bad mask name → never ran
        events = filter_log(read_log(log_path), kind="event")
        names = [e["data"]["name"] for e in events]
        assert "unknown_mask_name_in_delegation" in names

    def test_mask_name_is_injected_into_specialist_task(self, tmp_path):
        """When the Coordinator supplies a mask_name, the specialist's task
        string must begin with a structured handoff line — that line is the
        first thing the specialist's LLM sees, eliminating the guessing
        failure mode."""
        import numpy as np
        coord_llm = MockLLM(responses=[
            json.dumps({"delegate_to": "verifier",
                        "task":        "verify the segmented mask",
                        "mask_name":   "aorta_v1",
                        "why":         "single mask in workspace"}),
            json.dumps({"done": True, "summary": "verified"}),
        ])
        captured_task: list[str] = []

        class _SpyVerifier(Specialist):
            def handle(self, task, *, workspace, audit, verbose_callback=None):
                captured_task.append(task)
                return {"done": True, "report": "spy report",
                        "tools_called": [], "n_steps": 1}

        spy = _SpyVerifier(name="verifier", system_prompt="x",
                           tool_names=[], llm=MockLLM(responses=[]),
                           max_rounds=1)
        ws = Workspace()
        ws.masks["aorta_v1"] = np.ones((4, 4, 4), dtype=bool)
        log = AuditLog(tmp_path / "log.jsonl")
        Coordinator(coord_llm, {"verifier": spy}, ws, log,
                    max_delegations=5).run("test")
        log.close()
        assert captured_task, "spy specialist was never invoked"
        assert captured_task[0].startswith(
            '[Coordinator handoff] mask_name = "aorta_v1"')

    def test_recon_retry_blocked_when_one_shot(self, tmp_path):
        """When supports_reconstruction_retry=False (phantom/AS4DF/no-fresh-recon),
        a second delegation to reconstruction must be rejected rather than
        consuming the specialist's budget on a no-op load."""
        coord_llm = MockLLM(responses=[
            json.dumps({"delegate_to": "reconstruction", "task": "load",
                        "why": "first load"}),
            # Coordinator attempts to retry reconstruction
            json.dumps({"delegate_to": "reconstruction",
                        "task": "run with more iterations",
                        "why": "verifier failed"}),
            json.dumps({"done": True, "summary": "gave up after retry blocked"}),
        ])
        recon_spy_calls: list[str] = []

        class _SpyRecon(Specialist):
            def handle(self, task, *, workspace, audit, verbose_callback=None):
                recon_spy_calls.append(task)
                return {"done": True, "report": "loaded once",
                        "tools_called": [], "n_steps": 1}

        spy = _SpyRecon(name="reconstruction", system_prompt="x",
                        tool_names=[], llm=MockLLM(responses=[]),
                        max_rounds=1)
        log_path = tmp_path / "log.jsonl"
        log = AuditLog(log_path)
        Coordinator(coord_llm, {"reconstruction": spy}, Workspace(), log,
                    max_delegations=5,
                    supports_reconstruction_retry=False).run("test")
        log.close()
        # First delegation went through, second was blocked before reaching spy
        assert len(recon_spy_calls) == 1
        events = filter_log(read_log(log_path), kind="event")
        names = [e["data"]["name"] for e in events]
        assert "recon_retry_blocked" in names

    def test_verify_fail_cap_blocks_after_threshold(self, tmp_path):
        """After max_verify_failures_per_mask consecutive failures on the
        same mask, the Coordinator must NOT be able to re-delegate to
        verifier with that mask name — it has to either produce a new
        mask under a different name, or emit done."""
        import numpy as np
        coord_llm = MockLLM(responses=[
            # Attempts: verify aorta_v1 twice, then a 3rd time (should be blocked),
            # then emit done.
            json.dumps({"delegate_to": "verifier", "task": "verify",
                        "mask_name": "aorta_v1", "why": "first"}),
            json.dumps({"delegate_to": "verifier", "task": "verify again",
                        "mask_name": "aorta_v1", "why": "second"}),
            json.dumps({"delegate_to": "verifier", "task": "verify again",
                        "mask_name": "aorta_v1", "why": "third — should be blocked"}),
            json.dumps({"done": True, "summary": "honest escalation"}),
        ])

        class _FailingVerifier(Specialist):
            def handle(self, task, *, workspace, audit, verbose_callback=None):
                # Simulate the verifier producing a workspace verdict of fail
                workspace.verdicts["aorta_v1"] = {"verdict": "fail",
                                                  "checks": {}}
                return {"done": True, "report": "failed",
                        "tools_called": ["verify"], "n_steps": 1}

        spy = _FailingVerifier(name="verifier", system_prompt="x",
                               tool_names=[], llm=MockLLM(responses=[]),
                               max_rounds=1)
        ws = Workspace()
        ws.masks["aorta_v1"] = np.ones((4, 4, 4), dtype=bool)
        log_path = tmp_path / "log.jsonl"
        log = AuditLog(log_path)
        Coordinator(coord_llm, {"verifier": spy}, ws, log,
                    max_delegations=8,
                    max_verify_failures_per_mask=2).run("test")
        log.close()
        events = filter_log(read_log(log_path), kind="event")
        names = [e["data"]["name"] for e in events]
        # First two verifies went through; third was capped
        assert names.count("delegation") == 2
        assert "verify_fail_cap_blocked" in names

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