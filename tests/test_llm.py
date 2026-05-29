"""Tests for the LLM backend abstraction (Stage 3a).

MockLLM tests are pure unit tests — fast, deterministic. The single
live-Ollama test is auto-skipped when the daemon isn't running.
"""
from __future__ import annotations

import json
import os
import sys

import pytest
import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utility.llm import (
    LLM,
    LLMResponse,
    OllamaLLM,
    ClaudeLLM,
    MockLLM,
    get_llm,
)


# ----- LLMResponse -----------------------------------------------------------

class TestLLMResponse:
    def test_json_parses_valid(self):
        r = LLMResponse(text='{"a": 1, "b": [2, 3]}', model="m", latency_ms=0)
        assert r.json() == {"a": 1, "b": [2, 3]}

    def test_json_raises_on_bad(self):
        r = LLMResponse(text="not json", model="m", latency_ms=0)
        with pytest.raises(json.JSONDecodeError):
            r.json()


# ----- MockLLM ---------------------------------------------------------------

class TestMockLLM:
    def test_returns_responses_in_order(self):
        mock = MockLLM(responses=["one", "two", "three"])
        assert mock.chat([{"role": "user", "content": "x"}]).text == "one"
        assert mock.chat([{"role": "user", "content": "y"}]).text == "two"
        assert mock.chat([{"role": "user", "content": "z"}]).text == "three"

    def test_records_calls(self):
        mock = MockLLM(responses=["ok"])
        mock.chat(
            [{"role": "user", "content": "hi"}],
            temperature=0.7, max_tokens=100, json_mode=True,
        )
        assert len(mock.calls) == 1
        messages, opts = mock.calls[0]
        assert messages == [{"role": "user", "content": "hi"}]
        assert opts == {"temperature": 0.7, "max_tokens": 100, "json_mode": True}

    def test_exhausted_raises(self):
        mock = MockLLM(responses=["only one"])
        mock.chat([{"role": "user", "content": "1"}])
        with pytest.raises(RuntimeError, match="exhausted"):
            mock.chat([{"role": "user", "content": "2"}])

    def test_chat_json_helper(self):
        mock = MockLLM(responses=['{"verdict": "pass"}'])
        result = mock.chat_json([{"role": "user", "content": "go"}])
        assert result == {"verdict": "pass"}
        # The helper must have forwarded json_mode=True to chat()
        assert mock.calls[0][1]["json_mode"] is True

    def test_call_messages_are_deep_copied(self):
        # If the agent mutates messages later, the recorded call shouldn't change
        mock = MockLLM(responses=["ok"])
        msgs = [{"role": "user", "content": "x"}]
        mock.chat(msgs)
        msgs[0]["content"] = "MUTATED"
        assert mock.calls[0][0][0]["content"] == "x"


# ----- ClaudeLLM construction (no network calls) ----------------------------

class TestClaudeLLM:
    def test_requires_api_key(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
            ClaudeLLM()

    def test_uses_env_var(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        c = ClaudeLLM()
        assert c.api_key == "sk-test"

    def test_explicit_key_overrides_env(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "from-env")
        c = ClaudeLLM(api_key="explicit")
        assert c.api_key == "explicit"


# ----- Factory --------------------------------------------------------------

class TestGetLLM:
    def test_ollama(self):
        assert isinstance(get_llm("ollama"), OllamaLLM)

    def test_mock(self):
        assert isinstance(get_llm("mock", responses=["x"]), MockLLM)

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown backend"):
            get_llm("bogus")

    def test_passes_kwargs(self):
        m = get_llm("ollama", model="custom-model", host="http://x:1234")
        assert m.model == "custom-model"
        assert m.host == "http://x:1234"


# ----- Live Ollama integration (skipped if daemon down) ---------------------

def _ollama_up() -> bool:
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


@pytest.mark.skipif(not _ollama_up(), reason="Ollama daemon not reachable on localhost:11434")
class TestOllamaLive:
    # Qwen 3.6 is a reasoning model — internal "thinking" easily uses
    # 100+ tokens before the actual reply, so test budgets must be generous.

    def test_basic_chat(self):
        llm = OllamaLLM(model="qwen3.6")
        resp = llm.chat(
            [{"role": "user", "content": "Reply with the single word OK and nothing else."}],
            temperature=0.0, max_tokens=500,
        )
        assert resp.text.strip(), "got empty response"
        assert resp.latency_ms > 0
        assert resp.model.startswith("qwen3.6")

    def test_json_mode(self):
        llm = OllamaLLM(model="qwen3.6")
        result = llm.chat_json(
            [{"role": "user", "content":
              'Return JSON with one field "ok" set to true. No other text.'}],
            temperature=0.0, max_tokens=500,
        )
        assert isinstance(result, dict)
        assert "ok" in result

    def test_reasoning_field_captured(self):
        # Reasoning models expose their chain-of-thought in a separate field;
        # the agent's audit log needs this even though conversation history doesn't.
        llm = OllamaLLM(model="qwen3.6")
        resp = llm.chat(
            [{"role": "user", "content": "What is 17 * 23? Show your work."}],
            temperature=0.0, max_tokens=800,
        )
        assert resp.text.strip()
        # Qwen 3.6 should populate reasoning for a math question
        assert resp.reasoning is not None and resp.reasoning.strip()
