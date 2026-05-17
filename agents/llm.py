"""LLM backend abstraction for the Stage 3 agent.

Two production backends and a MockLLM for tests. All implement the same
``LLM`` protocol so the rest of the agent code never depends on a vendor.

    OllamaLLM("qwen3.6")       — local development (free, slow)
    ClaudeLLM("claude-opus-4-7") — final evaluation runs (paid, fast)
    MockLLM(responses=[...])   — deterministic scripted responses for tests

Message format follows the OpenAI convention so it round-trips cleanly to
both backends:

    [{"role": "system"|"user"|"assistant", "content": "..."}]
"""
from __future__ import annotations

import json
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import requests


# ----- Message type ----------------------------------------------------------

Message = dict[str, str]  # {"role": str, "content": str}


@dataclass
class LLMResponse:
    """What every backend returns. The agent code reads ``.text`` for prose
    and ``.json`` for structured calls; ``reasoning`` is the model's internal
    chain-of-thought when the backend exposes it (Qwen 3.6, future Claude
    extended thinking) — useful for the audit log but not for conversation
    history. ``raw`` and token counts feed the audit log."""
    text: str
    model: str
    latency_ms: int
    reasoning: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    def json(self) -> Any:
        """Parse ``.text`` as JSON. Raises ``json.JSONDecodeError`` on bad output."""
        return json.loads(self.text)


# ----- Base class -----------------------------------------------------------

class LLM(ABC):
    """Stateless chat backend. Each ``chat()`` is independent."""

    model: str  # set by subclasses

    @abstractmethod
    def chat(
        self,
        messages: list[Message],
        *,
        temperature: float = 0.2,
        max_tokens: int = 2048,
        json_mode: bool = False,
    ) -> LLMResponse: ...

    # Convenience wrapper — same as chat() with json_mode=True + parse.
    def chat_json(
        self,
        messages: list[Message],
        *,
        temperature: float = 0.2,
        max_tokens: int = 2048,
    ) -> Any:
        resp = self.chat(messages, temperature=temperature,
                         max_tokens=max_tokens, json_mode=True)
        return resp.json()


# ----- Ollama backend (local, via HTTP API) ---------------------------------

class OllamaLLM(LLM):
    """Talks to a locally-running Ollama daemon (default localhost:11434).

    No extra Python package needed — uses the documented /api/chat endpoint.
    """

    def __init__(
        self,
        model: str = "qwen3.6",
        host: str = "http://localhost:11434",
        timeout_s: int = 600,
    ):
        self.model = model
        self.host = host.rstrip("/")
        self.timeout_s = timeout_s

    def chat(
        self,
        messages: list[Message],
        *,
        temperature: float = 0.2,
        max_tokens: int = 2048,
        json_mode: bool = False,
    ) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        if json_mode:
            payload["format"] = "json"

        t0 = time.time()
        r = requests.post(f"{self.host}/api/chat", json=payload, timeout=self.timeout_s)
        r.raise_for_status()
        data = r.json()
        latency_ms = int((time.time() - t0) * 1000)

        # Ollama returns {"message": {"role": "assistant", "content": "...",
        #                              "thinking": "..."  # reasoning models only
        #                             }, ...}
        msg = data["message"]
        return LLMResponse(
            text=msg["content"],
            reasoning=msg.get("thinking"),
            model=data.get("model", self.model),
            latency_ms=latency_ms,
            prompt_tokens=data.get("prompt_eval_count"),
            completion_tokens=data.get("eval_count"),
            raw=data,
        )


# ----- Claude backend (cloud, anthropic SDK is optional) --------------------

class ClaudeLLM(LLM):
    """Talks to the Claude API via the official ``anthropic`` SDK.

    The SDK is an optional dependency (only needed for paid eval runs). If
    it's not installed, the import error is deferred until first use.
    """

    def __init__(
        self,
        model: str = "claude-opus-4-7",
        api_key: str | None = None,
    ):
        self.model = model
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY not set. Either pass api_key= or export the env var."
            )
        self._client = None  # lazy

    def _get_client(self):
        if self._client is None:
            try:
                import anthropic
            except ImportError as e:
                raise ImportError(
                    "ClaudeLLM requires the 'anthropic' package. "
                    "Install with: pip install anthropic"
                ) from e
            self._client = anthropic.Anthropic(api_key=self.api_key)
        return self._client

    def chat(
        self,
        messages: list[Message],
        *,
        temperature: float = 0.2,
        max_tokens: int = 2048,
        json_mode: bool = False,
    ) -> LLMResponse:
        client = self._get_client()

        # Claude API splits system from the conversation
        system_parts = [m["content"] for m in messages if m["role"] == "system"]
        chat_msgs = [m for m in messages if m["role"] != "system"]
        system = "\n\n".join(system_parts) if system_parts else None

        if json_mode:
            extra = "\n\nRespond ONLY with valid JSON. No prose, no markdown fences."
            system = (system + extra) if system else extra

        t0 = time.time()
        resp = client.messages.create(
            model=self.model,
            system=system or "",
            messages=chat_msgs,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        latency_ms = int((time.time() - t0) * 1000)

        text = "".join(b.text for b in resp.content if hasattr(b, "text"))
        return LLMResponse(
            text=text.strip(),
            model=resp.model,
            latency_ms=latency_ms,
            prompt_tokens=resp.usage.input_tokens,
            completion_tokens=resp.usage.output_tokens,
            raw={"id": resp.id, "stop_reason": resp.stop_reason},
        )


# ----- Mock backend (for tests) ---------------------------------------------

class MockLLM(LLM):
    """Returns scripted responses in order. Records every call for inspection.

    Useful for testing agent logic without a live LLM:

        mock = MockLLM(responses=["Hi", '{"action": "verify"}'])
        out1 = mock.chat([{"role": "user", "content": "..."}]).text  # "Hi"
        out2 = mock.chat_json([...])                                 # {"action": "verify"}
        assert mock.calls[0][0][0]["content"] == "..."  # first call's messages
    """

    def __init__(self, responses: list[str] | None = None, model: str = "mock"):
        self.model = model
        self.responses = list(responses or [])
        self.calls: list[tuple[list[Message], dict]] = []

    def chat(
        self,
        messages: list[Message],
        *,
        temperature: float = 0.2,
        max_tokens: int = 2048,
        json_mode: bool = False,
    ) -> LLMResponse:
        self.calls.append((
            [dict(m) for m in messages],
            {"temperature": temperature, "max_tokens": max_tokens, "json_mode": json_mode},
        ))
        if not self.responses:
            raise RuntimeError(
                f"MockLLM exhausted after {len(self.calls)} calls — pass more responses=."
            )
        return LLMResponse(
            text=self.responses.pop(0),
            model=self.model,
            latency_ms=0,
        )


# ----- Convenience factory --------------------------------------------------

def get_llm(backend: str = "ollama", **kwargs) -> LLM:
    """Construct a backend by name. Used by the playground / CLI / agent config."""
    if backend == "ollama":
        return OllamaLLM(**kwargs)
    if backend == "claude":
        return ClaudeLLM(**kwargs)
    if backend == "mock":
        return MockLLM(**kwargs)
    raise ValueError(f"Unknown backend {backend!r}; expected 'ollama'|'claude'|'mock'")
