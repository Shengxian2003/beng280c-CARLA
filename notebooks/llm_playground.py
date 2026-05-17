"""Interactive REPL for the LLM backend (Stage 3a).

Lets you chat with the LLM, see latency / token counts, and try JSON mode —
useful for sanity-checking model behavior before wiring it into the agent.

    python notebooks/llm_playground.py
    python notebooks/llm_playground.py --backend ollama --model qwen3.6
    python notebooks/llm_playground.py --backend claude --model claude-opus-4-7

Slash commands inside the REPL:
    /system <text>   set/replace the system prompt
    /json on|off     toggle JSON output mode
    /temp <0..1>     set sampling temperature
    /think on|off    show or hide the model's internal reasoning (Qwen 3.6)
    /reset           clear conversation history
    /show            print current history
    /save <path>     save conversation as JSON
    /quit            exit (also Ctrl-D)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agents.llm import get_llm, LLM


def banner(llm: LLM, backend: str, json_mode: bool, temperature: float) -> str:
    return (
        f"\n{'='*64}\n"
        f"  LLM Playground — backend={backend} model={llm.model}\n"
        f"  json_mode={json_mode}  temperature={temperature}\n"
        f"  Type /help for commands, /quit to exit.\n"
        f"{'='*64}\n"
    )


HELP = textwrap.dedent("""\
    Slash commands:
        /system <text>   set/replace system prompt
        /json on|off     toggle JSON output mode
        /temp <float>    set temperature (0.0 = deterministic, 1.0 = creative)
        /think on|off    show / hide model's internal reasoning (Qwen 3.6)
        /reset           clear conversation
        /show            print conversation history
        /save <path>     save conversation to JSON file
        /help            show this help
        /quit            exit
""")


def repl(llm: LLM, backend: str, *, system: str | None, json_mode: bool,
         temperature: float, show_thinking: bool = False):
    history: list[dict] = []
    if system:
        history.append({"role": "system", "content": system})

    print(banner(llm, backend, json_mode, temperature))
    if system:
        print(f"[system] {system}\n")

    while True:
        try:
            line = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue

        # ---- Slash commands ------------------------------------------------
        if line.startswith("/"):
            cmd, _, arg = line[1:].partition(" ")
            cmd = cmd.lower()

            if cmd in ("quit", "exit", "q"):
                break
            if cmd in ("help", "?"):
                print(HELP); continue
            if cmd == "reset":
                history = [m for m in history if m["role"] == "system"]
                print("[history cleared]"); continue
            if cmd == "show":
                for m in history:
                    print(f"  [{m['role']}] {m['content'][:200]}")
                continue
            if cmd == "system":
                # Replace any existing system message
                history = [m for m in history if m["role"] != "system"]
                if arg:
                    history.insert(0, {"role": "system", "content": arg})
                    print(f"[system set] {arg}")
                else:
                    print("[system cleared]")
                continue
            if cmd == "json":
                json_mode = arg.lower() in ("on", "true", "1", "yes")
                print(f"[json_mode = {json_mode}]"); continue
            if cmd == "temp":
                try:
                    temperature = float(arg)
                    print(f"[temperature = {temperature}]")
                except ValueError:
                    print(f"[bad temperature: {arg!r}]")
                continue
            if cmd == "think":
                show_thinking = arg.lower() in ("on", "true", "1", "yes")
                print(f"[show_thinking = {show_thinking}]"); continue
            if cmd == "save":
                if not arg:
                    print("[usage: /save <path>]"); continue
                Path(arg).write_text(json.dumps(history, indent=2))
                print(f"[saved {len(history)} messages → {arg}]")
                continue
            print(f"[unknown command: /{cmd} — try /help]"); continue

        # ---- Regular message ----------------------------------------------
        history.append({"role": "user", "content": line})
        try:
            resp = llm.chat(history, temperature=temperature, json_mode=json_mode)
        except Exception as e:
            print(f"[error] {type(e).__name__}: {e}")
            # Don't keep the user message in history if the call failed
            history.pop()
            continue

        # Pretty-print JSON output for readability
        text = resp.text
        if json_mode:
            try:
                text = json.dumps(json.loads(resp.text), indent=2)
            except json.JSONDecodeError:
                text = f"[!! invalid JSON returned !!]\n{resp.text}"

        if show_thinking and resp.reasoning:
            print(f"\n[thinking]\n{resp.reasoning}")
        print(f"\n{text}\n")
        meta = f"[{resp.latency_ms} ms"
        if resp.prompt_tokens is not None or resp.completion_tokens is not None:
            meta += f", in={resp.prompt_tokens} out={resp.completion_tokens} tok"
        if resp.reasoning and not show_thinking:
            meta += f", thinking={len(resp.reasoning)} chars — /think on to view"
        meta += "]"
        print(meta + "\n")

        history.append({"role": "assistant", "content": resp.text})


def main():
    p = argparse.ArgumentParser(description="Interactive LLM playground")
    p.add_argument("--backend", choices=["ollama", "claude", "mock"], default="ollama")
    p.add_argument("--model", default=None,
                   help="Model name. Defaults: qwen3.6 (ollama), claude-opus-4-7 (claude)")
    p.add_argument("--system", default=None, help="Initial system prompt")
    p.add_argument("--json", action="store_true", help="Start in JSON output mode")
    p.add_argument("--temp", type=float, default=0.2, help="Sampling temperature (default 0.2)")
    p.add_argument("--think", action="store_true",
                   help="Show the model's internal reasoning by default (Qwen 3.6)")
    args = p.parse_args()

    kwargs = {}
    if args.model:
        kwargs["model"] = args.model
    if args.backend == "mock":
        kwargs.setdefault("responses", ["mock reply 1", "mock reply 2", "mock reply 3"])
    llm = get_llm(args.backend, **kwargs)

    repl(llm, args.backend, system=args.system, json_mode=args.json,
         temperature=args.temp, show_thinking=args.think)


if __name__ == "__main__":
    main()
