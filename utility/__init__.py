"""MEDICT utility package — non-agent infrastructure.

Houses everything the multi-agent system depends on but that is itself NOT
an LLM-driven agent:
    - audit       : append-only JSONL audit log
    - llm         : OllamaLLM / ClaudeLLM / MockLLM backends
    - tools       : ToolSpec registry + Workspace + call_tool dispatcher
    - project_context : canonical paths + acquisition defaults injected into prompts
    - input_modes : InputProfile registry (real_scan / phantom / ...)
    - plan_policy : deterministic gate that converts a PlanCritique into an action

The `agents/` package depends on `utility/`. The reverse never happens.
"""
