"""Hermes plugin: codemode -- Code-as-Orchestration.

Registers one tool, ``codemode``. The model passes JavaScript; the script runs
in a fresh QuickJS sandbox (child process) where the exposed Hermes tools are
callable as ``tools.<name>(args)``. Tool schemas are not preloaded: scripts
discover them with ``searchTools()`` / ``describeTool()``. Calls go back
through ``ctx.dispatch_tool`` so Hermes' own approvals, credentials and
error handling stay in charge.

Exposed tools: every MCP tool (``mcp__*``) plus the allowlist in the
``extra_tools`` setting (default: web_search, web_extract).
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

try:
    from .codemode import CodemodeRunner
except ImportError:  # imported outside a package (e.g. pytest collection)
    from codemode import CodemodeRunner  # type: ignore[no-redef]

logger = logging.getLogger("plugins.codemode")

TOOL_NAME = "codemode"
TOOLSET = "codemode"
MCP_PREFIX = "mcp__"
DEFAULT_EXTRA = ["web_search", "web_extract"]
# Never expose these inside the sandbox (recursion / agent control surfaces).
NEVER = {TOOL_NAME, "delegate_task", "clarify", "execute_code", "memory",
         "skill_manage", "cronjob", "send_message"}

CODEMODE_SCHEMA: Dict[str, Any] = {
    "name": TOOL_NAME,
    "description": (
        "Run JavaScript that orchestrates many tool calls in ONE step "
        "(Code-as-Orchestration). Inside the script every exposed tool is "
        "tools.<name>(args) and returns parsed JSON. Discover tools with "
        "searchTools('keywords') (BM25) and describeTool(name) for the JSON "
        "schema -- schemas are not preloaded. Loop, filter, join and aggregate "
        "INSIDE the script; only the final `return` value comes back. "
        "Synchronous JS (no await needed), console.log allowed, no require/"
        "fs/network. Prefer this over sequential calls when a task needs 3+ "
        "tool calls or large intermediate results."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "code": {"type": "string",
                     "description": "JavaScript function body; use `return` for the result."},
            "timeout": {"type": "number",
                        "description": "Seconds before the sandbox is killed (default 30, max 120)."},
        },
        "required": ["code"],
    },
}

_ctx = None
_cache: Dict[str, CodemodeRunner] = {}


def _setting(key: str, default: Any) -> Any:
    try:
        val = _ctx.get_config(key, default) if _ctx is not None else default
    except Exception:  # noqa: BLE001
        val = default
    return default if val is None else val


def _exposed_entries() -> List[Any]:
    from tools.registry import registry

    extra = set(_setting("extra_tools", DEFAULT_EXTRA) or [])
    out = []
    for e in registry.get_all_entries():
        name = getattr(e, "name", "")
        if not name or name in NEVER:
            continue
        if not (name.startswith(MCP_PREFIX) or name in extra):
            continue
        check = getattr(e, "check_fn", None)
        if callable(check):
            try:
                if not check():
                    continue
            except Exception:  # noqa: BLE001
                continue
        out.append(e)
    return out


def _runner(dispatch_kwargs: Dict[str, Any]) -> CodemodeRunner:
    entries = _exposed_entries()
    key = ",".join(sorted(e.name for e in entries))
    runner = _cache.get(key)
    if runner is None:
        tools = []
        for e in entries:
            schema = getattr(e, "schema", {}) or {}
            params = schema.get("parameters", schema)
            desc = getattr(e, "description", "") or schema.get("description", "")
            tools.append((e.name, desc, params, None))
        runner = CodemodeRunner(
            tools=tools,
            time_limit_s=float(_setting("timeout_s", 30)),
            memory_limit_mb=int(_setting("memory_mb", 256)),
            max_tool_calls=int(_setting("max_tool_calls", 200)),
        )
        _cache.clear()
        _cache[key] = runner
        logger.info("codemode: runner with %d tools", len(tools))
    # bind the dispatcher for this call (per-call kwargs like task_id)
    for js, (reg_name, _) in list(runner._calls.items()):
        runner._calls[js] = (reg_name, (lambda n: lambda args: _ctx.dispatch_tool(
            n, args, **dispatch_kwargs))(reg_name))
    return runner


def _handle(args: Dict[str, Any], **kwargs) -> str:
    try:
        code = (args or {}).get("code", "")
        if not isinstance(code, str) or not code.strip():
            return json.dumps({"ok": False, "error": "code must be a non-empty string"})
        try:
            timeout = float((args or {}).get("timeout") or _setting("timeout_s", 30))
        except (TypeError, ValueError):
            timeout = 30.0
        timeout = max(1.0, min(timeout, 120.0))
        fwd = {k: v for k, v in kwargs.items() if k in ("task_id", "session_id")}
        runner = _runner(fwd)
        if not runner.tool_names:
            return json.dumps({"ok": False, "error": "no tools exposed to codemode "
                               "(connect an MCP server or set extra_tools)"})
        return json.dumps(runner.run(code, timeout_s=timeout), default=str)
    except Exception as e:  # noqa: BLE001 - handlers must never raise
        logger.exception("codemode failed")
        return json.dumps({"ok": False, "error": f"{type(e).__name__}: {e}"})


def _check() -> bool:
    try:
        import quickjs  # noqa: F401
        return True
    except ImportError:
        return False


def register(ctx) -> None:
    global _ctx
    _ctx = ctx
    ctx.register_tool(
        name=TOOL_NAME,
        toolset=TOOLSET,
        schema=CODEMODE_SCHEMA,
        handler=_handle,
        check_fn=_check,
        description=CODEMODE_SCHEMA["description"],
        emoji="⚙️",
    )
