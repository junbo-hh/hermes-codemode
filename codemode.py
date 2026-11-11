"""Codemode core: Code-as-Orchestration runtime for Hermes Agent.

Inspired by Pi's Codemode / Cloudflare Code Mode:
- The model writes JavaScript instead of emitting sequential JSON tool calls.
- Tool schemas are NOT preloaded into context; scripts discover tools on demand
  via BM25-ranked searchTools() / describeTool().
- Every execution runs in a FRESH QuickJS context inside a short-lived child
  process (memory-capped, hard-killed on timeout) with zero host access: no
  require, no filesystem, no network -- only the injected tool callbacks.
- Only the script's return value re-enters the model's context.

Process model:
    parent (Hermes)  --init {code, catalog}-->  child (QuickJS)
    parent           <--{"call": name, args}--  child   (tool call request)
    parent           --{"ok", result|error}-->  child   (parent dispatches)
    parent           <--{"done": ok, ...}-----  child
On timeout the parent kills the child, so a runaway `while(true){}` cannot
keep burning CPU or memory inside the Hermes process.

This module is harness-agnostic: CodemodeRunner takes a list of
(name, description, schema, call_fn) tuples. Run `python codemode.py --worker`
only as the child (the runner does this itself).
"""

from __future__ import annotations

import difflib
import json
import math
import os
import re
import select
import subprocess
import sys
import time
from collections import Counter
from typing import Any, Callable, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# JS identifier sanitization (invalid chars -> "_")
# ---------------------------------------------------------------------------

_JS_IDENT_RE = re.compile(r"[^A-Za-z0-9_$]")


def sanitize_js_name(name: str) -> str:
    safe = _JS_IDENT_RE.sub("_", name)
    if safe and safe[0].isdigit():
        safe = "_" + safe
    return safe or "_unnamed"


# ---------------------------------------------------------------------------
# Minimal BM25 (no external dependency)
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> List[str]:
    # split snake/camel/path-ish names too: mcp__github__list_prs -> mcp github list prs
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    return _TOKEN_RE.findall(text.lower())


class BM25Index:
    """Tiny BM25 over (doc_id, text) documents. k1=1.5, b=0.75."""

    def __init__(self, docs: List[Tuple[str, str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b, self.docs = k1, b, docs
        self._tf: List[Counter] = []
        self._len: List[int] = []
        df: Counter = Counter()
        for _, text in docs:
            toks = _tokenize(text)
            self._tf.append(Counter(toks))
            self._len.append(len(toks))
            df.update(set(toks))
        n = len(docs)
        self._idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}
        self._avg = (sum(self._len) / n) if n else 1.0

    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        scores: Dict[int, float] = {}
        for term in set(_tokenize(query)):
            idf = self._idf.get(term)
            if idf is None:
                continue
            for i, tf_counter in enumerate(self._tf):
                tf = tf_counter.get(term, 0)
                if not tf:
                    continue
                denom = tf + self.k1 * (1 - self.b + self.b * self._len[i] / (self._avg or 1))
                scores[i] = scores.get(i, 0.0) + idf * tf * (self.k1 + 1) / denom
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[: max(1, top_k)]
        return [(self.docs[i][0], s) for i, s in ranked]


class Catalog:
    """Tool metadata + discovery. Lives in the child (and parent, for hints)."""

    def __init__(self, entries: List[Dict[str, Any]]):
        # entries: [{"js": js_name, "name": registry_name, "description", "schema"}]
        self.by_js: Dict[str, Dict[str, Any]] = {e["js"]: e for e in entries}
        self.index = BM25Index([(e["js"], f'{e["name"]} {e["description"]}') for e in entries])

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        out = []
        for js, score in self.index.search(query, top_k):
            e = self.by_js[js]
            out.append({"name": js, "description": e["description"][:300],
                        "score": round(score, 3)})
        return out

    def describe(self, name: str) -> Dict[str, Any]:
        e = self.by_js.get(sanitize_js_name(name))
        if e is None:
            return {"ok": False, "error": f"unknown tool: {name}{self.hint(name)}"}
        return {"ok": True, "name": e["js"], "description": e["description"],
                "schema": e["schema"]}

    def hint(self, name: str) -> str:
        js = sanitize_js_name(name)
        keys = list(self.by_js)
        lower = {k.lower(): k for k in keys}
        m = [lower[c] for c in difflib.get_close_matches(js.lower(), list(lower), n=3, cutoff=0.6)]
        if not m:
            hits = [h["name"] for h in self.search(name.replace("_", " "), 3)]
            m = hits
        return f" (did you mean: {', '.join(m)}?)" if m else ""


# ---------------------------------------------------------------------------
# Parent side
# ---------------------------------------------------------------------------

ToolDef = Tuple[str, str, Dict[str, Any], Callable[[Dict[str, Any]], Any]]
MAX_RESULT_CHARS = 100_000


class CodemodeRunner:
    def __init__(self, tools: List[ToolDef], memory_limit_mb: int = 256,
                 time_limit_s: float = 30, max_tool_calls: int = 200,
                 python: Optional[str] = None):
        self.memory_limit_mb = memory_limit_mb
        self.time_limit_s = time_limit_s
        self.max_tool_calls = max_tool_calls
        self.python = python or sys.executable
        self._calls: Dict[str, Tuple[str, Callable]] = {}
        entries = []
        for name, desc, schema, fn in tools:
            js, base, n = sanitize_js_name(name), sanitize_js_name(name), 2
            while js in self._calls:
                js, n = f"{base}_{n}", n + 1
            self._calls[js] = (name, fn)
            entries.append({"js": js, "name": name, "description": desc or "",
                            "schema": schema or {}})
        self._entries = entries
        self.catalog = Catalog(entries)

    @property
    def tool_names(self) -> List[str]:
        return list(self._calls)

    # convenience for hosts / tests
    def search_tools(self, query: str, top_k: int = 5):
        return self.catalog.search(query, top_k)

    def _dispatch(self, js: str, args: Any) -> Dict[str, Any]:
        entry = self._calls.get(js)
        if entry is None:
            return {"ok": False, "error": f"unknown tool: {js}{self.catalog.hint(js)}"}
        _, fn = entry
        try:
            res = fn(args if isinstance(args, dict) else {})
        except Exception as e:  # noqa: BLE001 - surface to the script
            return {"ok": False, "error": f"{type(e).__name__}: {e}"}
        if isinstance(res, str):  # Hermes tool handlers return JSON strings
            try:
                res = json.loads(res)
            except ValueError:
                pass
        if isinstance(res, dict) and set(res) == {"error"}:
            return {"ok": False, "error": str(res["error"])}
        return {"ok": True, "result": res}

    def run(self, code: str, timeout_s: Optional[float] = None) -> Dict[str, Any]:
        timeout = float(timeout_s or self.time_limit_s)
        started = time.monotonic()
        stats = {"tool_calls": 0, "calls": []}
        # Minimal env; propagate the parent's import path so the child finds
        # quickjs wherever Hermes installed plugin dependencies.
        env = {k: v for k, v in os.environ.items() if k in ("PATH", "LANG", "LC_ALL", "HOME")}
        env["PYTHONPATH"] = os.pathsep.join(p for p in sys.path if p and os.path.isdir(p))
        try:
            proc = subprocess.Popen(
                [self.python, os.path.abspath(__file__), "--worker"],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                env=env, cwd=os.path.dirname(os.path.abspath(__file__)),
                text=True, bufsize=1)
        except OSError as e:
            return {"ok": False, "error": f"could not start sandbox: {e}"}

        def finish(out: Dict[str, Any]) -> Dict[str, Any]:
            if proc.poll() is None:
                proc.kill()
            try:
                proc.wait(timeout=2)
            except Exception:  # noqa: BLE001
                pass
            for f in (proc.stdin, proc.stdout, proc.stderr):
                try:
                    f.close()
                except Exception:  # noqa: BLE001
                    pass
            out["stats"] = {"tool_calls": stats["tool_calls"], "calls": stats["calls"][:50],
                            "elapsed_s": round(time.monotonic() - started, 3)}
            return out

        try:
            proc.stdin.write(json.dumps({"code": code, "catalog": self._entries,
                                         "memory_mb": self.memory_limit_mb}) + "\n")
            proc.stdin.flush()
            while True:
                remaining = timeout - (time.monotonic() - started)
                if remaining <= 0:
                    return finish({"ok": False, "error": f"timeout after {timeout:g}s "
                                   "(sandbox process killed)"})
                ready, _, _ = select.select([proc.stdout], [], [], remaining)
                if not ready:
                    continue
                line = proc.stdout.readline()
                if not line:
                    err = (proc.stderr.read() or "").strip()[-500:]
                    return finish({"ok": False, "error": "sandbox exited unexpectedly"
                                   + (f": {err}" if err else "")})
                msg = json.loads(line)
                if "call" in msg:
                    stats["tool_calls"] += 1
                    stats["calls"].append(msg["call"])
                    if stats["tool_calls"] > self.max_tool_calls:
                        reply = {"ok": False, "error": f"tool call budget exceeded "
                                 f"({self.max_tool_calls})"}
                    else:
                        reply = self._dispatch(msg["call"], msg.get("args"))
                    proc.stdin.write(json.dumps(reply, default=str) + "\n")
                    proc.stdin.flush()
                elif "done" in msg:
                    if msg["done"]:
                        result = msg.get("result")
                        text = json.dumps(result, default=str)
                        if len(text) > MAX_RESULT_CHARS:
                            return finish({"ok": True, "truncated": True,
                                           "result": text[:MAX_RESULT_CHARS]})
                        return finish({"ok": True, "result": result})
                    return finish({"ok": False, "error": msg.get("error", "unknown error")})
        except (BrokenPipeError, ValueError, OSError) as e:
            return finish({"ok": False, "error": f"sandbox I/O error: {type(e).__name__}: {e}"})


# ---------------------------------------------------------------------------
# Child side (QuickJS sandbox)
# ---------------------------------------------------------------------------

_PRELUDE = r"""
var tools = {};
function __registerTool(jsName) {
  tools[jsName] = function (args) {
    var env = JSON.parse(__call_tool(jsName, JSON.stringify(args === undefined ? {} : args)));
    if (!env.ok) { throw new Error(env.error); }
    return env.result;
  };
}
function searchTools(query, topK) { return JSON.parse(__search_tools(String(query), topK || 5)); }
function describeTool(name) { return JSON.parse(__describe_tool(String(name))); }
var console = { log: function () {
  __log(Array.prototype.map.call(arguments, function (a) {
    return typeof a === 'string' ? a : JSON.stringify(a); }).join(' ')); } };
console.info = console.warn = console.error = console.log;
function __sealTools() {
  var real = tools;
  tools = new Proxy(real, { get: function (t, k) {
    if (typeof k !== 'string' || k in t) { return t[k]; }
    throw new Error('unknown tool: tools.' + k + __hint(k));
  }});
}
"""


def _worker() -> None:
    import quickjs  # only the child needs it

    stdin, stdout = sys.stdin, sys.stdout
    init = json.loads(stdin.readline())
    catalog = Catalog(init["catalog"])
    logs: List[str] = []

    def send(obj: Dict[str, Any]) -> None:
        stdout.write(json.dumps(obj, default=str) + "\n")
        stdout.flush()

    def call_tool(js: str, args_json: str) -> str:
        try:
            args = json.loads(args_json) if args_json else {}
        except ValueError:
            return json.dumps({"ok": False, "error": "args must be JSON-serializable"})
        if js not in catalog.by_js:
            return json.dumps({"ok": False, "error": f"unknown tool: {js}{catalog.hint(js)}"})
        send({"call": js, "args": args})
        return stdin.readline()

    def log(s: str) -> None:
        if len(logs) < 100:
            logs.append(str(s)[:2000])

    ctx = quickjs.Context()
    ctx.set_memory_limit(int(init.get("memory_mb", 256)) * 1024 * 1024)
    ctx.set_max_stack_size(1024 * 1024)
    ctx.add_callable("__call_tool", call_tool)
    ctx.add_callable("__search_tools", lambda q, k=5: json.dumps(catalog.search(q, int(k))))
    ctx.add_callable("__describe_tool", lambda n: json.dumps(catalog.describe(n)))
    ctx.add_callable("__log", log)
    ctx.add_callable("__hint", lambda n: catalog.hint(str(n)))
    try:
        ctx.eval(_PRELUDE)
        for js in catalog.by_js:
            ctx.eval(f"__registerTool({json.dumps(js)});")
        ctx.eval("__sealTools();")
        wrapped = ("(function(){ var __r = (function(){\n%s\n})();"
                   " return __r === undefined ? 'null' : JSON.stringify(__r); })()") % init["code"]
        raw = ctx.eval(wrapped)
        result = json.loads(raw) if isinstance(raw, str) else raw
        send({"done": True, "result": {"value": result, "logs": logs} if logs else result})
    except Exception as e:  # noqa: BLE001 - JSException and friends
        msg = str(e).strip()
        send({"done": False, "error": f"{type(e).__name__}: {msg}"
              + (f" | logs: {logs[-5:]}" if logs else "")})


if __name__ == "__main__" and "--worker" in sys.argv:
    _worker()
