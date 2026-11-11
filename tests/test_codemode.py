"""Tests for the codemode runtime (no Hermes needed): `pytest tests/`."""

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from codemode import BM25Index, CodemodeRunner, sanitize_js_name  # noqa: E402

ISSUES = [{"id": i, "title": f"issue {i}", "labels": ["urgent"] if i % 3 == 0 else []}
          for i in range(1, 31)]


def make_runner(**kw):
    def list_issues(args):
        return json.dumps(ISSUES)  # Hermes handlers return JSON strings

    def get_issue(args):
        return json.dumps({"id": args["id"], "body": "x" * 5000})

    def boom(args):
        raise RuntimeError("upstream 500")

    return CodemodeRunner(tools=[
        ("mcp__tracker__list_issues", "List issues in the bug tracker", {}, list_issues),
        ("mcp__tracker__get_issue", "Get one issue with full body by id", {"type": "object"}, get_issue),
        ("mcp__weather__forecast", "Weather forecast for a city", {}, lambda a: {"t": 20}),
        ("mcp__flaky__boom", "Always fails", {}, boom),
    ], **kw)


def test_chain_filter_and_only_return_value_escapes():
    r = make_runner()
    out = r.run("""
        var urgent = tools.mcp__tracker__list_issues({}).filter(function (i) {
            return i.labels.indexOf('urgent') >= 0; });
        var bodies = urgent.map(function (i) { return tools.mcp__tracker__get_issue({id: i.id}); });
        return {count: urgent.length, totalBodyChars: bodies.reduce(function (s, b) {
            return s + b.body.length; }, 0)};
    """)
    assert out["ok"], out
    assert out["result"] == {"count": 10, "totalBodyChars": 50000}
    assert out["stats"]["tool_calls"] == 11          # 11 calls, one model turn
    assert len(json.dumps(out)) < 2000                # 50KB stayed in the sandbox


def test_bm25_search_and_describe():
    r = make_runner()
    out = r.run("return [searchTools('weather forecast')[0].name, describeTool('mcp__tracker__get_issue').schema];")
    assert out["ok"], out
    assert out["result"] == ["mcp__weather__forecast", {"type": "object"}]


def test_tool_error_is_catchable():
    out = make_runner().run("try { tools.mcp__flaky__boom({}); } catch (e) { return 'caught: ' + e.message; }")
    assert out["ok"] and "upstream 500" in out["result"]


def test_unknown_tool_suggests_close_match():
    out = make_runner().run("return describeTool('mcp__tracker__list_issue');")
    assert "did you mean: mcp__tracker__list_issues" in out["result"]["error"]


def test_timeout_kills_runaway_script():
    r = make_runner()
    t = time.monotonic()
    out = r.run("while (true) {}", timeout_s=1.5)
    assert not out["ok"] and "timeout" in out["error"]
    assert time.monotonic() - t < 4


def test_memory_cap():
    out = make_runner(memory_limit_mb=16).run(
        "var a = []; while (true) { a.push(new Array(100000).fill('x')); }", timeout_s=20)
    assert not out["ok"]
    assert "memory" in out["error"].lower() or "InternalError" in out["error"]


def test_no_host_access():
    out = make_runner().run(
        "return [typeof require, typeof process, typeof fetch, typeof std, typeof os, typeof XMLHttpRequest];")
    assert out["ok"] and out["result"] == ["undefined"] * 6


def test_fresh_context_per_run():
    r = make_runner()
    assert r.run("globalThis.leak = 42; return 1;")["ok"]
    assert r.run("return typeof leak;")["result"] == "undefined"


def test_tool_call_budget():
    out = make_runner(max_tool_calls=3).run(
        "for (var i = 0; i < 10; i++) tools.mcp__weather__forecast({}); return 'unreachable';")
    assert not out["ok"] and "budget" in out["error"]


def test_console_log_and_syntax_error():
    r = make_runner()
    out = r.run("console.log('hi', {a: 1}); return 7;")
    assert out["result"] == {"value": 7, "logs": ["hi {\"a\":1}"]}
    bad = r.run("return (;")
    assert not bad["ok"] and "SyntaxError" in bad["error"]


def test_helpers():
    assert sanitize_js_name("mcp-server.tool") == "mcp_server_tool"
    assert sanitize_js_name("9lives") == "_9lives"
    idx = BM25Index([("a", "list github pull requests"), ("b", "weather")])
    assert idx.search("pull requests")[0][0] == "a"
