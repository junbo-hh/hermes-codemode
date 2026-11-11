# codemode — Code-as-Orchestration for Hermes Agent

**One tool call instead of twenty.** `codemode` lets the model write a short
JavaScript snippet that calls your Hermes tools in a loop, filters the results
and returns only the answer. The intermediate data never touches the context
window.

```js
// one codemode call = search + 5 page fetches + aggregation, ONE model turn
var hits = tools.web_search({query: "QuickJS release notes", limit: 5}).data.web;
var pages = tools.web_extract({urls: hits.map(function (h) { return h.url; })}).results;
return pages.map(function (p) {
  return {url: p.url, chars: (p.content || "").length,
          mentionsBigInt: /bigint/i.test(p.content || "")};
});
```

Without codemode the model makes one inference per tool call and every raw
page lands in its context. With codemode the loop runs inside a sandbox and
only the ~200-byte summary comes back.

## Why a plugin, not a prompt

A prompt can ask a model to "batch" calls. It cannot give the model a runtime.
This plugin ships one:

| Piece | What it does |
|---|---|
| **QuickJS sandbox** | Each run gets a fresh JS context in a **separate child process**: 256 MB heap cap, **hard-killed on timeout** (a runaway `while(true){}` never keeps burning CPU), no `require`, no filesystem, no network, minimal env. |
| **Tool bridge** | Every MCP tool (`mcp__*`) plus an allowlist of built-ins (`web_search`, `web_extract` by default) becomes `tools.<name>(args)`. Calls go back through Hermes' public `ctx.dispatch_tool`, so **Hermes' approvals, credentials and error handling stay in charge**. |
| **On-demand schemas** | Tool schemas are *not* dumped into the prompt. Scripts find tools with `searchTools("keywords")` (self-contained BM25) and read one schema with `describeTool(name)`. |
| **Recovery hints** | A misspelled tool throws `unknown tool: ... (did you mean: ...)`. Tool errors throw catchable JS errors. |
| **Budgets** | Per-script timeout (default 30 s, max 120 s), tool-call budget (default 200), result cap (100 KB). |

It is the Hermes take on the "code mode" idea from
[Pi](https://github.com/badlogic/pi-mono), Cloudflare's Code Mode and the
CodeAct paper: models are much better at writing code than at emitting long
chains of tool-call JSON.

## Install

```bash
hermes plugins install junbo-hh/hermes-codemode --enable
```

The only dependency is [`quickjs`](https://pypi.org/project/quickjs/)
(`>=1.19.2,<2`); Hermes asks for consent and installs it.

**Restart Hermes after installing** (quit and reopen the Desktop app, or
restart your gateway). A running Hermes process keeps its old dependency
environment, so it will not see `codemode` until it restarts.

Then ask for something that needs several tool calls. The model sees one new
tool, `codemode`. With Tool Search on (the default), it is listed in the
`tool_search` catalog with its one-line description. Set
`tools.tool_search.enabled: off` if you want it in the tool list directly.

Optional: force all MCP usage through codemode by hiding the raw MCP tools:

```yaml
disabled_toolsets: ["mcp"]
```

## Settings

`plugins.entries.codemode.settings` in `config.yaml` (also editable in the
Desktop Plugins tab):

| Key | Default | Meaning |
|---|---|---|
| `extra_tools` | `[web_search, web_extract]` | Non-MCP tools exposed in the sandbox. MCP tools are always exposed. |
| `timeout_s` | `30` | Default script timeout (hard max 120). |
| `memory_mb` | `256` | QuickJS heap limit per run. |
| `max_tool_calls` | `200` | Tool-call budget per run. |

Never exposed, regardless of settings: `codemode` itself, `delegate_task`,
`clarify`, `execute_code`, `memory`, `skill_manage`, `cronjob`,
`send_message`.

## Sandbox API

| In JS | Returns |
|---|---|
| `tools.<name>(args)` | The tool's result, parsed from JSON. Throws on tool error. |
| `searchTools(query, topK=5)` | `[{name, description, score}]`, BM25-ranked |
| `describeTool(name)` | `{ok, name, description, schema}` |
| `console.log(...)` | Collected and returned next to the value as `logs` |
| `return <value>` | The only thing that goes back to the model |

The tool result is `{ok, result, stats: {tool_calls, calls, elapsed_s}}` or
`{ok: false, error, stats}`.

JS is synchronous: tool calls block, no `await` needed.

## Disclosure

- **Child process:** every run starts a short-lived Python child process
  (`python codemode.py --worker`) that hosts QuickJS. It gets a minimal
  environment (`PATH`, `HOME`, locale, Python path; no API keys) and is
  killed when the script finishes or times out.
- **No network, no file access** from the plugin itself. Anything the script
  does goes through the Hermes tools you exposed, with their normal behaviour
  and approvals.
- No telemetry, no stored credentials, no background processes.

## Tests

```bash
pip install quickjs pytest
pytest          # 12 tests: chaining, BM25, errors, did-you-mean hints, timeout
                # kill, memory cap, no host access, fresh context, budgets, logs
```

## Files

| File | Role |
|---|---|
| `plugin.yaml` | Manifest (`provides_tools: [codemode]`, `quickjs` dependency, settings) |
| `__init__.py` | `register(ctx)`: builds the tool catalog from the registry, dispatches through `ctx.dispatch_tool` |
| `codemode.py` | Harness-agnostic runtime: BM25 catalog, sandbox process protocol, QuickJS worker |
| `tests/` | pytest suite (no Hermes needed) |
| `demo/bench.py` | A/B run of one task on a real Hermes agent, with and without codemode |

## License

MIT © Junbo Hu
