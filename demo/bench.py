"""A/B benchmark: the same task with and without codemode, on a real Hermes agent.

Run from the hermes-agent checkout with Hermes' Python (the env with quickjs):

    cd ~/.hermes/hermes-agent
    HERMES_HOME=~/.hermes/profiles/<profile> <hermes-venv>/bin/python - < demo/bench.py

Prints model turns (API calls), total tokens and wall time for each mode.
"""

import json
import os
import sys
import time


from hermes_cli.config import load_config  # noqa: E402
from hermes_cli.plugins import discover_plugins  # noqa: E402
from run_agent import AIAgent  # noqa: E402

TASK = os.environ.get("BENCH_TASK") or (
    "Search the web for the 6 most relevant pages about 'QuickJS JavaScript engine'. "
    "Fetch every one of those pages, and tell me for each page: url, title, and whether "
    "the page mentions 'BigInt'. Finish with a one-line verdict.")

discover_plugins()
cfg = load_config()
model = cfg["model"]["default"]
provider = cfg["model"].get("provider")

MODES = {
    "baseline (web tools only)": ["web"],
    "codemode": ["codemode", "web"],
}
HINT = {"codemode": " Use the codemode tool and do the whole job inside ONE script."}

results = {}
for label, toolsets in MODES.items():
    agent = AIAgent(model=model, provider=provider, enabled_toolsets=toolsets,
                    quiet_mode=True, skip_memory=True, skip_context_files=True,
                    platform="cli", max_iterations=30)
    t = time.time()
    out = agent.run_conversation(TASK + HINT.get(label.split()[0], ""))
    tools_used = [tc["function"]["name"] for m in out["messages"] if m.get("role") == "assistant"
                  for tc in (m.get("tool_calls") or [])]
    results[label] = {
        "model_turns": out.get("api_calls"),
        "tool_calls_seen_by_model": len(tools_used),
        "tools": tools_used,
        "total_tokens": agent.session_total_tokens,
        "input_tokens": agent.session_input_tokens,
        "seconds": round(time.time() - t, 1),
        "answer": (out.get("final_response") or "")[:1200],
    }
    print(f"\n=== {label} ===")
    print(json.dumps({k: v for k, v in results[label].items() if k != "answer"}, indent=1))
    print(results[label]["answer"])

json.dump(results, open(os.environ.get("BENCH_OUT", "bench_result.json"), "w"), indent=1)
