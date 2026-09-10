#!/usr/bin/env bash
# Probe `tokenaut mcp` end to end over stdio in one session: handshake, tools/list
# and one call per tool. Prints `pass|fail|skip <check>` per line, exits 0 only
# when nothing failed. Uses a throwaway HOME so the real index is untouched.
set -u
BIN="${1:-tokenaut}"
export PROBE_HOME
PROBE_HOME="$(mktemp -d)"
trap 'rm -rf "$PROBE_HOME"' EXIT

python3 - "$BIN" <<'PY'
import json, os, socket, subprocess, sys, tempfile

binary = sys.argv[1]
env = dict(os.environ, HOME=os.environ["PROBE_HOME"])
proc = subprocess.Popen([binary, "mcp"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                        stderr=subprocess.DEVNULL, env=env, text=True)
next_id = [0]
results = []

def send(method, params=None, notify=False):
    msg = {"jsonrpc": "2.0", "method": method}
    if params is not None:
        msg["params"] = params
    if not notify:
        next_id[0] += 1
        msg["id"] = next_id[0]
    proc.stdin.write(json.dumps(msg) + "\n")
    proc.stdin.flush()
    if notify:
        return None
    while True:
        line = proc.stdout.readline()
        if not line:
            raise SystemExit("server closed stdout")
        reply = json.loads(line)
        if reply.get("id") == msg["id"]:
            return reply

def call(name, arguments):
    reply = send("tools/call", {"name": name, "arguments": arguments})
    if "error" in reply:
        return "ERROR: " + json.dumps(reply["error"])
    return "".join(c.get("text", "") for c in reply["result"]["content"])

def check(name, ok, detail=""):
    results.append(ok)
    print(("pass" if ok else "fail") + " " + name + (f"  ({detail})" if detail and not ok else ""))

def online():
    try:
        socket.create_connection(("example.com", 443), timeout=3).close()
        return True
    except OSError:
        return False

init = send("initialize", {"protocolVersion": "2025-06-18", "capabilities": {},
                           "clientInfo": {"name": "probe", "version": "0"}})
check("initialize names tokenaut", init.get("result", {}).get("serverInfo", {}).get("name") == "tokenaut")
send("notifications/initialized", notify=True)

tools = sorted(t["name"] for t in send("tools/list")["result"]["tools"])
expected = ["ctx_batch_execute", "ctx_execute", "ctx_execute_file", "ctx_fetch_and_index", "ctx_search"]
check("tools/list lists the five tools", tools == expected, str(tools))

text = call("ctx_execute", {"language": "shell", "code": "printf 'a\\nb\\n'"})
check("ctx_execute shell", "exit 0" in text and "a\nb" in text, text[:120])

text = call("ctx_execute", {"language": "python", "code": "print(sum(range(10)))"})
check("ctx_execute python", "\n45" in text or text.strip().endswith("45"), text[:120])

text = call("ctx_execute", {"language": "javascript", "code": "console.log(process.version)"})
check("ctx_execute javascript", any(l.startswith("v") for l in text.splitlines()), text[:120])

text = call("ctx_execute", {"language": "shell", "intent": "needle",
                            "code": "for i in $(seq 1 800); do echo \"filler line $i padding padding\"; done; echo 'the needle is here'"})
check("ctx_execute large output is indexed", len(text) < 6000 and "## needle" in text and "needle is here" in text,
      f"{len(text)} bytes")

with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
    f.write("x" * 123)
    sample = f.name
text = call("ctx_execute_file", {"path": sample, "language": "shell", "code": "printf '%s' \"$FILE_CONTENT\" | wc -c"})
check("ctx_execute_file exposes FILE_CONTENT", "123" in text, text[:120])
os.unlink(sample)

if online():
    text = call("ctx_fetch_and_index", {"url": "https://example.com", "source": "example"})
    check("ctx_fetch_and_index example.com", "Example Domain" in text and "chunks" in text, text[:160])
    text = call("ctx_search", {"queries": ["Example Domain"], "source": "fetch:example"})
    check("ctx_search filtered by source", "--- [fetch:example" in text, text[:160])
else:
    print("skip ctx_fetch_and_index (offline)")
    print("skip ctx_search filtered by source (offline)")

text = call("ctx_batch_execute", {"commands": [{"label": "one", "command": "printf 'same text\\n'"},
                                               {"label": "two", "command": "printf 'same text\\n'"}],
                                  "queries": ["same"]})
check("ctx_batch_execute dedups repeated output", text.count("same text") == 2 and "same output as" in text and "already shown above" in text, text[:200])

proc.stdin.close()
proc.wait(timeout=10)
sys.exit(0 if all(results) else 1)
PY
