#!/usr/bin/env bash
#
# Bootstrap and launch the `hostrun` MCP server.
#
# Lets a freshly cloned checkout serve its tests to sandboxed coding agents in one
# command. Idempotent: safe to re-run, and re-running is the normal way to restart the
# server after changing its flags.
#
#   ./scripts/run_host_bridge_mcp.sh                 # bootstrap, then serve on 127.0.0.1:8765
#   ./scripts/run_host_bridge_mcp.sh --port 9000     # a different port
#   ./scripts/run_host_bridge_mcp.sh --setup-only    # bootstrap and exit without serving
#
# What it does, in order:
#   1. verifies uv and python3
#   2. ensures .venv exists (the interpreter the *tests* run under)
#   3. creates .host-bridge-mcp-token if absent, 0600
#   4. merges a `hostrun` entry into .mcp.json, preserving any other servers
#   5. adds the generated files to .git/info/exclude
#   6. allows the port through the sandbox network policy, if `sbx` is present
#   7. execs the server
#
# Note the server itself does NOT run inside .venv. It runs via `uv run --script` against
# the PEP 723 header in host_bridge_mcp.py, so its dependencies never enter the project
# environment. .venv matters only because the server spawns tests with it.

set -euo pipefail

HOST="127.0.0.1"
PORT="8765"
MODEL=""
SETUP_ONLY=0

usage() {
    # Print the header comment block, stopping at the first line that is not a comment,
    # so editing the block above never desynchronises this.
    awk 'NR>2 && /^#/ { sub(/^# ?/, ""); print; next } NR>2 { exit }' "$0"
    cat <<'EOF'

Options:
  --host ADDR        Bind address (default 127.0.0.1; keep on loopback).
  --port N           Bind port (default 8765).
  --model ID         Model id for the ${model} placeholder (default: the server's).
  --setup-only       Do everything except starting the server.
  -h, --help         Show this help.
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        --host)  HOST="$2"; shift 2 ;;
        --port)  PORT="$2"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        --setup-only) SETUP_ONLY=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "$(basename "$0"): unknown option $1" >&2; usage >&2; exit 2 ;;
    esac
done

# Resolve the repository root from this script's location, following symlinks, so the
# script works when invoked through a PATH entry or from any working directory.
SOURCE="${BASH_SOURCE[0]}"
while [ -L "$SOURCE" ]; do
    DIR="$(cd -P "$(dirname "$SOURCE")" && pwd)"
    SOURCE="$(readlink "$SOURCE")"
    [ "${SOURCE#/}" = "$SOURCE" ] && SOURCE="$DIR/$SOURCE"
done
SCRIPT_DIR="$(cd -P "$(dirname "$SOURCE")" && pwd)"
REPO="$(cd -P "$SCRIPT_DIR/.." && pwd)"

SERVER="$SCRIPT_DIR/host_bridge_mcp.py"
VENV_PYTHON="$REPO/.venv/bin/python"
TOKEN_PATH="$REPO/.host-bridge-mcp-token"
MCP_JSON="$REPO/.mcp.json"

say()  { printf '\033[1m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[33m==> %s\033[0m\n' "$*" >&2; }
die()  { printf '\033[31m==> %s\033[0m\n' "$*" >&2; exit 1; }

[ -f "$SERVER" ] || die "server not found: $SERVER"

# ---------------------------------------------------------------------------- 1. tools
command -v uv      >/dev/null 2>&1 || die "uv not found. Install: https://docs.astral.sh/uv/"
command -v python3 >/dev/null 2>&1 || die "python3 not found."

# ----------------------------------------------------------------------------- 2. venv
# This is the interpreter the tests run under, not the server's. It must exist before a
# run is attempted, and a fresh clone has no .venv.
if [ ! -x "$VENV_PYTHON" ]; then
    warn "No .venv found at $REPO/.venv"
    echo "    The server runs tests with that interpreter, so it is required."
    echo "    Installing this project's dependencies downloads several GB (torch, mlx, ...)."
    echo
    # `uv sync` is not used: this project declares no [build-system], so uv would try to
    # build it as a package and fail. Installing from pyproject.toml directly is the
    # documented working path.
    SETUP_CMDS="uv venv '$REPO/.venv' && uv pip install -r '$REPO/pyproject.toml' --python '$VENV_PYTHON'"
    if [ -t 0 ]; then
        printf "    Create it now? [Y/n] "
        read -r reply
        case "$reply" in
            [Nn]*) die "Aborted. Run this when ready:\n    $SETUP_CMDS" ;;
        esac
        say "Creating .venv (this will take a while)"
        uv venv "$REPO/.venv"
        uv pip install -r "$REPO/pyproject.toml" --python "$VENV_PYTHON"
    else
        die "Not a terminal, so not installing unprompted. Run:\n    $SETUP_CMDS"
    fi
    [ -x "$VENV_PYTHON" ] || die "venv creation did not produce $VENV_PYTHON"
    say "Created .venv"
else
    say "Found .venv"
fi

# ---------------------------------------------------------------------------- 3. token
# The server creates this too, but doing it here means the token exists before any agent
# session starts. Claude Code runs the .mcp.json headersHelper at connect time, which
# fails outright if the file is missing.
if [ -f "$TOKEN_PATH" ]; then
    chmod 600 "$TOKEN_PATH"
    say "Reusing existing token ($TOKEN_PATH)"
else
    # Written 0600 from the start via os.open, never world-readable even momentarily.
    python3 - "$TOKEN_PATH" <<'PY'
import os, secrets, sys
path = sys.argv[1]
fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
with os.fdopen(fd, "w") as handle:
    handle.write(secrets.token_urlsafe(32))
PY
    say "Generated token ($TOKEN_PATH)"
fi

# -------------------------------------------------------------------------- 4. mcp.json
# Merged rather than overwritten: .mcp.json holds every project-scoped MCP server, so
# clobbering it would silently drop a user's other servers. Generated with json.dumps so
# the headersHelper quoting is correct by construction.
python3 - "$MCP_JSON" "$TOKEN_PATH" "$PORT" <<'PY'
import json, pathlib, sys

mcp_path, token_path, port = pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3]

helper = (
    'python3 -c "import json,pathlib; '
    "print(json.dumps({'Authorization': 'Bearer ' + "
    f"pathlib.Path('{token_path}').read_text().strip()}}))\""
)

entry = {
    "type": "http",
    # Templated so the sandbox can fall back to the gateway's IPv4 literal, or to
    # 127.0.0.1 behind a socat shim, without editing this file.
    "url": f"http://${{PREEMPT_HOSTRUN_HOST:-host.docker.internal}}:${{PREEMPT_HOSTRUN_PORT:-{port}}}/mcp",
    "headersHelper": helper,
    # Above the client's 60s time-to-first-byte timer, with headroom over the server's
    # own 120s maximum inline wait.
    "timeout": 180000,
}

config, existing = {}, None
if mcp_path.exists():
    try:
        config = json.loads(mcp_path.read_text())
    except json.JSONDecodeError:
        backup = mcp_path.with_suffix(".json.bak")
        mcp_path.rename(backup)
        print(f"    existing .mcp.json was not valid JSON; moved to {backup.name}")
        config = {}
    existing = config.get("mcpServers", {}).get("hostrun")

servers = config.setdefault("mcpServers", {})
others = [name for name in servers if name != "hostrun"]
servers["hostrun"] = entry
mcp_path.write_text(json.dumps(config, indent=2) + "\n")

action = "unchanged" if existing == entry else ("updated" if existing else "added")
note = f", preserved {len(others)} other server(s)" if others else ""
print(f"    hostrun entry {action}{note}")
PY
say "Wrote $MCP_JSON"

# --------------------------------------------------------------------------- 5. ignores
# This repo keeps machine-local ignores in .git/info/exclude rather than .gitignore, so
# generated files never show up in a diff.
EXCLUDE="$REPO/.git/info/exclude"
if [ -d "$REPO/.git" ]; then
    mkdir -p "$(dirname "$EXCLUDE")"; touch "$EXCLUDE"
    for pattern in ".host-bridge/" ".host-bridge-mcp-token" ".mcp.json"; do
        grep -qxF "$pattern" "$EXCLUDE" || printf '%s\n' "$pattern" >> "$EXCLUDE"
    done
    say "Local ignores present in .git/info/exclude"
else
    warn "Not a git repository; skipping .git/info/exclude"
fi

# ---------------------------------------------------------------------- 6. network policy
# Sandboxes reach the host over this port only once the policy allows it, and the grant
# does not survive a reboot. `sbx` exists on the host but not inside a sandbox.
if command -v sbx >/dev/null 2>&1; then
    if sbx policy allow network "localhost:$PORT" >/dev/null 2>&1; then
        say "Allowed localhost:$PORT through the sandbox network policy"
    else
        warn "Could not set the network policy. Run manually:"
        echo "    sbx policy allow network localhost:$PORT"
    fi
else
    warn "sbx not found. If you use sandboxed agents, run this on the host:"
    echo "    sbx policy allow network localhost:$PORT"
fi

if [ "$SETUP_ONLY" -eq 1 ]; then
    say "Setup complete. Start the server with: $0 --port $PORT"
    exit 0
fi

# ---------------------------------------------------------------------------- 7. serve
ARGS=(--repo "$REPO" --host "$HOST" --port "$PORT")
[ -n "$MODEL" ] && ARGS+=(--default-model "$MODEL")

say "Starting server on http://$HOST:$PORT/mcp  (Ctrl-C to stop)"
echo

# VIRTUAL_ENV is cleared rather than activated: `uv run --script` resolves dependencies
# from the script's PEP 723 header, and an ambient venv only makes uv warn about the
# mismatch. exec so Ctrl-C reaches the server directly.
unset VIRTUAL_ENV
exec uv run --script "$SERVER" "${ARGS[@]}"
