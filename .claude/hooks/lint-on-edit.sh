#!/bin/bash
# Lint on Edit Hook
# Type-checks Python files with pyright after they are written or edited.
#
# Claude Code passes the hook payload as JSON on stdin (NOT via env vars).
# The edited path lives at .tool_input.file_path.

PAYLOAD=$(cat)

if command -v jq >/dev/null 2>&1; then
	FILE_PATH=$(printf '%s' "$PAYLOAD" | jq -r '.tool_input.file_path // empty')
else
	# macOS has no jq by default; fall back to the stdlib json module.
	FILE_PATH=$(printf '%s' "$PAYLOAD" | python3 -c \
		'import json,sys; print(json.load(sys.stdin).get("tool_input",{}).get("file_path",""))')
fi

# Only type-check Python files that actually exist on disk.
if [[ -f "$FILE_PATH" && "$FILE_PATH" =~ \.py$ ]]; then
	# `uv tool run` == `uvx`, but is present wherever `uv` itself is on PATH.
	# In the Linux sandbox, point pyright at the throwaway venv so project
	# imports (attrs, pydantic, pyarrow, ...) resolve; without it every edit
	# reports bogus unresolved-import diagnostics. On the macOS host that venv
	# does not exist and pyright falls back to its default resolution.
	VENV_PY=/home/agent/.venvs/preempt-linux/bin/python
	if [[ -x "$VENV_PY" ]]; then
		uv tool run pyright --pythonpath "$VENV_PY" "$FILE_PATH" 2>/dev/null
	else
		uv tool run pyright "$FILE_PATH" 2>/dev/null
	fi
fi

# Never block the edit on a linter failure.
exit 0
