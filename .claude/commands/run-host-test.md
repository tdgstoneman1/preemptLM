---
description: Run a test script or pytest on the macOS host through the host-bridge MCP server
argument-hint: [script path or test target, e.g. tests/integration/mlx_single_router_layer.py]
---

# Run Host Test

Run a script under `tests/` on the macOS host through the `host-bridge` MCP server, then report
the result. This is the only way to exercise MLX/Metal code from a Linux sandbox, where the
project's `.venv` and MLX cannot run.

## What This Command Does

1. **Verifies the host-side server is running** — and stops with setup instructions if it is not,
   since the server cannot be started from inside the sandbox.
2. **Lists runnable targets** to resolve the requested script to a real path.
3. **Starts the run** as a job, receiving a run id and a log path.
4. **Reads the log directly** through the bind mount while the run proceeds.
5. **Polls to completion** and reports status, exit code, duration, and peak swap.

## Usage

```bash
# Run one integration script
/run-host-test tests/integration/mlx_single_router_layer.py

# Run the layer-resolution smoke script
/run-host-test tests/integration/mlx_layer_search.py

# No argument: list what is runnable and ask which to run
/run-host-test
```

## Implementation Steps

### 1. Confirm the host-side server is running — ALWAYS DO THIS FIRST

The MCP server runs on the **macOS host, outside this sandbox**. It cannot be started from here.

Call `mcp__hostrun__list_targets` (no arguments).

**If the call succeeds**: the server is up. Note `default_model`, the discovered `scripts`, and
`swap_used_gb`. Proceed to step 2.

**If the call fails, the tool is not registered, or the tools do not appear at all**: confirm with
a direct health check using the Bash tool:

```bash
curl -sS -m 5 http://host.docker.internal:8765/healthz
```

A healthy server returns `{"ok":true,"active_run":null,"swap_used_gb":2.77}`.

**If `/healthz` succeeds but the tool call failed, the server is up and the problem is
client-side authentication — do NOT tell the user to start the server.** The most common cause
is a stale token: `headersHelper` reads the token file once at connect time, so if `.mcp.json`
or the token file changed during the session, the client is still sending the old value. A tool
error mentioning `HTTP 404`, `Invalid OAuth error response`, or `Not Found` is this, not a
missing server — the client received a 401 and fell back to OAuth discovery, which 404s because
this server has no OAuth endpoints.

Confirm by testing the token the client should be using:

```bash
curl -sS -m 5 -o /dev/null -w 'http=%{http_code}\n' -X POST http://host.docker.internal:8765/mcp \
  -H "Authorization: Bearer $(cat .host-bridge-mcp-token)" \
  -H 'Content-Type: application/json' -H 'Accept: application/json, text/event-stream' \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/list"}'
```

`http=200` means the token on disk is valid and only the client's cached copy is stale. Report
that and ask the user to reconnect the server from `/mcp`, or reload Claude Code. Then stop.
`http=401` means the token on disk does not match the running server — ask the user to restart
the server with `./scripts/run_host_bridge_mcp.sh`, which reuses the token file on disk.

If `/healthz` itself fails, **STOP and report to the user**. Do NOT attempt to start the server, do NOT
run the MLX script locally, and do NOT fall back to the sandbox's Linux venv for an MLX test —
it will fail with a confusing import or architecture error. Instead, output exactly this:

```
The host-bridge MCP server is not reachable, so host tests cannot run.

The server runs on your Mac, outside this sandbox, so I cannot start it. Please run these
in a terminal on the host:

    # first time only — makes the launcher executable
    ./scripts/setup.sh

    # bootstrap and serve; leave this running in its own tab
    ./scripts/run_host_bridge_mcp.sh

The launcher creates .host-bridge-mcp-token, merges the server entry into .mcp.json, adds the
generated files to .git/info/exclude, and allows the port through the sandbox network policy.

Two things to expect:
  - `sbx policy allow network localhost:8765` is re-run by the launcher, but the grant does
    not survive a host reboot.
  - If Claude Code has never connected to this server, it will prompt to approve the
    project-scoped MCP server, and may need `/mcp` -> reconnect.

Tell me once it is running and I will re-run this command.
```

Then end the command. Do not proceed to step 2.

### 2. Resolve the target script

If the user supplied an argument, match it against the `scripts` list from `list_targets`.
Paths may be repo-relative (`tests/integration/mlx_layer_search.py`) or tests-relative
(`integration/mlx_layer_search.py`). Only paths under `tests/` can be executed.

If no argument was supplied, present the `scripts` list and ask the user which to run. Stop and
wait for the answer.

If the argument matches no known script, report the mismatch, show the available scripts, and stop.

### 3. Read the script before running it

Use the Read tool on the resolved script. Determine its actual `argparse` flags rather than
assuming — these scripts change. Specifically check whether it accepts `--output`, and whether
that default is a temporary directory.

**`tests/integration/mlx_single_router_layer.py` defaults `--output` to a `TemporaryDirectory`
that is deleted on exit.** Always pass `--output "${run_dir}/<name>.parquet"` for it, or the
artifact will not survive for inspection.

### 4. Start the run

Call `mcp__hostrun__run_script` with `wait_s: 90`.

Two placeholders are expanded server-side, so host-only values never need to be hardcoded:

- `${model}` — the host's default model id
- `${run_dir}` — this run's output directory, readable from the sandbox at the same path

Example for the router-capture script:

```
mcp__hostrun__run_script(
  script = "tests/integration/mlx_single_router_layer.py",
  args = ["--model", "${model}",
          "--config", "tests/integration/qwen3_6-35b-mlx-single_layer.toml",
          "--output", "${run_dir}/router-events.parquet"],
  wait_s = 90
)
```

Loading a 35B model takes roughly 60–90 seconds, so expect `status: "running"` on the first
return for MLX scripts.

### 5. Follow the run

If `status` is `"running"`, use the Read tool on the returned `log_path`. The repository is
bind-mounted at the same absolute path on both sides, so that file is readable from here and
updates live. Prefer reading the log over repeated polling — it holds complete output, whereas
`tail` is truncated.

Then call `mcp__hostrun__poll_run` with the `run_id` and `wait_s: 60`, repeating until `status`
is no longer `"running"`.

To stop a run early, call `mcp__hostrun__cancel_run` with the `run_id`.

### 6. Verify artifacts

If the script wrote an artifact into `${run_dir}`, confirm it exists and inspect it rather than
trusting the script's own success message. For a Parquet trace, check row count and that the
record's contract holds — for example that `expert_ids` has `top_k` entries, `expert_weights`
sum to ~1.0, and `gate_logits` spans all experts.

### 7. Report

State the final `status`, `returncode`, `duration_s`, and `peak_swap_gb`, plus the `log_path` and
any artifact path. Quote the relevant log lines for a failure rather than paraphrasing them.

## Important Notes

- **NEVER try to start the MCP server from the sandbox.** It runs on the host by design. The only
  correct response to an unreachable server is the instruction block in step 1.
- **NEVER run an MLX test with the sandbox's Linux interpreter.** MLX and Metal do not exist there
  and the project `.venv` holds macOS wheels. Per the project rules, never install into or modify
  that `.venv` from the sandbox.
- **Only one run executes at a time.** A second run is refused with an error naming the active run.
  Wait with `poll_run` or stop it with `cancel_run` — do not retry in a loop.
- **High swap is normal, not a failure.** A single 35B router-capture run peaks around 22.5 GB of
  swap on this 32 GB machine and completes fine; the machine deliberately runs past physical RAM.
  Only `status: "aborted_memory"` indicates the watchdog intervened.
- **Status `orphaned`** means the server restarted while that run was in flight. Its result is
  unknown — re-run it.
- **Tool names come from the `.mcp.json` server key**, currently `hostrun`, giving
  `mcp__hostrun__*`. If that key is renamed the prefix changes; resolve the actual tool names
  before assuming.

## Error Handling

If `run_script` returns a `Refused:` error:

- **Path outside `tests/`** — only `tests/` is executable. Report it and call `list_targets`.
- **Another run active** — report the active run id and its log path; offer to poll or cancel.
- **Swap at the runaway ceiling** — report the reported figures; something on the host is
  consuming memory.

If the run finishes with a non-zero `returncode`, read the full log at `log_path`, quote the
traceback, and report it. Do NOT automatically re-run — surface the failure and ask how to proceed.

Note that pytest exit code `5` means "no tests collected", which is not a crash.

## Example Output

```
## Host test: tests/integration/mlx_single_router_layer.py

**Status**: passed (exit 0, 76.0s, peak swap 22.47 GB)

**Log**: .host-bridge/20260801-114208-70e5-mlx_single_router_layer/run.log

Resolved one router layer: path='language_model.model.layers.0.mlp',
layer_class='Qwen3NextSparseMoeBlock', layer_idx=0
Captured and wrote 1 router record(s).
Verified Parquet file: 1 row(s), 1 row group(s).

**Artifact**: .host-bridge/20260801-114208-70e5-mlx_single_router_layer/router-events.parquet
  1 row; expert_ids has 8 entries, weights sum to 0.997, gate_logits spans 256 experts.
```
