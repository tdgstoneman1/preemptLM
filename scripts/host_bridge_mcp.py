# /// script
# requires-python = ">=3.12"
# dependencies = ["fastmcp>=3.0,<4", "attrs>=25.0"]
# ///
"""
`hostrun` — an MCP server that runs `preempt`'s tests on the macOS host.

Coding agents for this project run inside a Linux sandbox where MLX, Metal, and the
macOS `.venv` cannot execute, so the MLX integration scripts under `tests/` are
unrunnable from there. This server runs on the host and exposes those scripts over
streamable HTTP, which is the transport the sandbox boundary can actually cross.

Two properties of the setup shape the whole design:

- The repository is bind-mounted into the sandbox at the *same absolute path*, so a log
  file written here is readable by the agent with its ordinary file-reading tool. Run
  output is therefore never pushed through the MCP protocol — only a short tail is.
- HTTP MCP clients enforce a time-to-first-byte timer, so a tool call cannot block for
  the several minutes a 35B model load takes. Runs are therefore jobs: start, poll, cancel.

Everything executable is restricted to the repository's `tests/` directory. This is a
scoping control, not a security boundary — an agent that can write into `tests/` through
the mount can run what it wrote. It bounds a confused agent to the repo and leaves an
audit trail; it does not contain a hostile one.

Launch (from `$HOME`, so uv cannot pick up the surrounding project):

    cd ~ && uv run --script ./preempt/scripts/host_bridge_mcp.py \
      --repo ./preempt --host 127.0.0.1 --port 8765

Pair that, once per host boot, with `sbx policy allow network localhost:8765`.
"""

# TODO refactor!

from __future__ import annotations

from typing import Annotated, Literal
from collections.abc import Sequence

import argparse
import asyncio
import json
import logging
import os
import re
import secrets
import shlex
import signal
import subprocess
from contextlib import suppress
from datetime import datetime
from pathlib import Path

import attrs
from fastmcp import Context, FastMCP
from fastmcp.exceptions import ToolError
from fastmcp.server.auth.providers.jwt import StaticTokenVerifier
from pydantic import BaseModel, Field
from starlette.requests import Request
from starlette.responses import JSONResponse

LOG_PREFIX = "##HOST_BRIDGE"
TAIL_WINDOW_BYTES = 65_536
TAIL_LINE_CHARS = 500
SWAP_SAMPLE_SECONDS = 5.0
SWAP_BREACH_SAMPLES = 6
CANCEL_GRACE_SECONDS = 10.0
RECENT_RUNS = 10


logger = logging.getLogger("host_bridge")


@attrs.define(frozen=True)
class Config:
    """Immutable server configuration resolved from the command line."""

    # TODO use pydantic
    # TODO move to config/

    repo: Path
    tests_dir: Path
    runs_dir: Path
    token_path: Path
    python: Path
    host: str
    port: int
    default_model: str
    max_swap_gb: float
    abort_swap_gb: float


def _parse_args(argv: Sequence[str] | None = None) -> Config:
    """
    Parse command-line arguments into a `Config`.

    Parameters
    ----------
    argv : Sequence[str] | None
        Argument vector, or None to read `sys.argv`.

    Returns
    -------
    Config
        Validated configuration.

    Raises
    ------
    SystemExit
        If the repository, its `tests/` directory, or the project virtualenv's
        interpreter is missing. Failing here is far cheaper than failing on the
        first tool call.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--repo",
        type=Path,
        required=True,
        help="Absolute path to the preempt repository.",
    )
    parser.add_argument(
        "--host", default="127.0.0.1", help="Bind address. Keep on loopback."
    )
    parser.add_argument("--port", type=int, default=8765, help="Bind port.")
    parser.add_argument(
        "--default-model",
        default="unsloth/Qwen3.6-35B-A3B-MLX-8bit",
        help="Model id substituted for the `${model}` placeholder in run arguments.",
    )
    # Measured 2026-08-01: one `mlx_single_router_layer` run against
    # Qwen3.6-35B-A3B-MLX-8bit peaked at 22.77GB of swap and was still climbing, against
    # a ~3GB idle baseline. Swap here is the intended mechanism for exceeding 32GB of
    # physical memory, not a warning sign, so these thresholds exist only to catch genuine
    # runaway — never to police normal operation.
    parser.add_argument(
        "--max-swap-gb",
        type=float,
        default=32.0,
        help="Soft swap threshold. Exceeding it is logged and reported, never enforced.",
    )
    parser.add_argument(
        "--abort-swap-gb",
        type=float,
        default=64.0,
        help="Hard swap ceiling. A run is terminated if swap stays above this for ~30s.",
    )
    args = parser.parse_args(argv)

    repo = args.repo.expanduser().resolve()
    tests_dir = repo / "tests"
    python = repo / ".venv" / "bin" / "python"

    for path, what in (
        (repo, "repository"),
        (tests_dir, "tests directory"),
        (python, "virtualenv interpreter"),
    ):
        if not path.exists():
            parser.error(f"{what} not found: {path}")

    return Config(
        repo=repo,
        tests_dir=tests_dir.resolve(),
        runs_dir=repo / ".host-bridge",
        token_path=repo / ".host-bridge-mcp-token",
        python=python,
        host=args.host,
        port=args.port,
        default_model=args.default_model,
        max_swap_gb=args.max_swap_gb,
        abort_swap_gb=args.abort_swap_gb,
    )


# This is a single-file script, never imported as a library, so arguments are resolved at
# module scope. The tool decorators below close over `CFG`, and the auth provider needs
# the token before the `FastMCP` instance exists.
CFG = _parse_args()


def _load_or_create_token(path: Path) -> str:
    """
    Return the shared bearer token, creating it on first run.

    The token is stable across restarts so the client's `headersHelper`, which reads this
    same file through the bind mount, keeps working without reconfiguration.

    Parameters
    ----------
    path : Path
        Location of the token file.

    Returns
    -------
    str
        The token value.
    """
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        os.chmod(path, 0o600)  # in case it was created with looser permissions
        return path.read_text().strip()

    token = secrets.token_urlsafe(32)
    with os.fdopen(fd, "w") as handle:
        handle.write(token)
    return token


TOKEN = _load_or_create_token(CFG.token_path)

# Swap telemetry
_SWAP_FIELD = re.compile(r"used\s*=\s*([0-9.]+)([KMG])", re.IGNORECASE)
_SWAP_SCALE = {"K": 1 / 1024 / 1024, "M": 1 / 1024, "G": 1.0}


def swap_used_gb() -> float | None:
    """
    Return macOS swap usage in GiB, or None if it cannot be determined.

    Reads `sysctl -n vm.swapusage`, which reports e.g.
    `total = 6144.00M  used = 3210.50M  free = 2933.50M  (encrypted)`.

    Notes
    -----
    This machine deliberately runs past physical RAM — a single router-capture run peaks
    near 35 GB on 32 GB of unified memory and completes fine on swap. Swap usage, not
    resident size, is therefore the meaningful health signal.

    Returns
    -------
    float | None
        Swap used in GiB, or None on any failure (non-macOS, sysctl missing, parse error).
    """
    try:
        out = subprocess.run(
            ["sysctl", "-n", "vm.swapusage"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None

    match = _SWAP_FIELD.search(out)
    if match is None:
        return None
    return float(match.group(1)) * _SWAP_SCALE[match.group(2).upper()]


# Path allowlist
def resolve_under_tests(user_path: str, *, must_be_file: bool) -> Path:
    """
    Resolve a caller-supplied path and assert it lands inside the repo's `tests/` tree.

    Symlinks are resolved *before* the containment check, so neither `../` traversal nor a
    symlink planted inside `tests/` can escape.

    Parameters
    ----------
    user_path : str
        Repo-relative (`tests/integration/x.py`) or tests-relative (`integration/x.py`)
        path. Absolute paths are accepted but must still resolve inside `tests/`.
    must_be_file : bool
        Whether the target must be an existing `.py` file. Directories are permitted for
        pytest targets.

    Returns
    -------
    Path
        The resolved absolute path.

    Raises
    ------
    ToolError
        If the path escapes `tests/`, or `must_be_file` is set and it is not a `.py` file.
    """
    raw = Path(user_path)
    base = (
        CFG.repo if raw.is_absolute() or raw.parts[:1] == ("tests",) else CFG.tests_dir
    )
    candidate = (base / raw).resolve()

    if not candidate.is_relative_to(CFG.tests_dir):
        raise ToolError(
            f"Refused: {user_path!r} resolves outside {CFG.tests_dir}. "
            f"Only paths under tests/ may be run. Call list_targets to see what is available."
        )
    if must_be_file and not (candidate.is_file() and candidate.suffix == ".py"):
        raise ToolError(
            f"Refused: {user_path!r} is not an existing .py file. Call list_targets to see runnable scripts."
        )
    return candidate


def check_argv_tokens(tokens: Sequence[str]) -> list[str]:
    """
    Validate caller-supplied argv tokens.

    Tokens are passed to `create_subprocess_exec` as a vector, never through a shell, so
    quoting and injection are structurally impossible. This only rejects embedded NULs,
    which would truncate an argument at the syscall boundary.

    Parameters
    ----------
    tokens : Sequence[str]
        Raw argument tokens.

    Returns
    -------
    list[str]
        The validated tokens.

    Raises
    ------
    ToolError
        If a token is not a string or contains a NUL byte.
    """
    checked: list[str] = []
    for token in tokens:
        if not isinstance(token, str):
            raise ToolError(f"Refused: argument {token!r} is not a string.")
        if "\x00" in token:
            raise ToolError("Refused: arguments may not contain NUL bytes.")
        checked.append(token)
    return checked


RunState = Literal[
    "running", "passed", "failed", "cancelled", "aborted_memory", "orphaned"
]


# TODO Move to engine/model_executor.py
@attrs.define
class HostRun:
    """In-process record of one child process. Wire-facing shape is `RunStatus`."""

    run_id: str
    argv: list[str]
    run_dir: Path
    log_path: Path
    started_at: datetime
    status: RunState = "running"
    returncode: int | None = None
    duration_s: float | None = None
    peak_swap_gb: float = 0.0
    cancel_requested: bool = False
    process: asyncio.subprocess.Process | None = None
    waiter: asyncio.Task[None] | None = None


_RUNS: dict[str, HostRun] = {}
_ACTIVE: HostRun | None = None
_START_LOCK = asyncio.Lock()


def _tail(path: Path, lines: int) -> tuple[list[str], int]:
    """
    Read the last `lines` lines of a file without reading the whole thing.

    Seeks to a fixed window from the end, so cost is independent of log size.

    Parameters
    ----------
    path : Path
        File to read.
    lines : int
        Maximum number of lines to return.

    Returns
    -------
    tuple[list[str], int]
        The trailing lines (each truncated to `TAIL_LINE_CHARS`) and the file size in bytes.
    """
    try:
        size = path.stat().st_size
    except OSError:
        return [], 0

    with path.open("rb") as handle:
        start = max(0, size - TAIL_WINDOW_BYTES)
        handle.seek(start)
        chunk = handle.read()

    text = chunk.decode("utf-8", errors="replace")
    split = text.splitlines()
    if start > 0 and split:
        split = split[1:]  # drop the partial line the window started mid-way through
    return [line[:TAIL_LINE_CHARS] for line in split[-lines:]], size


def _write_meta(run: HostRun) -> None:
    """Persist run metadata so a restarted server can recognise orphans."""
    meta = {
        "run_id": run.run_id,
        "argv": run.argv,
        "started_at": run.started_at.isoformat(),
        "status": run.status,
        "returncode": run.returncode,
        "duration_s": run.duration_s,
        "peak_swap_gb": run.peak_swap_gb,
    }
    (run.run_dir / "meta.json").write_text(json.dumps(meta, indent=2))


def _append_log(run: HostRun, line: str) -> None:
    """Append one `##HOSTRUN` control line to a run's log, ignoring I/O errors."""
    try:
        with run.log_path.open("a") as handle:
            handle.write(f"{LOG_PREFIX} {line}\n")
    except OSError:
        logger.warning("could not append to %s", run.log_path)


# Run lifecycle


def _expand_placeholders(tokens: Sequence[str], run_dir: Path) -> list[str]:
    """
    Substitute host-side placeholders in argument tokens.

    `${model}` becomes the configured default model id, so callers need not know it.
    `${run_dir}` becomes this run's directory, which is the only way an artifact survives:
    `mlx_single_router_layer.py` defaults `--output` to a temporary directory that is
    deleted when it exits.

    Parameters
    ----------
    tokens : Sequence[str]
        Raw argument tokens.
    run_dir : Path
        The run's directory.

    Returns
    -------
    list[str]
        Tokens with placeholders expanded.
    """
    return [
        t.replace("${model}", CFG.default_model).replace("${run_dir}", str(run_dir))
        for t in tokens
    ]


async def _sample_swap(run: HostRun) -> None:
    """
    Watch swap usage for the lifetime of a run and abort it on a sustained breach.

    A single healthy run on this machine swaps a few GB, so an instantaneous spike must
    not kill it. The abort requires `SWAP_BREACH_SAMPLES` consecutive samples above the
    hard ceiling — roughly 30 seconds — before terminating the process group.

    Parameters
    ----------
    run : HostRun
        The run to watch.
    """
    breaches = 0
    elapsed = 0.0
    since_log = 0.0

    while run.process is not None and run.process.returncode is None:
        await asyncio.sleep(SWAP_SAMPLE_SECONDS)
        elapsed += SWAP_SAMPLE_SECONDS
        since_log += SWAP_SAMPLE_SECONDS

        used = swap_used_gb()
        if used is None:
            continue
        run.peak_swap_gb = max(run.peak_swap_gb, used)

        if since_log >= 60.0:
            _append_log(
                run,
                f"mem t={int(elapsed)}s swap_used={used:.2f}GB peak={run.peak_swap_gb:.2f}GB",
            )
            since_log = 0.0

        breaches = breaches + 1 if used > CFG.abort_swap_gb else 0
        if breaches >= SWAP_BREACH_SAMPLES:
            _append_log(
                run,
                f"abort reason=swap swap_used={used:.2f}GB ceiling={CFG.abort_swap_gb:.2f}GB "
                f"sustained={int(SWAP_BREACH_SAMPLES * SWAP_SAMPLE_SECONDS)}s",
            )
            run.status = "aborted_memory"
            await _terminate(run)
            return


async def _terminate(run: HostRun) -> None:
    """
    Kill a run's whole process group, escalating to SIGKILL after a grace period.

    Group kill matters: loading a 35B model spawns worker threads and can wedge inside a
    native call, where signalling only the parent leaves children holding tens of GB.

    Parameters
    ----------
    run : HostRun
        The run to terminate.
    """
    process = run.process
    if process is None or process.returncode is not None:
        return

    try:
        pgid = os.getpgid(process.pid)
    except ProcessLookupError:
        return

    try:
        os.killpg(pgid, signal.SIGTERM)
        await asyncio.wait_for(process.wait(), timeout=CANCEL_GRACE_SECONDS)
    except asyncio.TimeoutError:
        with suppress(ProcessLookupError):
            os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        return


async def _await_completion(run: HostRun, sampler: asyncio.Task[None]) -> None:
    """
    Wait for a child to exit, then finalise its status, log sentinel, and metadata.

    Parameters
    ----------
    run : HostRun
        The run being awaited.
    sampler : asyncio.Task[None]
        The swap watchdog task, cancelled once the child exits.
    """
    global _ACTIVE

    assert run.process is not None
    returncode = await run.process.wait()
    sampler.cancel()

    run.returncode = returncode
    run.duration_s = (datetime.now().astimezone() - run.started_at).total_seconds()
    if run.status not in ("aborted_memory",):
        if run.cancel_requested:
            run.status = "cancelled"
        else:
            run.status = "passed" if returncode == 0 else "failed"

    _append_log(
        run,
        f"end returncode={returncode} duration_s={run.duration_s:.1f} "
        f"peak_swap={run.peak_swap_gb:.2f}GB status={run.status}",
    )
    _write_meta(run)
    if _ACTIVE is run:
        _ACTIVE = None


async def _start(argv: list[str], label: str) -> HostRun:
    """
    Spawn a child process as a new job.

    Only one run executes at a time. A single run already consumes essentially the whole
    swap budget on this machine, so a second concurrent load has nowhere to go.

    Parameters
    ----------
    argv : list[str]
        Fully-resolved argument vector. Never passed through a shell.
    label : str
        Short slug used in the run id.

    Returns
    -------
    HostRun
        The started run.

    Raises
    ------
    ToolError
        If another run is active, or swap is already above the soft budget.
    """
    global _ACTIVE

    async with _START_LOCK:
        if _ACTIVE is not None and _ACTIVE.status == "running":
            raise ToolError(
                f"Refused: run {_ACTIVE.run_id} is still active (log: {_ACTIVE.log_path}). "
                f"Only one run executes at a time on this machine. "
                f"Wait for it with poll_run, or stop it with cancel_run."
            )

        # Only refuse when the machine is already past the runaway ceiling. An absolute
        # check against the soft threshold is useless here: normal operation swaps tens of
        # GB, so any budget small enough to detect "something else is hogging memory"
        # would also refuse every legitimate run.
        used = swap_used_gb()
        if used is not None and used >= CFG.abort_swap_gb:
            raise ToolError(
                f"Refused: swap is already at {used:.2f}GB, at or above the "
                f"{CFG.abort_swap_gb:.2f}GB runaway ceiling. Something on the host is "
                f"consuming memory; free it before starting a run."
            )
        if used is not None and used > CFG.max_swap_gb:
            logger.warning(
                "starting %s with swap already at %.2fGB (soft threshold %.2fGB)",
                label,
                used,
                CFG.max_swap_gb,
            )

        run_id = (
            f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(2)}-{label}"
        )
        run_dir = CFG.runs_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        argv = _expand_placeholders(argv, run_dir)
        log_path = run_dir / "run.log"
        started_at = datetime.now().astimezone()

        child_env = os.environ | {"PYTHONPATH": str(CFG.repo), "PYTHONUNBUFFERED": "1"}

        # The header is flushed before the child starts so the exact invocation is on
        # record even if the spawn itself fails.
        with log_path.open("a") as handle:
            handle.write(
                f"{LOG_PREFIX} run_id={run_id}\n"
                f"{LOG_PREFIX} started_at={started_at.isoformat()}\n"
                f"{LOG_PREFIX} cwd={CFG.repo}\n"
                f"{LOG_PREFIX} env=PYTHONPATH={CFG.repo} PYTHONUNBUFFERED=1\n"
                f"{LOG_PREFIX} argv={json.dumps(argv)}\n"
                f"{LOG_PREFIX} reproduce={shlex.join(argv)}\n"
                f"{LOG_PREFIX} begin\n"
            )
            handle.flush()

        run = HostRun(
            run_id=run_id,
            argv=argv,
            run_dir=run_dir,
            log_path=log_path,
            started_at=started_at,
        )

        # The raw fd goes straight to the child as both stdout and stderr: the kernel
        # writes output into the file with no pump task in between, which is what makes
        # the log readable live from the sandbox.
        log_handle = log_path.open("ab")
        try:
            run.process = await asyncio.create_subprocess_exec(
                *argv,
                stdout=log_handle,
                stderr=asyncio.subprocess.STDOUT,
                cwd=CFG.repo,
                env=child_env,
                start_new_session=True,
            )
        except OSError as exc:
            log_handle.close()
            run.status = "failed"
            _append_log(
                run,
                f"end returncode=None duration_s=0.0 status=failed spawn_error={exc}",
            )
            _write_meta(run)
            raise ToolError(f"Could not start {argv[0]}: {exc}") from exc
        finally:
            log_handle.close()  # the child holds its own duplicated descriptor

        _RUNS[run_id] = run
        _ACTIVE = run
        _write_meta(run)

        sampler = asyncio.create_task(_sample_swap(run))
        run.waiter = asyncio.create_task(_await_completion(run, sampler))
        logger.info("started %s: %s", run_id, shlex.join(argv))
        return run


def _recover_orphans() -> None:
    """
    Mark runs left `running` by a previous server process as `orphaned`.

    In-process state does not survive a restart and adopting a stale pid is a whole class
    of bugs for no benefit here, so orphans are reported rather than reattached.
    """
    if not CFG.runs_dir.exists():
        return

    for meta_path in sorted(CFG.runs_dir.glob("*/meta.json")):
        try:
            meta = json.loads(meta_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if meta.get("status") != "running":
            continue

        meta["status"] = "orphaned"
        try:
            meta_path.write_text(json.dumps(meta, indent=2))
            with (meta_path.parent / "run.log").open("a") as handle:
                handle.write(
                    f"{LOG_PREFIX} end returncode=None status=orphaned reason=server-restart\n"
                )
        except OSError:
            continue
        logger.warning(
            "marked %s as orphaned (server restarted while it ran)", meta.get("run_id")
        )


_NOTE = (
    "Read log_path directly with your file-reading tool — the repository is mounted at "
    "this same absolute path inside your sandbox, and the log updates live."
)


# TODO move to datamodel/
class RunStatus(BaseModel):
    """Status of one host run. Returned by every run, poll, and cancel call."""

    run_id: str
    status: RunState
    returncode: int | None
    argv: list[str]
    log_path: str
    run_dir: str
    started_at: str
    duration_s: float | None
    log_bytes: int
    tail: list[str]
    swap_used_gb: float | None
    peak_swap_gb: float
    note: str = _NOTE


class ScriptInfo(BaseModel):
    """One runnable script discovered under `tests/`."""

    path: str
    abs_path: str
    size_bytes: int


class TargetsInfo(BaseModel):
    """Everything a caller needs to construct a valid run request."""

    repo: str
    tests_dir: str
    python: str
    default_model: str
    scripts: list[ScriptInfo]
    placeholders: dict[str, str]
    swap_used_gb: float | None
    max_swap_gb: float
    abort_swap_gb: float
    runs_dir_bytes: int
    active_run: RunStatus | None
    recent_runs: list[RunStatus]


def _status(run: HostRun, tail_lines: int = 40) -> RunStatus:
    """Project a `HostRun` into its wire representation."""
    tail, size = _tail(run.log_path, tail_lines)
    return RunStatus(
        run_id=run.run_id,
        status=run.status,
        returncode=run.returncode,
        argv=run.argv,
        log_path=str(run.log_path),
        run_dir=str(run.run_dir),
        started_at=run.started_at.isoformat(),
        duration_s=run.duration_s,
        log_bytes=size,
        tail=tail,
        swap_used_gb=swap_used_gb(),
        peak_swap_gb=run.peak_swap_gb,
    )


async def _wait_for(run: HostRun, wait_s: int) -> None:
    """
    Block until a run finishes or `wait_s` elapses, whichever comes first.

    The waiter task is shielded so that a caller giving up never cancels the finaliser
    that records the run's exit status.

    Parameters
    ----------
    run : HostRun
        The run to wait on.
    wait_s : int
        Maximum seconds to wait.
    """
    if wait_s <= 0 or run.waiter is None or run.waiter.done():
        return
    try:
        await asyncio.wait_for(asyncio.shield(run.waiter), timeout=wait_s)
    except asyncio.TimeoutError:
        return


def _lookup(run_id: str) -> HostRun:
    """Return a tracked run by id, or raise a `ToolError` naming the known ids."""
    run = _RUNS.get(run_id)
    if run is None:
        known = ", ".join(sorted(_RUNS)[-RECENT_RUNS:]) or "none"
        raise ToolError(
            f"Unknown run_id {run_id!r}. Known runs this server session: {known}."
        )
    return run


# MCP server

mcp = FastMCP(
    name="hostrun",
    instructions=(
        "Runs this project's tests on the macOS host, outside your sandbox, where MLX and "
        "Metal are available. The repository is bind-mounted at the same absolute path on "
        "both sides, so when a tool returns a `log_path`, read that file directly with your "
        "own file-reading tool instead of polling for output — it updates live while the run "
        "proceeds, and holds the complete output rather than the truncated tail. Runs are "
        "jobs: a run tool returns as soon as it has a run id, and you poll for the result. "
        "Only one run executes at a time, because a single model load consumes nearly all of "
        "the host's memory budget. Only paths under `tests/` can be executed."
    ),
    auth=StaticTokenVerifier(
        tokens={TOKEN: {"client_id": "preempt-sandbox", "scopes": []}}
    ),
)


@mcp.custom_route("/healthz", methods=["GET"])
async def healthz(request: Request) -> JSONResponse:
    """Unauthenticated liveness probe. Deliberately returns no secrets."""
    active = (
        _ACTIVE.run_id if _ACTIVE is not None and _ACTIVE.status == "running" else None
    )
    return JSONResponse(
        {"ok": True, "active_run": active, "swap_used_gb": swap_used_gb()}
    )


@mcp.tool(annotations={"readOnlyHint": True})
async def list_targets() -> TargetsInfo:
    """
    List every script runnable on the host, plus current memory headroom and run history.

    Discovery walks the filesystem under `tests/`, so a script added there is immediately
    runnable with no server change. Returns paths, the default model id, the placeholders
    accepted in run arguments, live swap usage, and the most recent runs. Runs nothing.
    """
    # Only advertise what would actually be accepted: `rglob` follows symlinks, so a link
    # inside tests/ pointing elsewhere would otherwise be listed here and then refused by
    # `resolve_under_tests` at run time.
    scripts = [
        ScriptInfo(
            path=str(path.relative_to(CFG.repo)),
            abs_path=str(path),
            size_bytes=path.stat().st_size,
        )
        for path in sorted(CFG.tests_dir.rglob("*.py"))
        if "__pycache__" not in path.parts
        and path.resolve().is_relative_to(CFG.tests_dir)
    ]

    runs_bytes = (
        sum(f.stat().st_size for f in CFG.runs_dir.rglob("*") if f.is_file())
        if CFG.runs_dir.exists()
        else 0
    )
    recent = [_status(r, tail_lines=3) for r in list(_RUNS.values())[-RECENT_RUNS:]]

    return TargetsInfo(
        repo=str(CFG.repo),
        tests_dir=str(CFG.tests_dir),
        python=str(CFG.python),
        default_model=CFG.default_model,
        scripts=scripts,
        placeholders={
            "${model}": CFG.default_model,
            "${run_dir}": "this run's output directory; write artifacts here so they survive",
        },
        swap_used_gb=swap_used_gb(),
        max_swap_gb=CFG.max_swap_gb,
        abort_swap_gb=CFG.abort_swap_gb,
        runs_dir_bytes=runs_bytes,
        active_run=_status(_ACTIVE) if _ACTIVE is not None else None,
        recent_runs=recent,
    )


WaitSeconds = Annotated[
    int,
    Field(
        ge=0,
        le=120,
        description=(
            "Seconds to wait inline before returning. The call returns as soon as the run "
            "finishes, so a generous value costs nothing on a fast run. Keep at or below 90 "
            "to avoid the client backgrounding the call."
        ),
    ),
]


@mcp.tool
async def run_script(
    script: Annotated[
        str,
        Field(
            description=(
                "Path to a .py file under tests/, either repo-relative "
                "('tests/integration/mlx_layer_search.py') or tests-relative "
                "('integration/mlx_layer_search.py')."
            )
        ),
    ],
    args: Annotated[
        list[str] | None,
        Field(
            description=(
                "Arguments passed to the script as an argv vector, e.g. "
                "['--model', '${model}', '--output', '${run_dir}/out.parquet']. Never a shell "
                "string. '${model}' expands to the host's default model id and '${run_dir}' to "
                "this run's output directory."
            )
        ),
    ] = None,
    wait_s: WaitSeconds = 30,
    ctx: Context | None = None,
) -> RunStatus:
    """
    Run one Python script from tests/ on the macOS host and return its status.

    Starts the script under the project's own virtualenv with PYTHONPATH set, then waits
    up to `wait_s`. Returns a run id, the absolute `log_path`, and a short tail. If the run
    is still going, read `log_path` directly and call `poll_run` for the exit code.

    Use `run_pytest` to run a test suite; this executes a single script directly and does
    not collect or report tests. Only paths under tests/ are permitted.
    """
    target = resolve_under_tests(script, must_be_file=True)
    tokens = check_argv_tokens(args or [])
    argv = [str(CFG.python), "-u", str(target), *tokens]

    run = await _start(argv, label=target.stem)
    if ctx is not None:
        await ctx.report_progress(progress=0, total=1, message=f"started {run.run_id}")
    await _wait_for(run, wait_s)
    return _status(run)


@mcp.tool
async def run_pytest(
    paths: Annotated[
        list[str] | None,
        Field(
            description=(
                "Test paths under tests/, each a file, directory, or node id "
                "('tests/unit/test_x.py::test_y'). Defaults to ['tests']."
            )
        ),
    ] = None,
    k: Annotated[
        str | None, Field(description="Value for pytest's -k expression filter.")
    ] = None,
    markers: Annotated[
        str | None, Field(description="Value for pytest's -m marker filter.")
    ] = None,
    extra_args: Annotated[
        list[str] | None,
        Field(
            description="Further pytest flags as an argv vector, e.g. ['--collect-only', '-q']."
        ),
    ] = None,
    wait_s: WaitSeconds = 30,
    ctx: Context | None = None,
) -> RunStatus:
    """
    Run pytest on the macOS host against paths under tests/ and return its status.

    Runs under the project's own virtualenv with PYTHONPATH set, so MLX-dependent tests
    work here even though they cannot run in your sandbox. Returns a run id, the absolute
    `log_path`, and a short tail; read the log directly for full output.

    Use `run_script` to execute a single script directly. Note that pytest exit code 5
    means no tests were collected, which is not a crash.
    """
    targets = [
        str(resolve_under_tests(p.split("::", 1)[0], must_be_file=False))
        for p in (paths or ["tests"])
    ]
    # Re-attach any node-id suffix that was stripped for the allowlist check.
    resolved = [
        f"{target}::{original.split('::', 1)[1]}" if "::" in original else target
        for target, original in zip(targets, paths or ["tests"])
    ]

    argv = [str(CFG.python), "-u", "-m", "pytest", "-p", "no:cacheprovider", *resolved]
    if k is not None:
        argv += ["-k", *check_argv_tokens([k])]
    if markers is not None:
        argv += ["-m", *check_argv_tokens([markers])]
    argv += ["-q", *check_argv_tokens(extra_args or [])]

    run = await _start(argv, label="pytest")
    if ctx is not None:
        await ctx.report_progress(progress=0, total=1, message=f"started {run.run_id}")
    await _wait_for(run, wait_s)
    return _status(run)


@mcp.tool(annotations={"readOnlyHint": True})
async def poll_run(
    run_id: Annotated[
        str, Field(description="Run id returned by run_script or run_pytest.")
    ],
    wait_s: WaitSeconds = 0,
    tail_lines: Annotated[
        int,
        Field(
            ge=1,
            le=200,
            description="Lines of log tail to include. The full log is at log_path.",
        ),
    ] = 40,
) -> RunStatus:
    """
    Return the current status of a run, optionally waiting for it to finish first.

    Defaults to returning immediately. Pass `wait_s` to block until the run ends or the
    time expires. Status is one of running, passed, failed, cancelled, aborted_memory, or
    orphaned. Only reports; use `cancel_run` to stop a run.
    """
    run = _lookup(run_id)
    await _wait_for(run, wait_s)
    return _status(run, tail_lines=tail_lines)


@mcp.tool(annotations={"idempotentHint": True})
async def cancel_run(
    run_id: Annotated[
        str, Field(description="Run id returned by run_script or run_pytest.")
    ],
) -> RunStatus:
    """
    Stop a running job by terminating its whole process group.

    Sends SIGTERM, escalating to SIGKILL after a grace period, so a model load wedged in
    native code cannot leave orphaned processes holding memory. Returns the final status;
    calling this on an already-finished run is a no-op that returns its status unchanged.
    """
    run = _lookup(run_id)
    if run.status != "running":
        return _status(run)

    run.cancel_requested = True
    _append_log(run, "cancel requested=client")
    await _terminate(run)
    await _wait_for(run, wait_s=int(CANCEL_GRACE_SECONDS) + 5)
    return _status(run)


def main() -> None:
    """Configure logging, recover orphaned runs, and serve until interrupted."""
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    CFG.runs_dir.mkdir(parents=True, exist_ok=True)
    handlers.append(logging.FileHandler(CFG.runs_dir / "server.log"))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        handlers=handlers,
    )

    _recover_orphans()

    # Every host:port form the client might present must be listed, or the Host guard
    # answers 421. The sandbox reaches this server as host.docker.internal, may fall back
    # to the gateway's IPv4 literal, and uses 127.0.0.1 when tunnelled through socat.
    allowed_hosts = [
        f"{name}:{CFG.port}"
        for name in ("127.0.0.1", "localhost", "host.docker.internal", "169.254.1.1")
    ]

    app = mcp.http_app(
        host_origin_protection=True,
        allowed_hosts=allowed_hosts,
        allowed_origins=[],  # no browser origin is ever legitimate for this server
        stateless_http=True,  # survives server restarts without stale-session errors
    )

    logger.info("repo=%s python=%s", CFG.repo, CFG.python)
    logger.info("default_model=%s", CFG.default_model)
    logger.info(
        "swap budget: soft=%.1fGB abort=%.1fGB (current=%s)",
        CFG.max_swap_gb,
        CFG.abort_swap_gb,
        swap_used_gb(),
    )
    logger.info("token file: %s", CFG.token_path)
    logger.info("listening on http://%s:%d/mcp", CFG.host, CFG.port)

    import uvicorn

    uvicorn.run(app, host=CFG.host, port=CFG.port, log_level="warning")


if __name__ == "__main__":
    main()
