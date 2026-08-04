# v1 Inference Pipeline Implementation Plan (Phases 1–2)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move the inference machinery inlined in `tests/integration/mlx_all_router_layers.py` into the source tree behind backend-agnostic contracts, so a small script loads one TOML and runs an instrumented pipeline; then build the packed expert store that streaming (phases 3–4) will read from.

**Architecture:** Capability-scoped `typing.Protocol` contracts in `core/`, composed by a platform-agnostic engine (`engine/generation.py` loop + `engine/pipeline.py` facade), wired only at composition roots (`main.py`, integration scripts). New `storage/` package owns the backend-neutral packed expert container; per-source-format extractors live in their backend. Spec: `.claude/docs/superpowers/specs/2026-08-01-v1-inference-pipeline-design.md`.

**Tech Stack:** Python 3.12, attrs, pydantic v2, pyarrow, pytest + pytest-asyncio, MLX/mlx_lm (host only), numpy.

## Global Constraints

Copied from CLAUDE.md / the spec — every task implicitly includes these:

- **EXACT INFERENCE:** predictions/instrumentation never change which tokens are produced. Never `mx.eval` in the capture path.
- **Layering:** `core/` imports nothing internal; `engine/`, `storage/`, `predictors/` import only `core/` + `datamodel/`; `backends/` is imported only by composition roots. No `__init__.py` re-exports — always import by full path.
- **attrs (`@attrs.define`) for internal records, pydantic `BaseModel` for anything crossing a process/file boundary.** Every attrs field uses `attrs.field(...)`; every pydantic field uses `pydantic.Field(...)`; pydantic models use `ConfigDict(extra="forbid", frozen=True)`.
- **Typing:** full annotations; builtin generics (`list[int]`); `collections.abc` for `Sequence`/`Callable`/etc. Import order: `typing` then `collections.abc` first (after `__future__`), stdlib, third-party, `preempt` last, groups separated by blank lines.
- **Docstrings:** NumPy style, single backticks for inline code.
- **Sandbox pytest:** never touch the workspace `.venv`. Use the throwaway env (setup below) with absolute paths.
- **MLX code cannot run in the sandbox.** Anything under `backends/mlx_metal/` and `tests/integration/` is verified on the host via the hostrun MCP tools (`mcp__hostrun__run_script` / `run_pytest`, then poll and read `log_path` directly). One host run at a time; write host artifacts to `${run_dir}`.
- **Commits:** conventional-commit style, message ends with `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`. Work on branch `v1-inference-pipeline`.

## One-time sandbox test-environment setup

Run once before Task 1 (skip if `/home/agent/.venvs/preempt-linux` exists):

```bash
uv venv /home/agent/.venvs/preempt-linux --python 3.12
uv pip install -r pyproject.toml --python /home/agent/.venvs/preempt-linux/bin/python
```

All `pytest` invocations in this plan mean:

```bash
PYTHONPATH=/Users/davidstoneman/venvs/preempt \
  /home/agent/.venvs/preempt-linux/bin/pytest <args>
```

Note: `mlx` cannot install on Linux; if the bulk install fails on it, install the needed subset instead: `attrs pydantic pyarrow pytest pytest-asyncio numpy`.

---

# Phase 1 — Pipeline extraction

### Task 1: pytest bootstrap + pure-layer tests (layer resolution, target config)

**Files:**
- Modify: `pyproject.toml` (append `[tool.pytest.ini_options]` only — do NOT touch `[project].dependencies`)
- Create: `tests/unit/test_layer_resolution.py`
- Create: `tests/unit/test_target_layers.py`

**Interfaces:**
- Consumes: `preempt.engine.layer_resolution` (`LayerCandidate`, `layer_is_match`, `match_target_layers`, `resolve_target_layers`, `ensure_no_target_layer_overlap`), `preempt.config.target_layers` (`TargetLayerConfig`, `TargetLayerSpec`, `TargetLayerSearchParams`).
- Produces: working `pytest` config all later tasks rely on (`asyncio_mode = "auto"`, `slow`/`integration` markers).

- [x] **Step 1: Add pytest config to `pyproject.toml`**

```toml
[tool.pytest.ini_options]
testpaths = ["tests/unit"]
asyncio_mode = "auto"
markers = [
    "slow: long-running tests, excluded from the default fast loop",
    "integration: tests needing real weights or the macOS host",
]
```

- [x] **Step 2: Write `tests/unit/test_layer_resolution.py`**

```python
import pytest

from preempt.config.target_layers import TargetLayerConfig, TargetLayerSearchParams
from preempt.engine.layer_resolution import (
    LayerCandidate,
    ensure_no_target_layer_overlap,
    layer_is_match,
    match_target_layers,
    resolve_target_layers,
)


def make_candidates() -> tuple[LayerCandidate, ...]:
    return tuple(
        LayerCandidate(
            layer_path=f"model.layers.{i}.mlp",
            layer_class="SparseMoeBlock" if i % 2 == 0 else "DenseMlp",
            layer_idx=i,
        )
        for i in range(6)
    )


def test_layer_is_match_by_class() -> None:
    params = TargetLayerSearchParams(layer_class="SparseMoeBlock")
    candidates = make_candidates()
    assert layer_is_match(candidates[0], params)
    assert not layer_is_match(candidates[1], params)


def test_layer_is_match_conjunction_of_criteria() -> None:
    params = TargetLayerSearchParams(
        layer_class="SparseMoeBlock", layer_path_glob="model.layers.*.mlp", layer_idx=2
    )
    candidates = make_candidates()
    assert layer_is_match(candidates[2], params)
    assert not layer_is_match(candidates[0], params)  # class+glob match, idx doesn't


def test_match_target_layers_count_mismatch_raises() -> None:
    params = TargetLayerSearchParams(layer_class="SparseMoeBlock", count=2)
    with pytest.raises(ValueError, match="unexpected number"):
        match_target_layers(make_candidates(), params)


def test_match_target_layers_no_match_raises() -> None:
    params = TargetLayerSearchParams(layer_class="DoesNotExist")
    with pytest.raises(ValueError, match="Could not find"):
        match_target_layers(make_candidates(), params)


def test_resolve_target_layers_groups_by_spec_name() -> None:
    config = TargetLayerConfig.model_validate(
        {
            "target_layers": [
                {"name": "moe", "search_params": {"layer_class": "SparseMoeBlock", "count": 3}},
                {"name": "dense", "search_params": {"layer_class": "DenseMlp", "count": 3}},
            ]
        }
    )
    resolved = resolve_target_layers(make_candidates(), config)
    assert set(resolved) == {"moe", "dense"}
    assert [c.layer_idx for c in resolved["moe"]] == [0, 2, 4]


def test_ensure_no_target_layer_overlap_raises_on_shared_layer() -> None:
    config = TargetLayerConfig.model_validate(
        {
            "target_layers": [
                {"name": "a", "search_params": {"layer_class": "SparseMoeBlock"}},
                {"name": "b", "search_params": {"layer_idx": 0}},
            ]
        }
    )
    resolved = resolve_target_layers(make_candidates(), config)
    with pytest.raises(ValueError, match="matches both"):
        ensure_no_target_layer_overlap(resolved)
```

- [x] **Step 3: Write `tests/unit/test_target_layers.py`**

```python
import pytest
from pydantic import ValidationError

from preempt.config.target_layers import TargetLayerConfig, TargetLayerSearchParams


def test_search_params_require_at_least_one_criterion() -> None:
    with pytest.raises(ValidationError, match="At least one"):
        TargetLayerSearchParams()


def test_search_params_reject_unknown_keys() -> None:
    with pytest.raises(ValidationError):
        TargetLayerSearchParams(layer_class="X", not_a_field=1)  # type: ignore[call-arg]


def test_config_rejects_duplicate_target_names() -> None:
    spec = {"name": "same", "search_params": {"layer_idx": 0}}
    with pytest.raises(ValidationError, match="unique"):
        TargetLayerConfig.model_validate({"target_layers": [spec, spec]})


def test_config_round_trips_from_toml_shaped_dict() -> None:
    config = TargetLayerConfig.model_validate(
        {
            "version": 1,
            "target_layers": [
                {"name": "r", "search_params": {"layer_class": "Blk", "count": 40}}
            ],
        }
    )
    assert config.target_layers[0].search_params.count == 40
```

- [x] **Step 4: Run and verify all pass** — `pytest tests/unit -v`. Expected: all PASS (these test existing code; any failure is a real finding — stop and report, don't "fix" the test).

- [x] **Step 5: Commit** — `test: bootstrap pytest and cover layer resolution + target config`

### Task 2: pure-layer tests for Arrow schema derivation and sink batching

**Files:**
- Create: `tests/unit/test_expert_routing_arrow.py`
- Create: `tests/unit/test_sinks.py`

**Interfaces:**
- Consumes: `ExpertRoutingEvent`, `EventMetadata`, `LayerIdentifiers` (`preempt.datamodel.tracing.expert_routing`); `TraceRunContext`, `TraceStepContext` (`preempt.datamodel.tracing.context`); `ParquetEventSink` (`preempt.core.sinks`).
- Produces: nothing new — regression floor under the refactors that follow.

- [x] **Step 1: Write `tests/unit/test_expert_routing_arrow.py`**

```python
from datetime import UTC, datetime

import pyarrow as pa
import pytest

from preempt.datamodel.tracing.context import TraceRunContext, TraceStepContext
from preempt.datamodel.tracing.expert_routing import (
    EXPERT_ROUTING_EVENT_TYPE,
    EXPERT_ROUTING_SCHEMA_VERSION,
    EventMetadata,
    ExpertRoutingEvent,
    LayerIdentifiers,
)


def make_event(*, gate_logits: tuple[float, ...] | None = None) -> ExpertRoutingEvent:
    return ExpertRoutingEvent(
        run_context=TraceRunContext(
            run_id="r-1", model_id="m", model_architecture="arch"
        ),
        step_context=TraceStepContext(sequence_id=0, token_idx=3, token_id=7),
        event_metadata=EventMetadata(event_idx=0, timestamp=datetime.now(UTC)),
        layer_identifiers=LayerIdentifiers(
            layer_path="model.layers.0.mlp", layer_class="Blk", layer_idx=0
        ),
        expert_ids=(4, 9),
        expert_weights=(0.7, 0.3),
        gate_logits=gate_logits,
    )


def test_schema_and_record_share_flat_field_names() -> None:
    schema = ExpertRoutingEvent.arrow_schema()
    record = make_event().as_arrow_record()
    assert set(record) == set(schema.names)


def test_schema_carries_event_type_and_version_metadata() -> None:
    metadata = ExpertRoutingEvent.arrow_schema().metadata
    assert metadata[b"preempt.event_type"] == EXPERT_ROUTING_EVENT_TYPE.encode()
    assert metadata[b"preempt.schema_version"] == str(EXPERT_ROUTING_SCHEMA_VERSION).encode()


def test_record_batch_round_trips_through_arrow() -> None:
    schema = ExpertRoutingEvent.arrow_schema()
    rows = [make_event().as_arrow_record(), make_event(gate_logits=(0.1, 0.9)).as_arrow_record()]
    batch = pa.RecordBatch.from_pylist(rows, schema)
    assert batch.num_rows == 2
    assert batch.to_pylist()[0]["expert_ids"] == [4, 9]
    assert batch.to_pylist()[0]["gate_logits"] is None


def test_mismatched_ids_and_weights_rejected() -> None:
    event = make_event()
    with pytest.raises(ValueError, match="same number"):
        ExpertRoutingEvent(
            run_context=event.run_context,
            step_context=event.step_context,
            event_metadata=event.event_metadata,
            layer_identifiers=event.layer_identifiers,
            expert_ids=(1, 2, 3),
            expert_weights=(0.5, 0.5),
            gate_logits=None,
        )
```

- [x] **Step 2: Write `tests/unit/test_sinks.py`**

```python
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from preempt.core.sinks import ParquetEventSink

SCHEMA = pa.schema([pa.field("a", pa.int64(), nullable=False)])


async def test_batching_creates_one_row_group_per_full_batch(tmp_path: Path) -> None:
    path = tmp_path / "out.parquet"
    async with ParquetEventSink(path, schema=SCHEMA, batch_size=2) as sink:
        for i in range(5):
            await sink.write({"a": i})
    parquet_file = pq.ParquetFile(path)
    assert parquet_file.metadata.num_rows == 5
    # 2 full batches + 1 partial written at aclose()
    assert parquet_file.metadata.num_row_groups == 3
    assert sink.records_written == 5


async def test_refuses_overwrite_by_default(tmp_path: Path) -> None:
    path = tmp_path / "out.parquet"
    path.touch()
    sink = ParquetEventSink(path, schema=SCHEMA, batch_size=1)
    with pytest.raises(FileExistsError):
        await sink.write({"a": 1})


async def test_write_after_close_raises(tmp_path: Path) -> None:
    sink = ParquetEventSink(tmp_path / "out.parquet", schema=SCHEMA)
    await sink.aclose()
    with pytest.raises(RuntimeError, match="closed"):
        await sink.write({"a": 1})


def test_rejects_nonpositive_batch_size(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        ParquetEventSink(tmp_path / "x.parquet", schema=SCHEMA, batch_size=0)
```

- [x] **Step 3: Run and verify all pass** — `pytest tests/unit -v`. Same rule as Task 1: failures against existing code are findings, not test bugs.

- [ ] **Step 4: Commit** — `test: cover arrow schema derivation and parquet sink batching`

### Task 3: `ExpertKey` (core identity)

**Files:**
- Create: `preempt/core/identity.py`
- Test: `tests/unit/test_identity.py`

**Interfaces:**
- Produces: `ExpertKey(model_fingerprint: str, layer_idx: int, expert_idx: int, variant: str = "all")` — frozen, kw-only, hashable. Every later task keys caches/stores/residency with it.

- [x] **Step 1: Write the failing test `tests/unit/test_identity.py`**

```python
import pytest

from preempt.core.identity import ExpertKey


def test_key_is_hashable_and_value_equal() -> None:
    a = ExpertKey(model_fingerprint="fp", layer_idx=3, expert_idx=17)
    b = ExpertKey(model_fingerprint="fp", layer_idx=3, expert_idx=17)
    assert a == b
    assert len({a, b}) == 1
    assert a.variant == "all"


def test_distinct_fingerprints_are_distinct_keys() -> None:
    a = ExpertKey(model_fingerprint="fp1", layer_idx=0, expert_idx=0)
    b = ExpertKey(model_fingerprint="fp2", layer_idx=0, expert_idx=0)
    assert a != b


@pytest.mark.parametrize(
    "kwargs",
    [
        {"model_fingerprint": "", "layer_idx": 0, "expert_idx": 0},
        {"model_fingerprint": "fp", "layer_idx": -1, "expert_idx": 0},
        {"model_fingerprint": "fp", "layer_idx": 0, "expert_idx": -1},
        {"model_fingerprint": "fp", "layer_idx": 0, "expert_idx": 0, "variant": ""},
    ],
)
def test_invalid_fields_rejected(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        ExpertKey(**kwargs)
```

- [x] **Step 2: Run to verify it fails** — `pytest tests/unit/test_identity.py -v`. Expected: `ModuleNotFoundError: preempt.core.identity`.

- [x] **Step 3: Implement `preempt/core/identity.py`**

```python
from __future__ import annotations

import attrs
from attrs import field, validators


@attrs.define(kw_only=True, frozen=True)
class ExpertKey:
    """Fully qualified identity of one routed expert's weights.

    The cache, store, and residency key everywhere in the engine; bare
    integer expert ids never cross a module boundary. `variant` names the
    weight variant a key refers to — `"all"` is the full fused expert blob.
    """

    model_fingerprint: str = field(validator=validators.min_len(1))
    layer_idx: int = field(validator=validators.ge(0))
    expert_idx: int = field(validator=validators.ge(0))
    variant: str = field(default="all", validator=validators.min_len(1))
```

- [x] **Step 4: Run to verify pass** — `pytest tests/unit/test_identity.py -v`
- [ ] **Step 5: Commit** — `feat(core): add ExpertKey fully-qualified expert identity` (deferred: commits are handled by the user)

### Task 4: core protocols + `AllResidentProvider`

**Files:**
- Create: `preempt/core/protocols/__init__.py` (empty), `preempt/core/protocols/runner.py`, `preempt/core/protocols/provider.py`, `preempt/core/protocols/store.py`, `preempt/core/protocols/residency.py`, `preempt/core/protocols/predictor.py`
- Create: `preempt/engine/scheduler.py` (currently empty placeholder — gets `AllResidentProvider`)
- Test: `tests/unit/test_protocols.py`

**Interfaces:**
- Produces (exact, later tasks depend on these):
  - `TokenCodec`: `encode(text: str) -> list[int]`, `decode(tokens: Sequence[int]) -> str`
  - `ModelRunner`: `prepare() -> None`, `step(tokens: Sequence[int]) -> int`
  - `ExpertProvider`: `acquire(keys: Sequence[ExpertKey]) -> None`
  - `ReadPriority` (`IntEnum`): `DEMAND = 0`, `PREFETCH = 1`
  - `ExpertPayload`: frozen attrs — `key: ExpertKey`, `data: bytes`, `encoding: str`
  - `ExpertStore`: `async read(key: ExpertKey, priority: ReadPriority) -> ExpertPayload`
  - `ExpertResidency`: `install(key, payload) -> None`, `evict(key) -> None`, `is_resident(key) -> bool`, `resident_bytes() -> int`
  - `PredictionRequest` (frozen attrs): `layer_idx: int`, `observed: tuple[ExpertKey, ...]`, `horizon: int`
  - `PredictionResponse` (frozen attrs): `predicted: tuple[ExpertKey, ...]`
  - `PredictionHandle`: `async poll_until(deadline: float) -> PredictionResponse | None`
  - `Predictor`: `submit(request: PredictionRequest) -> PredictionHandle`
  - `AllResidentProvider` (engine/scheduler.py): no-op `ExpertProvider`

All protocols are `@runtime_checkable class X(Protocol)`.

- [x] **Step 1: Write the failing test `tests/unit/test_protocols.py`**

```python
from collections.abc import Sequence

from preempt.core.identity import ExpertKey
from preempt.core.protocols.provider import ExpertProvider
from preempt.core.protocols.residency import ExpertResidency
from preempt.core.protocols.runner import ModelRunner, TokenCodec
from preempt.core.protocols.store import ExpertPayload, ExpertStore, ReadPriority
from preempt.engine.scheduler import AllResidentProvider

KEY = ExpertKey(model_fingerprint="fp", layer_idx=0, expert_idx=0)


class FakeRunner:
    def prepare(self) -> None: ...
    def step(self, tokens: Sequence[int]) -> int:
        return 1


class FakeCodec:
    def encode(self, text: str) -> list[int]:
        return [1]

    def decode(self, tokens: Sequence[int]) -> str:
        return "x"


class FakeResidency:
    def install(self, key: ExpertKey, payload: ExpertPayload) -> None: ...
    def evict(self, key: ExpertKey) -> None: ...
    def is_resident(self, key: ExpertKey) -> bool:
        return True

    def resident_bytes(self) -> int:
        return 0


class FakeStore:
    async def read(self, key: ExpertKey, priority: ReadPriority) -> ExpertPayload:
        return ExpertPayload(key=key, data=b"\x00", encoding="fake")


def test_fakes_satisfy_protocols_structurally() -> None:
    assert isinstance(FakeRunner(), ModelRunner)
    assert isinstance(FakeCodec(), TokenCodec)
    assert isinstance(FakeResidency(), ExpertResidency)
    assert isinstance(FakeStore(), ExpertStore)
    assert isinstance(AllResidentProvider(), ExpertProvider)


def test_demand_orders_before_prefetch() -> None:
    assert ReadPriority.DEMAND < ReadPriority.PREFETCH


def test_all_resident_provider_acquire_is_noop() -> None:
    AllResidentProvider().acquire((KEY,))  # must not raise or block
```

- [x] **Step 2: Run to verify it fails** — expected `ModuleNotFoundError`.

- [x] **Step 3: Implement the five protocol modules**

`preempt/core/protocols/runner.py`:

```python
from __future__ import annotations

from typing import Protocol, runtime_checkable
from collections.abc import Sequence


@runtime_checkable
class TokenCodec(Protocol):
    """Minimal tokenizer surface the engine needs."""

    def encode(self, text: str) -> list[int]: ...

    def decode(self, tokens: Sequence[int]) -> str: ...


@runtime_checkable
class ModelRunner(Protocol):
    """One synchronous forward pass + greedy sample.

    Sync by design: MLX decode is sync; the engine wraps `step` in
    `asyncio.to_thread` so the event loop stays free for I/O.
    """

    def prepare(self) -> None:
        """Reset per-sequence state (e.g. the prompt cache). Call before the
        first `step` of each generation."""
        ...

    def step(self, tokens: Sequence[int]) -> int:
        """Run one forward pass over `tokens`; return the greedy next token id."""
        ...
```

`preempt/core/protocols/provider.py`:

```python
from __future__ import annotations

from typing import Protocol, runtime_checkable
from collections.abc import Sequence

from preempt.core.identity import ExpertKey


@runtime_checkable
class ExpertProvider(Protocol):
    """Sync in-forward hook: block until the given experts are resident.

    Called from the runner thread by instrumented MoE blocks right after
    top-k selection — the correctness path. Implementations must be
    thread-safe with respect to the engine's event loop.
    """

    def acquire(self, keys: Sequence[ExpertKey]) -> None: ...
```

`preempt/core/protocols/store.py`:

```python
from __future__ import annotations

from typing import Protocol, runtime_checkable

from enum import IntEnum

import attrs
from attrs import field, validators

from preempt.core.identity import ExpertKey


class ReadPriority(IntEnum):
    """Lower value = more urgent. Ordering lives in the scheduler's queue;
    stores just read what they are told."""

    DEMAND = 0
    PREFETCH = 1


@attrs.define(kw_only=True, frozen=True)
class ExpertPayload:
    """Opaque expert weight bytes plus the encoding tag needed to decode them."""

    key: ExpertKey = field()
    data: bytes = field()
    encoding: str = field(validator=validators.min_len(1))


@runtime_checkable
class ExpertStore(Protocol):
    async def read(self, key: ExpertKey, priority: ReadPriority) -> ExpertPayload: ...
```

`preempt/core/protocols/residency.py`:

```python
from __future__ import annotations

from typing import Protocol, runtime_checkable

from preempt.core.identity import ExpertKey
from preempt.core.protocols.store import ExpertPayload


@runtime_checkable
class ExpertResidency(Protocol):
    """Backend-owned: where payload bytes become live device tensors."""

    def install(self, key: ExpertKey, payload: ExpertPayload) -> None: ...

    def evict(self, key: ExpertKey) -> None: ...

    def is_resident(self, key: ExpertKey) -> bool: ...

    def resident_bytes(self) -> int: ...
```

`preempt/core/protocols/predictor.py`:

```python
from __future__ import annotations

from typing import Protocol, runtime_checkable

import attrs
from attrs import field, validators

from preempt.core.identity import ExpertKey


@attrs.define(kw_only=True, frozen=True)
class PredictionRequest:
    """Observed routing at `layer_idx`; predict routing `horizon` layers ahead."""

    layer_idx: int = field(validator=validators.ge(0))
    observed: tuple[ExpertKey, ...] = field()
    horizon: int = field(validator=validators.ge(1))


@attrs.define(kw_only=True, frozen=True)
class PredictionResponse:
    predicted: tuple[ExpertKey, ...] = field()


@runtime_checkable
class PredictionHandle(Protocol):
    async def poll_until(self, deadline: float) -> PredictionResponse | None:
        """Return the prediction, or `None` if unavailable by `deadline`
        (`time.monotonic()` seconds). Callers fall back to the heuristic."""
        ...


@runtime_checkable
class Predictor(Protocol):
    def submit(self, request: PredictionRequest) -> PredictionHandle: ...
```

`preempt/engine/scheduler.py`:

```python
from __future__ import annotations

from collections.abc import Sequence

from preempt.core.identity import ExpertKey


class AllResidentProvider:
    """`ExpertProvider` for fully-resident models: every acquire is a no-op.

    The phase-1 stub — the loop, hooks, and wrapper signature are final from
    day one; streaming (phase 3) swaps this for the real scheduler and
    nothing above it changes.
    """

    def acquire(self, keys: Sequence[ExpertKey]) -> None:
        return None
```

- [x] **Step 4: Run to verify pass** — `pytest tests/unit/test_protocols.py -v`
- [ ] **Step 5: Commit** — `feat(core): capability protocols for runner, store, residency, provider, predictor`

### Task 5: `PipelineConfig`

**Files:**
- Create: `preempt/config/pipeline.py`
- Test: `tests/unit/test_pipeline_config.py`

**Interfaces:**
- Consumes: `TargetLayerConfig`, `TargetLayerSpec` (`preempt.config.target_layers`); parsed via existing `read_and_validate_toml` (`preempt.utils.io_utils`).
- Produces: `PipelineConfig` with fields `version: int`, `model: ModelConfig` (`id`, `backend`, `architecture`, `revision: str | None`), `generation: GenerationConfig` (`max_tokens`), `tracing: TracingConfig | None` (`output_path`, `batch_size`, `capture_gate_logits`, `run_id_prefix`, `overwrite_output`, `targets`, method `to_target_layer_config() -> TargetLayerConfig`), `streaming: StreamingConfig | None` (`store_path`, `resident_bytes_budget`).

- [x] **Step 1: Write the failing test `tests/unit/test_pipeline_config.py`**

```python
from pathlib import Path

import pytest
from pydantic import ValidationError

from preempt.config.pipeline import PipelineConfig
from preempt.utils.io_utils import read_and_validate_toml

MINIMAL = {
    "model": {"id": "some/model", "backend": "mlx_metal", "architecture": "qwen3-next"}
}

TRACED = {
    **MINIMAL,
    "tracing": {
        "output_path": "out/events.parquet",
        "targets": [
            {"name": "r", "search_params": {"layer_class": "Blk", "count": 40}}
        ],
    },
}


def test_minimal_config_defaults() -> None:
    config = PipelineConfig.model_validate(MINIMAL)
    assert config.tracing is None
    assert config.streaming is None
    assert config.generation.max_tokens >= 1


def test_tracing_targets_convert_to_target_layer_config() -> None:
    config = PipelineConfig.model_validate(TRACED)
    assert config.tracing is not None
    tl_config = config.tracing.to_target_layer_config()
    assert tl_config.target_layers[0].search_params.count == 40


def test_duplicate_tracing_target_names_rejected_at_parse() -> None:
    bad = {
        **MINIMAL,
        "tracing": {
            "output_path": "x.parquet",
            "targets": [
                {"name": "same", "search_params": {"layer_idx": 0}},
                {"name": "same", "search_params": {"layer_idx": 1}},
            ],
        },
    }
    with pytest.raises(ValidationError, match="unique"):
        PipelineConfig.model_validate(bad)


def test_unknown_keys_rejected() -> None:
    with pytest.raises(ValidationError):
        PipelineConfig.model_validate({**MINIMAL, "surprise": 1})


def test_streaming_requires_positive_budget() -> None:
    bad = {**MINIMAL, "streaming": {"store_path": "s", "resident_bytes_budget": 0}}
    with pytest.raises(ValidationError):
        PipelineConfig.model_validate(bad)


def test_round_trips_from_toml_file(tmp_path: Path) -> None:
    toml = tmp_path / "p.toml"
    toml.write_text(
        '[model]\nid = "m"\nbackend = "mlx_metal"\narchitecture = "a"\n'
        "[generation]\nmax_tokens = 4\n"
        '[tracing]\noutput_path = "o.parquet"\n'
        '[[tracing.targets]]\nname = "r"\n'
        "[tracing.targets.search_params]\nlayer_class = \"Blk\"\n"
    )
    config = read_and_validate_toml(toml, PipelineConfig)
    assert config.generation.max_tokens == 4
    assert config.tracing is not None
```

- [x] **Step 2: Run to verify it fails** — expected `ModuleNotFoundError: preempt.config.pipeline`.

- [x] **Step 3: Implement `preempt/config/pipeline.py`**

```python
from __future__ import annotations

from typing import Self

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

from preempt.config.target_layers import TargetLayerConfig, TargetLayerSpec


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(min_length=1)
    backend: str = Field(min_length=1)
    architecture: str = Field(min_length=1)
    revision: str | None = Field(default=None)


class GenerationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    max_tokens: int = Field(default=16, ge=1)


class TracingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    output_path: Path = Field()
    batch_size: int = Field(default=1024, ge=1)
    capture_gate_logits: bool = Field(default=False)
    run_id_prefix: str = Field(default="trace", min_length=1)
    overwrite_output: bool = Field(default=False)
    targets: tuple[TargetLayerSpec, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_targets_as_layer_config(self) -> Self:
        # Runs TargetLayerConfig's own validators (e.g. unique names) at parse
        # time instead of deferring the failure to resolve time.
        self.to_target_layer_config()
        return self

    def to_target_layer_config(self) -> TargetLayerConfig:
        return TargetLayerConfig(target_layers=self.targets)


class StreamingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    store_path: Path = Field()
    resident_bytes_budget: int = Field(gt=0)


class PipelineConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    version: int = Field(default=1, ge=1)
    model: ModelConfig
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    tracing: TracingConfig | None = Field(default=None)
    streaming: StreamingConfig | None = Field(default=None)
```

- [x] **Step 4: Run to verify pass**, then run the whole suite: `pytest tests/unit -v`
- [ ] **Step 5: Commit** — `feat(config): PipelineConfig TOML model for config-driven runs` (deferred: commits are handled by the user)

### Task 6: metrics + generation loop

**Files:**
- Create: `preempt/engine/metrics.py`, `preempt/engine/generation.py`
- Delete: `preempt/engine/model_executor.py` (empty placeholder superseded by these)
- Test: `tests/unit/test_generation.py`

**Interfaces:**
- Consumes: `ModelRunner` protocol (Task 4), `BaseEventRecorder` (`preempt.engine.recorder`), `BaseEventSink` (`preempt.core.sinks`), `TraceStepContext`.
- Produces:
  - `StepMetrics` (attrs): `step_idx: int`, `n_tokens: int`, `duration_s: float`
  - `GenerationMetrics` (attrs): `steps: list[StepMetrics]`, `records_written: int`, `cache_hits: int`, `cache_misses: int`, `demand_stall_s: float`, `prefetched_bytes: int`, `wasted_prefetch_bytes: int`; properties `tokens_forwarded: int`, `total_duration_s: float`
  - `async generate_greedy(*, runner, prompt_ids, max_tokens, recorder=None, sink=None, sequence_id=0, on_step=None) -> tuple[list[int], GenerationMetrics]`

- [x] **Step 1: Write the failing test `tests/unit/test_generation.py`**

```python
import asyncio
from typing import Any
from collections.abc import Mapping, Sequence

import pytest

from preempt.core.sinks import BaseEventSink
from preempt.datamodel.tracing.context import TraceRunContext, TraceStepContext
from preempt.engine.generation import generate_greedy
from preempt.engine.metrics import StepMetrics
from preempt.engine.recorder import BaseEventRecorder


class ScriptedRunner:
    """Returns queued token ids; records every `step` call's inputs."""

    def __init__(self, outputs: list[int]) -> None:
        self.outputs = list(outputs)
        self.prepared = False
        self.calls: list[list[int]] = []

    def prepare(self) -> None:
        self.prepared = True

    def step(self, tokens: Sequence[int]) -> int:
        assert self.prepared, "step() before prepare()"
        self.calls.append(list(tokens))
        return self.outputs.pop(0)


class SpyRecorder(BaseEventRecorder):
    """Buffers one fake capture per step; counts flushes."""

    def __init__(self) -> None:
        super().__init__(
            TraceRunContext(run_id="r", model_id="m", model_architecture="a")
        )
        self.step_contexts: list[TraceStepContext] = []

    def start_step(self, step_context: TraceStepContext) -> None:
        super().start_step(step_context)
        self.step_contexts.append(step_context)

    def capture(self, **kwargs: Any) -> None: ...

    async def flush(self, sink: BaseEventSink) -> int:
        self.end_step()
        return 3  # pretend 3 records per step


class NullSink(BaseEventSink):
    async def write(self, event: Mapping[str, Any]) -> None: ...
    async def flush(self) -> None: ...
    async def aclose(self) -> None: ...


async def test_prefill_then_single_token_decode_steps() -> None:
    runner = ScriptedRunner([11, 12, 13])
    tokens, metrics = await generate_greedy(
        runner=runner, prompt_ids=[1, 2, 3], max_tokens=3
    )
    assert tokens == [11, 12, 13]
    assert runner.calls == [[1, 2, 3], [11], [12]]
    assert metrics.tokens_forwarded == 5
    assert [s.n_tokens for s in metrics.steps] == [3, 1, 1]


async def test_traced_run_stamps_step_contexts_and_counts_records() -> None:
    runner = ScriptedRunner([9, 8])
    recorder = SpyRecorder()
    tokens, metrics = await generate_greedy(
        runner=runner,
        prompt_ids=[5, 6],
        max_tokens=2,
        recorder=recorder,
        sink=NullSink(),
    )
    assert metrics.records_written == 6
    prefill, decode = recorder.step_contexts
    assert (prefill.token_idx, prefill.token_id) == (0, None)  # multi-token: id null
    assert (decode.token_idx, decode.token_id) == (2, 9)


async def test_recorder_without_sink_rejected() -> None:
    with pytest.raises(ValueError, match="both"):
        await generate_greedy(
            runner=ScriptedRunner([1]),
            prompt_ids=[1],
            max_tokens=1,
            recorder=SpyRecorder(),
        )


async def test_on_step_callback_sees_every_step() -> None:
    seen: list[StepMetrics] = []
    await generate_greedy(
        runner=ScriptedRunner([1, 2]),
        prompt_ids=[3],
        max_tokens=2,
        on_step=seen.append,
    )
    assert [s.step_idx for s in seen] == [0, 1]


async def test_runner_runs_off_event_loop() -> None:
    loop = asyncio.get_running_loop()

    class LoopAsserter(ScriptedRunner):
        def step(self, tokens: Sequence[int]) -> int:
            with pytest.raises(RuntimeError):
                asyncio.get_running_loop()  # no loop in the worker thread
            return super().step(tokens)

    await generate_greedy(runner=LoopAsserter([1]), prompt_ids=[1], max_tokens=1)
    assert loop is asyncio.get_running_loop()
```

- [x] **Step 2: Run to verify it fails** — expected `ModuleNotFoundError`.

- [x] **Step 3: Implement `preempt/engine/metrics.py`**

```python
from __future__ import annotations

import attrs
from attrs import field, validators


@attrs.define(kw_only=True, frozen=True)
class StepMetrics:
    """Timing for one forward pass (prefill or decode)."""

    step_idx: int = field(validator=validators.ge(0))
    n_tokens: int = field(validator=validators.ge(1))
    duration_s: float = field(validator=validators.ge(0.0))


@attrs.define(kw_only=True)
class GenerationMetrics:
    """Per-generation counters.

    The streaming counters (`cache_*`, `demand_stall_s`, `*_bytes`) stay zero
    until the scheduler lands in phase 3; they exist now so the shape of
    `GenerationResult` is stable across phases.
    """

    steps: list[StepMetrics] = field(factory=list)
    records_written: int = field(default=0)
    cache_hits: int = field(default=0)
    cache_misses: int = field(default=0)
    demand_stall_s: float = field(default=0.0)
    prefetched_bytes: int = field(default=0)
    wasted_prefetch_bytes: int = field(default=0)

    @property
    def tokens_forwarded(self) -> int:
        return sum(step.n_tokens for step in self.steps)

    @property
    def total_duration_s(self) -> float:
        return sum(step.duration_s for step in self.steps)
```

- [x] **Step 4: Implement `preempt/engine/generation.py`**

```python
from __future__ import annotations

from collections.abc import Callable, Sequence

import asyncio
import time

from preempt.core.protocols.runner import ModelRunner
from preempt.core.sinks import BaseEventSink
from preempt.datamodel.tracing.context import TraceStepContext
from preempt.engine.metrics import GenerationMetrics, StepMetrics
from preempt.engine.recorder import BaseEventRecorder


async def generate_greedy(
    *,
    runner: ModelRunner,
    prompt_ids: Sequence[int],
    max_tokens: int,
    recorder: BaseEventRecorder | None = None,
    sink: BaseEventSink | None = None,
    sequence_id: int = 0,
    on_step: Callable[[StepMetrics], None] | None = None,
) -> tuple[list[int], GenerationMetrics]:
    """Greedily decode `max_tokens` ids: one batched prefill, then decode steps.

    The forward runs via `asyncio.to_thread` so the event loop stays free for
    I/O — the structural disk-parallel-compute overlap, in place before any
    real I/O exists. When `recorder` and `sink` are given, each step is
    bracketed by `start_step`/`flush`; both must be given together.

    Returns
    -------
    tuple[list[int], GenerationMetrics]
        Generated token ids and per-step metrics.
    """
    if max_tokens < 1:
        raise ValueError("`max_tokens` must be at least 1.")

    if not prompt_ids:
        raise ValueError("`prompt_ids` must be non-empty.")

    if (recorder is None) != (sink is None):
        raise ValueError("`recorder` and `sink` must be provided both or neither.")

    runner.prepare()

    generated: list[int] = []
    metrics = GenerationMetrics()
    tokens = list(prompt_ids)
    token_idx = 0

    for step_idx in range(max_tokens):
        started = time.perf_counter()

        if recorder is not None:
            recorder.start_step(
                TraceStepContext(
                    sequence_id=sequence_id,
                    token_idx=token_idx,
                    # A prefill spans many tokens; a single `token_id` is only
                    # meaningful for a one-token forward.
                    token_id=tokens[0] if len(tokens) == 1 else None,
                )
            )

        next_token = await asyncio.to_thread(runner.step, tokens)

        if recorder is not None and sink is not None:
            metrics.records_written += await recorder.flush(sink)

        step_metrics = StepMetrics(
            step_idx=step_idx,
            n_tokens=len(tokens),
            duration_s=time.perf_counter() - started,
        )
        metrics.steps.append(step_metrics)

        if on_step is not None:
            on_step(step_metrics)

        generated.append(next_token)
        token_idx += len(tokens)
        tokens = [next_token]

    return generated, metrics
```

- [x] **Step 5: Delete `preempt/engine/model_executor.py`** (deleted with `rm`; user stages the deletion)
- [x] **Step 6: Run to verify pass** — `pytest tests/unit -v`
- [ ] **Step 7: Commit** — `feat(engine): platform-agnostic greedy generation loop and metrics` (deferred: commits are handled by the user)

### Task 7: `InferencePipeline` facade

**Files:**
- Create: `preempt/engine/pipeline.py`
- Test: `tests/unit/test_pipeline.py`

**Interfaces:**
- Consumes: `generate_greedy`, `GenerationMetrics`, `StepMetrics` (Task 6), `ModelRunner`/`TokenCodec` (Task 4), `BaseEventRecorder`, `BaseEventSink`.
- Produces:
  - `GenerationResult` (attrs): `token_ids: list[int]`, `text: str`, `metrics: GenerationMetrics`
  - `InferencePipeline(*, runner, tokenizer, max_tokens, recorder=None, sink=None, on_step=None)` with `async generate(prompt: str, *, max_tokens: int | None = None) -> GenerationResult`

- [x] **Step 1: Write the failing test `tests/unit/test_pipeline.py`** (reuses fakes by defining them inline; each test file stays standalone)

```python
from typing import Any
from collections.abc import Mapping, Sequence

import pytest

from preempt.core.sinks import BaseEventSink
from preempt.datamodel.tracing.context import TraceRunContext, TraceStepContext
from preempt.engine.pipeline import InferencePipeline
from preempt.engine.recorder import BaseEventRecorder


class ScriptedRunner:
    def __init__(self, outputs: list[int]) -> None:
        self.outputs = list(outputs)
        self.prepare_calls = 0

    def prepare(self) -> None:
        self.prepare_calls += 1

    def step(self, tokens: Sequence[int]) -> int:
        return self.outputs.pop(0)


class StubCodec:
    def encode(self, text: str) -> list[int]:
        return [ord(c) for c in text]

    def decode(self, tokens: Sequence[int]) -> str:
        return "".join(chr(t) for t in tokens)


class SpyRecorder(BaseEventRecorder):
    def __init__(self) -> None:
        super().__init__(
            TraceRunContext(run_id="r", model_id="m", model_architecture="a")
        )

    def capture(self, **kwargs: Any) -> None: ...

    async def flush(self, sink: BaseEventSink) -> int:
        self.end_step()
        return 1


class SpySink(BaseEventSink):
    def __init__(self) -> None:
        self.closed = False

    async def write(self, event: Mapping[str, Any]) -> None: ...
    async def flush(self) -> None: ...

    async def aclose(self) -> None:
        self.closed = True


async def test_generate_encodes_decodes_and_counts() -> None:
    pipeline = InferencePipeline(
        runner=ScriptedRunner([65, 66]), tokenizer=StubCodec(), max_tokens=2
    )
    result = await pipeline.generate("hi")
    assert result.token_ids == [65, 66]
    assert result.text == "AB"
    assert result.metrics.tokens_forwarded == 3  # 2 prompt + 1 decode


async def test_traced_generate_closes_sink_and_is_single_use() -> None:
    sink = SpySink()
    pipeline = InferencePipeline(
        runner=ScriptedRunner([65, 66]),
        tokenizer=StubCodec(),
        max_tokens=1,
        recorder=SpyRecorder(),
        sink=sink,
    )
    result = await pipeline.generate("hi")
    assert result.metrics.records_written == 1
    assert sink.closed
    with pytest.raises(RuntimeError, match="single"):
        await pipeline.generate("again")


async def test_max_tokens_override_and_recorder_sink_pairing() -> None:
    pipeline = InferencePipeline(
        runner=ScriptedRunner([1, 2, 3]), tokenizer=StubCodec(), max_tokens=1
    )
    result = await pipeline.generate("xyz", max_tokens=3)
    assert len(result.token_ids) == 3

    with pytest.raises(ValueError, match="both"):
        InferencePipeline(
            runner=ScriptedRunner([1]),
            tokenizer=StubCodec(),
            max_tokens=1,
            recorder=SpyRecorder(),
        )
```

- [x] **Step 2: Run to verify it fails** — `ModuleNotFoundError: No module named 'preempt.engine.pipeline'`.

- [x] **Step 3: Implement `preempt/engine/pipeline.py`**

```python
from __future__ import annotations

from collections.abc import Callable

import attrs
from attrs import field

from preempt.core.protocols.runner import ModelRunner, TokenCodec
from preempt.core.sinks import BaseEventSink
from preempt.engine.generation import generate_greedy
from preempt.engine.metrics import GenerationMetrics, StepMetrics
from preempt.engine.recorder import BaseEventRecorder


@attrs.define(kw_only=True)
class GenerationResult:
    token_ids: list[int] = field()
    text: str = field()
    metrics: GenerationMetrics = field()


class InferencePipeline:
    """Facade over one configured generation setup.

    Built at a composition root from injected concretes. When constructed
    with a `recorder`/`sink` pair the pipeline is single-use: the sink owns
    one output file and is closed when `generate` returns.
    """

    def __init__(
        self,
        *,
        runner: ModelRunner,
        tokenizer: TokenCodec,
        max_tokens: int,
        recorder: BaseEventRecorder | None = None,
        sink: BaseEventSink | None = None,
        on_step: Callable[[StepMetrics], None] | None = None,
    ) -> None:
        if (recorder is None) != (sink is None):
            raise ValueError("`recorder` and `sink` must be provided both or neither.")

        self._runner = runner
        self._tokenizer = tokenizer
        self._max_tokens = max_tokens
        self._recorder = recorder
        self._sink = sink
        self._on_step = on_step
        self._sink_consumed = False

    async def generate(
        self, prompt: str, *, max_tokens: int | None = None
    ) -> GenerationResult:
        if self._sink is not None and self._sink_consumed:
            raise RuntimeError(
                "A traced pipeline is single-use: its sink already wrote and "
                "closed one output file. Build a new pipeline to trace again."
            )

        prompt_ids = self._tokenizer.encode(prompt)
        budget = max_tokens if max_tokens is not None else self._max_tokens

        if self._sink is not None:
            self._sink_consumed = True
            async with self._sink as sink:
                token_ids, metrics = await generate_greedy(
                    runner=self._runner,
                    prompt_ids=prompt_ids,
                    max_tokens=budget,
                    recorder=self._recorder,
                    sink=sink,
                    on_step=self._on_step,
                )
        else:
            token_ids, metrics = await generate_greedy(
                runner=self._runner,
                prompt_ids=prompt_ids,
                max_tokens=budget,
                on_step=self._on_step,
            )

        return GenerationResult(
            token_ids=token_ids,
            text=self._tokenizer.decode(token_ids),
            metrics=metrics,
        )
```

- [x] **Step 4: Run to verify pass** — `pytest tests/unit -v` (41 passed)
- [ ] **Step 5: Commit** — `feat(engine): InferencePipeline facade` (deferred: commits are handled by the user)

### Task 8: MLX backend — loader, runner, wrapper/factory update

MLX cannot run in the sandbox: write the code, sanity-check with a syntax/type pass, then verify on the host by re-running the *existing* single-layer integration script (its call sites keep working because the factory's new parameters are keyword-only with defaults).

**Files:**
- Create: `preempt/backends/mlx_metal/loader.py`, `preempt/backends/mlx_metal/runner.py`
- Modify: `preempt/backends/mlx_metal/instrumented/qwen3_next_moe.py`

**Interfaces:**
- Consumes: `TokenCodec`, `ModelRunner`, `ExpertProvider`, `ExpertKey`; `mlx_lm.load`, `mlx_lm.generate.generation_stream`, `mlx_lm.models.cache.make_prompt_cache`, `mlx.utils.tree_flatten`.
- Produces:
  - `MlxLoadedModel` (attrs, `eq=False`): `model: nn.Module`, `tokenizer: TokenCodec`
  - `load_mlx_model(model_id: str) -> MlxLoadedModel`
  - `MlxModelRunner(model: nn.Module)` implementing `ModelRunner`
  - `make_qwen3next_moe_wrapper_factory(recorder, *, capture_gate_logits=False, provider=None, model_fingerprint=None)` — `recorder` now `MlxExpertRoutingRecorder | None`

- [x] **Step 1: Implement `preempt/backends/mlx_metal/loader.py`**

```python
from __future__ import annotations

from typing import cast

import attrs
from attrs import field

import mlx.nn as nn
from mlx_lm import load

from preempt.core.protocols.runner import TokenCodec


@attrs.define(kw_only=True, frozen=True, eq=False)
class MlxLoadedModel:
    model: nn.Module = field()
    tokenizer: TokenCodec = field()


def load_mlx_model(model_id: str) -> MlxLoadedModel:
    """Load an MLX model + tokenizer via `mlx_lm`.

    `mlx_lm`'s `TokenizerWrapper` forwards `encode`/`decode` through an
    unannotated `__getattr__`; the cast to `TokenCodec` restores real
    signatures rather than silencing the diagnostic.
    """
    model, raw_tokenizer = load(model_id)  # type: ignore[misc]
    return MlxLoadedModel(model=model, tokenizer=cast(TokenCodec, raw_tokenizer))
```

- [x] **Step 2: Implement `preempt/backends/mlx_metal/runner.py`** — this is `forward_and_argmax` from the test script, verbatim discipline included (stream, explicit cache-state eval, `mx.clear_cache()`):

```python
from __future__ import annotations

from typing import Any
from collections.abc import Sequence

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from mlx_lm.generate import generation_stream
from mlx_lm.models.cache import make_prompt_cache


class MlxModelRunner:
    """`ModelRunner` for MLX: one forward pass + greedy argmax per `step`.

    Mirrors the memory discipline of `mlx_lm.generate.generate_step`: compute
    runs on the generation stream and the cache state is evaluated explicitly.
    That eval is load-bearing — this model's linear-attention layers return
    recurrent state as a *sibling* of the layer output, so evaluating only the
    sampled token leaves the state an unevaluated graph and every subsequent
    step re-derives the whole chain from step 0.
    """

    def __init__(self, model: nn.Module) -> None:
        self._model = model
        self._cache: list[Any] | None = None

    def prepare(self) -> None:
        self._cache = make_prompt_cache(self._model)

    def step(self, tokens: Sequence[int]) -> int:
        if self._cache is None:
            raise RuntimeError("`prepare()` must be called before `step()`.")

        with mx.stream(generation_stream):
            input_ids = mx.array([list(tokens)], dtype=mx.int32)
            logits = self._model(input_ids, cache=self._cache)
            next_token = mx.argmax(logits[:, -1, :], axis=-1)
            mx.eval(next_token)

            state_arrays = [
                value
                for _, value in tree_flatten([entry.state for entry in self._cache])
                if isinstance(value, mx.array)
            ]
            if state_arrays:
                mx.eval(state_arrays)

        mx.clear_cache()
        return int(next_token.item())
```

- [x] **Step 3: Update the instrumented wrapper** in `preempt/backends/mlx_metal/instrumented/qwen3_next_moe.py`:
  - `__init__` signature becomes `(self, inner, recorder: MlxExpertRoutingRecorder | None, capture_gate_logits: bool, layer_path: str, layer_idx: int, provider: ExpertProvider | None = None, model_fingerprint: str | None = None)`; raise `ValueError("`model_fingerprint` is required when a `provider` is given.")` if `provider is not None and model_fingerprint is None`. Update the class attribute annotations to match.
  - In `__call__`, replace the unconditional capture block with:

```python
        # Record state (lazily -- no eval in the capture path)
        if self.recorder is not None:
            self.recorder.capture(
                layer_path=self.layer_path,
                layer_class=self.inner.__class__.__name__,
                layer_idx=self.layer_idx,
                expert_ids=inds,
                expert_weights=scores,
                gate_logits=logits if self.capture_gate_logits else None,
            )

        # Demand sync point: only a streaming run wires a provider, and only
        # then do we pay the eval that materializing the routed ids forces.
        if self.provider is not None:
            assert self.model_fingerprint is not None
            unique_ids = sorted({int(e) for e in inds.flatten().tolist()})
            self.provider.acquire(
                tuple(
                    ExpertKey(
                        model_fingerprint=self.model_fingerprint,
                        layer_idx=self.layer_idx,
                        expert_idx=expert_idx,
                    )
                    for expert_idx in unique_ids
                )
            )
```

  - Update the factory (also delete its stale `# TODO Get rid of this` comment and give the `candidate.layer_idx is None` branch a real message: `ValueError(f"Cannot instrument {candidate.layer_path!r}: no transformer block index.")`):

```python
def make_qwen3next_moe_wrapper_factory(
    recorder: MlxExpertRoutingRecorder | None,
    *,
    capture_gate_logits: bool = False,
    provider: ExpertProvider | None = None,
    model_fingerprint: str | None = None,
) -> MlxWrapperFactory:
```

  passing the new arguments through to `InstrumentedQwen3NextMoE`. Add imports: `from preempt.core.identity import ExpertKey`, `from preempt.core.protocols.provider import ExpertProvider`.

- [x] **Step 4: Sanity-check in the sandbox** — `/home/agent/.venvs/preempt-linux/bin/python -m py_compile preempt/backends/mlx_metal/loader.py preempt/backends/mlx_metal/runner.py preempt/backends/mlx_metal/instrumented/qwen3_next_moe.py` (imports of `mlx` will fail at runtime here; `py_compile` checks syntax only).

- [x] **Step 5: Verify on the host** — run the existing single-layer smoke script unchanged via `mcp__hostrun__run_script` with script `tests/integration/mlx_single_router_layer.py` and args `--model ${model} --output ${run_dir}`; poll with `mcp__hostrun__poll_run`, read `log_path` on completion. Expected: passes exactly as before (wrapper defaults preserve old behavior).

- [ ] **Step 6: Commit** — `feat(backends/mlx): loader, ModelRunner, provider-aware capture wrapper`

### Task 9: composition root `main.py` + example config

**Files:**
- Create: `main.py` (repo root)
- Create: `configs/pipeline-qwen3_6-35b-mlx.toml`

**Interfaces:**
- Consumes: everything above. `resolve_mlx_target_layers` + `ensure_no_target_layer_overlap` for target resolution; `mlx_instrument_model` + `make_qwen3next_moe_wrapper_factory` for instrumentation; `ParquetEventSink`; `TraceRunContext.with_generated_run_id`; `ExpertRoutingEvent.arrow_schema()`.
- Produces: `build_pipeline(config: PipelineConfig) -> InferencePipeline` — the one function wiring concretes; `main()` CLI (`--config`, `--prompt`, `--max-tokens`).

- [x] **Step 1: Write `configs/pipeline-qwen3_6-35b-mlx.toml`**

```toml
version = 1

[model]
id = "unsloth/Qwen3.6-35B-A3B-UD-MLX-4bit"
backend = "mlx_metal"
architecture = "qwen3-next"

[generation]
max_tokens = 8

[tracing]
output_path = "out/router-events.parquet"
overwrite_output = true

[[tracing.targets]]
name = "Qwen3.6-35B-expert-router"

[tracing.targets.search_params]
layer_class = "Qwen3NextSparseMoeBlock"
count = 40
```

- [x] **Step 2: Write `main.py`**

```python
"""Composition root: the one place concrete backends, sinks, and the engine meet.

Usage (macOS host)::

    python main.py --config configs/pipeline-qwen3_6-35b-mlx.toml \
        --prompt "The capital of France is"
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from preempt.config.pipeline import PipelineConfig
from preempt.engine.metrics import StepMetrics
from preempt.engine.pipeline import InferencePipeline
from preempt.utils.io_utils import read_and_validate_toml


def _print_step(step: StepMetrics) -> None:
    print(
        f"  step {step.step_idx}: {step.n_tokens} token(s) in {step.duration_s:6.1f}s",
        flush=True,
    )


def build_pipeline(config: PipelineConfig) -> InferencePipeline:
    """Wire concrete backend components per `config` into an `InferencePipeline`."""

    if config.streaming is not None:
        raise NotImplementedError("Streaming lands in phase 3 of the v1 plan.")

    if config.model.backend != "mlx_metal":
        raise ValueError(f"Unknown backend: {config.model.backend!r}")

    # Backend imports stay inside the branch: only the composition root may
    # import concrete backends, and only for the backend actually selected.
    from preempt.backends.mlx_metal.instrument import mlx_instrument_model
    from preempt.backends.mlx_metal.instrumented.qwen3_next_moe import (
        make_qwen3next_moe_wrapper_factory,
    )
    from preempt.backends.mlx_metal.layer_discovery import resolve_mlx_target_layers
    from preempt.backends.mlx_metal.loader import load_mlx_model
    from preempt.backends.mlx_metal.recorder import MlxExpertRoutingRecorder
    from preempt.backends.mlx_metal.runner import MlxModelRunner
    from preempt.core.sinks import ParquetEventSink
    from preempt.datamodel.tracing.context import TraceRunContext
    from preempt.datamodel.tracing.expert_routing import ExpertRoutingEvent
    from preempt.engine.layer_resolution import ensure_no_target_layer_overlap

    print(f"Loading model: {config.model.id}")
    loaded = load_mlx_model(config.model.id)

    recorder = None
    sink = None

    if config.tracing is not None:
        resolved = resolve_mlx_target_layers(
            loaded.model, config.tracing.to_target_layer_config()
        )
        ensure_no_target_layer_overlap(resolved)
        layers = [candidate for matches in resolved.values() for candidate in matches]

        run_context = TraceRunContext.with_generated_run_id(
            run_id_prefix=config.tracing.run_id_prefix,
            model_id=config.model.id,
            model_architecture=config.model.architecture,
            model_revision=config.model.revision,
        )
        recorder = MlxExpertRoutingRecorder(run_context=run_context)

        mlx_instrument_model(
            loaded.model,
            layers,
            make_qwen3next_moe_wrapper_factory(
                recorder, capture_gate_logits=config.tracing.capture_gate_logits
            ),
        )
        print(f"Instrumented {len(layers)} router layer(s).")

        sink = ParquetEventSink(
            path=config.tracing.output_path,
            schema=ExpertRoutingEvent.arrow_schema(),
            batch_size=config.tracing.batch_size,
            overwrite=config.tracing.overwrite_output,
        )

    return InferencePipeline(
        runner=MlxModelRunner(loaded.model),
        tokenizer=loaded.tokenizer,
        max_tokens=config.generation.max_tokens,
        recorder=recorder,
        sink=sink,
        on_step=_print_step,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a configured inference pipeline.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-tokens", type=int, default=None)
    args = parser.parse_args()

    config = read_and_validate_toml(args.config, PipelineConfig)
    pipeline = build_pipeline(config)

    result = asyncio.run(pipeline.generate(args.prompt, max_tokens=args.max_tokens))

    print(f"Output: {result.text!r}")
    print(
        f"{result.metrics.tokens_forwarded} token position(s) forwarded, "
        f"{result.metrics.records_written} trace record(s), "
        f"{result.metrics.total_duration_s:.1f}s total."
    )


if __name__ == "__main__":
    main()
```

- [x] **Step 3: Sanity-check config parses in the sandbox**

```bash
PYTHONPATH=/Users/davidstoneman/venvs/preempt /home/agent/.venvs/preempt-linux/bin/python -c "
from preempt.config.pipeline import PipelineConfig
from preempt.utils.io_utils import read_and_validate_toml
c = read_and_validate_toml('configs/pipeline-qwen3_6-35b-mlx.toml', PipelineConfig)
print(c.model.id, len(c.tracing.targets))"
```

- [ ] **Step 4: Commit** — `feat: main.py composition root and example pipeline config`

Note: `main.py` itself is exercised end-to-end in Task 10's host run (the host bridge only executes paths under `tests/`, and the new integration script drives the same wiring).

### Task 10: shrink `mlx_all_router_layers.py` + host exactness gate (phase 1 gate)

**Files:**
- Modify: `tests/integration/mlx_all_router_layers.py`
- Create: `tests/integration/qwen3_6-35b-mlx-pipeline.toml` (copy of `configs/pipeline-qwen3_6-35b-mlx.toml` with `max_tokens = 2` and `output_path = "out/router-events.parquet"` — the script overrides output at runtime anyway)

**Interfaces:**
- Consumes: `PipelineConfig`, `load_mlx_model`, `MlxModelRunner`, `InferencePipeline`, `GenerationResult`, `resolve_mlx_target_layers`, `ensure_no_target_layer_overlap`, `mlx_instrument_model`, `make_qwen3next_moe_wrapper_factory`, `MlxExpertRoutingRecorder`, `ParquetEventSink`, `TraceRunContext`.
- Produces: the phase-1 acceptance evidence. The script remains its own composition root (it needs reference-then-instrument sequencing `main.build_pipeline` doesn't expose).

- [x] **Step 1: Rewrite the script.** Keep unchanged: module docstring (update the usage block to point at the new TOML), `TraceVerificationError`, `parse_args` (drop `--model`; the model id now comes from `--config`, keep an optional `--model` override), `describe_router_topology`, `get_modules_by_path`, `instrument_router_layers`, `verify_trace`, `main()`'s temp-dir handling. Delete: `TokenCodec` protocol (import from `preempt.core.protocols.runner`), `resolve_router_layers` body's plumbing stays but now reads targets from `PipelineConfig.tracing`, `forward_and_argmax`, `report_step`, `generate_greedy`, `generate_greedy_traced` (all now live in `preempt/`). The new `run()` core:

```python
async def run(args: argparse.Namespace, output_path: Path) -> None:
    config = read_and_validate_toml(args.config, PipelineConfig)
    assert config.tracing is not None, "This script requires a [tracing] table."
    model_id = args.model if args.model is not None else config.model.id

    print(f"Loading model: {model_id}")
    loaded = load_mlx_model(model_id)

    resolved = resolve_mlx_target_layers(
        loaded.model, config.tracing.to_target_layer_config()
    )
    ensure_no_target_layer_overlap(resolved)
    layers = sort_and_check_layers(resolved)  # the old resolve_router_layers checks
    top_k, num_experts, norm_topk_prob = describe_router_topology(loaded.model, layers)

    runner = MlxModelRunner(loaded.model)

    reference: GenerationResult | None = None
    if args.verify_exactness:
        print(f"Reference pass (uninstrumented), {args.max_tokens} token(s)...")
        reference_pipeline = InferencePipeline(
            runner=runner, tokenizer=loaded.tokenizer,
            max_tokens=args.max_tokens, on_step=make_step_printer("reference"),
        )
        reference = await reference_pipeline.generate(args.prompt)
        print(f"Reference output: {reference.text!r}")

    run_context = TraceRunContext.with_generated_run_id(
        run_id_prefix="mlx-all-router-layers",
        model_id=model_id, model_architecture=config.model.architecture,
    )
    recorder = MlxExpertRoutingRecorder(run_context=run_context)
    instrument_router_layers(
        model=loaded.model, layers=layers, recorder=recorder,
        capture_gate_logits=not args.no_gate_logits,
    )
    print(f"Instrumented all {len(layers)} router layer(s).")

    sink = ParquetEventSink(
        path=output_path, schema=ExpertRoutingEvent.arrow_schema(),
        batch_size=args.batch_size, overwrite=True,
    )
    traced_pipeline = InferencePipeline(
        runner=runner, tokenizer=loaded.tokenizer, max_tokens=args.max_tokens,
        recorder=recorder, sink=sink, on_step=make_step_printer("traced"),
    )
    print(f"Traced pass (instrumented), {args.max_tokens} token(s)...")
    traced = await traced_pipeline.generate(args.prompt)
    print(f"Traced output: {traced.text!r}")
```

  followed by the existing row-count checks (`records_written` and `sink.records_written` vs `len(layers) * traced.metrics.tokens_forwarded`), the exactness comparison (`traced.token_ids != reference.token_ids` → `TraceVerificationError`), and the existing `verify_trace(...)` call with `expected_token_idxs=set(range(traced.metrics.tokens_forwarded))`. `sort_and_check_layers` is the tail of the old `resolve_router_layers` (≥2 layers, no missing `layer_idx`, sort by `(layer_idx, layer_path)`). `make_step_printer(label)` returns a closure printing `f"  {label} step {s.step_idx}: {s.n_tokens} token(s) in {s.duration_s:6.1f}s | peak {mx.get_peak_memory() / 1e9:5.2f} GB"`.

- [x] **Step 2: Sanity-check syntax in the sandbox** — `py_compile` the script.

- [x] **Step 3: Run on the host with exactness** — `mcp__hostrun__run_script`, script `tests/integration/mlx_all_router_layers.py`, args: `--config tests/integration/qwen3_6-35b-mlx-pipeline.toml --output ${run_dir}/router-events.parquet --verify-exactness --max-tokens 2`. Poll; read the log. Expected output includes `Exactness verified` and `Verified Parquet file: ... 40 layer(s) x N token(s)`.

- [x] **Step 4: Run the full sandbox suite once more** — `pytest tests/unit -v`.

- [ ] **Step 5: Commit** — `refactor(tests): all-router-layers script drives the extracted pipeline` — and note in the commit body that this is the phase-1 gate (EXACT INFERENCE re-verified post-extraction).

---

# Phase 2 — Packed expert store

### Task 11: `ExpertStoreManifest`

**Files:**
- Create: `preempt/storage/__init__.py` (empty), `preempt/storage/manifest.py`
- Test: `tests/unit/test_store_manifest.py`

**Interfaces:**
- Consumes: `ExpertKey`.
- Produces (exact — writer/reader/converter all depend on these):
  - Constants: `MANIFEST_FILENAME = "manifest.json"`, `EXPERTS_FILENAME = "experts.bin"`, `STORE_SCHEMA_VERSION = 1`
  - `TensorSpec(BaseModel)`: `name: str`, `dtype: str` (numpy dtype name), `shape: tuple[int, ...]` (per-expert), `nbytes: int`
  - `ExpertTopology(BaseModel)`: `moe_layer_idxs: tuple[int, ...]`, `num_experts: int`, `top_k: int`
  - `ExpertBlobRecord(BaseModel)`: `layer_idx: int`, `expert_idx: int`, `variant: str = "all"`, `offset: int`, `length: int`
  - `ExpertStoreManifest(BaseModel)`: `schema_version`, `model_id`, `model_fingerprint`, `payload_encoding`, `alignment: int`, `tensor_specs`, `topology`, `blobs`; methods `expert_nbytes() -> int`, `blob_index() -> dict[ExpertKey, ExpertBlobRecord]`, `save(store_dir: Path) -> Path`, classmethod `load(store_dir: Path) -> Self`
  - `StoreCompatibilityError(RuntimeError)`

- [x] **Step 1: Write the failing test `tests/unit/test_store_manifest.py`**

```python
from pathlib import Path

import pytest
from pydantic import ValidationError

from preempt.core.identity import ExpertKey
from preempt.storage.manifest import (
    ExpertBlobRecord,
    ExpertStoreManifest,
    ExpertTopology,
    TensorSpec,
)


def make_manifest() -> ExpertStoreManifest:
    specs = (
        TensorSpec(name="gate_proj.weight", dtype="uint32", shape=(4, 8), nbytes=128),
        TensorSpec(name="gate_proj.scales", dtype="float16", shape=(4, 2), nbytes=16),
    )
    return ExpertStoreManifest(
        model_id="some/model",
        model_fingerprint="fp",
        payload_encoding="mlx-affine-q4-g64",
        tensor_specs=specs,
        topology=ExpertTopology(moe_layer_idxs=(1, 3), num_experts=2, top_k=1),
        blobs=(
            ExpertBlobRecord(layer_idx=1, expert_idx=0, offset=0, length=144),
            ExpertBlobRecord(layer_idx=1, expert_idx=1, offset=4096, length=144),
        ),
    )


def test_expert_nbytes_is_sum_of_tensor_specs() -> None:
    assert make_manifest().expert_nbytes() == 144


def test_blob_length_must_match_tensor_specs() -> None:
    manifest = make_manifest()
    bad_blob = {"layer_idx": 1, "expert_idx": 0, "offset": 0, "length": 7}
    with pytest.raises(ValidationError, match="length"):
        ExpertStoreManifest.model_validate(
            manifest.model_dump() | {"blobs": [bad_blob]}
        )


def test_duplicate_blob_identity_rejected() -> None:
    manifest = make_manifest()
    blob = {"layer_idx": 1, "expert_idx": 0, "offset": 0, "length": 144}
    with pytest.raises(ValidationError, match="[Dd]uplicate"):
        ExpertStoreManifest.model_validate(
            manifest.model_dump() | {"blobs": [blob, blob]}
        )


def test_blob_index_keys_by_expert_key() -> None:
    manifest = make_manifest()
    index = manifest.blob_index()
    key = ExpertKey(model_fingerprint="fp", layer_idx=1, expert_idx=1)
    assert index[key].offset == 4096


def test_save_load_round_trip(tmp_path: Path) -> None:
    manifest = make_manifest()
    manifest.save(tmp_path)
    assert ExpertStoreManifest.load(tmp_path) == manifest
```

- [x] **Step 2: Run to verify it fails**

- [x] **Step 3: Implement `preempt/storage/manifest.py`**

```python
from __future__ import annotations

from typing import Final, Self

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

from preempt.core.identity import ExpertKey

MANIFEST_FILENAME: Final[str] = "manifest.json"
EXPERTS_FILENAME: Final[str] = "experts.bin"
STORE_SCHEMA_VERSION: Final[int] = 1


class StoreCompatibilityError(RuntimeError):
    """Raised when a packed store does not match the loaded model."""


class TensorSpec(BaseModel):
    """Shape/dtype of one per-expert tensor inside a blob, in blob order."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    dtype: str = Field(min_length=1)
    shape: tuple[int, ...] = Field(min_length=1)
    nbytes: int = Field(gt=0)


class ExpertTopology(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    moe_layer_idxs: tuple[int, ...] = Field(min_length=1)
    num_experts: int = Field(ge=1)
    top_k: int = Field(ge=1)


class ExpertBlobRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    layer_idx: int = Field(ge=0)
    expert_idx: int = Field(ge=0)
    variant: str = Field(default="all", min_length=1)
    offset: int = Field(ge=0)
    length: int = Field(gt=0)


class ExpertStoreManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = Field(default=STORE_SCHEMA_VERSION, ge=1)
    model_id: str = Field(min_length=1)
    model_fingerprint: str = Field(min_length=1)
    payload_encoding: str = Field(min_length=1)
    alignment: int = Field(default=4096, ge=1)
    tensor_specs: tuple[TensorSpec, ...] = Field(min_length=1)
    topology: ExpertTopology
    blobs: tuple[ExpertBlobRecord, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_blobs(self) -> Self:
        expected = self.expert_nbytes()
        identities = set()

        for blob in self.blobs:
            if blob.length != expected:
                raise ValueError(
                    f"Blob (layer {blob.layer_idx}, expert {blob.expert_idx}) has "
                    f"length {blob.length}; tensor specs total {expected}."
                )
            identity = (blob.layer_idx, blob.expert_idx, blob.variant)
            if identity in identities:
                raise ValueError(f"Duplicate blob identity: {identity!r}")
            identities.add(identity)

        return self

    def expert_nbytes(self) -> int:
        return sum(spec.nbytes for spec in self.tensor_specs)

    def blob_index(self) -> dict[ExpertKey, ExpertBlobRecord]:
        return {
            ExpertKey(
                model_fingerprint=self.model_fingerprint,
                layer_idx=blob.layer_idx,
                expert_idx=blob.expert_idx,
                variant=blob.variant,
            ): blob
            for blob in self.blobs
        }

    def save(self, store_dir: Path) -> Path:
        store_dir.mkdir(parents=True, exist_ok=True)
        path = store_dir / MANIFEST_FILENAME
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, store_dir: Path) -> Self:
        raw = (store_dir / MANIFEST_FILENAME).read_text(encoding="utf-8")
        return cls.model_validate_json(raw)
```

- [x] **Step 4: Run to verify pass** — `pytest tests/unit/test_store_manifest.py -v`
- [ ] **Step 5: Commit** — `feat(storage): packed expert store manifest`

### Task 12: `PackedStoreWriter`

**Files:**
- Create: `preempt/storage/writer.py`
- Test: `tests/unit/test_store_writer.py`

**Interfaces:**
- Consumes: Task 11's manifest types + constants.
- Produces: `PackedStoreWriter(store_dir: Path, *, model_id, model_fingerprint, payload_encoding, tensor_specs, topology, alignment=4096, overwrite=False)` — context manager; `add_expert(*, layer_idx: int, expert_idx: int, data: bytes, variant: str = "all") -> None`; `finalize() -> ExpertStoreManifest` (writes `manifest.json`; called by `__exit__` on clean exit).

- [x] **Step 1: Write the failing test `tests/unit/test_store_writer.py`**

```python
from pathlib import Path

import pytest

from preempt.storage.manifest import (
    EXPERTS_FILENAME,
    ExpertStoreManifest,
    ExpertTopology,
    TensorSpec,
)
from preempt.storage.writer import PackedStoreWriter

SPECS = (TensorSpec(name="w", dtype="uint8", shape=(3,), nbytes=3),)
TOPOLOGY = ExpertTopology(moe_layer_idxs=(0,), num_experts=2, top_k=1)


def make_writer(tmp_path: Path, **kwargs) -> PackedStoreWriter:
    return PackedStoreWriter(
        tmp_path / "store",
        model_id="m",
        model_fingerprint="fp",
        payload_encoding="test-enc",
        tensor_specs=SPECS,
        topology=TOPOLOGY,
        alignment=8,
        **kwargs,
    )


def test_blobs_are_aligned_and_readable_at_recorded_offsets(tmp_path: Path) -> None:
    with make_writer(tmp_path) as writer:
        writer.add_expert(layer_idx=0, expert_idx=0, data=b"aaa")
        writer.add_expert(layer_idx=0, expert_idx=1, data=b"bbb")
    manifest = ExpertStoreManifest.load(tmp_path / "store")

    raw = (tmp_path / "store" / EXPERTS_FILENAME).read_bytes()
    for blob, expected in zip(manifest.blobs, (b"aaa", b"bbb"), strict=True):
        assert blob.offset % 8 == 0
        assert raw[blob.offset : blob.offset + blob.length] == expected


def test_wrong_length_data_rejected(tmp_path: Path) -> None:
    with make_writer(tmp_path) as writer:
        with pytest.raises(ValueError, match="3 bytes"):
            writer.add_expert(layer_idx=0, expert_idx=0, data=b"toolong")
        writer.add_expert(layer_idx=0, expert_idx=0, data=b"ok!")


def test_duplicate_expert_rejected(tmp_path: Path) -> None:
    with make_writer(tmp_path) as writer:
        writer.add_expert(layer_idx=0, expert_idx=0, data=b"aaa")
        with pytest.raises(ValueError, match="[Dd]uplicate"):
            writer.add_expert(layer_idx=0, expert_idx=0, data=b"aaa")


def test_existing_store_not_overwritten_by_default(tmp_path: Path) -> None:
    with make_writer(tmp_path) as writer:
        writer.add_expert(layer_idx=0, expert_idx=0, data=b"aaa")
    with pytest.raises(FileExistsError):
        make_writer(tmp_path)
    with make_writer(tmp_path, overwrite=True) as writer:
        writer.add_expert(layer_idx=0, expert_idx=0, data=b"zzz")
```

- [x] **Step 2: Run to verify it fails**

- [x] **Step 3: Implement `preempt/storage/writer.py`**

```python
from __future__ import annotations

from types import TracebackType
from typing import BinaryIO, Self

from pathlib import Path

from preempt.storage.manifest import (
    EXPERTS_FILENAME,
    ExpertBlobRecord,
    ExpertStoreManifest,
    ExpertTopology,
    TensorSpec,
)


class PackedStoreWriter:
    """Sole authority on the packed store container layout.

    Appends page-aligned expert blobs to `experts.bin` and writes the
    manifest on `finalize()`. Converters (per source format, living in their
    backend) extract tensors and feed `add_expert`; nothing else in the
    project writes this layout.
    """

    def __init__(
        self,
        store_dir: Path,
        *,
        model_id: str,
        model_fingerprint: str,
        payload_encoding: str,
        tensor_specs: tuple[TensorSpec, ...],
        topology: ExpertTopology,
        alignment: int = 4096,
        overwrite: bool = False,
    ) -> None:
        self._store_dir = Path(store_dir)
        self._model_id = model_id
        self._model_fingerprint = model_fingerprint
        self._payload_encoding = payload_encoding
        self._tensor_specs = tensor_specs
        self._topology = topology
        self._alignment = alignment
        self._expected_length = sum(spec.nbytes for spec in tensor_specs)

        bin_path = self._store_dir / EXPERTS_FILENAME
        if bin_path.exists() and not overwrite:
            raise FileExistsError(f"Store already exists at `{self._store_dir}`.")

        self._store_dir.mkdir(parents=True, exist_ok=True)
        self._file: BinaryIO = bin_path.open("wb")
        self._offset = 0
        self._blobs: list[ExpertBlobRecord] = []
        self._identities: set[tuple[int, int, str]] = set()
        self._finalized = False

    def add_expert(
        self, *, layer_idx: int, expert_idx: int, data: bytes, variant: str = "all"
    ) -> None:
        if len(data) != self._expected_length:
            raise ValueError(
                f"Expert blob must be exactly {self._expected_length} bytes "
                f"(sum of tensor specs); got {len(data)}."
            )

        identity = (layer_idx, expert_idx, variant)
        if identity in self._identities:
            raise ValueError(f"Duplicate expert blob: {identity!r}")
        self._identities.add(identity)

        padding = -self._offset % self._alignment
        if padding:
            self._file.write(b"\x00" * padding)
            self._offset += padding

        self._file.write(data)
        self._blobs.append(
            ExpertBlobRecord(
                layer_idx=layer_idx,
                expert_idx=expert_idx,
                variant=variant,
                offset=self._offset,
                length=len(data),
            )
        )
        self._offset += len(data)

    def finalize(self) -> ExpertStoreManifest:
        if self._finalized:
            raise RuntimeError("Writer already finalized.")

        self._file.close()
        self._finalized = True

        manifest = ExpertStoreManifest(
            model_id=self._model_id,
            model_fingerprint=self._model_fingerprint,
            payload_encoding=self._payload_encoding,
            alignment=self._alignment,
            tensor_specs=self._tensor_specs,
            topology=self._topology,
            blobs=tuple(self._blobs),
        )
        manifest.save(self._store_dir)
        return manifest

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if exc_type is not None:
            self._file.close()  # abandon a partial store; no manifest written
            return
        if not self._finalized:
            self.finalize()
```

- [x] **Step 4: Run to verify pass** — `pytest tests/unit -v`
- [ ] **Step 5: Commit** — `feat(storage): PackedStoreWriter`

### Task 13: `PackedExpertStore` reader

**Files:**
- Create: `preempt/storage/packed_store.py`
- Test: `tests/unit/test_packed_store.py`

**Interfaces:**
- Consumes: Tasks 11–12; `ExpertStore`/`ExpertPayload`/`ReadPriority`; `StoreCompatibilityError`.
- Produces: `PackedExpertStore(store_dir: Path)` — sync context manager satisfying `ExpertStore`; `manifest` property; `ensure_compatible(*, model_id: str, num_experts: int, top_k: int, moe_layer_idxs: tuple[int, ...]) -> None`; `async read(key, priority) -> ExpertPayload`; `close() -> None`. Raises `KeyError` for unknown keys (a demand read for a key the store lacks is fatal upstream, per the spec's error handling).

- [x] **Step 1: Write the failing test `tests/unit/test_packed_store.py`**

```python
from pathlib import Path

import pytest

from preempt.core.identity import ExpertKey
from preempt.core.protocols.store import ExpertStore, ReadPriority
from preempt.storage.manifest import (
    ExpertTopology,
    StoreCompatibilityError,
    TensorSpec,
)
from preempt.storage.packed_store import PackedExpertStore
from preempt.storage.writer import PackedStoreWriter


@pytest.fixture()
def store_dir(tmp_path: Path) -> Path:
    with PackedStoreWriter(
        tmp_path / "store",
        model_id="m",
        model_fingerprint="fp",
        payload_encoding="test-enc",
        tensor_specs=(TensorSpec(name="w", dtype="uint8", shape=(3,), nbytes=3),),
        topology=ExpertTopology(moe_layer_idxs=(0, 2), num_experts=2, top_k=1),
        alignment=8,
    ) as writer:
        writer.add_expert(layer_idx=0, expert_idx=0, data=b"aaa")
        writer.add_expert(layer_idx=0, expert_idx=1, data=b"bbb")
        writer.add_expert(layer_idx=2, expert_idx=0, data=b"ccc")
    return tmp_path / "store"


async def test_round_trips_written_blobs(store_dir: Path) -> None:
    with PackedExpertStore(store_dir) as store:
        assert isinstance(store, ExpertStore)
        key = ExpertKey(model_fingerprint="fp", layer_idx=2, expert_idx=0)
        payload = await store.read(key, ReadPriority.DEMAND)
        assert payload.data == b"ccc"
        assert payload.encoding == "test-enc"
        assert payload.key == key


async def test_unknown_key_raises_key_error(store_dir: Path) -> None:
    with PackedExpertStore(store_dir) as store:
        missing = ExpertKey(model_fingerprint="fp", layer_idx=9, expert_idx=9)
        with pytest.raises(KeyError):
            await store.read(missing, ReadPriority.PREFETCH)


def test_ensure_compatible_accepts_matching_topology(store_dir: Path) -> None:
    with PackedExpertStore(store_dir) as store:
        store.ensure_compatible(
            model_id="m", num_experts=2, top_k=1, moe_layer_idxs=(0, 2)
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"model_id": "other", "num_experts": 2, "top_k": 1, "moe_layer_idxs": (0, 2)},
        {"model_id": "m", "num_experts": 4, "top_k": 1, "moe_layer_idxs": (0, 2)},
        {"model_id": "m", "num_experts": 2, "top_k": 8, "moe_layer_idxs": (0, 2)},
        {"model_id": "m", "num_experts": 2, "top_k": 1, "moe_layer_idxs": (0,)},
    ],
)
def test_ensure_compatible_rejects_mismatch(store_dir: Path, kwargs: dict) -> None:
    with PackedExpertStore(store_dir) as store:
        with pytest.raises(StoreCompatibilityError):
            store.ensure_compatible(**kwargs)


async def test_read_after_close_raises(store_dir: Path) -> None:
    store = PackedExpertStore(store_dir)
    store.close()
    key = ExpertKey(model_fingerprint="fp", layer_idx=0, expert_idx=0)
    with pytest.raises(RuntimeError, match="closed"):
        await store.read(key, ReadPriority.DEMAND)
```

- [x] **Step 2: Run to verify it fails**

- [x] **Step 3: Implement `preempt/storage/packed_store.py`**

```python
from __future__ import annotations

from types import TracebackType
from typing import Self

import asyncio
import os
from pathlib import Path

from preempt.core.identity import ExpertKey
from preempt.core.protocols.store import ExpertPayload, ReadPriority
from preempt.storage.manifest import (
    EXPERTS_FILENAME,
    ExpertBlobRecord,
    ExpertStoreManifest,
    StoreCompatibilityError,
)


class PackedExpertStore:
    """`ExpertStore` over a packed store directory: one `pread` per expert.

    Priority is accepted per the protocol but ignored here — read *ordering*
    is the scheduler's job; the store just reads what it is told.
    """

    def __init__(self, store_dir: Path) -> None:
        self._manifest = ExpertStoreManifest.load(store_dir)
        self._index: dict[ExpertKey, ExpertBlobRecord] = self._manifest.blob_index()
        self._fd: int | None = os.open(store_dir / EXPERTS_FILENAME, os.O_RDONLY)

    @property
    def manifest(self) -> ExpertStoreManifest:
        return self._manifest

    def ensure_compatible(
        self,
        *,
        model_id: str,
        num_experts: int,
        top_k: int,
        moe_layer_idxs: tuple[int, ...],
    ) -> None:
        """Refuse to serve a store that does not describe the loaded model."""

        observed = {
            "model_id": model_id,
            "num_experts": num_experts,
            "top_k": top_k,
            "moe_layer_idxs": moe_layer_idxs,
        }
        expected = {
            "model_id": self._manifest.model_id,
            "num_experts": self._manifest.topology.num_experts,
            "top_k": self._manifest.topology.top_k,
            "moe_layer_idxs": self._manifest.topology.moe_layer_idxs,
        }
        mismatches = {
            name: (expected[name], observed[name])
            for name in expected
            if expected[name] != observed[name]
        }
        if mismatches:
            raise StoreCompatibilityError(
                f"Packed store does not match the loaded model: {mismatches!r}"
            )

    async def read(self, key: ExpertKey, priority: ReadPriority) -> ExpertPayload:
        if self._fd is None:
            raise RuntimeError("Store is closed.")

        blob = self._index[key]  # KeyError for unknown keys, by design
        data = await asyncio.to_thread(os.pread, self._fd, blob.length, blob.offset)

        if len(data) != blob.length:
            raise IOError(
                f"Short read for {key!r}: got {len(data)} of {blob.length} bytes."
            )

        return ExpertPayload(
            key=key, data=data, encoding=self._manifest.payload_encoding
        )

    def close(self) -> None:
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
```

- [x] **Step 4: Run to verify pass** — `pytest tests/unit -v`
- [ ] **Step 5: Commit** — `feat(storage): PackedExpertStore reader`

### Task 14: blob assembly + MLX converter + host smoke test

**Files:**
- Create: `preempt/storage/blob.py` (backend-neutral numpy blob assembly — sandbox-testable)
- Create: `preempt/backends/mlx_metal/convert.py` (MLX-layout-aware extractor CLI)
- Create: `tests/integration/mlx_convert_store.py` (host smoke: convert a few layers, byte-verify)
- Test: `tests/unit/test_blob.py`

**Interfaces:**
- Consumes: `TensorSpec`, `PackedStoreWriter`, `PackedExpertStore`, `ExpertKey`, `ReadPriority`.
- Produces:
  - `derive_tensor_specs(arrays: Mapping[str, np.ndarray], order: Sequence[str]) -> tuple[TensorSpec, ...]` — specs from one expert's arrays, in `order`
  - `assemble_expert_blob(arrays: Mapping[str, np.ndarray], specs: Sequence[TensorSpec]) -> bytes` — concatenation in spec order, validating dtype/shape per spec
  - `convert_mlx_model_to_store(model_dir: Path, store_dir: Path, *, max_layers: int | None = None, overwrite: bool = False) -> ExpertStoreManifest` and a `main()` CLI (`--model`, `--output`, `--max-layers`, `--overwrite`)
  - `compute_mlx_fingerprint(model_dir: Path) -> str`

- [x] **Step 1: Write the failing test `tests/unit/test_blob.py`**

```python
import numpy as np
import pytest

from preempt.storage.blob import assemble_expert_blob, derive_tensor_specs


def make_arrays() -> dict[str, np.ndarray]:
    return {
        "w": np.arange(6, dtype=np.uint32).reshape(2, 3),
        "s": np.ones((2, 1), dtype=np.float16),
    }


def test_derive_specs_preserves_order_and_metadata() -> None:
    specs = derive_tensor_specs(make_arrays(), order=("s", "w"))
    assert [spec.name for spec in specs] == ["s", "w"]
    assert specs[0].dtype == "float16"
    assert specs[1].shape == (2, 3)
    assert specs[1].nbytes == 24


def test_assemble_concatenates_in_spec_order() -> None:
    arrays = make_arrays()
    specs = derive_tensor_specs(arrays, order=("s", "w"))
    blob = assemble_expert_blob(arrays, specs)
    assert blob == arrays["s"].tobytes() + arrays["w"].tobytes()
    assert len(blob) == sum(spec.nbytes for spec in specs)


def test_assemble_rejects_spec_mismatch() -> None:
    arrays = make_arrays()
    specs = derive_tensor_specs(arrays, order=("s", "w"))
    arrays["w"] = arrays["w"].astype(np.uint8)  # dtype drift
    with pytest.raises(ValueError, match="dtype"):
        assemble_expert_blob(arrays, specs)


def test_derive_specs_requires_all_names_present() -> None:
    with pytest.raises(KeyError):
        derive_tensor_specs(make_arrays(), order=("s", "missing"))
```

- [x] **Step 2: Run to verify it fails**, then implement `preempt/storage/blob.py`

```python
from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from preempt.storage.manifest import TensorSpec


def derive_tensor_specs(
    arrays: Mapping[str, np.ndarray], order: Sequence[str]
) -> tuple[TensorSpec, ...]:
    """Build `TensorSpec`s for one expert's arrays, in the declared blob order."""

    return tuple(
        TensorSpec(
            name=name,
            dtype=str(arrays[name].dtype),
            shape=tuple(arrays[name].shape),
            nbytes=arrays[name].nbytes,
        )
        for name in order
    )


def assemble_expert_blob(
    arrays: Mapping[str, np.ndarray], specs: Sequence[TensorSpec]
) -> bytes:
    """Concatenate one expert's arrays into a blob, validating against `specs`.

    Raises
    ------
    ValueError
        If any array's dtype or shape drifts from its spec — a silent drift
        here would corrupt every read of the resulting store.
    """
    parts: list[bytes] = []

    for spec in specs:
        array = arrays[spec.name]
        if str(array.dtype) != spec.dtype:
            raise ValueError(
                f"Tensor {spec.name!r}: dtype {array.dtype} != spec {spec.dtype}."
            )
        if tuple(array.shape) != spec.shape:
            raise ValueError(
                f"Tensor {spec.name!r}: shape {tuple(array.shape)} != spec {spec.shape}."
            )
        parts.append(np.ascontiguousarray(array).tobytes())

    return b"".join(parts)
```

- [ ] **Step 3: Run to verify pass**, commit — `feat(storage): backend-neutral expert blob assembly`

- [x] **Step 4: Implement `preempt/backends/mlx_metal/convert.py`** (host-only code; `py_compile` in the sandbox). Key structure:

```python
"""Convert an MLX-quantized MoE checkpoint into a packed expert store.

Offline, one-time-per-model. MLX-layout-aware by design: this module knows
mlx_lm's stacked `[num_experts, ...]` tensor naming; the container layout it
feeds belongs entirely to `preempt.storage.writer.PackedStoreWriter`.

Usage (macOS host)::

    python -m preempt.backends.mlx_metal.convert \
        --model <hf-id-or-local-dir> --output out/expert-store
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np

import mlx.core as mx

from preempt.storage.blob import assemble_expert_blob, derive_tensor_specs
from preempt.storage.manifest import ExpertStoreManifest, ExpertTopology
from preempt.storage.writer import PackedStoreWriter

# Blob order: for each projection, weight then scales then biases.
EXPERT_PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
QUANT_PARTS = ("weight", "scales", "biases")
TENSOR_ORDER = tuple(
    f"{projection}.{part}" for projection in EXPERT_PROJECTIONS for part in QUANT_PARTS
)

_EXPERT_TENSOR_RE = re.compile(
    r"^model\.layers\.(?P<layer>\d+)\.mlp\.switch_mlp\."
    r"(?P<projection>gate_proj|up_proj|down_proj)\.(?P<part>weight|scales|biases)$"
)


def resolve_model_dir(model: str) -> Path:
    path = Path(model)
    if path.exists():
        return path
    from huggingface_hub import snapshot_download  # mlx_lm transitive dependency

    return Path(snapshot_download(model))


def compute_mlx_fingerprint(model_dir: Path) -> str:
    """Cheap, stable fingerprint: sha256 of `config.json` bytes plus the sorted
    `(name, size)` list of safetensors shards. Catches shape/layout/quant
    changes; does not detect in-place bit flips (acceptable — deep hashing
    20 GB per run is not)."""
    digest = hashlib.sha256((model_dir / "config.json").read_bytes())
    for shard in sorted(model_dir.glob("*.safetensors")):
        digest.update(f"{shard.name}:{shard.stat().st_size}".encode())
    return digest.hexdigest()


def build_tensor_shard_index(model_dir: Path) -> dict[str, Path]:
    """Map expert tensor name -> shard file, from the safetensors index (or the
    single `model.safetensors` when unsharded)."""
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        weight_map = json.loads(index_path.read_text())["weight_map"]
        return {
            name: model_dir / shard
            for name, shard in weight_map.items()
            if _EXPERT_TENSOR_RE.match(name)
        }
    single = model_dir / "model.safetensors"
    return {
        name: single
        for name in mx.load(str(single))
        if _EXPERT_TENSOR_RE.match(name)
    }


def convert_mlx_model_to_store(
    model_dir: Path,
    store_dir: Path,
    *,
    max_layers: int | None = None,
    overwrite: bool = False,
) -> ExpertStoreManifest:
    ...
```

  `convert_mlx_model_to_store` implementation outline (bounded memory — one layer in flight):

  1. `shard_index = build_tensor_shard_index(model_dir)`; group tensor names by layer index via `_EXPERT_TENSOR_RE`; sort layer indices; truncate to `max_layers` if set.
  2. Read `config.json`: `top_k = config["num_experts_per_tok"]`; `quant = config.get("quantization")`; `payload_encoding = f"mlx-affine-q{quant['bits']}-g{quant['group_size']}"` if quant else `"mlx-unquantized"`.
  3. For the first layer, load its 9 tensors (`mx.load` per shard, cache the last-loaded shard dict since consecutive layers usually share one), slice expert 0, convert each with `np.array(tensor[0])`, and `derive_tensor_specs(arrays, TENSOR_ORDER)`; `num_experts = stacked.shape[0]`.
  4. Open `PackedStoreWriter(store_dir, model_id=model_dir.name, model_fingerprint=compute_mlx_fingerprint(model_dir), payload_encoding=..., tensor_specs=..., topology=ExpertTopology(moe_layer_idxs=tuple(layer_idxs), num_experts=..., top_k=...), overwrite=overwrite)`.
  5. Per layer, per expert `e`: `arrays = {name_suffix: np.array(stacked[name][e]) for ...}` → `writer.add_expert(layer_idx=..., expert_idx=e, data=assemble_expert_blob(arrays, specs))`. Free the layer's arrays before the next layer.
  6. Context-manager exit finalizes; return the manifest. `main()` wires argparse (`--model` required, `--output` required, `--max-layers`, `--overwrite`) and prints blob count + total bytes.

- [x] **Step 5: Write `tests/integration/mlx_convert_store.py`** (host smoke, argparse CLI following the existing scripts' shape). Flow: `--model` (default `${model}` passed by the runner), `--output` (point at `${run_dir}`), `--layers` (default 2). It:
  1. calls `convert_mlx_model_to_store(resolve_model_dir(args.model), args.output / "store", max_layers=args.layers, overwrite=True)`;
  2. reopens with `PackedExpertStore`; asserts manifest topology (`num_experts == 256`, `top_k == 8` for the default model) and `len(manifest.blobs) == args.layers * manifest.topology.num_experts`;
  3. byte-verifies a sample: for 3 sampled `(layer, expert)` pairs, re-load the source tensors via the same shard index, `assemble_expert_blob`, and assert equality with `asyncio.run(store.read(key, ReadPriority.DEMAND)).data`;
  4. asserts every blob offset is `alignment`-aligned; prints a summary line per check.

- [x] **Step 6: Run the smoke on the host** — `mcp__hostrun__run_script`, script `tests/integration/mlx_convert_store.py`, args `--model ${model} --output ${run_dir} --layers 2`. Poll; read `log_path`. Expected: topology + byte-verification lines all pass. (2 layers ≈ 512 blobs, well under a minute of I/O.)

- [ ] **Step 7: Commit** — `feat(backends/mlx): safetensors -> packed expert store converter with host smoke test`

### Task 15: Phase-2 gate — full conversion + phases 3–4 planning checkpoint

**Files:** none created in-repo (store lands outside the repo).

- [ ] **Step 1: Full conversion on the host.** Ask the user to pick a store location on the 8 TB SSD (e.g. `~/preempt-stores/qwen3_6-35b-a3b-mlx-4bit`), then run via `mcp__hostrun__run_script` with `tests/integration/mlx_convert_store.py` and args `--model ${model} --output <chosen-dir> --layers 40`. Expect all 40 layers × 256 experts = 10,240 blobs. Record from the log: total bytes, wall time, and implied write bandwidth.
- [ ] **Step 2: Re-run the whole sandbox suite** (`pytest tests/unit -v`) and the two tracing host scripts, confirm green.
- [ ] **Step 3: Planning checkpoint.** Phases 3 (residency + demand path + scheduler LRU) and 4 (prefetch + heuristic predictor) get their own implementation plan, written only now — the spec defers two decisions to empirical input this phase produces: the MLX install mechanism (in-place row writes vs. slot pool — needs a micro-benchmark against the real stacked quantized tensors) and prefetch parameters (needs measured per-layer compute time vs. store read latency from the phase-1/2 artifacts). Deliverable: run a brainstorm/planning session against `.claude/docs/superpowers/specs/2026-08-01-v1-inference-pipeline-design.md` §"MLX residency" and §"Engine", producing `.claude/docs/superpowers/plans/<date>-v1-streaming-phase3-4.md`.

---

## Verification (end-to-end, after Task 15)

1. Sandbox: `pytest tests/unit -v` — all green, no `.venv` touched, no new deps.
2. Host: `tests/integration/mlx_single_router_layer.py` (unchanged behavior), `tests/integration/mlx_all_router_layers.py --verify-exactness` (prints `Exactness verified`), `tests/integration/mlx_convert_store.py --layers 2` (byte-verified store).
3. `python main.py --config configs/pipeline-qwen3_6-35b-mlx.toml --prompt "..."` on the host produces text output + a Parquet trace (user-run, since the host bridge only executes under `tests/`).
4. Layering audit: `grep -rn "import mlx\|from mlx" preempt/ --include="*.py" | grep -v backends/` returns nothing; `grep -rn "from preempt.backends" preempt/ --include="*.py"` returns nothing (backends imported only by `main.py` and `tests/`).
