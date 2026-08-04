# v1 Inference Pipeline — Design

**Date:** 2026-08-01
**Status:** Draft for review
**Scope decision:** Full streaming v1 — one architecture design, phased implementation.

## Context

The router-tracing "lens" is proven: `tests/integration/mlx_all_router_layers.py`
instruments every MoE block in Qwen3.6-35B, captures router events to Parquet, and
verifies EXACT INFERENCE. But all of the reusable machinery — model loading, layer
resolution, instrumentation, the greedy generation loop with its MLX memory
discipline, traced stepping, sink wiring — lives inline in a 500+ line test script.
`engine/scheduler.py` and `engine/model_executor.py` are empty placeholders.

This design moves that machinery into the source tree behind backend-agnostic
contracts and extends it into the full v1 engine: expert weights streamed from SSD,
a resident expert cache in leftover RAM, and a demand/prefetch scheduler. The
milestone's bar: **a small Python script loads one TOML config and runs an
instrumented inference pipeline.**

MLX remains the prototyping backend, but llama.cpp and/or vLLM integration is
planned soon — no backend's file-format or runtime idiosyncrasies may leak into
core contracts.

## Goals

1. Config-driven pipeline: TOML → load → (optionally) instrument → generate →
   (optionally) trace.
2. Expert streaming: per-expert SSD reads, resident cache under a byte budget,
   demand reads preempting prefetch, heuristic-lookahead prefetch baseline.
3. Every contract testable with fakes — no weights or Metal required for the
   scheduler, loop, store, or config tests.
4. EXACT INFERENCE preserved and re-verified at every phase: identical greedy
   tokens with and without instrumentation, and with full vs. starved residency.

## Non-goals (deferred, per CLAUDE.md v1 scope)

Serving/API, speculative decoding, KV persistence, CUDA, learned predictor
training (the `Predictor` *contract* ships now; only the heuristic impl is built),
multi-model support, plugin ABI.

## Architecture (Approach B: capability-scoped protocols)

Small `typing.Protocol` contracts in `core/`, composed by a platform-agnostic
engine, wired only at the composition root. Backends implement 3–4 small
protocols rather than one monolith; each is independently fakeable. Rejected
alternatives: one fat `InferenceBackend` interface (entangles tracing, execution,
residency; giant fakes); top-level async actor/dataflow pipeline (premature —
async machinery is confined inside the scheduler and store instead).

### Module map

```
preempt/
  core/
    identity.py        ExpertKey (frozen attrs): model fingerprint, layer_idx,
                       expert_idx, weight variant. The cache/store/residency key
                       everywhere; bare ints never cross a boundary.
    protocols/
      runner.py        ModelRunner, TokenCodec
      store.py         ExpertStore, ExpertPayload, ReadPriority
      residency.py     ExpertResidency
      provider.py      ExpertProvider (sync in-forward hook)
      predictor.py     Predictor, PredictionRequest/Response, PredictionHandle
    sinks.py           unchanged; known layering deviation stands
  engine/
    pipeline.py        InferencePipeline facade + GenerationResult
    generation.py      generation loop (step lifecycle, recorder hooks)
    scheduler.py       ExpertScheduler: LRU + budget + demand/prefetch queue;
                       implements ExpertProvider
    metrics.py         attrs counters: hits, misses, stall time, wasted bytes,
                       per-layer timings
  storage/
    manifest.py        ExpertStoreManifest (pydantic — crosses a file boundary)
    packed_store.py    PackedExpertStore (ExpertStore impl)
    writer.py          PackedStoreWriter — sole authority on container layout
  backends/mlx_metal/
    loader.py          model + tokenizer load via mlx_lm
    runner.py          MlxModelRunner (forward_and_argmax + eval discipline)
    residency.py       MlxExpertResidency (stacked-tensor surgery)
    convert.py         MLX safetensors → PackedStoreWriter extractor
  predictors/
    heuristic.py       lookahead baseline (Predictor impl #1)
  config/
    pipeline.py        PipelineConfig (top-level TOML model)
main.py                composition root
```

Dependency direction unchanged: `core/` imports nothing internal; `engine/`,
`storage/`, `predictors/` import only `core/` + `datamodel/`; `backends/`
implements core protocols and is imported only by `main.py`. `storage/` is a new
top-level package: the packed store is backend-neutral but still concrete I/O —
it belongs neither in `core/` (how sinks got mislaid) nor in a backend.

Protocols are `typing.Protocol`, not ABCs, so backends and fakes inherit nothing
from core. `BaseEventRecorder` stays an ABC (it carries real lifecycle state).

### Core contracts

- **`ModelRunner`** — `prepare()` (cache setup) + `step(tokens) -> int` (one
  forward pass, greedy sample). Sync, because MLX decode is sync; the engine
  wraps it in `asyncio.to_thread` so the event loop stays free for I/O.
- **`TokenCodec`** — `encode`/`decode`; promoted from the test script.
- **`ExpertStore`** — `async read(key, priority) -> ExpertPayload`. Payload is
  opaque bytes + the manifest's payload-encoding tag. `ReadPriority` enum
  (`DEMAND`, `PREFETCH`) expresses preemption at the contract level; *ordering*
  lives in the scheduler's queue, the store just reads what it's told.
- **`ExpertResidency`** — `install(key, payload)`, `evict(key)`,
  `is_resident(key)`, `resident_bytes()`. Sync, backend-owned; where bytes
  become live device tensors.
- **`ExpertProvider`** — `acquire(keys) -> None`, sync, blocks until the given
  experts are resident. The one hook the instrumented wrapper calls after top-k
  selection. Called from the runner thread.
- **`Predictor`** — deadline-aware async contract per CLAUDE.md:
  `submit(request) -> handle`, `handle.poll_until(deadline)`; heuristic fallback
  on expiry.

## Engine

**`InferencePipeline`** is the facade: constructed at the composition root from
config + injected concretes; exposes `async generate(prompt) -> GenerationResult`.
The milestone script is ~15 lines: parse args, load TOML, `build_pipeline(config)`,
`asyncio.run(pipeline.generate(...))`, print.

**Generation loop** (`engine/generation.py`) — what `generate_greedy_traced`
becomes: prefill step, then decode steps. Each step: `recorder.start_step` (if
tracing) → `asyncio.to_thread(runner.step, ...)` → `recorder.flush(sink)` →
metrics update. Running the forward in a thread is the structural disk-∥-compute
overlap and is in place from phase 1, before any real I/O exists.

**Demand path.** Router decisions happen mid-forward, inside each MoE layer,
while the runner is off in its thread — so the correctness path cannot route
through the engine. The instrumented wrapper calls the injected `ExpertProvider`
right after top-k selection. On a hit, `acquire` is a dict lookup. On a miss, it
files a `DEMAND` read and blocks the runner thread until store + residency
complete it (bridged to the event loop via `run_coroutine_threadsafe`). The
backend never imports the scheduler; it calls the injected provider, exactly as
it receives the recorder today.

**Prefetch path.** Every `acquire` call *is* the observation "layer L routed to
these experts." The scheduler forwards it to the predictor with deadline =
estimated time until layer L+Δ (rolling per-layer compute-time average) and
schedules `PREFETCH` reads for the predicted set. On deadline expiry, fall back
to the heuristic. Wrong predictions cost only wasted bytes — `acquire` at
layer L+Δ catches anything missing. This is the CLAUDE.md invariant in
mechanism form: predictions affect *when* weights stage, never *which tokens*.

**Scheduler internals** (`engine/scheduler.py`): one asyncio priority queue over
store reads (demand preempts prefetch); per-layer LRU keyed by `ExpertKey` under
a byte budget from config; predicted-soon experts protected from eviction;
prefetch depth backs off when the wasted-bytes ratio climbs. Fully
contract-testable with fake store/residency/predictor and scripted routing.

**Phase 1 stub:** `AllResidentProvider` with a no-op `acquire`. Loop, hooks, and
wrapper signature are final from day one; streaming swaps the provider and
nothing above it changes.

**Tracing is optional; metrics are separate.** The existing sinks were built for
training-dataset capture, so recorder + sink are an opt-in pipeline feature
(config-driven) — a plain inference run wires no sink. The wrapper's two
injection points decouple: recorder optional (tracing runs), provider always
present once streaming lands (correctness path). Metrics are in-process attrs
counters summarized on `GenerationResult`, never events forced through the
dataset-oriented sinks. (Metric *datasets* later = a new event type through the
same sink machinery; nothing here requires it.)

## Storage: packed expert store

**On-disk layout** — one store directory per converted model:

```
<store>/
  manifest.json        ExpertStoreManifest (pydantic)
  experts.bin          all expert payloads, concatenated, page-aligned
```

Manifest carries: model fingerprint (hash of source weights + conversion
params), expert topology (layer count, experts/layer, top-k), payload encoding
tag (e.g. `mlx-affine-q4-g64`), schema version, and the
`ExpertKey → (offset, length)` table. One expert = one contiguous blob = its
tensors concatenated in a declared order (for MLX quant: gate/up/down ×
weight/scales/biases), so a cache miss is exactly one `pread`.

**Compatibility check:** at pipeline build time the reader validates the
manifest fingerprint against the loaded model and refuses to run on mismatch —
same spirit as predictor-artifact compat rules.

**`PackedStoreWriter`** (`storage/writer.py`) is the *only* code that knows the
container layout (blob order, alignment, manifest writing). Converters are
per-source-format extractors living in their backend:
`backends/mlx_metal/convert.py` slices experts out of mlx_lm's stacked
`[num_experts, …]` safetensors tensors and feeds the writer; a future
`backends/gguf/convert.py` feeds the same writer. Extraction lives with its
format; the container is written and read only by `storage/`.

**`PackedExpertStore`** opens `experts.bin` once and serves
`read(key, priority)` via `asyncio.to_thread(os.pread, …)`.

Cost: one offline conversion per model, ~20 GB duplicate disk for the working
model (trivial on 8 TB).

## MLX residency

**`MlxExpertResidency`** is the hard backend-specific piece. mlx_lm stores each
layer's experts stacked in single quantized tensors, so "install expert 137"
means writing its rows into the stacked `weight`/`scales`/`biases` arrays of the
three switch projections. Two candidate mechanisms, to be decided empirically in
phase 3:

1. in-place row assignment on the live arrays, or
2. slot pool: stacked tensor as fixed slots + `ExpertKey → slot` indirection
   (llama.cpp-PoC style).

The protocol doesn't care which; the choice stays inside the backend. Eviction
is bookkeeping-only — a slot becomes free, bytes are not zeroed.

**Dev-machine validation trick:** the 4-bit model fits in 32 GB, so streaming is
exercised by configuring a starved resident budget. EXACT INFERENCE gives the
acceptance test: identical greedy tokens with full vs. starved residency, only
slower.

## Config and composition root

**`PipelineConfig`** (`config/pipeline.py`, pydantic, parsed with the existing
`read_and_validate_toml`):

```toml
[model]              # id, backend = "mlx_metal"
[generation]         # max_tokens; greedy is the only mode
[tracing]            # optional table; absent = no recorder/sink wired
[[tracing.targets]]  # embeds existing TargetLayerConfig entries
[streaming]          # optional table; absent = AllResidentProvider
                     # store_path, resident_budget, prefetch settings
```

**`main.py`** at repo root is the composition root (the one place concretes
meet): maps `backend = "mlx_metal"` to concrete loader/runner/residency, builds
store/scheduler/recorder per config presence, hands everything to
`InferencePipeline`.

## Error handling

- Wrong target `count`, overlapping targets: existing loud resolve-time failures
  retained.
- Manifest/model fingerprint mismatch: refuse to build the pipeline.
- Store read failure on a `DEMAND` read: fatal — correctness path, no fallback.
- Store read failure on `PREFETCH`: logged, counted (wasted-read metrics), never
  fatal.
- Predictor deadline expiry or error: fall back to heuristic; never blocks the
  forward.
- Config validation errors: pydantic, at startup, before any model load.

## Testing

**pytest, Linux-runnable, no weights** (also converts the pure-layer tests
CLAUDE.md lists as outstanding):

- Scheduler contract tests: fake store/residency/predictor + scripted routing —
  demand preemption, LRU + budget, predicted-soon eviction protection, prefetch
  back-off, stall accounting.
- Generation-loop lifecycle: fake runner/recorder — step ordering, optional
  tracing, metrics.
- Packed store round-trip: writer → reader on synthetic blobs; offset/alignment;
  manifest compat rejection.
- `PipelineConfig` validation; `engine/layer_resolution.py` matching;
  `datamodel/arrow.py` schema derivation; sink batching against a fake schema.

**Host-side integration** (`/run-host-test`):

- Existing two scripts stay; `mlx_all_router_layers.py` shrinks to config +
  assertions once its machinery lives in source.
- New `mlx_streaming_pipeline.py`: real pipeline, starved budget, asserts
  token-exactness vs. full residency — the EXACT INFERENCE gate for streaming.

## Build phases

Each phase ends green (pytest + host integration where applicable).

1. **Pipeline extraction** — core protocols, engine loop, `PipelineConfig`,
   `main.py`, `AllResidentProvider`; test script shrinks against it. No behavior
   change; exactness re-verified.
2. **Packed store** — writer, manifest, reader, MLX extractor; round-trip tests;
   converted store for Qwen3.6-35B on disk.
3. **Residency + demand path** — `MlxExpertResidency`; wrapper calls `acquire`;
   scheduler with LRU/budget but *no prefetch*; starved-budget exactness test
   passes (slow is fine).
4. **Prefetch + predictor** — heuristic predictor, prefetch queue, back-off,
   metrics; benchmark hit rate and tok/s vs. phase 3.

## Open questions (deliberately deferred to their phase)

- MLX install mechanism: in-place row writes vs. slot pool (phase 3, empirical).
- Prefetch horizon Δ and back-off thresholds: initial values picked in phase 4
  from measured bandwidth vs. per-layer compute time.
- Whether `ExpertPayload` needs zero-copy buffers (memoryview vs. bytes) —
  measure before optimizing.
