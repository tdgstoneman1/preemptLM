# preempt

**preempt** is a standalone, open-source MoE inference engine with a **trainable expert prefetcher**. Its goal: run huge Mixture-of-Experts models whose weights vastly exceed available RAM by streaming expert weights from SSD, and eliminate the resulting disk bottleneck by *predicting* which experts the router will select and prefetching them ahead of demand.

The project's differentiator versus llama.cpp, ds4, and colibri: the prefetcher is a **first-class, user-trainable, swappable component**. Users can capture routing traces from their own workloads, fine-tune their own expert predictor, and hot-swap it into the engine. This is the core product; the inference engine is infrastructure around it.

**Central invariant: EXACT INFERENCE, ADAPTIVE SPEED.** The router's decisions are always authoritative. Predictions only affect *when* expert weights are staged into memory — never *which tokens are produced*. Wrong predictions cost wasted disk bandwidth, never correctness or quality. Never silently change model precision or router semantics; any precision compromise must be explicit and validated.

## The pitch

Running a multi-trillion-parameter model on a laptop sounds impossible only if you assume all parameters must reside in memory at once. Mixture-of-Experts models break that assumption.

A sparse MoE model activates only a small fraction of its parameters per token. Kimi K3, for example, has ~2.8 trillion total parameters, but its router selects only 16 of 896 experts per MoE layer — roughly **104B parameters active per token**, under 4% of the total. The rest sit idle.

At MXFP4 precision (~4.25 bits per parameter), the full model occupies 1.4 TB — too large for any consumer machine's RAM, but trivial for a modern laptop SSD (this project's dev machine has 8 TB). The crucial number is the **per-token working set**: the weights actually touched to produce one token. The dense backbone (attention, embeddings, shared experts) at int8/MXFP8 is on the order of 10–20 GB and stays permanently resident. The routed experts needed per token add up to roughly **25 GB of weight traffic per token** in the worst case (a fully cold cache):

$$
\text{expert bytes/token} = \underbrace{16\ \text{experts}}_{\text{top-}k} \times \underbrace{L_{\text{MoE}}\ \text{layers}}_{\text{MoE layer count}} \times \underbrace{B_{\text{expert}}}_{\text{bytes per expert @ MXFP4}} \approx 25\ \text{GB}
$$

That reframes the entire problem: **the model doesn't need to fit in memory — it needs to be placed.** VRAM, RAM, and SSD become one managed memory hierarchy. Weights are not resident state; they are data staged across the hierarchy exactly when the router proves they're needed. This is the same bet colibri made running a 744B MoE on 25 GB RAM (~11 GB/token expert traffic, ~4 GB/s consumer NVMe, ~0.4 tok/s cold). K3 merely scales the same arithmetic by ~2–3×.

The catch: a naive implementation reads ~25 GB from disk for every token — at ~4–6 GB/s SSD bandwidth, that's several seconds per token:

$$
\text{cold decode speed} \approx \frac{\text{SSD bandwidth}}{\text{expert bytes/token}} = \frac{4\ \text{GB/s}}{25\ \text{GB/token}} \approx 0.16\ \text{tok/s}
$$

Three facts make it practical:

1. **Routing has structure.** Empirically, expert selection is highly predictable from earlier layers and consecutive tokens (colibri measures ~72% top-8 predictability one layer ahead on GLM-5.2; academic work reports 85–97% accuracy for learned predictors). Structure is cacheable and predictable.
2. **Workloads are repetitive.** A small LRU cache of hot experts absorbs a large share of recurring routing; the engine gets faster the more it's used on a given workload.
3. **Compute and I/O can overlap.** While resident experts compute on the GPU, the disk can be streaming predicted experts for upcoming layers concurrently. If prefetch + cache absorbs a fraction $h$ of the per-token traffic, effective decode scales accordingly:

$$
\text{effective tok/s} \approx \frac{\text{SSD bandwidth}}{(1 - h)\times \text{expert bytes/token}}
$$

At $h = 0.8$ — an ambitious but plausible target for learned prefetch + LRU — that's ~0.8 tok/s; enough for interactive use, and the number this project exists to push upward.

**This project exists to close the gap between that naive ~25 GB/token floor and near-resident effective bandwidth**, by replacing heuristic lookahead with a *learned, user-trainable* predictor that hides the read latency behind compute.

## Primary author context

- Solo developer, Python-native, working on an **M1 Max MacBook Pro (32 GB unified memory, 8 TB SSD)**.
- Owns a working PyTorch implementation of **dense Qwen3 (all sizes)** — included in `preempt/models/qwen3/`. Its role is intentionally undecided; candidate uses include:
  - **Validation oracle**: token-exactness comparison against engine output (teacher-forced).
  - **Expert predictor base**: a fine-tuned small Qwen3 variant (e.g., 0.4B) as the transformer-family predictor, *if* empirical tests show it beats cheaper baselines (heuristic, linear probe) on routing-prediction accuracy per unit cost.
  - Evaluate empirically before committing to either role; the architecture must support both without entanglement.
- Docker-comfortable for production/training jobs; will rent cloud GPU for predictor fine-tuning when needed.
- Apple Silicon is the **development platform of convenience** — fast SSD, unified memory, and MLX make it the fastest place to build and iterate on the PoC. This is **not** an MLX-centric project: the engine's core logic, scheduler, predictor contract, and training API are platform-agnostic by design, and future backends (llama.cpp/ggml, CUDA) are expected.

## Hard constraints (do not violate)

1. **Correctness path never waits behind a guess.** Demand reads (router just picked an uncached expert) always preempt prefetch reads.
2. **Backend and platform APIs never leak above their layer.** The scheduler and core logic must not import MLX, PyTorch, Metal, or file-format-specific code.
3. **Predictor artifacts are compatibility-checked.** A predictor carries a manifest (model fingerprint, feature schema version, expert topology). Incompatible artifacts are rejected and the engine falls back to the heuristic predictor — never silently misbehave.
4. **Benchmark-driven.** Every optimization claim is measured. Maintain metrics: tok/s, expert cache hit rate, prefetch precision/recall vs. actual router choices, wasted bytes, demand-read stall time, per-layer time breakdown.
5. **No premature generalization.** One model architecture, one quant scheme, greedy decoding for v1. No serving API, no speculative decoding, no multi-model support until the research question is answered.

## Current architecture

### Module ownership map

| Module                          | Owns                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        | Must NOT import                                       |
| ------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------- |
| `preempt/core/`                 | Shared vocabulary:`enums.py` (`ParquetCompressionCodecs`), `protocols.py` (`IExpertCache` / `IExpertLoader` / `ITokenizer` / `IModelRunner` capability protocols).                              | MLX, PyTorch, backends, model adapters, training libs |
| `preempt/datamodel/`            | Shared data classes: `arrow.py` (attrs↔Arrow field-metadata machinery) and `tracing/` (`TraceRunContext`, `TraceStepContext`, `ExpertRoutingEvent`, `EventMetadata`, `LayerIdentifiers`). Engine-side records (`ModelManifest`, `LoadRequest`, `PredictionRequest`, …) are not written yet. `identity.py` (`ExpertKey`, `TensorSpec`)                                                                                                                                                                                                           | Backends, trainers, ORMs, concrete storage            |
| `preempt/config/`               | **Pydantic config models parsed from TOML.** `target_layers.py` (`TargetLayerConfig` / `TargetLayerSpec` / `TargetLayerSearchParams`). Concrete config files live in `configs/*.toml`.                                                                                                                                                                                                                                                                                                                                                      | Backends, models, engine internals                    |
| `preempt/expert_bank/`          | Operations and interfaces for expert bank on disk:`banks.py` (`PreadExpertBank` — one `pread` per expert, `F_NOCACHE` page-cache bypass, `MmapExpertBank` – reads expert blobs from memory-mapped `experts.bin` ), `manifest.py` (`ExpertBankManifest`, model MoE spec + blob index), `writer.py` (`ExpertBankWriter`), `blob.py` (helper functions), `encoding.py` (`PayloadEncoding` + `parse_payload_encoding_tag`)                                                                                                                                                                                                                                                         | Backends, engine internals                            |
| `preempt/engine/`               | **Decisions, inference, and coordination.** `layer_resolution.py` (platform-agnostic layer resolution — `LayerCandidate`, `match_target_layers`, `resolve_target_layers`), `recorder.py` (`BaseEventRecorder`, the trace-lifecycle ABC). The streaming scheduler is now written: `DiskBackedExpertLoader` (`scheduler.py`), `ExpertCacheManager` (`expert_cache.py`, LFRU eviction), chunked `generation.py`, `pipeline.py`, `BaseEventSink` async ABC + `ParquetEventSink` + `JsonlFileEventSink` (`sinks.py`), and `metrics.py`.                                             | `backends/*` concrete classes, model-specific code  |
| `preempt/backends/mlx_metal/`   | Platform-specific execution and instrumentation:`layer_discovery.py` (enumerate MLX modules), `instrument.py` (`mlx_instrument_model` via `update_modules`/`tree_unflatten`, plus `mlx_strip_instrumented_expert_weights`), `recorder.py` (`MoERecorder`), `types.py` (`MlxWrapperFactory` alias), `instrumented/` (forked-forward capture wrappers). Streaming/residency I/O now written: `expert_kernel.py` (the expert-major kernel), `residency.py` (`MlxExpertCache`), lazy expert load in `loader.py`. | Scheduler/policy logic                                |
| `preempt/predictors/`           | Predictor runtimes behind one contract: heuristic baseline, linear probe, transformer predictor later.**Empty.**                                                                                                                                                                                                                                                                                                                                                                                                                                      | Engine internals                                      |
| `preempt/training/`             | Trace → dataset → fine-tune → evaluate → export artifact.**Empty.**                                                                                                                                                                                                                                                                                                                                                                                                                                                                               | Engine runtime imports                                |
| `preempt/utils/`                | Helpers.`attrs_utils.py` (recursive attrs field walking), `io_utils.py` (`read_and_validate_toml`), `torch_utils.py`.                                                                                                                                                                                                                                                                                                                                                                                                                               | —                                                    |

**No `__init__.py` re-exports.** Every package `__init__.py` under `preempt/` is empty; modules are always imported by full path (`from preempt.engine.layer_resolution import LayerCandidate`). Keep it that way — it makes dependency direction visible at every import site.

### Dependency direction

- **`core/`** is the root — nothing in the project imports *into* it; it imports nothing internal.
- **`datamodel/`** imports only `core/`.
- **`engine/`**, **`models/`**, **`predictors/`**, **`training/`** may all import `core/` + `datamodel/`, but not each other's concrete implementations.
- **`backends/mlx_metal/`** implements the protocols defined in `core/`:
  - Nothing in the project imports it directly except the composition root (`main.py` / CLI).
  - The scheduler talks to it only through the injected interface.
- **Composition root** (`main.py` / CLI entry) is the only place that wires concrete backends, model adapters, and predictors together.
- This rule is enforceable later with an import-linter; keep new code compliant from the start.

### Router tracing pipeline (the "lens") — the main thing built so far

This is currently the most developed subsystem and the one most new work will touch. It captures a MoE model's real router decisions and writes them as Parquet training data. Six stages, each swappable, only two of which are platform-specific:

1. **Declare targets in TOML** — `configs/*.toml` → `TargetLayerConfig` (`preempt/config/target_layers.py`). A target names a layer set by `layer_class`, `layer_path_glob`, and/or `block_idx`, plus an optional `count` that asserts how many layers must match. Instrumentation targets are **configuration, never hardcoded paths**; a wrong `count` fails loudly at resolve time instead of silently tracing the wrong layers.
2. **Resolve targets, platform-agnostically** — `preempt/engine/layer_resolution.py`. Pure functions over an iterable of `LayerCandidate(layer_path, layer_class, block_idx)`. Knows nothing about MLX or torch.
3. **Enumerate candidates, per backend** — `preempt/backends/mlx_metal/layer_discovery.py` walks `model.named_modules()` and parses the block index out of dotted paths. This is the only MLX-aware half of target resolution.
4. **Replace modules with capture wrappers** — `instrument.py`'s `mlx_instrument_model()` swaps resolved modules via `update_modules(tree_unflatten(...))`, using an injected `MlxWrapperFactory` (aliased in `types.py`). Wrappers live in `backends/mlx_metal/instrumented/`, one module per upstream block type (`qwen3_x_moe.py` → `InstrumentedQwen3_xMoE`, built by `make_qwen3_x_moe_wrapper_factory`).
5. **Record into a lazy buffer** — `MoERecorder` (`backends/mlx_metal/recorder.py`) implements `BaseEventRecorder` (`engine/recorder.py`). Lifecycle is `start_step(step_context)` → many `capture(...)` → `flush(sink)` → `end_step()`.
6. **Write through an async sink** — `BaseEventSink` / `ParquetEventSink` / `JsonlFileEventSink` (`engine/sinks.py`). Async context managers; all blocking I/O goes through `asyncio.to_thread`, writes serialized by an `asyncio.Lock`, batched so one row group isn't created per event.

Three conventions in this pipeline matter more than the code itself:

- **Wrappers fork the upstream forward — verbatim router + tail, re-derived expert half when streaming.** `InstrumentedQwen3_xMoE.__call__` copies `mlx_lm`'s `Qwen3NextSparseMoeBlock.__call__` with a `capture(...)` inserted after top-k selection — not a subclass, not a hook (MLX has no `register_forward_hook`). On the **non-streaming** path it is a line-for-line copy that delegates to `self.inner` for all weights. On the **streaming** path only the router half (gate → softmax → top-k → renorm) and the tail stay verbatim; the **expert-application half is deliberately re-derived** as an expert-major loop (`sequential_run_selected_experts`) instead of delegating to `inner.switch_mlp`, because experts are streamed one at a time rather than held in a stacked `gather_qmm` tensor — see `.claude/docs/expert-major-precision-decision.md`. **Cite the upstream source URL and line in the docstring** (the existing one does) so the verbatim halves can be re-diffed when `mlx_lm` moves, and mark which parts are re-derived. A silently drifted fork of the verbatim halves is the highest-risk failure mode here: it changes model output, violating EXACT INFERENCE.
- **Never `mx.eval` in the capture path.** The recorder buffers *unevaluated* `mx.array`s and defers a single batched `mx.eval` to flush time. Forcing evaluation inside the forward pass would serialize the graph and make every measurement meaningless. Conversion to Python (`.tolist()`) and the batch×token explosion into per-token `ExpertRoutingEvent`s also happen at flush.
- **Arrow schema is declared on the attrs fields, not written by hand.** `preempt/datamodel/arrow.py` attaches `arrow_metadata(pa_type, serializer=…, nullable=…)` to each attrs field; `ExpertRoutingEvent.arrow_schema()` and `.as_arrow_record()` derive schema and row by walking those fields, flattening nested attrs classes (`TraceRunContext`, `TraceStepContext`, `EventMetadata`, `LayerIdentifiers`) into a flat Parquet row. **Add a field by adding an attrs field with `arrow_metadata` — never by editing a schema literal.** Bump `EXPERT_ROUTING_SCHEMA_VERSION` when the layout changes; it is stamped into the file metadata alongside `preempt.event_type`.

### Key design contracts (already decided)

- **Expert identity is fully qualified**: model fingerprint / layer index / expert index / weight variant. Bare global integers never leak into traces or artifacts.
- **Deadline-aware, async-capable predictor API** from day one, even if v1 executes synchronously: `submit(request) → handle`, then `poll_until(deadline)`; on expiry, fall back to heuristic. Same pattern for expert loads: `LoadRequest` with priority + completion handle.
- **Serializable records everywhere**: prediction requests/responses, trace records, artifact manifests are plain data with explicit schema versions. This is what lets a llama.cpp backend, a sidecar process, or a cloud trainer plug in later without rewriting.
- **Trace format is the most important public API**: features + actual router outputs + cache/timing context per layer/step. Store features rather than raw prompts (privacy + size). Append-only chunked files (Arrow/Parquet) with per-session manifests. **This now has a concrete implementation** — `ExpertRoutingEvent` in `preempt/datamodel/tracing/expert_routing.py` is the canonical record; treat changes to it as public-API changes. Note the record's own docstring contract: `expert_ids`/`expert_weights` are the *post-normalization* top-k selection, while `gate_logits` (optional, training-only) is the *full* `num_routed_experts`-wide distribution. Run/step provenance is carried by the nested `TraceRunContext` (run id, model fingerprint, architecture, revision, lens version) and `TraceStepContext` (sequence id, token index, token id) rather than by loose columns.
- **Cache policy**: per-layer LRU baseline + predictor-informed eviction (a predicted-soon expert is protected even if cold). The engine is disk-bound, so the I/O scheduler also adapts prefetch depth to measured disk bandwidth vs. per-layer compute time, and backs off aggressiveness when misprediction wastes reads.

### Scheduler behavior (v1 spec)

1. Merge demand reads and prefetch reads into one queue; demand preempts prefetch.
2. Prefetch horizon adapts to measured bandwidth vs. per-layer compute time.
3. On misprediction-heavy workloads, back off prefetch aggressiveness so wasted reads never starve demand reads.
4. Emit metrics on every decision (hit/miss/precision/wasted bytes/stall time).

## v1 scope (cut hard)

- **Rapid iteration and prototyping** are a **top priority**. Once these features are more mature much later on, the focus will shift toward building a comprehensive LLM inference framework.
- **One small MoE** as the first streaming target. The concrete working target is **Qwen3.6-35B-A3B** (`unsloth/Qwen3.6-35B-A3B-UD-MLX-4bit`), whose MLX implementation is `mlx_lm`'s `Qwen3NextSparseMoeBlock` — 40 MoE layers, 256 experts, top-k 8. The in-repo Qwen3 PyTorch code is dense-only and **is not used for this model**; we currently instrument `mlx_lm`'s implementation rather than porting an MoE variant into `preempt/models/`.
- MLX execution backend on Apple Silicon; int4 experts / int8 dense quantization initially.
- LRU + heuristic lookahead baseline; learned predictor integrated behind the same contract afterward.
- **Fine-tuning API included in v1.** Since training scripts must be written anyway, the trace → dataset → fine-tune → export loop ships as a first-class Python package from the start (LoRA on the chosen predictor base; linear probe as the cheap baseline). This is the feature that makes the project forkable rather than just notable — do not defer it.
- Greedy decoding, CLI chat, trace capture tooling.
- **Launch benchmark story**: tok/s vs. LRU and vs. heuristic-lookahead baselines on a 32 GB Mac, predictor overhead <2%, zero quality change.

Explicitly deferred: serving/API, speculative decoding, KV persistence, dual-SSD mirroring, grammar forcing, CUDA, plugin ABI.

## Long-term roadmap (context, not v1 scope)

- **Kimi K3** (~2.8T-param MoE, ~104B activated, MXFP4 ≈ 1.4 TB weights) is the aspirational target. Wait for community maturation (GGUF support, KDA/MLA kernels in reference runtimes) before committing engine support. Its hybrid attention (KDA recurrent state + gated MLA compressed KV, plus AttnRes) means two cache types and earlier-layer state retention — a `ModelAdapter`-level project, not a scheduler rewrite.
- **llama.cpp backend (future)**: after the MLX PoC proves predictor value, port the policy to a llama.cpp fork with MoE disk-paging hooks (async expert staging, priority + cancellation, router observation, completion events). Keep learned policy, training API, and artifacts in this repo; optionally upstream only *generic* staging/telemetry primitives to llama.cpp later as narrowly scoped PRs. The fork/bridge is v2 work — do not scaffold for it now beyond keeping the backend protocol residency-oriented.
- **Open-core business angle**: local trace capture and self-hosted training stay first-class; a future hosted service would sit exactly at the training/artifact-registry boundary. Nothing in v1 may depend on a network service.

## Rules and Guidelines

1. **Avoid unnecessary boilerplate and KISS:**

   To stay on track with our short target development timeline, we need to avoid adding bloat or fancy features that will take time to integrate, test, and debug. Local LLM inference is notoriously *fraught* with quirks, silent errors, and countless other unique challenges. The more moving parts there are, the more we will inevitably get bogged down with debugging. So keep it simple, stupid!
2. **Avoid planned rewrites**

   Minimizing boilerplate and prioritizing simplicity **does not** mean cutting corners. Even if it means more work upfront, we invest in stable boundaries early, keep implementations provisional, and minimize tight coupling so that swapping or adding a backend is substitution—never an overhaul!

   - **Async-first interfaces**: scheduler, predictor, and storage contracts are  async  from day one—overlap (disk ∥ compute, predictor ∥ decode) is structural, and retrofitting  await  into a sync call stack is a rewrite, not a refactor.
3. **Python guidelines and coding styl**

- Use `typing.Protocol` for contracts; explicit dependency injection at the composition root.
- **Use `attrs` (`@attrs.define`) for data classes** — chosen over builtin `dataclasses.dataclass` for speed and flexibility.
- **Pydantic where heavier validation is needed** — predictor artifact manifests, config parsing, or anything crossing a process/file boundary with compatibility rules.
- Plain `attrs` classes for hot-path internal records where Pydantic overhead would matter.
- **`pydantic.BaseModel` vs. `attrs.define`, decided by trust boundary, not convenience:**

  - **`pydantic.BaseModel`** for configs and any external/user-provided data that needs validation — TOML-parsed config, predictor artifact manifests, anything crossing a process/file boundary. Every field annotated with `pydantic.Field(...)`.
  - **`attrs` (`@attrs.define`)** for internal dataclasses — records passed between trusted in-process code, hot-path structures — since it has much lower overhead than Pydantic and runs faster. Every field annotated with `attrs.field(...)`.
- **Typing**

  - All classes should have annotations
  - All function input and output signatures should have type hints
  - Use Python's built-in `list`, `tuple`, `dict` and `|` instead of the equivalent type from the `typing` module (e.g. `list[int]` instead of `typing.List[int]`)
  - Import types like `Generator`, `Sequence`, `Awaitable`, etc. from `collections.abc` instead of from `typing`
- **Import ordering**

  - Place imports from `typing` then `collections.abc` first before any others (except for `__future__.annotations`).
  - Place local `preempt` imports last.
  - Imports should be grouped logically and/or by parent module, and groups should be separated by empty lines for clarity. Example:

    ```python
    from __future__ import annotations

    from typing import Optional
    from collections.abc import Generator

    import torch
    import torch.nn as nn

    from .qwen3_config import Qwen3Config
    from .qwen3_block import GQATransformerBlock, RMSNorm

    from ..kv_cache import KVCache
    from ..ops.rope import compute_rope_params
    ```

- **Docstrings**
  Use the 'NumPy style' to format docstrings. Inline code should be wrapped in single backticks, or in rare cases, you may use triple backticks and a language identifer for fenced code blocks--*but don't use double backticks*. Here's an example of how docstrings should look:

  ```python
  def compose_dataclass_from_protos(
      protocols: Sequence[type],
      *,
      cls_name: str = "NewDataclass",
      defaults: Optional[dict[str, Any]] = None,
      kw_only: bool = True,
      slots: bool = False,
  ) -> type:
      """
      Returns a new `dataclasses.dataclass` from the union of all field annotations
      for the classes given in `protocols`.

      Parameters
      ----------
      protocols : Sequence[type]
          Sequence of protocol classes
      cls_name : str
          The name of the new dataclass
      defaults : Optional[dict[str, Any]]
          Optional field defaults (not otherwise inferred if none provided),
          by default None
      kw_only : bool
          Whether the new dataclass's constructor should accept keyword arguments only,
          by default True
      slots : bool
          Whether the new dataclass should use slots, by default False

      Returns
      -------
      type
          A new concrete `dataclasses.dataclass`


      Raises
      ------
      TypeError
          If 2 or more protocols have fields with the same name but different types.
      """
  ```

- **Testing and review guidelines**

  - **Reference-oracle discipline**: before optimizing, every supported model's forward output and router choices must match a reference implementation (e.g. `transformers` or the in-repo PyTorch Qwen3 assuming updates are made to support MoE architectures) token-for-token under teacher-forcing. *But build the test harness first.*
  - **Contract tests with fakes**: scheduler priority, cancellation, deadline, and fallback behavior must be testable with a fake disk, fake executor, and fake predictor — no model weights or Metal required.
  - **Measure before optimizing**: `iobench`-style disk profiling and per-turn telemetry exist before any cleverness lands.
  - **Framework:** We use `pytest` for all testing.
  - **Current state of `tests/` (read this before assuming a suite exists):** there are **no `pytest` tests yet**. `tests/integration/` holds hand-run smoke *scripts* — `mlx_layer_search.py` (resolve a TOML target against a real model and print matches) and `mlx_single_router_layer.py` (wrap one router layer, run one token, assert exactly one Parquet row round-trips). They are `argparse` CLIs with `main()`, not `test_*.py`. They require macOS + MLX + real weights, so they cannot execute *in* a Linux sandbox — but a sandboxed agent can now run them *on the host* through the host-bridge MCP server (below). Converting the pure layers (`engine/layer_resolution.py` matching, `datamodel/arrow.py` schema derivation, `config/target_layers.py` validation, sink batching against a fake schema) into real `pytest` tests is outstanding work — all four are already free of MLX imports and testable without weights.
  - **Running MLX tests from a sandbox — the host-bridge MCP server.** `scripts/host_bridge_mcp.py` is a small MCP server that runs **on the macOS host** and executes anything under `tests/` there, so sandboxed agents can exercise MLX/Metal code they cannot run locally. Use the `/run-host-test` slash command, which encodes the whole workflow.

    Start it on the host (not from a sandbox — it cannot be started from inside one):

    ```bash
    ./scripts/setup.sh                  # first time only: makes the launcher executable
    ./scripts/run_host_bridge_mcp.sh    # bootstrap + serve; leave running in its own tab
    ```

    The launcher is idempotent: it ensures `.venv` exists, creates `.host-bridge-mcp-token` (0600, stable across restarts), merges a `hostrun` entry into `.mcp.json` **without clobbering other servers**, adds the generated files to `.git/info/exclude`, and runs `sbx policy allow network localhost:8765`. That policy grant does not survive a host reboot.

    Key properties to know before using it:

    - **Only `tests/` is executable.** This is a scoping control, not a security boundary — an agent that can write into `tests/` through the mount can run what it wrote.
    - **Runs are jobs, not blocking calls**, because HTTP MCP clients enforce a 60s time-to-first-byte timer and a 35B model load takes ~60–90s. A run tool returns a `run_id` and a `log_path`; poll for the result.
    - **Read `log_path` directly** rather than polling for output. The repo is bind-mounted at the *same absolute path* on both sides, so the log is readable live from the sandbox and holds complete output; the returned `tail` is truncated.
    - **Write artifacts to `${run_dir}`.** `mlx_single_router_layer.py` defaults `--output` to a `TemporaryDirectory` that is deleted on exit, so anything not directed at `${run_dir}` vanishes before it can be inspected. `${model}` expands to the host's default model id.
    - **One run at a time**, since a single model load consumes most of the host's memory budget.
    - The server runs via `uv run --script` against its PEP 723 header, so its dependencies **never enter the project `.venv`**.
  - Write pure functions that are easily testable with input fixtures in pytest.
  - Test edge cases for dependency injection and state immutability.
  - Provide descriptive docstrings for exported functions (purpose, params, return, throws). Comment the "why," not the "what."
- **Running pytest in a Linux sandbox**
  The host machine is macOS, so the workspace `.venv` contains macOS wheels and its
  interpreter cannot run in a Linux sandbox (agent sandboxes, containers, CI images). Running
  `pytest` there needs a throwaway Linux environment.

  **Never touch the workspace `.venv` from a Linux sandbox.** It belongs to the host and its
  wheels are platform-specific — installing into it, upgrading it, or recreating it corrupts
  the host's working environment. Every command below targets a separate interpreter
  explicitly, and `.venv/`, `pyproject.toml` and `uv.lock` are left unmodified.

  Create the throwaway environment **outside the repo** so it can never be mistaken for
  `.venv` and never lands in `git status`:

  ```bash
  uv venv /home/agent/.venvs/preempt-linux --python 3.12
  uv pip install -r pyproject.toml --python /home/agent/.venvs/preempt-linux/bin/python
  ```

  `uv pip install -r pyproject.toml` reads the `[project].dependencies` list directly. Do
  **not** try `uv pip install -e .` — editable installs fail here on setuptools package
  discovery. Because nothing is installed as a package, `preempt` is imported off
  `PYTHONPATH` instead:

  ```bash
  PYTHONPATH=/Users/davidstoneman/venvs/preempt \
    /home/agent/.venvs/preempt-linux/bin/pytest -m "not slow and not integration" -q
  ```

Notes:

- Always invoke the venv's `pytest` by absolute path, or pass `--python <venv>/bin/python`
  to `uv`. Activating the environment is not enough — a bare `pytest`/`uv pip install` can
  resolve against `.venv` and silently write to it.
- The Linux env can only exercise MLX-free code. Anything under `backends/mlx_metal/`, and both
  scripts in `tests/integration/`, need the macOS host — run those through the host-bridge MCP
  server (`/run-host-test`) rather than trying to make them work in the sandbox. Running the same
  MLX-free tests in both places is a useful cross-check.
- Subagents run in the same sandbox and hit the same constraint: give them the absolute
  `pytest` path and the `PYTHONPATH` prefix in their brief, or they will reach for `.venv`.

## Reference projects (for study, not dependency)

- [**colibri**](https://github.com/JustVugg/colibri): 744B MoE on 25 GB RAM, single C file, per-expert contiguous `pread` reads, per-layer LRU, heuristic router-lookahead prefetch, token-exact validation against transformers. Model for the streaming/storage layer and for README/benchmark honesty.
- [**ds4**](https://github.com/antirez/ds4): model-specific C/Metal engine, asymmetric quantization (aggressive on experts only), disk-backed KV. Model for narrow-scope engines and native Metal organization.
- [**waste**](https://github.com/sqliteai/waste): "Weight-Aware Streaming Tensor Engine" — embeddable C engine running the full 2.78T Kimi K3 on a 64 GB MacBook at ~0.45–0.62 tok/s. Closest prior art to preempt's thesis: trunk resident, one aligned read per expert, bounded LRU expert cache, reads overlapped with compute, and a **lookahead router that prefetches the next layer's experts while the real router stays authoritative** — the same "changes timing, not results" invariant. Model for container layout and for measurement honesty.
- [**kimi-k3-in-c**](https://github.com/FareedKhan-dev/kimi-k3-in-c): 2.78T K3 in portable C99 on one CPU in 8.24 GB peak RSS, ~33 s/token. No prefetching cleverness — it is the *naive streaming floor* this project exists to beat, and a readable reference for K3's architecture.
- [**llama.cpp**](https://github.com/ggml-org/llama.cpp): target of the future backend; its MoE disk-paging PoC (slot pool + `pread` sidecar + `MTLSharedEvent`) is the integration substrate. Not a dependency of v1.

### Local clones of the reference repos

All five of these are cloned read-only for study at **`/home/agent/repos/`** (`colibri`, `ds4`, `kimi-k3-in-c`, `llama.cpp`, `warp`). Notes before using them:

- **Sandbox-local, not part of this repo.** The path exists inside the Linux agent sandbox, is outside the git worktree, and nothing in `preempt/` may import, vendor, or build against it. Clones drift — re-check `git log -1` before citing a line number.
- **Read, never copy.** These are C engines under their own licences (Apache-2.0 / MIT etc.). Borrow *designs and measurements*; do not paste code.
- Highest-value docs for current work:
  - `warp/docs/ENGINE.md`, `FORMAT.md`, `EFFICIENCY.md` — streaming design, on-disk expert layout, per-stage cost breakdown; `waste/docs/LEARNED.md` is an append-only, dated measurement log that keeps its own refuted hypotheses (a good model for our benchmark discipline).
  - `colibri/docs/routing-telemetry.md` — its `.coli_usage` / `ROUTE_TRACE` routing-trace format, the nearest existing analogue to `ExpertRoutingEvent`; `colibri/docs/CACHE_ROUTE.md` and `tuning.md` — cache-aware routing and its learning cache. Note `CACHE_ROUTE` **changes which experts run**; that is exactly the line preempt does not cross (EXACT INFERENCE), so read it as a contrast, not a template.
  - `kimi-k3-in-c/docs/ARCHITECTURE.md` and `docs/kimi-k3-tech-report.pdf` — K3 architecture background for the long-term roadmap.
  - `ds4/README.md` + `MODEL_CARD.md` — asymmetric expert-only quantization and SSD-streaming behaviour on Metal.
  - `llama.cpp/tools/server/README.md` (`-ot/--override-tensor`, `-ncmoe/--n-cpu-moe`) — upstream's *static* CPU/GPU expert-placement flags; today it's "pin the first N MoE layers to CPU," not disk-streaming or prediction. Useful as the substrate/vocabulary the future backend would extend, and as a baseline to benchmark against, not as a working disk-paging PoC — no such PoC exists in this clone as of `git log -1` (2026-08-07); re-verify before citing specifics.
