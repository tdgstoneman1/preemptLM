# Naming review — instrumentation & tracing path

Review of `backends/mlx_metal/` → `engine/recorder.py` → `core/sinks.py` → Parquet, focused on
names that will confuse a new reader or that disagree with the data they hold.

Ordered by **cost of delay**, not effort. Tier 1 items are already written into Parquet output and
get more expensive with every trace captured.

---

## Tier 1 — baked into persisted output

* [X] 1.1 `gate_logits` does not contain logits

**`backends/mlx_metal/instrumented/qwen3_next_moe.py:53-69`**

```python
gates = self.inner.gate(x)                          # logits
gates = mx.softmax(gates, axis=-1, precise=True)    # ← now a probability distribution
...
self.accumulator.capture(
    ...
    gate_logits=gates if self.capture_gate_logits else None,   # ← mislabelled
)
```

The captured array is post-softmax. `ExpertRoutingEvent` then documents it as:

> `gate_logits` corresponds to full-size logits array (used for training)

A trainer that trusts the field name and applies `softmax` / `log_softmax` again is silently wrong,
and nothing downstream can detect it — a softmaxed-twice distribution is still a valid distribution.

**Two options:**

|                             | change                                                     | effect                                                                                                   |
| --------------------------- | ---------------------------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| **(a)** *preferred* | capture`self.inner.gate(x)` **before** the softmax | field name becomes true; logits are the more useful training target and probabilities remain recoverable |
| (b)                         | rename field to`gate_probs` / `router_probs`           | honest, but discards information you'd likely want later                                                 |

Touches `qwen3_next_moe.py`, `datamodel/tracing/expert_routing.py:88`, `backends/mlx_metal/recorder.py`.

> Note the sibling field `expert_weights` **is** correctly named and documented — it's the top-k
> scores after optional `norm_topk_prob` renormalization. Only the full-width field is wrong.

---

* [X] 1.2 `instrumentation_version` → `instrumentation_version`

**`datamodel/tracing/context.py:14`**

Nothing in this project is a lens — no projection into vocabulary space. The field means *"version
of the capture machinery that produced this trace."* Since it's a per-run provenance field it lands
in every Parquet row.

Remaining "lens" language to purge at the same time:

- `core/sinks.py:206` — `"""Append lens events to a UTF-8 JSON Lines file."""`
- `core/sinks.py:212` — `"...so multiple lens wrappers cannot interleave bytes"`
- `backends/mlx_metal/recorder.py:75` — `RuntimeError("No lens capture step currently active.")`
- `tests/integration/mlx_single_router_lens.py` — filename

---

* [X] 1.3 `"expert_router_event"` → `"expert_routing_event"`

**`datamodel/tracing/expert_routing.py:16`**

The class is `ExpertRoutingEvent`; the constant value says `router`. It's written into Parquet
schema metadata as `preempt.event_type`, so it's the string any downstream reader dispatches on.

---

## Tier 2 — actively misleading

* [X] 2.1 `engine/profiler.py` does no profiling

It owns `LayerCandidate`, `layer_is_match`, `match_target_layers`, `resolve_target_layers`,
`ensure_no_target_layer_overlap` — that is **layer resolution**, not performance measurement.

Two problems, not one: the name is wrong today, *and* `CLAUDE.md` commits to a real profiler later
("per-layer time breakdown", tok/s, demand-read stall time) which will want exactly this name.

**Suggested:** `engine/layer_resolution.py`.

Consider renaming the backend half at the same time so the pair reads as one job:

```
backends/mlx_metal/layer_search.py  →  layer_discovery.py   # enumerate candidates (backend)
engine/profiler.py                  →  layer_resolution.py  # match against config (agnostic)
```

---

* [X] 2.2 `accumulator` → `recorder`

**`instrumented/qwen3_next_moe.py:22,30,38,63,92,106`** and `tests/integration/mlx_single_router_lens.py:92,98,164,172,178,191`

The type is already `MlxExpertRoutingRecorder` — only the attribute name lags. "Accumulator" implies
a fold or reduction; this appends discrete records.

---

* [X] 2.3 `MlxExpertRoutingEventBuffer` names a container but is an element

**`backends/mlx_metal/recorder.py:25`**

```python
class MlxExpertRoutingEventBuffer:   # holds ONE capture
    ...

_buffer: list[MlxExpertRoutingEventBuffer]   # reads as "a list of buffers"
```

**Suggested: `PendingExpertRoutingEvent`.** This pairs it explicitly with the finalized
`ExpertRoutingEvent` in `datamodel/`, and names the real distinction — unevaluated MLX arrays
awaiting `mx.eval`, versus materialized Python scalars ready for a sink. The pipeline then documents
itself:

```
PendingExpertRoutingEvent  ──mx.eval──▶  ExpertRoutingEvent  ──▶  sink
   (MLX arrays, backend)                 (scalars, datamodel)
```

Also: `_eval_arrays_in_queue(self, queue: ...)` at line 150 calls `self._buffer` a `queue`. One word
per thing — make it `buffer`.

---

* [X] 2.4 `start_step()` / `end_step()` operate on a *step*

**`engine/recorder.py:24,33`**

They take and clear a `TraceStepContext`, but "trace" everywhere else means the whole run
(`TraceRunContext`, `run_id`). Starting a "trace" per decode step is a level mismatch.

**Suggested:** `start_step()` / `end_step()`.

---

## Tier 3 — consistency

* [X] 3.1 Split the `record` verb between the two layers

Currently inverted relative to the class names:

| class                 | method         | when           |
| --------------------- | -------------- | -------------- |
| `BaseEventRecorder` | `.capture()` | sync, hot path |
| `BaseEventSink`     | `.record()`  | async, storage |

**Suggested:** rename `BaseEventSink.record` → **`write`**. It matches the existing
`records_written` property, and leaves `capture` as the unambiguous hot-path verb:

```python
recorder.capture(...)   # in the forward pass
await sink.write(...)   # on flush
```

Touches `core/sinks.py` (ABC + `ParquetEventSink` + `JsonlFileEventSink`) and
`backends/mlx_metal/recorder.py:90`.

---

* [ ] 3.2 One layer concept, three field names

| concept            | `LayerCandidate` | `TargetLayerSearchParams` | `LayerIdentifiers` |
| ------------------ | ------------------ | --------------------------- | -------------------- |
| dotted module path | `path`           | `layer_path_glob`         | `layer_path`       |
| Python class name  | `class_name`     | `class_name`              | `layer_class`      |

The value flows candidate → event unchanged, but renames itself twice on the way. Standardize on
**`layer_path`** and **`class_name`**.

---

* [X] 3.3 `Qwen3NextMoEWrapperFactory` is a PascalCase function

**`instrumented/qwen3_next_moe.py:91`** — it's a factory *builder* returning a closure, so the
PascalCase reads as a class at every call site.

**Suggested:** `build_qwen3_next_moe_factory` (snake_case). And optionally
`Qwen3NextMoEWrapper` → `InstrumentedQwen3NextMoE`, matching the `instrumented/` package it lives in
and stating the contract (transparent wrapper, semantics unchanged) in the name.

---

* [X] 3.4 `MlxWrapperFactory` / `ModuleT` defined twice

Identical aliases in `instrument.py:17-19` and `instrumented/qwen3_next_moe.py:86-88`. Define once
(suggest `instrument.py`) and import — these will drift.

---

* [X] 3.5 `replace_modules` → `instrument_model`

**`instrument.py:22`** — the module is named for the concept; the function should be too. Gives the
readable call site `instrument_model(model, candidates, factory)`.

---

* [ ] 3.6 Test files are neither tests nor consistently named

`tests/integration/mlx_single_router_lens.py`:

- it's an argparse smoke script, not pytest-discoverable (`CLAUDE.md` specifies pytest / `test_*.py`)
- its own header comment on line 1 calls it `scripts/test_mlx_single_router_capture.py`
- the filename says `lens`

Three names for one file. Either move both under `scripts/` with honest names, or convert to
`test_*.py` pytest cases.

---

## Adjacent — not naming, found while reading

- [X] **Incompatible override.** `BaseEventRecorder.flush(self) -> int` vs
  `MlxExpertRoutingRecorder.flush(self, sink) -> int` (`engine/recorder.py:42`,
  `backends/mlx_metal/recorder.py:73`). The sink should be either a constructor dependency or part
  of the ABC signature.
- [X] **Star imports.** `from ..arrow import *` in `tracing/context.py:6` and
  `tracing/expert_routing.py:10` hide where `arrow_metadata` / `serialize_utc` come from — working
  against the clarity this pass is buying. `arrow.py` already defines `__all__`, so explicit imports
  are cheap.
- [ ] `core/constants.py` and `core/protocols.py` are empty.

---

## If we proceed

Renames only, no behavior change. One commit per numbered item so any single rename is revertible.
Suggested order: Tier 1 → Tier 2 → Tier 3.

**Verification**

1. `rg -n 'lens|accumulator|profiler|expert_router_event|replace_modules' --glob '!.venv'` — clean
   outside intentional prose.
2. Import sweep (no `__init__.py` re-exports, so every full-path import must resolve):
   ```
   python -c "import pkgutil,importlib; [importlib.import_module(m.name) for m in pkgutil.walk_packages(['preempt'],'preempt.')]"
   ```
3. Run both `tests/integration/` scripts against the Qwen3.6-35B MLX model; confirm Parquet still writes.
4. `pq.read_schema(path)` shows `instrumentation_version`, the corrected gate field, and
   `preempt.event_type == b"expert_routing_event"`.
5. Existing `traces/*.parquet` carry the old schema — regenerate or delete. **Note:**
   `.notes/revised_analysis.ipynb` reads `traces/part-00000.parquet`, so it will need a re-run.
