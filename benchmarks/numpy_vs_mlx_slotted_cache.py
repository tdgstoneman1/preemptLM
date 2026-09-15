#!/usr/bin/env python3
"""
Benchmark: MLX-centric vs NumPy-centric slotted expert cache,
with materialize-once / reuse investigation and hit-rate crossover analysis.

Models an inference engine that streams sparse MoE expert weights from disk
(represented here as raw `bytes` payloads) into a fixed-slot LFRU cache.

  Path A (MLX-centric):   np.frombuffer -> mx.asarray -> slot in mx bank -> slice -> matmul
  Path B (NumPy-centric): np.frombuffer -> slot in np bank -> slice -> mx.asarray -> matmul
  Path C (MLX reuse):     like A, but the slot view is materialized ONCE at admission
                          and the resolved weight is reused on every subsequent hit.

Key MLX nuance under test: after a lazy expression (bank slice) is evaluated,
its array object holds the result. Two questions decide whether Path C works:
  1) Is re-evaluating / consuming an already-materialized view free?
  2) Does rewriting the bank slot afterwards leave the old snapshot STALE
     (private copy) or LIVE (aliases the bank)?

NOTE: mx.eval() evaluates in place and returns None (current MLX). Never
assign from mx.eval(); keep a reference to the lazy array and eval it.

Weights are synthesized as (in_features, out_features) = (hidden, ffn),
so the proxy GEMM is y = x @ W (no transpose).

------------------------------- TRADEOFF ------------------------------------
The dominant design tradeoff this benchmark quantifies is HIT-RATE DEPENDENT.

Each engine pays a fixed cost per access that decomposes (to first order) as:

    cost(r) = r * HIT + (1 - r) * MISS

where r is the steady-state cache hit rate. Empirically (Apple Silicon, MLX
with zero-copy DLPack import of numpy arrays, 256 x 2048x512 bf16 experts at
64-slot LFRU):

  - MLX reuse wins on HITS: a materialized slot snapshot resolves to the GEMM
    floor (~0.24 ms) because re-evaluating an evaluated array object is ~free,
    while the NumPy path still pays a fresh dtype-view/alias wrap (~0.03+ ms).
  - MLX reuse loses on MISSES: admission materializes the write AND the
    snapshot AND the dtype resolution (~0.93 ms admission vs ~0.54 ms for
    NumPy, which only does one memcpy + one zero-copy alias wrap).
  - Snapshots are PRIVATE/IMMUTABLE: a materialized view does NOT see later
    bank writes (staleness probe). Eviction is therefore just dropping a ref.
  - numpy -> mlx conversion ALIASES memory on this build (copy=False works),
    so Cache-path eviction over numpy-backed storage needs an eval barrier
    before recycling a slot that a pending consumer may still read.

Because engine A beats engine B exactly when per-access cost_A < cost_B, the
crossover hit rate r* = (M_B - M_A) / (H_A - H_B + M_B - M_A) decides which rec-
ommended data path to ship. r* is machine-, version-, and workload-dependent,
so `breakeven_hit_rate()` below recomputes it from THIS run's observed values.
------------------------------------------------------------------------------

Run on Apple Silicon:
    python bench_slotted_cache.py
    python bench_slotted_cache.py --experts 256 --slots 64 --hidden 2048 --ffn 512 --csv out.csv
"""

import argparse
import gc
import time

import mlx.core as mx
import numpy as np

# ----------------------------------------------------------------------------
# Timing utilities


def percentile_stats(samples_ns):
    a = np.asarray(samples_ns, dtype=np.float64) / 1e6  # ns -> ms
    return (np.median(a), np.percentile(a, 95), np.percentile(a, 99), a.mean())


def time_op(fn, iters, warmup=5):
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter_ns()
        fn()
        t1 = time.perf_counter_ns()
        samples.append(t1 - t0)
    return percentile_stats(samples)


def fmt_row(label, stats, unit_bytes=None):
    p50, p95, p99, mean = stats
    extra = ""
    if unit_bytes:
        gbs = unit_bytes / (p50 / 1e3) / 1e9 if p50 > 0 else float("inf")
        extra = f" | {gbs:9.1f} GB/s"
    return f"{label:<46} {p50:10.4f} {p95:10.4f} {p99:10.4f} {mean:10.4f}{extra}"


def print_header():
    print(
        f"{'operation':<46} {'p50 ms':>10} {'p95 ms':>10} {'p99 ms':>10} {'mean ms':>10}"
    )


def sync_eval(arr):
    """Force evaluation; return the array (never rely on mx.eval's return)."""
    mx.eval(arr)
    return arr


# ----------------------------------------------------------------------------
# Tradeoff analysis: crossover hit rate between two engines


def breakeven_hit_rate(hit_a, miss_a, hit_b, miss_b):
    """Solve for the hit rate r* at which engine A and engine B break even.

    Model: per-access cost(engine) = r * HIT + (1 - r) * MISS.

    Cost_A(r) = Cost_B(r)  =>  r* = (miss_B - miss_A) / (hit_A - hit_B + miss_B - miss_A)

    Interpretation of the two "slopes":
      slope(engine) = d(cost)/d(hit_rate) = HIT - MISS  (< 0 normally: hits are cheaper)
      The STEEPER (more negative) slope wins as r -> 1; the FLATTER slope wins
      as r -> 0, because it pays less on every miss.

    Returns dict with r* in [0, 1] clamped, plus a dominance verdict when no
    crossover exists in the valid range (one engine is better at every hit rate).
    All latencies in the same units (ms recommended).
    """
    denom = (hit_a - hit_b) + (miss_b - miss_a)
    out = dict(slope_a=hit_a - miss_a, slope_b=hit_b - miss_b)
    if abs(denom) < 1e-12:
        out.update(r_star=None, verdict="parallel cost curves; no crossover exists")
        return out
    r = (miss_b - miss_a) / denom
    if 0.0 <= r <= 1.0:
        # For r > r*: the engine with the steeper slope (lower hit cost) wins.
        winner_high = "A" if hit_a < hit_b else "B"
        out.update(
            r_star=r,
            verdict=(
                f"crossover at r* = {100*r:.1f}%: engine {winner_high} wins above, "
                f"engine {'B' if winner_high == 'A' else 'A'} wins below"
            ),
        )
    else:
        # No interior crossover: whoever is cheaper at r -> 1 wins everywhere
        # (the steeper engine dominates the flatter one over the whole range).
        uniform = "A" if hit_a < hit_b else "B"
        out.update(
            r_star=r,
            verdict=(
                f"no crossover in [0,1] (extrapolated r* = {r:+.2f}): "
                f"engine {uniform} dominates at every hit rate"
            ),
        )
    return out


def report_crossover(e2e, stream_len):
    """Plug observed hit/miss p50s from the end-to-end run into the tradeoff model."""
    print("## Crossover analysis (from observed p50 latencies)\n")
    for mode, r in e2e.items():
        print(
            f"  {mode:<11} linear model: cost(r) = {r['miss_p50']:.4f} - {abs(r['hit_p50'] - r['miss_p50']):.4f} * r   "
            f"(slope {r['hit_p50'] - r['miss_p50']:+.4f} ms per +100% hit rate)"
        )
    print()

    a, b = e2e["numpy"], e2e["mlx_reuse"]
    res = breakeven_hit_rate(a["hit_p50"], a["miss_p50"], b["hit_p50"], b["miss_p50"])
    print(f"  numpy (A) vs mlx_reuse (B): {res['verdict']}")
    if res["r_star"] is not None and 0.0 <= res["r_star"] <= 1.0:
        for mode, r in e2e.items():
            margin = r["hit_rate"] - res["r_star"]
            side = "ABOVE" if margin >= 0 else "below"
            print(
                f"    measured hit rate for {mode}: {100*r['hit_rate']:.1f}% -> {side} the crossover "
                f"by {100*abs(margin):.1f} pts"
            )
    # Sanity check the linear model against total-time observations.
    for mode, r in e2e.items():
        modeled = r["hit_rate"] * r["hit_p50"] + (1 - r["hit_rate"]) * r["miss_p50"]
        observed = 1e3 * r["total_s"] / stream_len
        resid = observed - modeled
        print(
            f"    model check {mode}: modeled {modeled:.4f} ms/access, observed {observed:.4f} "
            f"(residual {resid:+.4f}; >0 = warmup/eviction overhead not in p50 model)"
        )
    print()


# ----------------------------------------------------------------------------
# Probe: is numpy -> mlx conversion a copy in this MLX build?


def probe_zerocopy():
    print("## Zero-copy probe: mx.asarray(np.ndarray)\n")
    print_header()
    sizes = [64 << 10, 1 << 20, 8 << 20, 64 << 20]
    timed = {}
    for nbytes in sizes:
        src = np.arange(nbytes // 2, dtype=np.uint16)
        stats = time_op(lambda: sync_eval(mx.asarray(src)), iters=30, warmup=5)
        timed[nbytes] = stats[0]
        print(
            fmt_row(
                f"asarray + eval  ({nbytes >> 20:>3} MiB)", stats, unit_bytes=nbytes
            )
        )

    t_small, t_large = timed[sizes[0]], timed[sizes[-1]]
    looks_like_copy = t_large > 50 * t_small
    print(
        f"\n  copy scaling heuristic (64MiB / 64KiB time ratio): {t_large / max(t_small, 1e-9):.0f}x"
    )
    print(
        f"  -> conversion appears to be: {'COPY (bandwidth-bound)' if looks_like_copy else 'NEAR-FLAT (check sharing below)'}"
    )

    # Mutation-visibility: if mlx aliases numpy memory, a numpy write shows up in mlx.
    src = np.arange(8, dtype=np.uint16)
    arr = mx.asarray(src)
    mx.eval(arr)
    src[0] = 999
    mx.eval(arr)  # no-op if materialized; forces any pending op otherwise
    aliased = int(arr[0].item()) == 999  # type: ignore
    print(
        f"  mutation-visibility probe: {'ALIASES numpy memory' if aliased else 'independent storage (copied)'}"
    )

    # copy=False: raises if zero-copy import is impossible for this input.
    try:
        zc = mx.asarray(src, copy=False)
        mx.eval(zc)
        print("  mx.asarray(np, copy=False): SUCCEEDED (zero-copy import available)")
    except Exception as e:
        print(
            f"  mx.asarray(np, copy=False): RAISED ({type(e).__name__}) -> numpy inputs are copied"
        )
    print()


# ----------------------------------------------------------------------------
# Probe: materialize-once, then reuse. Cost + staleness.


def probe_view_reuse(args, shape_u16):
    """Does a materialized bank slice become free to reuse? Does it go stale?"""
    n = shape_u16[0] * shape_u16[1]
    bank = mx.zeros((args.slots,) + shape_u16, dtype=mx.uint16)
    mx.eval(bank)

    # Distinct, deterministic patterns so staleness is observable.
    pattern_a = (
        (np.arange(n, dtype=np.uint32) % 65521).astype(np.uint16).reshape(shape_u16)
    )
    pattern_b = (
        ((np.arange(n, dtype=np.uint32) + 7) % 65521)
        .astype(np.uint16)
        .reshape(shape_u16)
    )

    bank[0] = mx.asarray(pattern_a)
    mx.eval(bank)

    snapshot = bank[0]
    mx.eval(snapshot)  # materialize once -- admission-time cost

    print("## Materialize-once / reuse probe (MLX bank slice)\n")
    print_header()

    stats_fresh = time_op(lambda: sync_eval(bank[0]), args.iters)
    print(fmt_row("fresh slice + eval            (every hit)", stats_fresh))

    stats_reuse = time_op(lambda: sync_eval(snapshot), args.iters)
    print(fmt_row("re-eval SAME object           (reuse)", stats_reuse))

    bf_view = snapshot.view(mx.bfloat16)
    mx.eval(bf_view)
    stats_bf = time_op(lambda: sync_eval(snapshot.view(mx.bfloat16)), args.iters)
    print(fmt_row("bf16 view of resolved slice   (per hit)", stats_bf))

    stats_bf_reuse = time_op(lambda: sync_eval(bf_view), args.iters)
    print(fmt_row("re-eval resolved bf16 view    (reuse)", stats_bf_reuse))

    # Staleness: rewrite slot with pattern B, then inspect the old snapshot.
    bank[0] = mx.asarray(pattern_b)
    mx.eval(bank)
    mx.eval(snapshot)  # in case re-evaluation is triggered by bank change
    v_now = int(snapshot[0, 0].item())  # type: ignore
    v_bank = int(bank[0, 0, 0].item())  # type: ignore
    a00, b00 = int(pattern_a[0, 0]), int(pattern_b[0, 0])
    stale = (v_now == a00) and (v_now != b00)
    print()
    print(
        f"  snapshot[0,0]={v_now}  (pattern A={a00}, pattern B={b00}, bank now={v_bank})"
    )
    if stale:
        print(
            "  -> materialized slice is a PRIVATE SNAPSHOT: stale after slot rewrite."
        )
        print(
            "     Reuse design => slot views are standalone arrays; bank writes do NOT"
        )
        print("     propagate. Protect: drop the snapshot reference at eviction.")
    elif v_now == b00 == v_bank:
        print("  -> materialized slice TRACKS bank writes (aliases / re-evaluates).")
        print(
            "     Reuse design => view stays current, but hits re-pay any materialization."
        )
    else:
        print("  -> UNEXPECTED state; inspect manually before trusting reuse.")
    print()
    return dict(
        fresh=stats_fresh, reuse=stats_reuse, bf_view=stats_bf, bf_reuse=stats_bf_reuse
    )


# ----------------------------------------------------------------------------
# Synthetic MoE layer


def make_expert_payloads(n_experts, hidden, ffn, seed=0):
    """Raw bytes per expert: [w1(h,f), w2(f,h)] as bf16-width payloads."""
    rng = np.random.default_rng(seed)
    # Random uint16 bits are fine: benchmarks only move bytes, never values.
    return [
        (rng.bytes(hidden * ffn * 2), rng.bytes(ffn * hidden * 2))
        for _ in range(n_experts)
    ]


# ----------------------------------------------------------------------------
# Slotted LFRU eviction


class LFRU:
    """Min frequency, ties broken by least-recent use."""

    def __init__(self, n_slots):
        self.n_slots = n_slots
        self.slot_of = {}  # expert_id -> slot
        self.owner = [None] * n_slots
        self.freq = np.zeros(n_slots, dtype=np.int64)
        self.tick = np.zeros(n_slots, dtype=np.int64)
        self.clock = 0

    def lookup(self, expert_id):
        self.clock += 1
        slot = self.slot_of.get(expert_id)
        if slot is None:
            return None
        self.freq[slot] += 1
        self.tick[slot] = self.clock
        return slot

    def admit(self, expert_id):
        free = [s for s in range(self.n_slots) if self.owner[s] is None]
        if free:
            slot = free[0]
        else:
            order = np.lexsort((self.tick, self.freq))
            slot = int(order[0])
            evicted = self.owner[slot]
            if evicted is not None:
                del self.slot_of[evicted]
        self.owner[slot] = expert_id
        self.slot_of[expert_id] = slot
        self.freq[slot] = 1
        self.tick[slot] = self.clock
        return slot


# ----------------------------------------------------------------------------
# Consumption helpers


def as_bf16_weight(mx_u16):
    """uint16 bit-cast -> bf16, materialized. Same-itemsize reinterpret, no value conversion."""
    w = mx_u16.view(mx.bfloat16)
    return sync_eval(w)


def proxy_forward(x, w_bf16):
    """Stand-in for the expert GEMM. w is stored (in, out) = (hidden, ffn), so y = x @ W."""
    y = x @ w_bf16
    mx.eval(y)  # returns None; keep y if needed
    return y


# ----------------------------------------------------------------------------
# Benchmarks


def bench_isolated(args, shape_u16, expert_bytes):
    hidden = args.hidden
    payload = expert_bytes[0][0]  # w1 bytes for expert 0
    decode_fn = lambda: np.frombuffer(payload, dtype=np.uint16).reshape(shape_u16)
    decoded = decode_fn()
    n_elems = decoded.size

    bank_np = np.empty((args.slots,) + shape_u16, dtype=np.uint16)
    bank_mx = mx.zeros((args.slots,) + shape_u16, dtype=mx.uint16)
    mx.eval(bank_mx)

    x = mx.random.normal((args.tokens, hidden), dtype=mx.float32).astype(mx.bfloat16)
    mx.eval(x)

    print("## Isolated operation costs (one expert weight matrix w1)\n")
    print_header()
    rows = {}

    stats = time_op(decode_fn, args.iters)
    rows["np.frombuffer -> u16 view (shared)"] = stats
    print(fmt_row("np.frombuffer -> u16 view (shared)", stats))

    stats = time_op(lambda: sync_eval(mx.asarray(decoded)), args.iters)
    rows["u16 -> mx.asarray + eval"] = stats
    print(fmt_row("u16 -> mx.asarray + eval", stats, unit_bytes=n_elems * 2))

    w_mx = sync_eval(mx.asarray(decoded))

    def insert_np():
        bank_np[0] = decoded

    def insert_mx():
        bank_mx[0] = w_mx
        mx.eval(bank_mx)

    stats = time_op(insert_np, args.iters)
    rows["slot write: numpy bank"] = stats
    print(fmt_row("slot write: numpy bank  (memcpy)", stats, unit_bytes=n_elems * 2))
    stats = time_op(insert_mx, args.iters)
    rows["slot write: mlx bank"] = stats
    print(
        fmt_row("slot write: mlx bank    (incl. eval)", stats, unit_bytes=n_elems * 2)
    )

    stats = time_op(lambda: bank_np[0], args.iters)
    rows["slot read: numpy slice"] = stats
    print(fmt_row("slot read: numpy slice  (view)", stats))
    stats = time_op(lambda: sync_eval(bank_mx[0]), args.iters)
    rows["slot read: mlx slice"] = stats
    print(fmt_row("slot read: mlx slice    (view + eval)", stats))

    w_bf = as_bf16_weight(mx.asarray(decoded))
    stats = time_op(lambda: proxy_forward(x, w_bf), args.iters)
    rows["proxy forward matmul (w resident)"] = stats
    print(fmt_row("proxy forward matmul    (mlx, w resident)", stats))
    print()
    return rows


def bench_hit_path(args, shape_u16, bank_np, bank_mx):
    """Returns dict of hit-path closures: naive MLX, NumPy, reuse-u16, reuse-bf16."""
    x = mx.random.normal((args.tokens, args.hidden), dtype=mx.float32).astype(
        mx.bfloat16
    )
    mx.eval(x)

    def hit_mlx_naive():
        w = as_bf16_weight(bank_mx[0])  # fresh slice + eval, every hit
        return proxy_forward(x, w)

    def hit_numpy():
        v = bank_np[0]
        w = as_bf16_weight(sync_eval(mx.asarray(v)))
        return proxy_forward(x, w)

    # Materialize-once variants: resolved at admission, reused here.
    slot_u16 = bank_mx[0]
    mx.eval(slot_u16)
    slot_bf16 = slot_u16.view(mx.bfloat16)
    mx.eval(slot_bf16)

    def hit_mlx_reuse_u16():
        w = as_bf16_weight(slot_u16)  # snapshot held; re-wrap dtype per hit
        return proxy_forward(x, w)

    def hit_mlx_reuse_bf16():
        return proxy_forward(x, slot_bf16)  # fully resolved weight; GEMM floor

    return {
        "HIT mlx-centric   (fresh slice -> matmul)": hit_mlx_naive,
        "HIT numpy-centric (slice -> asarray -> matmul)": hit_numpy,
        "HIT mlx-reuse-u16 (snapshot, re-wrap bf16)     ": hit_mlx_reuse_u16,
        "HIT mlx-reuse-bf16(snapshot resolved once)     ": hit_mlx_reuse_bf16,
    }


def load_slot(mode, bank_np, bank_mx, slot, payload, shape_u16, resolved):
    """Cache-miss fill. Returns consumed bf16 weight.

    mode="mlx_reuse" additionally materializes a slot snapshot and a resolved
    bf16 view at admission; both are cached into `resolved` for later hits.
    """
    v = np.frombuffer(payload, dtype=np.uint16).reshape(shape_u16)
    if mode == "numpy":
        bank_np[slot] = v
        return as_bf16_weight(sync_eval(mx.asarray(v)))
    wm = sync_eval(mx.asarray(v))
    bank_mx[slot] = wm
    mx.eval(bank_mx)
    if mode == "mlx_reuse":
        snap = bank_mx[slot]
        mx.eval(snap)
        w = as_bf16_weight(snap)
        resolved[slot] = w  # reuse on subsequent hits; drop at eviction
        return w
    return as_bf16_weight(wm)  # "mlx": bf16 over the freshly converted array


def bench_end_to_end(args, expert_bytes, shape_u16):
    n_experts = len(expert_bytes)
    bank_np = np.empty((args.slots,) + shape_u16, dtype=np.uint16)
    bank_mx = mx.zeros((args.slots,) + shape_u16, dtype=mx.uint16)
    sync_eval(bank_mx)

    x = mx.random.normal((args.tokens, args.hidden), dtype=mx.float32).astype(
        mx.bfloat16
    )
    mx.eval(x)

    # Zipf-like popularity stream, deterministic and identical across engines.
    rng = np.random.default_rng(42)
    ranks = np.arange(1, n_experts + 1)
    probs = 1.0 / ranks
    probs /= probs.sum()
    stream = rng.choice(n_experts, size=args.stream_len, p=probs)

    results = {}
    for mode in ("mlx", "numpy", "mlx_reuse"):
        lfru = LFRU(args.slots)
        resolved = {}  # slot -> materialized bf16 weight (mlx_reuse only)
        hit_lat, miss_lat = [], []

        for eid in stream:
            eid = int(eid)
            t0 = time.perf_counter_ns()
            slot = lfru.lookup(eid)
            if slot is None:
                slot = lfru.admit(eid)
                if mode == "mlx_reuse":
                    resolved.pop(slot, None)  # evict snapshot with the slot
                w = load_slot(
                    mode,
                    bank_np,
                    bank_mx,
                    slot,
                    expert_bytes[eid][0],
                    shape_u16,
                    resolved,
                )
                proxy_forward(x, w)
                miss_lat.append((time.perf_counter_ns() - t0) / 1e6)
            else:
                if mode == "numpy":
                    w = as_bf16_weight(sync_eval(mx.asarray(bank_np[slot])))
                elif mode == "mlx_reuse":
                    w = resolved[slot]  # admission-materialized weight, free hit
                else:
                    w = as_bf16_weight(bank_mx[slot])
                proxy_forward(x, w)
                hit_lat.append((time.perf_counter_ns() - t0) / 1e6)

        hit = np.asarray(
            hit_lat[args.slots :] if len(hit_lat) > args.slots else hit_lat
        )
        miss = np.asarray(miss_lat)
        n = len(hit_lat) + len(miss_lat)
        results[mode] = dict(
            hit_rate=len(hit_lat) / n,
            hit_p50=np.percentile(hit, 50) if hit.size else float("nan"),
            hit_p99=np.percentile(hit, 99) if hit.size else float("nan"),
            miss_p50=np.percentile(miss, 50) if miss.size else float("nan"),
            miss_p99=np.percentile(miss, 99) if miss.size else float("nan"),
            total_s=(hit.sum() + miss.sum()) / 1e3,
        )
        gc.collect()
    return results


# ----------------------------------------------------------------------------
# Main


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--experts", type=int, default=16)
    ap.add_argument("--slots", type=int, default=6)
    ap.add_argument("--hidden", type=int, default=2048)
    ap.add_argument("--ffn", type=int, default=512)
    ap.add_argument(
        "--tokens", type=int, default=8, help="batch dim for proxy forward matmul"
    )
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--stream-len", type=int, default=4000)
    ap.add_argument("--csv", type=str, default=None)
    args = ap.parse_args()

    shape_u16 = (args.hidden, args.ffn)
    n_bytes = args.hidden * args.ffn * 2
    print(
        f"slot payload (w1): {shape_u16} bf16 = {n_bytes / 1e6:.2f} MB | "
        f"bank: {args.slots} slots = {args.slots * n_bytes / 1e6:.0f} MB | experts: {args.experts}"
    )
    print(f"(defaults now match one Qwen3.6-35B-A3B expert projection: 2048x512)\n")

    probe_zerocopy()
    reuse_probe = probe_view_reuse(args, shape_u16)

    expert_bytes = make_expert_payloads(args.experts, args.hidden, args.ffn)

    rows = bench_isolated(args, shape_u16, expert_bytes)

    # --- Hit-path showdown ---
    bank_np = np.empty((args.slots,) + shape_u16, dtype=np.uint16)
    bank_mx = mx.zeros((args.slots,) + shape_u16, dtype=mx.uint16)
    mx.eval(bank_mx)
    seed_v = np.frombuffer(expert_bytes[0][0], dtype=np.uint16).reshape(shape_u16)
    bank_np[0] = seed_v
    bank_mx[0] = mx.asarray(seed_v)
    mx.eval(bank_mx)

    hit_paths = bench_hit_path(args, shape_u16, bank_np, bank_mx)
    print("## Cache-HIT path (slot read + proxy forward)\n")
    print_header()
    hit_stats = {}
    for label, fn in hit_paths.items():
        hit_stats[label] = time_op(fn, args.iters)
        print(fmt_row(label, hit_stats[label]))
    base_label = "HIT mlx-centric   (fresh slice -> matmul)"
    np_label = "HIT numpy-centric (slice -> asarray -> matmul)"
    delta = hit_stats[np_label][0] - hit_stats[base_label][0]
    print(f"\n  numpy-centric vs naive-mlx per hit (p50): {delta:+.4f} ms")
    for label, st in hit_stats.items():
        if "reuse" in label:
            print(
                f"  {label.strip()}  vs numpy: {st[0] - hit_stats[np_label][0]:+.4f} ms/hit"
            )
    print()

    # --- End-to-end LFRU stream + crossover analysis ---
    print(
        f"## End-to-end LFRU stream ({args.stream_len} accesses, {args.slots}/{args.experts} slots)\n"
    )
    e2e = bench_end_to_end(args, expert_bytes, shape_u16)
    print(
        f"{'engine':<11} {'hit rate':>9} {'hit p50':>9} {'hit p99':>9} {'miss p50':>9} {'miss p99':>9} {'total s':>9}"
    )
    for mode, r in e2e.items():
        print(
            f"{mode:<11} {r['hit_rate']:>9.1%} {r['hit_p50']:>9.4f} {r['hit_p99']:>9.4f} "
            f"{r['miss_p50']:>9.4f} {r['miss_p99']:>9.4f} {r['total_s']:>9.3f}"
        )
    print()
    for mode in ("numpy", "mlx_reuse"):
        d = e2e[mode]["total_s"] - e2e["mlx"]["total_s"]
        print(
            f"  total-time delta ({mode} - mlx-naive): {d:+.3f} s "
            f"({1000 * d / args.stream_len:+.5f} ms/access)"
        )
    print()

    report_crossover(e2e, args.stream_len)

    if args.csv:
        with open(args.csv, "w") as fh:
            fh.write("operation,p50_ms,p95_ms,p99_ms,mean_ms\n")
            for k, v in rows.items():
                fh.write(f"{k},{v[0]:.6f},{v[1]:.6f},{v[2]:.6f},{v[3]:.6f}\n")
            for k, v in hit_stats.items():
                fh.write(f"{k.strip()},{v[0]:.6f},{v[1]:.6f},{v[2]:.6f},{v[3]:.6f}\n")
            for k, v in reuse_probe.items():
                fh.write(
                    f"reuse_probe:{k},{v[0]:.6f},{v[1]:.6f},{v[2]:.6f},{v[3]:.6f}\n"
                )
        print(f"wrote {args.csv}")


if __name__ == "__main__":
    main()
