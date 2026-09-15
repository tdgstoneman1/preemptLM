# preempt

An experimental Mixture-of-Experts (MoE) inference engine built to stream model weights directly from NVMe SSDs to GPU memory, bypassing RAM capacity limits.

*Note: This is an active research project, currently being developed in collaboration for future publication.*

## The Research Focus: Predictive Prefetching

Running multi-trillion-parameter sparse models (like Kimi K3) locally is structurally bottlenecked by memory capacity and disk read speeds. Because an MoE model only activates a small fraction of its parameters per token (e.g., K3 activates **~104B** out of **~2.8T** total parameters), the full 1.4 TB of weights does not need to be resident in memory simultaneously.

`preempt` treats VRAM, system RAM, and the SSD as a single managed memory hierarchy. The core research focus—**which is actively in development**—is to replace standard heuristic lookahead strategies with a **user-trainable expert prefetcher**. By observing the router and predicting expert selection a layer ahead, the engine aims to stage weights into memory concurrently with active computation, effectively hiding disk I/O latency behind GPU compute.

## Current State & Engine Mechanics

The project is in active v1 development, built on MLX for Apple Silicon unified memory. The foundational streaming scheduler, cache layer, and tracing pipelines are operational. We currently run **Qwen3.6-35B-A3B** at full precision.

What works mechanically today:

* **Optimized MLX Stream Concurrency:** Precise management of thread-local MLX streams. Recent synchronization fixes resolved background thread race conditions between compute and I/O scheduling, resulting in a measured **~50% increase in raw inference throughput**.
* **Vectorized Memory Arenas (`SlottedGLU`):** A strict per-layer cache architecture that pre-allocates weight slots, reducing memory allocation overhead to near zero during rapid, sub-millisecond expert swapping.
* **Lock-Free Asynchronous I/O:** The `DiskBackedExpertLoader` operates entirely lock-free. Demand reads (cache misses) preempt prefetch reads without stalling the main execution thread.
* **Routing Telemetry Pipeline:** A non-blocking instrumentation layer captures the router's raw top-$k$ logits and flushes them to append-only Parquet files. This data is the ground-truth training set for the predictive component currently under development.

## Technical Stack

* **Compute:** MLX / Apple Metal
* **Language:** Python (Strictly typed, async-first architecture)
* **Storage:** Parquet (Tracing), raw contiguous binary (Expert Bank blobs)
