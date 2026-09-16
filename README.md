<div align="center">
<h1 align="center", style="fontsize=100px">preemptLM🔮</h1>
<h4 align="left">PreemptLM is Mixture-of-Experts (MoE) inference engine designed to bypass memory limits by predicting which experts will be routed in the forward pass and <i>preemptively</i> streaming them from disk.</h4>
<p align="left"><i>This is an active research project currently being developed in collaboration for publication.</i></p>
</div>

## Research Focus: Predictive Prefetching

Running multi-trillion-parameter sparse models (like Kimi K3) locally is structurally bottlenecked by memory capacity and disk read speeds. Because an MoE model only activates a small fraction of its parameters per token (e.g., K3 activates **~104B** out of **~2.8T** total parameters), the full 1.4 TB of weights does not need to be resident in memory simultaneously.

`preempt` treats VRAM, system RAM, and the SSD as a single managed memory hierarchy. The core research focus actively in development is to replace standard heuristic lookahead strategies with a lightweight expert prediction head. By predicting expert selections ahead of the forward pass, the engine aims to stage weights into memory concurrently with active computation, effectively hiding disk I/O latency behind GPU compute.

## Tech Stack

* **Language:** Python 3.12

* **Compute:** MLX / Apple Metal

## Current State & Engine Mechanics

The project is in active v1 development, built on MLX for Apple Silicon unified memory. The foundational streaming scheduler, cache layer, and tracing pipelines are operational. Engine currently supports `Qwen3-Next` and `Qwen3.x` MoE models. We currently run **Qwen3.6-35B-A3B** and **Qwen3.5-397B** at full precision for benchmarking on MacBook Pro M1 Max 32GB.

What works mechanically today:

* **Optimized MLX Stream Concurrency:** Precise management of thread-local MLX streams. Recent synchronization fixes resolved background thread race conditions between compute and I/O scheduling, resulting in a measured **~50% increase in raw inference throughput**.
* **Vectorized Memory Arenas (`SlottedGLU`):** A strict per-layer cache architecture that pre-allocates weight slots, reducing memory allocation overhead to near zero during rapid, sub-millisecond expert swapping.
* **Lock-Free Asynchronous I/O:** The `DiskBackedExpertLoader` operates entirely lock-free. Demand reads (cache misses) preempt prefetch reads without stalling the main execution thread.
* **Routing Telemetry Pipeline:** A non-blocking instrumentation layer captures the router's raw top-$k$ logits and flushes them to append-only Parquet files. This data is the ground-truth training set for the predictive component currently under development.

## Initial Setup & Prerequisites

Before running the engine, the dense model and its routed experts must be prepared:

1. **Torch Model Conversion:** If you are working with extremely large models PyTorch models that crash standard conversion utilities (like `mlx-lm`) due to memory constraints, use the provided memory-safe converter:

   ```bash
   python scripts/mlx_convert_hf_model.py --hf-path <hf_model_path_or_id> --mlx-path <output_path>
   ```

2. **Expert Bank Generation:** The engine requires experts to be extracted from the model and packed into a contiguous binary "Expert Bank" on disk for fast `pread` streaming. Generate this bank by running:

   ```bash
   python scripts/mlx_make_expert_bank.py --model <mlx_model_path_or_id> --output-dir <output_path>
   ```

3. **Running Inference:** Once the expert bank is prepared, you can start streaming generation using the provided runner script. Note that you must point to a valid pipeline configuration file:

   ```bash
   uv run scripts/mlx_generation.py \
       --config configs/mlx/base-qwen3.6-35b.toml \
       --stream-experts \
       --max-tokens 1024 \
       --prompt "Fire and fury like "
   (Appending --profile will capture live router decisions and flush them to a Parquet file for future predictor training).
