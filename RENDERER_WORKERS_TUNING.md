# `renderer_num_workers` tuning

Sweep of `renderer_num_workers` ∈ {1, 2, 4, 8} on top of the `run4b` config (`compile_mm_encoder` + `cudagraph_mm_encoder` + fixed 16 frames + `max_pixels=256·28²`). All runs on samples/large (1000 NExTQA + 990 MVBench = **N=1990**), same A100-80GB, bf16, `max_tokens=1`.

Workers > 1 also requires `mm_processor_cache_gb=0` (the MM processor cache isn't thread-safe; vllm 0.19.1 refuses `workers>1` otherwise).

| workers | cache | Engine load (s) | NExTQA wall (s) | MVBench wall (s) | Combined wall (s) | req/s | NExTQA E2E (s) | MVBench E2E (s) | NExTQA acc | MVBench acc |
|--------:|:------|----------------:|----------------:|-----------------:|------------------:|------:|---------------:|----------------:|-----------:|------------:|
| 1 (run4b) | on  | 39.4 | 352.5 | 341.1 | 693.6 | 2.87 | 0.122 | 0.317 | 79.4% | 65.1% |
| **2** ✓   | off | 39.2 | 353.2 | **320.8** | **674.0** | **2.95** | 0.121 | 0.456 | 79.6% | 65.3% |
| 4         | off | 39.9 | **397.9** | 336.2 | 734.1 | 2.71 | 0.121 | 0.319 | 79.6% | 65.2% |
| 8         | off | 41.1 | 397.3 | 361.8 | 759.1 | 2.62 | 0.125 | 0.204 | 79.6% | 65.3% |

## Observations

- **w=2 is the sweet spot.** −3% combined wall vs w=1; the whole improvement lives in MVBench (−6% MVBench wall, 0% NExTQA). Accuracy unchanged within noise at every worker count.
- **w≥4 regresses sharply on NExTQA** (397 s vs 353 s at w=1 or w=2, +12%). NExTQA clips are more uniform and larger on average; the thread pool's contention outweighs the overlap benefit.
- **MVBench E2E goes UP with workers even as wall goes down**. With w=2 MVBench E2E climbs 0.317→0.456 s, with w=8 it drops to 0.204 s. Interpretation: at moderate concurrency the engine has one pending request + one prefilling, so the in-flight request takes longer to finish (it shares the GPU); at high concurrency the engine batches more aggressively, so E2E drops but render congestion dominates wall.
- **Bottleneck is the single-threaded CPU code path, not "GPU starvation".** Workers help a bit where cv2's C++ threading doesn't already saturate a core; beyond w=2 we only add scheduling overhead + GIL pressure from the HF processor's numpy ops.

## Winner

**`renderer_num_workers = 2`** (with `mm_processor_cache_gb = 0`) is the final `run5` config. Baked into `PRESETS["run5"]` in `infer.py`.

## What didn't help

- Higher worker counts (4, 8) — GIL + cv2-internal threading + ThreadPoolExecutor overhead cancel out the parallelism.
- Disabling the MM processor cache alone — when w=1 we keep the cache on; cache hit rate was ~1-3% so turning it off doesn't matter much, but it's required once workers>1.

## Caveats

- All workers-sweep runs used **GPU 0** on the same node (GPU 4 was mid-run by another user). Same A100-80GB hardware, same driver, so timings are directly comparable to the `run1-run4b` numbers captured on GPU 4 in `LARGE_SPEED_LOG.md`.
