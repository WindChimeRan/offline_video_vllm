# Speed Log — vLLM Offline Inference (Large)

**Setup:** Qwen/Qwen2.5-VL-7B-Instruct · 1× A100-80GB (GPU 4) · bf16 · `max_tokens=1` · `max_model_len=65536`
**Data:** NExTQA MC test (1000) + MVBench stratified 55/subtask × 18 subtasks (990). Total **N = 1990**.

Same presets as [`MINI_SPEED_LOG.md`](MINI_SPEED_LOG.md), re-run at 10× scale so engine-load overhead is amortized and the per-request wins can surface in wall time.

| # | Preset | Change | Engine load (s) | Wall (s) | req/s | TTFT mean / p95 (s) | E2E mean (s) | Acc NExTQA / MVBench |
|---|--------|--------|----------------:|---------:|------:|---------------------:|-------------:|---------------------:|
| 1 | `run1`  | Baseline: `mm_processor_kwargs={"fps":0.5}`, `media_io_kwargs={"video":{"num_frames":64,"fps":0.5}}` (no pixel cap) | 41.0 | 1045.6 | 1.90 | 252.5 / 480.7 | NExTQA 0.510 · MVBench 2.130 | 79.9% / 60.3% |
| 2 | `run2`  | +`min_pixels=32*28²`, `max_pixels=256*28²` — caps 720p/1080p clips, NExTQA barely touched | 28.1 | **680.6** | **2.92** | 166.2 / 322.4 | 0.226 · 0.170 | 79.5% / 59.4% |
| 3 | `run3`  | Drop `fps`, fix `num_frames=16` uniform (same pixel caps) | 33.0 | 714.6 | 2.78 | 173.8 / 339.0 | 0.139 · 0.435 | 79.8% / **64.8%** |
| 4b | `run4b` | +`compile_mm_encoder=True`, `cudagraph_mm_encoder=True` (local shim over vllm PR #38997) | 39.4 | 693.6 | 2.87 | 169.6 / 329.0 | **0.122 · 0.317** | 79.4% / **65.1%** |
| 5 | `run5`  | +`renderer_num_workers=2`, `mm_processor_cache_gb=0` (thread-safety). Picked after a 1/2/4/8 sweep — see [`RENDERER_WORKERS_TUNING.md`](RENDERER_WORKERS_TUNING.md). w=2 is the sweet spot (−3% wall, the whole win is on MVBench); w≥4 regresses from GIL + cv2-internal-threading contention. | 39.2 | **674.0** | **2.95** | 165.9 / 318.2 | 0.121 · 0.456 | 79.6% / 65.3% |

_TTFT = arrival → first token (dominated by batch-queue wait in a single-shot batch). E2E = `last_token_ts − scheduled_ts`, monotonic clock, per-request inference time._

## Findings

**`max_pixels` cap is the biggest single lever.** Run 1 → Run 2: **−35% total wall**, accuracy essentially unchanged. The savings come almost entirely from **CPU-side preprocessing**: MVBench render phase 518 s → 280 s (halved) because resize work scales with pixel count. Downsampling 1080p to ~448 px/side before the HF processor touches it is the entire story.

**Fixed 16 frames (Run 3) trades speed for MVBench accuracy.** +5% wall but +5.4 pt MVBench accuracy vs Run 2. The extra frames help short MVBench clips (which `fps=0.5` starved), while not hurting NExTQA.

**Compile + CUDA-graph ViT (Run 4b) is a small win at scale, invisible at mini scale.** Run 4b vs Run 3: −3% wall, per-request E2E down 12% (NExTQA) / 27% (MVBench). The gain is small because **CPU rendering dominates the wall** (see below), not the ViT — graphing the ViT just means the GPU sits idle a bit less.

**The wall is CPU-bound.** Engine logs show the two phases are completely decoupled in offline-batch mode:

| | MVBench render (CPU-single-thread) | MVBench engine (GPU) | % render |
|---|---|---|---|
| Run 1 | 518 s | 43 s | 92% |
| Run 2 | 280 s | <1 s | ~100% |
| Run 3 | 352 s | <2 s | ~100% |
| Run 4b | ~315 s | <2 s | ~100% |

Once `max_pixels` is tuned, the GPU clears 990 requests in under a second — anything that makes it even faster is invisible in wall time. To cut further we need to **parallelize preprocessing** (e.g. `num_mm_processor_workers > 1`), **change the video backend** (decord / torchcodec), or **pre-cache decoded frames**.

Per-run artifacts live under `runs/<timestamp>_<preset>/{nextqa,mvbench}.jsonl` + `results.json`.
