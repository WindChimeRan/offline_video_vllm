# Speed Log — vLLM Offline Inference (Large)

**Setup:** Qwen/Qwen2.5-VL-7B-Instruct · 1× A100-80GB · bf16 · `max_tokens=1` · `max_model_len=65536`
**Data:** NExTQA MC test (1000) + MVBench stratified 55/subtask × 18 subtasks (990). Total **N = 1990**.

Same presets as [`MINI_SPEED_LOG.md`](MINI_SPEED_LOG.md), re-run at 10× scale so engine-load is amortized and per-request wins surface in wall time.

| #  | Preset | Config delta                                                  | Engine | Wall   | req/s | TTFT mean / p95 | E2E (NExT · MVB) | Acc NExT / MVB    |
|----|--------|----------------------------------------------------------------|------:|-------:|------:|----------------:|-----------------:|-------------------|
| 1  | run1   | Baseline: `fps=0.5`, `num_frames=64`, no pixel cap             | 41.0  | 1045.6 | 1.90  | 252.5 / 480.7   | 0.51 · 2.13      | 79.9% / 60.3%     |
| 2  | run2   | +`min/max_pixels` caps (downsamples 720p/1080p)                | 28.1  | **680.6** | **2.92** | 166.2 / 322.4 | 0.23 · 0.17 | 79.5% / 59.4%     |
| 3  | run3   | Drop `fps`; fix `num_frames=16` uniform                        | 33.0  | 714.6  | 2.78  | 173.8 / 339.0   | 0.14 · 0.44      | 79.8% / **64.8%** |
| 4b | run4b  | +`compile_mm_encoder` + `cudagraph_mm_encoder`                 | 39.4  | 693.6  | 2.87  | 169.6 / 329.0   | **0.12 · 0.32**  | 79.4% / **65.1%** |
| 5  | run5   | +`renderer_num_workers=2`, `mm_processor_cache_gb=0`           | 39.2  | **674.0** | **2.95** | 165.9 / 318.2 | 0.12 · 0.46  | 79.6% / 65.3%     |

_TTFT = arrival → first token. E2E = `last_token_ts − scheduled_ts`. Workers sweep behind run5 is in [`RENDERER_WORKERS_TUNING.md`](RENDERER_WORKERS_TUNING.md)._

## Key levers

- **`max_pixels=256·28²` is the biggest single lever** (run1→run2: −35 % wall). Resize cost scales with pixel count; downsampling 1080p before the HF processor touches it is the entire story.
- **Fixed `num_frames=16`** (run2→run3) trades 5 % wall for +5.4 pt MVBench. Short MVBench clips were starved at `fps=0.5`.
- **ViT compile + CUDA-graphs** (run3→run4b) trim per-request E2E 12-27 % but **wall is flat** — CPU rendering dominates, GPU is no longer the bottleneck.
- **Wall is CPU-bound from run2 onward.** Once `max_pixels` is tuned, MVBench's GPU phase is <2 s out of ~340 s wall:

  |        | MVBench render (CPU, single-thread) | MVBench engine (GPU) | % render |
  |--------|---:|---:|---:|
  | run1   | 518 s | 43 s | 92 % |
  | run2   | 280 s | <1 s | ~100 % |
  | run3   | 352 s | <2 s | ~100 % |
  | run4b  | ~315 s | <2 s | ~100 % |

  Further speedup has to come from cheaper preprocessing — parallelize the renderer (`run5`), change the video backend (decord / torchcodec / keyframe-only), or pre-cache decoded frames.

## Lossy acceleration: `pyav_keyframes_v2` backend

To push past run5's CPU-decode floor, [`pyav_keyframe_backend.py`](pyav_keyframe_backend.py) ships `pyav_keyframes_v2`: one demux pass to enumerate keyframe PTS (no decode), then seek + decode only the `num_frames` keyframes we keep. No B/P decode ever; decode work is `O(num_frames)` regardless of clip length. When `K_total < num_frames`, oversample the available keyframes (balanced via `np.round(np.linspace(...))`); metadata reports the true source-frame index of each returned keyframe so the temporal positional encoding stays honest.

`run6` = run4b config + `media_io_kwargs.video.video_backend = "pyav_keyframes_v2"`, no extra workers:

| Preset           | NExTQA wall · acc      | MVBench wall · acc      | Combined wall   | vs run4b cv2 |
|------------------|-----------------------:|------------------------:|----------------:|-------------:|
| run4b (cv2)      | 328.0 · 0.7960         | 340.9 · **0.6515**      | 668.9           | 1.00× |
| **run6**  | **183.4 · 0.7950**     | **197.1 · 0.5404**      | **380.5**       | **1.76× faster** |

Top MVBench subtask deltas vs cv2 — regression concentrates on motion/temporal-order tasks; non-motion subtasks are within ±2 pt or improve:

| Subtask                      | Δacc vs cv2      |
|------------------------------|-----------------:|
| `action_antonym`             | **−52.7 pt**     |
| `moving_attribute`           | −36.4 pt         |
| `object_existence`           | −36.4 pt         |
| `moving_count`               | −23.6 pt         |
| `counterfactual_inference`   | −23.6 pt         |
| **`action_prediction`**      | **+7.3 pt**      |
| `character_order`, `fine_grained_action` | +3.6 pt |

**`renderer_num_workers > 1` doesn't compose with v2** — adding workers=2 regressed combined wall 9 % (380 → 416 s). v2's per-clip decode is already 16-66 ms; thread coordination + the mandatory `mm_processor_cache_gb=0` cost more than the parallelism wins. Workers help only when single-thread decode is the bottleneck.

## Verdict

- **Motion-dense workloads** (action_*, moving_*, MVBench-style temporal reasoning) → `run5`: full accuracy, 1.55× over baseline `run1`.
- **NExTQA-style scene / state / identity QA, lossy OK** → `run6`: 2.75× over baseline `run1`, NExTQA preserved within noise, MVBench −6.3 pt overall (concentrated in subtasks that need intra-scene motion).

Per-run artifacts: `runs/<timestamp>_<preset>/{nextqa,mvbench}.jsonl` + `results.json`.
