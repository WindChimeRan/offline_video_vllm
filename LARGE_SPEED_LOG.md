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

## Decode-backend exploration (run4b base, GPU 7, 2026-05-13)

Four custom video loaders registered against `VIDEO_LOADER_REGISTRY` in [`pyav_keyframe_backend.py`](pyav_keyframe_backend.py), all anchored at run4b (`compile_mm_encoder` + `cudagraph_mm_encoder` + `num_frames=16` + `max_pixels=256·28²`). Unless noted, only `media_io_kwargs.video.video_backend` varies between rows.

| Preset            | Backend / policy                                          | Engine load | NExTQA wall | NExTQA acc | MVBench wall | MVBench acc | Combined wall | req/s |
|-------------------|------------------------------------------------------------|------------:|------------:|-----------:|-------------:|------------:|--------------:|------:|
| `run4b`           | cv2 default (uniform `cap.grab()`)                         | 59.5        | 328.0       | **0.7960** | 340.9        | **0.6515**  | **668.9**     | 2.98  |
| `run4b_pyav` †    | PyAV seek-based, **buggy** (vllm 0.20.x port)              | 44.0        | 171.8       | 0.7950     | 299.1        | 0.5354 (−11.6 pt) | 470.9 *(0.70×)* | 4.22  |
| `run4b_pyav` ✓    | PyAV seek-based, **fixed** (decode-forward-to-target_pts)  | 48.8        | 383.9       | 0.7960     | 2549.0       | 0.6273 (−2.4 pt)  | 2932.9 *(4.38×)* | 0.64  |
| `run4b_kf`        | PyAV keyframe-only, v1 (`skip_frame='NONKEY'` + np.stack)  | 48.9        | 475.0       | 0.7960     | 350.2        | 0.5606 (−9.1 pt)  | 825.2 *(1.23×)* | 2.41  |
| **`run4b_kf_v2`** | **PyAV keyframe-only, v2 (demux→pick→decode + oversample)**| **44.8**    | **183.4**   | **0.7950** | **197.1**    | **0.5404 (−11.1 pt)** | **380.5 (1.76× faster)** | **5.23** |
| `run4b_kf_v2_w2`  | v2 + `renderer_num_workers=2`                              | 47.3        | 210.1       | 0.7960     | 205.5        | 0.5374      | 415.6         | 4.79  |

† vllm 0.20.x default `pyav` behavior (port for our 0.19.1) — kept here as the "before" measurement; the bug fix has been filed upstream.
✓ Post-fix correctness regained (NExTQA accuracy matches cv2 exactly, MVBench within 2.4 pt), at the price of 4× wall time on long-GOP MVBench clips — see the v1 keyframe path's caveats below.

### Findings

**The "PyAV speedup" upstream sold was buying wrong frames.** Buggy `pyav_seek` (vllm 0.20.x default) is 1.42× faster than cv2 because PyAV's `seek(pts) + next(decode())` snaps to the keyframe at-or-before `pts`, returns that keyframe with the target's *label* in metadata, and never decodes B/P frames. On NExTQA (regular GOP=30) the snap distance is small → accuracy survives. On MVBench (long-GOP MVBench/Charades clips with K=2 per clip) 16 uniform targets collapse onto 2 distinct frames, repeated 8× each → **−11.6 pt** MVBench, with `action_antonym` going 89% → 47% (worse than random for a 2-option task). Filed upstream as a correctness bug.

**Once correct, PyAV seek is unambiguously slower than cv2.** The fix (decode-forward-to-exact-target-pts) restores accuracy but does ~`GOP/2` frame-decodes per target × 16 targets per clip. On long-GOP MVBench that's catastrophic — combined wall 2933 s (**4.4× slower than cv2**). The throughput win was *bought* by skipping required work. cv2's tight `cap.grab()` C++ loop turns out to be near-optimal for "exact uniform 16 frames."

**The v1 keyframe-only path didn't transfer the standalone-benchmark win.** Despite `sampling_video.py` measuring 7× per-clip, engine wall on N=1990 was **1.45× slower** than cv2 on NExTQA. Two reasons: (a) `skip_frame='NONKEY'` skips B/P *decode* but still demuxes every packet; (b) we `np.stack` every keyframe (often 20+) before picking 16, copying memory the engine never sees.

**The v2 keyframe path actually transfers the win.** Two changes vs v1:
1. **Single demux pass to enumerate keyframe PTS** (no decode), then **seek + decode only the `num_frames` we keep**. Decode work is `O(min(K_total, num_frames))` regardless of clip length.
2. **Oversample policy** when `K_total < num_frames`: pick keyframes via `np.round(linspace)` so a clip with K=2 returns 8 copies of KF[0] + 8 copies of KF[1] (balanced), each labeled with its true source frame index. Never falls back to B/P decode.

Result: **combined wall 380 s vs cv2's 669 s (1.76×)**, accuracy −11.1 pt MVBench / −0.1 pt NExTQA. Same accuracy hit as v1 (keyframe sampling is inherently lossy on motion-dense MVBench subtasks), at less than half the wall.

**`renderer_num_workers=2` doesn't compose with v2.** Adding parallel-decode workers (run5's winning trick on cv2) actually *regressed* v2 by 9% wall (380 → 416 s). Reason: v2's per-clip decode is already 16-66 ms — thread coordination + disabled MM-processor-cache cost more than the parallelism win. Workers only help when single-thread CPU decode is the bottleneck; v2 broke that bottleneck.

### Per-subtask MVBench breakdown for v2

Δacc = `run4b_kf_v2` − `run4b` (cv2 baseline). The regression is concentrated in motion-/temporal-order-sensitive subtasks, as expected from a keyframe-sampling policy.

| Subtask                  | cv2   | v2    | Δacc         |
|--------------------------|------:|------:|-------------:|
| `action_antonym`         | 0.891 | 0.364 | **−52.7 pt** |
| `moving_attribute`       | 0.927 | 0.564 | **−36.4 pt** |
| `object_existence`       | 0.927 | 0.564 | **−36.4 pt** |
| `counterfactual_inference` | 0.636 | 0.400 | **−23.6 pt** |
| `moving_count`           | 0.727 | 0.491 | −23.6 pt |
| `moving_direction`       | 0.527 | 0.345 | −18.2 pt |
| `object_interaction`     | 0.600 | 0.527 | −7.3 pt |
| `action_count`, `action_sequence`, `unexpected_action` | — | — | ±5 pt |
| **`action_prediction`**  | 0.691 | 0.764 | **+7.3 pt** |
| `character_order`, `fine_grained_action` | — | — | +3.6 pt |
| `scene_transition`, `state_change`, `egocentric_navigation`, `action_localization`, `object_shuffle` | — | — | ±2 pt |

Interpretation: when the answer depends on **dense temporal coverage of motion** (action_antonym, moving_*), keyframe sampling is fatal — it returns 2-16 scene-cut frames, the model can't see motion in between. When the answer depends on **scene/state/identity recognition** (scene_transition, character_order, action_prediction at the clip level), keyframe sampling is as good as or slightly better than uniform — KFs are placed at scene boundaries, which is *helpful* for those questions.

### Verdict & recommendation

- **For latency-sensitive video classification where MVBench-style motion tasks aren't critical**, ship `run4b_kf_v2` (the new v2 backend, no extra workers). Net: 1.76× wall, NExTQA parity, MVBench −11 pt (mostly recoverable for non-motion subtasks).
- **For accuracy-critical / motion-dense workloads**, stay on `run5` (cv2 + `renderer_num_workers=2` + `mm_processor_cache_gb=0`) — combined wall 674 s, full accuracy.
- **Don't use `pyav_seek` either way.** Buggy is faster but corrupts MVBench; fixed is correct but 4× slower than cv2.
- **Don't add `renderer_num_workers` to v2.** Composition is *negative* — workers help when decode is slow, v2 makes decode cheap.

### Action items

- [x] Filed upstream bug fix PR against vllm-project/vllm
- [x] Implemented and shipped v2 backend (`pyav_keyframes_v2`) — winner for lossy direction
- [ ] If a future workload tolerates more MVBench accuracy loss, can crank further: `num_frames=8` + v2 would probably go ~250 s combined wall

Per-run artifacts: `runs/20260513_*_run4b*/`.

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
