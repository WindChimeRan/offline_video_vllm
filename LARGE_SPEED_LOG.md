# Speed Log — `baseline` vs `keyframes`

**Setup:** Qwen/Qwen2.5-VL-7B-Instruct · 1× A100-80GB · bf16 · `max_tokens=1` · `max_model_len=65536`
**Data:** NExTQA MC test (1000) + MVBench stratified 55/subtask × 18 subtasks (990). **N = 1990.**

Two presets, identical except for the video loader (and the parallel-render
trick only the lossless path can use). The delta is the cost/benefit of the
`pyav_keyframes` keyframe-only loader on top of an already-optimized pipeline.

- **`baseline`** — best lossless config: cv2 decode, `max_pixels=256·28²`, fixed
  `num_frames=16`, ViT `compile_mm_encoder`/`cudagraph_mm_encoder`, parallel
  renderer (`renderer_num_workers=2`).
- **`keyframes`** — same pipeline, `video_backend=pyav_keyframes` (keyframe-only,
  lossy). Renderer workers drop to the default 1 (they don't compose with
  keyframe decode — see below).

| Preset | Loader | Wall (s) | req/s | TTFT mean / p95 (s) | NExTQA | MVBench |
|---|---|---:|---:|---:|---:|---:|
| `baseline` | cv2 (lossless) | 674.0 | 2.95 | 165.9 / 318.2 | 79.6% | 65.3% |
| **`keyframes`** | pyav_keyframes (lossy) | **380.5** | **5.23** | **87.0 / 167.2** | 79.5% | 54.0% |
| **Δ** | | **1.77× faster** | **+77%** | | **−0.1 pt** | **−11.3 pt** |

> Numbers are from the prior run (these rows were captured under the old
> `run5` / `run6` names). A clean back-to-back co-run under the renamed presets
> is pending. Git history (and the local `*.backup` files) keep the full
> `run1 → run6` tuning journey that found this `baseline`.

## The trade-off

`pyav_keyframes` does one demux-only pass to enumerate keyframe PTS, then
seek+decodes only the `num_frames` keyframes it keeps — decode cost is
`O(num_frames)` regardless of clip length, and no B/P frame is ever decoded.
The cost is temporal accuracy: frames land on GOP boundaries (scene cuts), not a
uniform stride. NExTQA-style scene/state QA is unaffected; the MVBench loss
concentrates in motion / temporal-order subtasks:

| Subtask | Δ acc vs `baseline` |
|---|---:|
| `action_antonym` | **−52.7 pt** |
| `moving_attribute` | −36.4 pt |
| `object_existence` | −36.4 pt |
| `moving_count` | −23.6 pt |
| `counterfactual_inference` | −23.6 pt |
| `action_prediction` | **+7.3 pt** |
| `character_order`, `fine_grained_action` | +3.6 pt |

10 of 18 MVBench subtasks stay within ±2 pt.

## Verdict

- **Scene / state / identity QA, lossy OK** (e.g. NExTQA) → `keyframes`: 1.77×
  throughput, accuracy within noise.
- **Motion-dense / temporal-order** (`action_*`, `moving_*`) → `baseline`: full
  lossless accuracy.

## Decode-speed microbench

The throughput gain above is end-to-end (HF resize/normalize still runs on every
frame, so it's smaller than the raw decode gain). For the isolated decode
cost — near-constant for keyframes vs growing with clip length for lossless
full-decode — see
[`upstream_bench/bench_real_loader.py`](upstream_bench/bench_real_loader.py).

Per-run artifacts: `runs/<timestamp>_<preset>/{nextqa,mvbench}.jsonl` + `results.json`.
