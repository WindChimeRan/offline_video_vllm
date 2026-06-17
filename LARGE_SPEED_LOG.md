# Speed Log — faithful 3-way video-loader comparison (Qwen2.5-VL)

**Setup:** Qwen/Qwen2.5-VL-7B-Instruct · 1× A100-80GB · bf16 · `max_tokens=1` ·
`max_model_len=65536` · `enforce_eager` · vLLM `0.23.1rc1+gdf07fd276` (from
source, cu128) · torch 2.11.0+cu128.
**Data:** NExTQA MC test (1000) + MVBench 55/subtask × 18 (990). **N = 1990.**
**Harness:** [`bench_matrix.py`](bench_matrix.py) (one cell per `(model, loader)`).

Three loaders, identical resolution/engine settings — the only varied factor is
the frame-sampling strategy:

- **`opencv`** — vLLM default, uniform `num_frames=32`. What Qwen2.5-VL silently
  falls back to today (it ships no `video_processor_type`).
- **`faithful`** — the processor-mapped `qwen2_vl` loader, fps=2 (matches HF
  transformers byte-for-byte). The correct baseline, added by
  [vllm-project/vllm#45555](https://github.com/vllm-project/vllm/pull/45555).
- **`keyframes`** — `pyav_keyframes`, keyframe-only `num_frames=16` (lossy, ours,
  [#45203](https://github.com/vllm-project/vllm/pull/45203)).

| Loader | NExTQA | MVBench | NExTQA req/s | MVBench req/s |
|---|---:|---:|---:|---:|
| `opencv` (uniform-32, default) | 81.1% | 66.8% | 1.98 | 1.93 |
| **`faithful`** (`qwen2_vl`, fps=2) | **82.7%** | **67.7%** | 1.06 | 1.73 |
| **`keyframes`** (kf-16, ours) | 79.6% | 53.2% | **5.58** | **5.14** |

## The trade-off (vs the faithful baseline)

`pyav_keyframes` vs `qwen2_vl` — the honest, apples-to-apples comparison
(#45203's accuracy cost should be stated against #45555's groundtruth, not the
uniform-32 default):

| | NExTQA | MVBench | NExTQA req/s | MVBench req/s |
|---|---:|---:|---:|---:|
| **Δ keyframes − faithful** | **−3.1 pt** | **−14.4 pt** | **5.2×** | **3.0×** |

NExTQA (scene/state QA) is within ~3 pt; the MVBench cost is real and
concentrates in motion / temporal-order subtasks:

| Subtask | Δ acc (keyframes − faithful) |
|---|---:|
| `action_antonym` | **−47.3 pt** (83.6% → 36.4%) |
| `moving_attribute` | −38.2 pt (94.5% → 56.4%) |
| `object_existence` | −36.4 pt (92.7% → 56.4%) |
| `counterfactual_inference` | −29.1 pt (69.1% → 40.0%) |
| `object_interaction` | −27.3 pt (80.0% → 52.7%) |
| `moving_direction` | −20.0 pt (52.7% → 32.7%) |
| `moving_count` | −18.2 pt (67.3% → 49.1%) |

Note the default `opencv` (uniform-32) is itself ~1 pt below `faithful` — even
the "lossless" default under-samples relative to the correct fps=2 policy, which
is the gap #45555 closes.

## Verdict

- **Scene / state / identity QA, throughput-bound** → `keyframes`: ~4× the
  faithful baseline's throughput, NExTQA within ~3 pt.
- **Motion / temporal-order** (`action_*`, `moving_*`) → `faithful`: full
  accuracy.

## Notes

- req/s is per-dataset under `enforce_eager` (no ViT compile / CUDA-graph), so
  absolute numbers are lower than a compiled engine; the **relative** loader
  comparison is the result. The keyframe decode win is CPU-side (no P/B decode),
  independent of the GPU engine config.
- **Qwen3-VL is deferred.** On the same harness/build it sits at ~0.42
  (near-chance) regardless of `_C` rebuild or transformers version (5.6.2 ≡
  5.12.1), while Qwen2.5-VL is correct — a Qwen3-VL-specific issue still open
  (see [`QWEN3_VL_DEFERRED.md`](QWEN3_VL_DEFERRED.md)). The faithful loader for
  Qwen3-VL is the separate `qwen3_vl` (#44412).

Per-run artifacts: `runs/<ts>_<model>_<loader>/results.json`. Aggregate with
[`aggregate_matrix.py`](aggregate_matrix.py).
