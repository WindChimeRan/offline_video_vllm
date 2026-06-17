# Qwen3-VL — deferred (open issue)

The faithful 3-way ([`LARGE_SPEED_LOG.md`](LARGE_SPEED_LOG.md)) ships **Qwen2.5-VL only**.
Qwen3-VL-4B-Instruct sits at **near-chance** on this build and the cause is still
open.

## Symptom

On the identical harness ([`bench_matrix.py`](bench_matrix.py)) that gives
**Qwen2.5-VL 0.83 NExTQA**, Qwen3-VL-4B gets **~0.40 NExTQA / ~0.48 MVBench**
(80-sample smoke). The model emits **clean single letters** — they're just wrong
(not a parsing / `max_tokens` artifact). So the harness and build are sound; the
problem is Qwen3-VL-specific.

## Ruled out (2026-06-17)

A prior note blamed the env (stale compiled `_C` vs newer Python). Both leading
hypotheses are now **disproven**:

1. **Python/`_C` skew** — rebuilt `_C` from source at HEAD `df07fd276` against
   torch 2.11.0+cu128 (consistent build, [`project_vllm_main_build_recipe`]).
   Result unchanged (0.375/0.475). Not the cause.
2. **transformers version** — 5.6.2 and 5.12.1 give **identical** predictions
   (0.375/0.475). Not the cause.

## Next diagnostics (for whoever picks this up)

- **Text-only control:** run the same questions with no video. If accuracy is
  unchanged, the video is being ignored (token-insertion / processing bug); if
  it drops to pure chance, the video is used but ineffective.
- Compare the `opencv` (uniform-32) vs `qwen3_vl` (fps=2) loaders for Qwen3-VL.
- Resolution / fps sweep (the `size` config), and Qwen3-VL's timestamp-aware
  video handling.
- A different Qwen3-VL checkpoint or a known-good transformers pin.

The faithful Qwen3-VL baseline loader is **`qwen3_vl`** (vllm#44412, merged) —
distinct from Qwen2.5-VL's `qwen2_vl` (#45555).
