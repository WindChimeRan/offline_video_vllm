# Speed Log — vLLM Offline Inference

**Setup:** Qwen/Qwen2.5-VL-7B-Instruct · 1× A100-80GB (GPU 4) · bf16 · `max_tokens=1` · `fps=0.5` · `max_frames=64`
**Data:** NExTQA MC test (100) + MVBench stratified 5/subtask (90; `fine_grained_pose` and `episodic_reasoning` skipped — NTU-licensed / TVQA frame-dir). Total N = 190.

| # | Change | Engine load (s) | Wall (s) | req/s | TTFT mean / p95 (s) | E2E mean (s) | Acc NExTQA / MVBench |
|---|--------|----------------:|---------:|------:|---------------------:|-------------:|---------------------:|
| 1 | Baseline: `llm.chat` + `file://` URLs; `mm_processor_kwargs={"fps":0.5}`; `media_io_kwargs={"video":{"num_frames":64,"fps":0.5}}`; `disable_log_stats=False` so per-request metrics are populated | 32.7 | 96.9 | 1.96 | 22.5 / 41.2 | 1.0 | 80.0% / 64.4% |
| 2 | +`mm_processor_kwargs.min_pixels = 32*28²`, `max_pixels = 256*28²` (frame budget ~158→448 px/side; downsamples MVBench 720p/1080p by 4-10×, NExTQA barely affected) | 29.6 | 73.6 | 2.58 | 17.4 / 33.6 | 0.22 | 81.0% / 60.0% |
| 3 | Drop `fps`, fix `num_frames=16` uniform in both dicts (deterministic prefill size). **NExTQA faster (fewer frames than fps=0.5 was giving for long clips); MVBench slower (16 > frame count fps gave for short clips).** | 30.0 | 71.1 | 2.67 | 16.3 / 30.6 | 0.24 | 79.0% / 61.1% |
| 4 | +`compilation_config.compile_mm_encoder=True` (torch.compile the ViT). `cudagraph_mm_encoder=True` intended but blocked by [vllm bug #38997](https://github.com/vllm-project/vllm/pull/38997) in 0.19.1 (missing `vllm.v1.worker.gpu.mm.encoder_cudagraph`; fix merged upstream, unreleased). Per-request E2E drops (NExTQA −12%, MVBench −32%) but engine load +38%; wall is flat. | 41.5 | 73.4 | 2.59 | 16.9 / 31.9 | 0.18 | 80.0% / 60.0% |
| 4b | Same as #4 + `cudagraph_mm_encoder=True` (local shim over PR #38997 paths; encoder CUDA graphs captured at 8192-token budget, 4 video items). Per-request E2E trims further on MVBench (0.188→0.176 s), but **wall is flat** — ViT time is no longer the bottleneck; remaining cost is CPU preprocessing + scheduling. | 36.9 | 75.0 | 2.53 | 17.6 / 32.8 | 0.18 | 80.0% / 60.0% |

_TTFT = arrival → first token (dominated by batch-queue wait in a single-shot batch). E2E = `last_token_ts − scheduled_ts`, monotonic clock, the actual per-request inference time._

Per-run artifacts live under `runs/<timestamp>/{nextqa,mvbench}.jsonl` + `results.json`.
