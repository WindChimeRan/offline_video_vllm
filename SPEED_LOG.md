# Speed Log — vLLM Offline Inference

**Setup:** Qwen/Qwen2.5-VL-7B-Instruct · 1× A100-80GB (GPU 4) · bf16 · `max_tokens=1` · `fps=0.5` · `max_frames=64`
**Data:** NExTQA MC test (100) + MVBench stratified 5/subtask (90; `fine_grained_pose` and `episodic_reasoning` skipped — NTU-licensed / TVQA frame-dir). Total N = 190.

| # | Change | Engine load (s) | Wall (s) | req/s | TTFT mean / p95 (s) | E2E mean (s) | Acc NExTQA / MVBench |
|---|--------|----------------:|---------:|------:|---------------------:|-------------:|---------------------:|
| 1 | Baseline: `llm.chat` + `file://` URLs; `mm_processor_kwargs={"fps":0.5}`; `media_io_kwargs={"video":{"num_frames":64,"fps":0.5}}`; `disable_log_stats=False` so per-request metrics are populated | 32.7 | 96.9 | 1.96 | 22.5 / 41.2 | 1.0 | 80.0% / 64.4% |
| 2 | +`mm_processor_kwargs.min_pixels = 32*28²`, `max_pixels = 256*28²` (frame budget ~158→448 px/side; downsamples MVBench 720p/1080p by 4-10×, NExTQA barely affected) | 29.6 | 73.6 | 2.58 | 17.4 / 33.6 | 0.22 | 81.0% / 60.0% |

_TTFT = arrival → first token (dominated by batch-queue wait in a single-shot batch). E2E = `last_token_ts − scheduled_ts`, monotonic clock, the actual per-request inference time._

Per-run artifacts live under `runs/<timestamp>/{nextqa,mvbench}.jsonl` + `results.json`.
