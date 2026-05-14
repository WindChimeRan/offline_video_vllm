# Speed Log — vLLM Offline Inference

**Setup:** Qwen/Qwen2.5-VL-7B-Instruct · 1× A100-80GB (GPU 4) · bf16 · `max_tokens=1`
**Data:** NExTQA MC test (100) + MVBench stratified 5/subtask (90; `fine_grained_pose` / `episodic_reasoning` skipped — NTU-licensed / TVQA frame-dir). Total N = 190.

| #  | Preset | Config delta                                                                                                  | Engine | Wall | req/s | TTFT mean / p95 | E2E  | Acc NExT / MVB |
|----|--------|---------------------------------------------------------------------------------------------------------------|------:|-----:|------:|----------------:|-----:|----------------|
| 1  | run1   | Baseline: `fps=0.5`, `num_frames=64`, no pixel cap                                                            | 32.7 | 96.9 | 1.96  | 22.5 / 41.2     | 1.0  | 80.0% / 64.4%  |
| 2  | run2   | +`min_pixels=32·28²`, `max_pixels=256·28²` (downsamples 720p/1080p ~4-10×)                                    | 29.6 | 73.6 | 2.58  | 17.4 / 33.6     | 0.22 | 81.0% / 60.0%  |
| 3  | run3   | Drop `fps`; fix `num_frames=16` uniform                                                                       | 30.0 | 71.1 | 2.67  | 16.3 / 30.6     | 0.24 | 79.0% / 61.1%  |
| 4  | run4   | +`compile_mm_encoder=True` (torch.compile the ViT)                                                            | 41.5 | 73.4 | 2.59  | 16.9 / 31.9     | 0.18 | 80.0% / 60.0%  |
| 4b | run4b  | +`cudagraph_mm_encoder=True` (local shim over [vllm PR #38997](https://github.com/vllm-project/vllm/pull/38997)) | 36.9 | 75.0 | 2.53  | 17.6 / 32.8     | 0.18 | 80.0% / 60.0%  |

_TTFT = arrival → first token. E2E = `last_token_ts − scheduled_ts`, per-request inference time._

At N=190 engine-load (~30 s) dominates wall; per-request wins (compile/cudagraph) don't surface until larger N. See [`LARGE_SPEED_LOG.md`](LARGE_SPEED_LOG.md) for the same ladder at N=1990 plus the lossy `pyav_keyframes_v2` result.
