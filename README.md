# video_vllm

Scoping vLLM offline inference for video QA — Qwen2.5-VL-7B-Instruct over [NExTQA](https://huggingface.co/datasets/lmms-lab/NExTQA) (MC) and [MVBench](https://huggingface.co/datasets/OpenGVLab/MVBench).

- [`infer.py`](infer.py) — the inference script (`llm.chat` + `file://` URLs, `mm_processor_kwargs={"fps":0.5}`, no hand-rolled decode).
- [`SPEED_LOG.md`](SPEED_LOG.md) — per-tuning-iteration speed/accuracy log (TTFT, req/s, wall, E2E, accuracy).

## Quick start

Requirements: Python 3.11, `uv`, 1× GPU with ≥40 GB VRAM, ~50 GB free disk.

```sh
# 1. Install deps
uv sync

# 2. Pick 100 NExTQA + 95 MVBench rows (deterministic, seed=0)
uv run python sample.py

# 3. Download + extract the videos (~24 GB; skips NTU-licensed fine_grained_pose)
uv run python fetch_videos.py

# 4. Run Qwen2.5-VL-7B-Instruct on GPU 4 (first run also pulls ~16 GB of weights)
env CUDA_VISIBLE_DEVICES=4 uv run python infer.py
```

Artifacts land under `runs/<timestamp>/` as `{nextqa,mvbench}.jsonl` + `results.json` (per-row predictions, per-subtask accuracy, TTFT / E2E / throughput). Videos, HF cache, samples, and runs are all gitignored.

Notes: MVBench's `episodic_reasoning` (TVQA frame-dirs) and `fine_grained_pose` (NTU RGB+D, manual download) are dropped — they're not shipped as single video files on HF.
