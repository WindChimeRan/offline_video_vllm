# video_vllm

Scoping vLLM offline inference for video QA — Qwen2.5-VL-7B-Instruct over [NExTQA](https://huggingface.co/datasets/lmms-lab/NExTQA) (MC) and [MVBench](https://huggingface.co/datasets/OpenGVLab/MVBench).

- [`infer.py`](infer.py) — the inference script (`llm.chat` + `file://` URLs, preset-driven config, no hand-rolled decode).
- [`pyav_keyframe_backend.py`](pyav_keyframe_backend.py) — custom vLLM video loaders. Ships **`pyav_keyframes_v2`**, the keyframe-only sampler used by our final preset `run4b_kf_v2`: single demux pass to enumerate keyframe PTS, then seek + decode only the `num_frames` keyframes we keep. Decode work is `O(num_frames)` regardless of clip length.
- [`MINI_SPEED_LOG.md`](MINI_SPEED_LOG.md) — small-scale (N≈190) tuning log.
- [`LARGE_SPEED_LOG.md`](LARGE_SPEED_LOG.md) — large-scale (N≈2000) tuning log + decode-backend exploration.

## Quick start — drop `pyav_keyframes_v2` into your own vLLM script

The loader is a single file with no native code; importing the module is the installation (the `@VIDEO_LOADER_REGISTRY.register(...)` decorator runs at import time and adds the loader to vLLM's registry). Three steps:

1. **Environment.** Need `av >= 12.0.0` (we use 17.0.1) and a `vllm` whose `vllm.multimodal.video` exposes `VIDEO_LOADER_REGISTRY` and `VideoLoader`. Verified on vllm 0.19.1 and 0.20.x. Sanity check:
   ```sh
   python -c "from vllm.multimodal.video import VIDEO_LOADER_REGISTRY, VideoLoader"
   ```

2. **Get the file.** Copy `pyav_keyframe_backend.py` somewhere on `PYTHONPATH` — typically next to whatever script constructs `LLM(...)`. No `pip install`, no setup.py.

3. **Wire it.** At the top of the script (must run **before** `LLM(...)` is constructed; otherwise the engine fails the registry lookup with `Extension class pyav_keyframes_v2 not found`):
   ```python
   import pyav_keyframe_backend  # noqa: F401  -- import for side-effect (registers loader)

   llm = LLM(
       model="Qwen/Qwen2.5-VL-7B-Instruct",
       media_io_kwargs={"video": {"video_backend": "pyav_keyframes_v2", "num_frames": 16}},
       ...
   )
   ```

That's the whole install. **Caveat:** this is a *lossy* sampler — frames are placed on GOP boundaries (scene cuts), not uniform stride. NExTQA-style scene QA is unaffected; motion-sensitive subtasks of MVBench can drop 35–50 pt. See the "Final call" table below for the empirical tradeoff before deciding.

## Reproduce the benchmark in this repo

Requirements: Python 3.11, `uv`, 1× GPU with ≥40 GB VRAM, ~50 GB free disk.

```sh
# 1. Install deps
uv sync

# 2. Pick 100 NExTQA + 95 MVBench rows (deterministic, seed=0)
uv run python sample.py

# 3. Download + extract the videos (~24 GB; skips NTU-licensed fine_grained_pose)
uv run python fetch_videos.py

# 4. Run Qwen2.5-VL-7B-Instruct on GPU 4 (first run also pulls ~16 GB of weights)
env CUDA_VISIBLE_DEVICES=4 uv run python infer.py --preset run4b_kf_v2
```

Artifacts land under `runs/<timestamp>_<preset>/` as `{nextqa,mvbench}.jsonl` + `results.json` (per-row predictions, per-subtask accuracy, TTFT / E2E / throughput). Videos, HF cache, samples, and runs are all gitignored.

## Final call: `run4b_kf_v2` (vs the `run1` baseline, N=1990)

`run1` = the original starting point: `mm_processor_kwargs={"fps":0.5}`, `num_frames=64`, no pixel cap, no `torch.compile`/CUDA-graph on the ViT, cv2 default decode.

| Metric              | `run1` (baseline) | **`run4b_kf_v2`** (final) | Change          |
|---------------------|------------------:|--------------------------:|----------------:|
| Combined wall (s)   | 1045.6            | **380.5**                 | **2.75× faster**|
| Throughput (req/s)  | 1.90              | **5.23**                  | +175 %          |
| TTFT mean (s)       | 252.5             | **91.4**                  | −64 %           |
| NExTQA accuracy     | 0.799             | 0.795                     | −0.4 pt (noise) |
| MVBench accuracy    | 0.603             | 0.540                     | −6.3 pt         |

What `run4b_kf_v2` stacks together: `max_pixels=256·28²` (downsample 1080p clips), fixed `num_frames=16` (drop the `fps=0.5` policy), `compile_mm_encoder + cudagraph_mm_encoder` (ViT compile + CUDA-graphs), **`pyav_keyframes_v2`** as the video backend (keyframe-only sampling, never pays B/P decode cost). The MVBench drop is concentrated in motion-sensitive subtasks (`action_antonym`, `moving_*`, `object_existence`) — keyframe sampling can't capture intra-scene motion. NExTQA-style scene/state QA is unaffected. If your downstream task is motion-dense, stay on `run5` (cv2 + parallel renderer) for full MVBench accuracy at 1.55× speedup over baseline; full breakdown in [`LARGE_SPEED_LOG.md`](LARGE_SPEED_LOG.md).

Notes: MVBench's `episodic_reasoning` (TVQA frame-dirs) and `fine_grained_pose` (NTU RGB+D, manual download) are dropped — they're not shipped as single video files on HF.
