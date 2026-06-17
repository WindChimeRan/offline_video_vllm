# video_vllm

Offline vLLM video-QA on [NExTQA](https://huggingface.co/datasets/lmms-lab/NExTQA) (MC) and [MVBench](https://huggingface.co/datasets/OpenGVLab/MVBench), Qwen2.5-VL-7B-Instruct.

The contribution is **`pyav_keyframes`** — a lossy, keyframe-only vLLM video loader that bounds decode cost by the frame budget regardless of clip length. Upstreamed as [vllm-project/vllm#45203](https://github.com/vllm-project/vllm/pull/45203).

- [`pyav_keyframe_backend.py`](pyav_keyframe_backend.py) — the loader (registers `pyav_keyframes`).
- [`infer.py`](infer.py) — the eval harness (`llm.chat` + `file://` URLs, two presets).
- [`upstream_bench/bench_real_loader.py`](upstream_bench/bench_real_loader.py) — synthetic decode-speed microbench (GPU-free).
- [`LARGE_SPEED_LOG.md`](LARGE_SPEED_LOG.md) — `baseline` vs `keyframes` results (N=1990).

## What `pyav_keyframes` does

One demux-only pass enumerates keyframe (I-frame) PTS — packet headers only, no
decode — then it seek+decodes *only* the `num_frames` keyframes it keeps. No P/B
frame is ever decoded, so decode cost is `O(num_frames)` whatever the clip
length. If a clip has fewer keyframes than `num_frames`, the available keyframes
are oversampled (balanced duplication) rather than falling back to non-keyframe
decode; `frames_indices` reports each returned keyframe's true source index, so
temporal positional encoding stays honest.

**Lossy:** returned frames sit on GOP boundaries (scene cuts), not a uniform
stride. Scene/state QA is unaffected; motion- and temporal-order tasks degrade
(see [`LARGE_SPEED_LOG.md`](LARGE_SPEED_LOG.md)).

## Quick start — drop `pyav_keyframes` into your own vLLM script

The loader is a single file, no native code; importing the module *is* the
install (the `@VIDEO_LOADER_REGISTRY.register(...)` decorator runs at import
time). Needs `av >= 12.0.0` (we use 17.0.1) and a vllm whose
`vllm.multimodal.video` exposes `VIDEO_LOADER_REGISTRY` and `VideoLoader`
(verified on 0.19.1 and 0.20.x). Import the module **before** `LLM(...)` — or the
engine fails the registry lookup — then point the loader at it:

```python
import pyav_keyframe_backend  # noqa: F401 — side-effect import (registers the loader)
from vllm import LLM

NUM_FRAMES = 16
llm = LLM(
    model="Qwen/Qwen2.5-VL-7B-Instruct",
    max_model_len=65536,
    limit_mm_per_prompt={"video": 1},
    dtype="bfloat16",
    trust_remote_code=True,
    allowed_local_media_path="/path/to/videos",   # if using file:// URLs
    mm_processor_kwargs={"num_frames": NUM_FRAMES, "max_pixels": 256 * 28 * 28},
    media_io_kwargs={"video": {"num_frames": NUM_FRAMES, "video_backend": "pyav_keyframes"}},
    compilation_config={"compile_mm_encoder": True, "cudagraph_mm_encoder": True},
)
```

`num_frames` appears in both kwarg blocks intentionally: the value in
`media_io_kwargs.video` tells the loader how many keyframes to return; the one in
`mm_processor_kwargs` tells the HF processor what to expect. They must match.

> Once #45203 merges, this loader ships inside vllm as `pyav_keyframes` — drop
> this file and just set `VLLM_VIDEO_LOADER_BACKEND=pyav_keyframes` (or the
> `video_backend` kwarg above).

## Reproduce the benchmark

Requirements: Python 3.11, `uv`, 1× GPU with ≥40 GB VRAM, ~50 GB free disk.

```sh
uv sync                                   # install deps
uv run python sample.py                   # pick the NExTQA + MVBench subset (seed=0)
uv run python fetch_videos.py             # download + extract videos (~24 GB)

# best lossless pipeline vs keyframe-only loader
env CUDA_VISIBLE_DEVICES=0 uv run python infer.py --preset baseline
env CUDA_VISIBLE_DEVICES=0 uv run python infer.py --preset keyframes
```

Artifacts land under `runs/<timestamp>_<preset>/` as `{nextqa,mvbench}.jsonl` +
`results.json` (per-row predictions, per-subtask accuracy, TTFT / E2E /
throughput). Videos, HF cache, samples, and runs are gitignored.

The decode-speed microbench needs no GPU and no videos:

```sh
uv run python upstream_bench/bench_real_loader.py
```

## Result (N=1990)

| Preset | Wall (s) | req/s | NExTQA | MVBench |
|---|---:|---:|---:|---:|
| `baseline` (cv2, lossless) | 674.0 | 2.95 | 79.6% | 65.3% |
| **`keyframes`** (lossy) | **380.5** | **5.23** | 79.5% | **54.0%** |

**1.77× throughput, NExTQA within noise (−0.1 pt), MVBench −11.3 pt** —
concentrated in motion / temporal-order subtasks (`action_antonym` −52.7 pt).
Full breakdown and verdict in [`LARGE_SPEED_LOG.md`](LARGE_SPEED_LOG.md).

Notes: MVBench's `episodic_reasoning` (TVQA frame-dirs) and `fine_grained_pose`
(NTU RGB+D, manual download) are dropped — not shipped as single video files on
HF.
