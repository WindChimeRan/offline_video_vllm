# video_vllm

Offline vLLM video-QA on [NExTQA](https://huggingface.co/datasets/lmms-lab/NExTQA) (MC) and [MVBench](https://huggingface.co/datasets/OpenGVLab/MVBench), Qwen2.5-VL-7B-Instruct.

The contribution is **`pyav_keyframes`** — a lossy, keyframe-only vLLM video loader that bounds decode cost by the frame budget regardless of clip length. Upstreamed as [vllm-project/vllm#45203](https://github.com/vllm-project/vllm/pull/45203).

- [`pyav_keyframe_backend.py`](pyav_keyframe_backend.py) — the loader (registers `pyav_keyframes`).
- [`bench_matrix.py`](bench_matrix.py) — 3-loader eval harness (opencv / faithful `qwen2_vl` / `pyav_keyframes`); [`aggregate_matrix.py`](aggregate_matrix.py) builds the table.
- [`infer.py`](infer.py) — shared `llm.chat` harness helpers (reused by `bench_matrix.py`).
- [`upstream_bench/bench_real_loader.py`](upstream_bench/bench_real_loader.py) — synthetic decode-speed microbench (GPU-free).
- [`LARGE_SPEED_LOG.md`](LARGE_SPEED_LOG.md) — faithful 3-way results (N=1990).

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

# faithful 3-way (needs a vllm exposing the qwen2_vl loader, #45555; we used a
# from-source 0.23.1rc1 cu128 build — see LARGE_SPEED_LOG.md for the setup)
for L in opencv faithful keyframes; do
  env CUDA_VISIBLE_DEVICES=0 VLLM_USE_FLASHINFER_SAMPLER=0 \
    .venv/bin/python bench_matrix.py --model qwen2.5 --loader "$L" --samples-dir samples/large
done
.venv/bin/python aggregate_matrix.py      # build the 3-way table
```

Artifacts land under `runs/<timestamp>_<model>_<loader>/` as `{nextqa,mvbench}.jsonl` +
`results.json` (per-row predictions, per-subtask accuracy, TTFT / E2E /
throughput). Videos, HF cache, samples, and runs are gitignored.

The decode-speed microbench needs no GPU and no videos:

```sh
uv run python upstream_bench/bench_real_loader.py
```

## Result — faithful 3-way (Qwen2.5-VL, N=1990)

Three loaders, identical resolution/engine settings (vLLM 0.23.1rc1, `enforce_eager`); only the frame-sampling differs:

| Loader | NExTQA | MVBench | NExTQA req/s | MVBench req/s |
|---|---:|---:|---:|---:|
| `opencv` (uniform-32, default) | 81.1% | 66.8% | 1.98 | 1.93 |
| **`faithful`** (`qwen2_vl`, fps=2) | **82.7%** | **67.7%** | 1.06 | 1.73 |
| **`keyframes`** (ours) | 79.6% | 53.2% | **5.58** | **5.14** |

Against the **faithful** baseline ([#45555](https://github.com/vllm-project/vllm/pull/45555) — the groundtruth for #45203), `pyav_keyframes` costs **−3.1 pt NExTQA / −14.4 pt MVBench** for **~3–5× throughput**, the MVBench loss concentrated in motion / temporal-order subtasks (`action_antonym` −47.3 pt). Full breakdown in [`LARGE_SPEED_LOG.md`](LARGE_SPEED_LOG.md).

> **Qwen3-VL is deferred** — near-chance on this build for a still-open Qwen3-VL-specific reason (`_C` rebuild and transformers version both ruled out).

Notes: MVBench's `episodic_reasoning` (TVQA frame-dirs) and `fine_grained_pose`
(NTU RGB+D, manual download) are dropped — not shipped as single video files on
HF.
