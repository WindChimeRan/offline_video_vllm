"""3x2 video-loader benchmark matrix.

Loaders x models:
  - opencv    : vLLM default, uniform num_frames=32 (the "bad" baseline)
  - faithful  : processor-mapped fps=2 loader (qwen2_vl / qwen3_vl) -- the
                correct baseline (#45555 / #44412)
  - keyframes : pyav_keyframes, keyframe-only num_frames=16 (ours, #45203)
  x {Qwen2.5-VL-7B, Qwen3-VL-4B}

One (model, loader) cell per invocation so cells run on separate GPUs. All cells
use enforce_eager + identical resolution/engine settings within a model, so the
only varied factor is the frame-sampling strategy.

Usage (flashinfer sampler off -> avoids the ninja JIT; identical for greedy):
  VLLM_USE_FLASHINFER_SAMPLER=0 CUDA_VISIBLE_DEVICES=2 \
    .venv/bin/python bench_matrix.py --model qwen2.5 --loader faithful \
      --samples-dir samples/large
"""
import argparse
import datetime as dt
import json
import time
from pathlib import Path

# infer.py sets HF_HOME and exposes the harness helpers; importing it also
# registers pyav_keyframes (side-effect import of pyav_keyframe_backend).
from infer import (
    build_conversation,  # noqa: F401  (used transitively via run_dataset)
    filter_to_real_files,
    load_samples,
    parse_letter,  # noqa: F401
    run_dataset,
    RUNS_DIR,
    VIDEOS_DIR,
)

from vllm import LLM, SamplingParams

ROOT = Path(__file__).parent
NUM_FRAMES_KF = 16
NUM_FRAMES_OPENCV = 32
FPS_FAITHFUL = 2
MIN_PIXELS = 32 * 28 * 28
MAX_PIXELS = 256 * 28 * 28

MODELS = {
    "qwen2.5": {
        "model_id": "Qwen/Qwen2.5-VL-7B-Instruct",
        "resolution_kwargs": {"min_pixels": MIN_PIXELS, "max_pixels": MAX_PIXELS},
        "faithful_backend": "qwen2_vl",
    },
    "qwen3": {
        "model_id": "Qwen/Qwen3-VL-4B-Instruct",
        "resolution_kwargs": {"size": {"shortest_edge": 4096, "longest_edge": 128 * 28 * 28}},
        "faithful_backend": "qwen3_vl",
    },
}


def build_kwargs(model_key: str, loader: str) -> tuple[dict, dict]:
    """Return (mm_processor_kwargs, media_io_kwargs) for a (model, loader) cell.

    Frame-count loaders (opencv/keyframes) carry num_frames in both blocks so
    the HF processor and the loader agree; the faithful loader is fps-driven.
    """
    m = MODELS[model_key]
    res = dict(m["resolution_kwargs"])
    if loader == "opencv":
        mm = {**res, "num_frames": NUM_FRAMES_OPENCV}
        video = {"video_backend": "opencv", "num_frames": NUM_FRAMES_OPENCV}
    elif loader == "faithful":
        mm = res  # fps-based; the loader drives the frame count
        video = {"video_backend": m["faithful_backend"], "fps": FPS_FAITHFUL}
    elif loader == "keyframes":
        mm = {**res, "num_frames": NUM_FRAMES_KF}
        video = {"video_backend": "pyav_keyframes", "num_frames": NUM_FRAMES_KF}
    else:
        raise ValueError(f"unknown loader {loader!r}")
    return mm, {"video": video}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--loader", required=True, choices=["opencv", "faithful", "keyframes"])
    ap.add_argument("--samples-dir", type=Path, default=ROOT / "samples" / "large")
    ap.add_argument("--limit", type=int, default=None, help="cap rows/dataset (smoke)")
    args = ap.parse_args()

    m = MODELS[args.model]
    mm_kwargs, media_kwargs = build_kwargs(args.model, args.loader)

    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RUNS_DIR / f"{ts}_{args.model}_{args.loader}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[matrix] model={args.model} loader={args.loader}")
    print(f"[matrix] mm_processor_kwargs={mm_kwargs}")
    print(f"[matrix] media_io_kwargs={media_kwargs}")

    import vllm
    print(f"[matrix] vllm {vllm.__version__} @ {vllm.__file__}", flush=True)

    t0 = time.perf_counter()
    llm = LLM(
        model=m["model_id"],
        max_model_len=65536,
        max_num_batched_tokens=65536,
        limit_mm_per_prompt={"video": 1},
        dtype="bfloat16",
        gpu_memory_utilization=0.9,
        tensor_parallel_size=1,
        enforce_eager=True,
        trust_remote_code=True,
        allowed_local_media_path=str(VIDEOS_DIR.resolve()),
        mm_processor_kwargs=mm_kwargs,
        media_io_kwargs=media_kwargs,
    )
    engine_load_s = time.perf_counter() - t0
    print(f"[matrix] engine load {engine_load_s:.1f}s", flush=True)

    sp = SamplingParams(temperature=0.0, max_tokens=1)
    results = {
        "model": m["model_id"],
        "model_key": args.model,
        "loader": args.loader,
        "mm_processor_kwargs": mm_kwargs,
        "media_io_kwargs": media_kwargs,
        "engine_load_s": engine_load_s,
        "enforce_eager": True,
    }
    for name in ("nextqa.jsonl", "mvbench.jsonl"):
        rows = filter_to_real_files(load_samples(args.samples_dir / name))
        if args.limit:
            rows = rows[: args.limit]
        results[name.replace(".jsonl", "")] = run_dataset(llm, sp, rows, run_dir / name)

    with open(run_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"[matrix] wrote {run_dir / 'results.json'}", flush=True)


if __name__ == "__main__":
    main()
