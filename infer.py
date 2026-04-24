"""Run Qwen2.5-VL-7B-Instruct on sampled NExTQA / MVBench rows via vLLM's
chat API. Videos are passed as `file://` URLs; vLLM's connector decodes them
and the HF processor handles temporal sampling + normalization. No manual
frame decoding or prompt-template assembly.

- GPU 4 only (set CUDA_VISIBLE_DEVICES=4 before running).
- max_tokens=1: only the single letter A/B/C/D/E is needed.
- Per CLAUDE.md: no silent try/except.

CLI:
    python infer.py --preset run4b --samples-dir samples/small
    python infer.py --preset run1  --samples-dir samples/large

Presets correspond to rows in MINI_SPEED_LOG.md / LARGE_SPEED_LOG.md.
"""

import argparse
import datetime as dt
import json
import os
import statistics
import time
from pathlib import Path

ROOT = Path(__file__).parent
CACHE_DIR = ROOT / "hf_cache"
os.environ.setdefault("HF_HOME", str(CACHE_DIR))
os.environ.setdefault("HF_HUB_CACHE", str(CACHE_DIR / "hub"))

from vllm import LLM, SamplingParams

MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
LETTERS = "ABCDE"

VIDEOS_DIR = ROOT / "videos"
RUNS_DIR = ROOT / "runs"

MIN_PIXELS = 32 * 28 * 28    # ~158 px/side floor; no upsampling of tiny clips
MAX_PIXELS = 256 * 28 * 28   # ~448 px/side ceil; downsamples 720p/1080p
NUM_FRAMES = 16              # fixed uniform sample budget

# Each preset corresponds to one row in the SPEED_LOG tables.
PRESETS: dict[str, dict] = {
    "run1": {
        "mm_processor_kwargs": {"fps": 0.5},
        "media_io_kwargs": {"video": {"num_frames": 64, "fps": 0.5}},
        "compilation_config": {},
    },
    "run2": {
        "mm_processor_kwargs": {
            "fps": 0.5,
            "min_pixels": MIN_PIXELS,
            "max_pixels": MAX_PIXELS,
        },
        "media_io_kwargs": {"video": {"num_frames": 64, "fps": 0.5}},
        "compilation_config": {},
    },
    "run3": {
        "mm_processor_kwargs": {
            "num_frames": NUM_FRAMES,
            "min_pixels": MIN_PIXELS,
            "max_pixels": MAX_PIXELS,
        },
        "media_io_kwargs": {"video": {"num_frames": NUM_FRAMES}},
        "compilation_config": {},
    },
    "run4b": {
        "mm_processor_kwargs": {
            "num_frames": NUM_FRAMES,
            "min_pixels": MIN_PIXELS,
            "max_pixels": MAX_PIXELS,
        },
        "media_io_kwargs": {"video": {"num_frames": NUM_FRAMES}},
        "compilation_config": {
            # Needs vllm PR #38997 import paths; local shim installed in
            # .venv/.../vllm/v1/worker/gpu/mm/ for 0.19.1.
            "compile_mm_encoder": True,
            "cudagraph_mm_encoder": True,
        },
    },
}


def resolve_video(dataset: str, video_name: str) -> Path:
    root = VIDEOS_DIR / dataset
    p = root / video_name
    if p.exists():
        return p.resolve()
    base = Path(video_name).name
    matches = list(root.rglob(base))
    if matches:
        return matches[0].resolve()
    if not Path(video_name).suffix:
        for ext in (".mp4", ".mkv", ".webm", ".avi"):
            matches = list(root.rglob(base + ext))
            if matches:
                return matches[0].resolve()
    raise FileNotFoundError(f"video {video_name!r} not found under {root}")


def build_conversation(row: dict) -> list[dict]:
    path = resolve_video(row["dataset"], row["video"])
    opts = "\n".join(f"{LETTERS[i]}) {c}" for i, c in enumerate(row["candidates"]))
    text = f"{row['question']}\nOptions:\n{opts}\nAnswer with a single letter."
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": [
                {"type": "video_url", "video_url": {"url": f"file://{path}"}},
                {"type": "text", "text": text},
            ],
        },
    ]


def parse_letter(text: str, n_candidates: int) -> str | None:
    for ch in text.strip().upper():
        if ch in LETTERS[:n_candidates]:
            return ch
    return None


def load_samples(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def filter_to_real_files(rows: list[dict]) -> list[dict]:
    kept, dropped = [], []
    for r in rows:
        p = resolve_video(r["dataset"], r["video"])
        if p.is_dir():
            dropped.append((r.get("subtask"), r["video"], str(p)))
            continue
        kept.append(r)
    if dropped:
        print(f"  dropping {len(dropped)} row(s) whose path is a directory:")
        for sub, name, path in dropped:
            print(f"    [{sub}] {name}  ->  {path}")
    return kept


def _percentile(values: list[float], p: float) -> float:
    s = sorted(values)
    k = max(0, min(len(s) - 1, int(round((p / 100) * (len(s) - 1)))))
    return s[k]


def run_dataset(llm: LLM, sp: SamplingParams, rows: list[dict], out_path: Path) -> dict:
    label = rows[0]["dataset"]
    print(f"\n{label}: {len(rows)} rows -> llm.chat ...", flush=True)
    conversations = [build_conversation(r) for r in rows]

    t0 = time.perf_counter()
    outputs = llm.chat(conversations, sampling_params=sp)
    total_time = time.perf_counter() - t0

    correct = 0
    per_subtask: dict[str, list[int]] = {}
    records = []
    ttfts: list[float] = []
    e2e_latencies: list[float] = []
    total_gen_tokens = 0
    for row, out in zip(rows, outputs):
        pred_text = out.outputs[0].text
        pred_letter = parse_letter(pred_text, len(row["candidates"]))
        is_correct = (pred_letter == row["gold_letter"])
        correct += int(is_correct)
        total_gen_tokens += len(out.outputs[0].token_ids)

        m = getattr(out, "metrics", None)
        ttft = None
        lat = None
        if m is not None:
            ttft = m.first_token_latency
            ttfts.append(ttft)
            if m.last_token_ts and m.scheduled_ts:
                lat = m.last_token_ts - m.scheduled_ts
                e2e_latencies.append(lat)

        records.append({
            **row,
            "pred_text": pred_text,
            "pred_letter": pred_letter,
            "correct": is_correct,
            "ttft_s": ttft,
            "e2e_latency_s": lat,
        })
        if row["dataset"] == "mvbench":
            per_subtask.setdefault(row["subtask"], []).append(int(is_correct))

    with open(out_path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    accuracy = correct / len(rows)
    req_per_sec = len(rows) / total_time
    tok_per_sec = total_gen_tokens / total_time

    print(f"  -> {out_path}")
    print(f"  accuracy: {correct}/{len(rows)} = {accuracy:.4f}")
    print(f"  total wall: {total_time:.2f} s")
    print(f"  requests/s: {req_per_sec:.3f}")
    print(f"  tokens generated: {total_gen_tokens}")
    print(f"  throughput:  {tok_per_sec:.3f} tok/s")
    if ttfts:
        print(f"  TTFT (s)  mean={statistics.mean(ttfts):.3f} "
              f"median={statistics.median(ttfts):.3f} "
              f"p95={_percentile(ttfts, 95):.3f} "
              f"min={min(ttfts):.3f} max={max(ttfts):.3f}")
    if e2e_latencies:
        print(f"  E2E  (s)  mean={statistics.mean(e2e_latencies):.3f} "
              f"median={statistics.median(e2e_latencies):.3f} "
              f"p95={_percentile(e2e_latencies, 95):.3f}", flush=True)

    summary = {
        "n": len(rows),
        "correct": correct,
        "accuracy": accuracy,
        "wall_time_s": total_time,
        "requests_per_sec": req_per_sec,
        "tokens_generated": total_gen_tokens,
        "throughput_tok_per_sec": tok_per_sec,
    }
    if ttfts:
        summary["ttft_s"] = {
            "mean": statistics.mean(ttfts),
            "median": statistics.median(ttfts),
            "p95": _percentile(ttfts, 95),
            "min": min(ttfts),
            "max": max(ttfts),
        }
    if e2e_latencies:
        summary["e2e_latency_s"] = {
            "mean": statistics.mean(e2e_latencies),
            "median": statistics.median(e2e_latencies),
            "p95": _percentile(e2e_latencies, 95),
        }
    if per_subtask:
        summary["per_subtask_accuracy"] = {
            sub: sum(v) / len(v) for sub, v in per_subtask.items()
        }
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", required=True, choices=list(PRESETS.keys()))
    ap.add_argument("--samples-dir", type=Path, default=ROOT / "samples" / "small")
    ap.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    args = ap.parse_args()

    cfg = PRESETS[args.preset]

    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.runs_dir / f"{ts}_{args.preset}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run dir: {run_dir}")
    print(f"Preset : {args.preset}")
    print(f"Model  : {MODEL_ID}")
    print(f"Samples: {args.samples_dir}")
    print(f"cfg    : {cfg}")
    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")

    t_load_start = time.perf_counter()
    llm = LLM(
        model=MODEL_ID,
        # 64k is enough for uncapped Qwen2.5-VL video (up to ~37k vision
        # tokens observed on 1080p MVBench clips with fps=0.5 + num_frames=64
        # and no max_pixels). Capped presets use far less, but we set it
        # once here so run1 baseline doesn't need a special case.
        max_model_len=65536,
        limit_mm_per_prompt={"video": 1},
        dtype="bfloat16",
        gpu_memory_utilization=0.9,
        tensor_parallel_size=1,
        trust_remote_code=True,
        allowed_local_media_path=str(VIDEOS_DIR.resolve()),
        disable_log_stats=False,
        **cfg,
    )
    engine_load_s = time.perf_counter() - t_load_start
    print(f"Engine load: {engine_load_s:.1f} s")
    sp = SamplingParams(temperature=0.0, max_tokens=1)

    nextqa_rows = filter_to_real_files(load_samples(args.samples_dir / "nextqa.jsonl"))
    mvbench_rows = filter_to_real_files(load_samples(args.samples_dir / "mvbench.jsonl"))

    results = {
        "model": MODEL_ID,
        "preset": args.preset,
        "samples_dir": str(args.samples_dir),
        "config": cfg,
        "max_tokens": 1,
        "engine_load_s": engine_load_s,
        "nextqa": run_dataset(llm, sp, nextqa_rows, run_dir / "nextqa.jsonl"),
        "mvbench": run_dataset(llm, sp, mvbench_rows, run_dir / "mvbench.jsonl"),
    }
    with open(run_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nWrote {run_dir / 'results.json'}")


if __name__ == "__main__":
    main()
