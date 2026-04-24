"""Run Qwen2.5-VL-7B-Instruct on the 100+95 sampled NExTQA / MVBench rows
via vLLM's chat API. Videos are passed as `file://` URLs; vLLM's connector
decodes them and the HF processor applies fps=0.5. No manual frame decode
or prompt-template assembly.

- GPU 4 only (set CUDA_VISIBLE_DEVICES=4 before running).
- max_tokens=1: only the single letter A/B/C/D/E is needed.
- Per CLAUDE.md: no silent try/except.
"""

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
TARGET_FPS = 0.5
MAX_FRAMES = 64
# Per-frame pixel budget after resize (Qwen2-VL patch basis = 28*28).
# 32*28*28 ≈ 25k pixels (~158×158) → floor so small clips don't upsample.
# 256*28*28 ≈ 200k pixels (~448×448) → ceil so big clips (1080p MVBench) downsample.
MIN_PIXELS = 32 * 28 * 28
MAX_PIXELS = 256 * 28 * 28
LETTERS = "ABCDE"

VIDEOS_DIR = ROOT / "videos"
SAMPLES_DIR = ROOT / "samples"
RUNS_DIR = ROOT / "runs"


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
    """Drop rows whose 'video' resolves to a directory (e.g. MVBench's
    episodic_reasoning uses TVQA frame-directories, not single mp4 files).
    Prints every dropped row so nothing is lost silently.
    """
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
            # RequestStateStats mixes wall-clock (arrival_time) and monotonic
            # (*_ts) fields; keep the clocks separate.
            ttft = m.first_token_latency  # precomputed TTFT
            ttfts.append(ttft)
            if m.last_token_ts and m.scheduled_ts:
                lat = m.last_token_ts - m.scheduled_ts  # in-engine inference time
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
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RUNS_DIR / ts
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run dir: {run_dir}")
    print(f"Model:   {MODEL_ID}")
    print(f"fps:     {TARGET_FPS} (frames capped at {MAX_FRAMES})")
    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")

    t_load_start = time.perf_counter()
    llm = LLM(
        model=MODEL_ID,
        max_model_len=32768,
        limit_mm_per_prompt={"video": 1},
        dtype="bfloat16",
        gpu_memory_utilization=0.9,
        tensor_parallel_size=1,
        trust_remote_code=True,
        allowed_local_media_path=str(VIDEOS_DIR.resolve()),
        mm_processor_kwargs={
            "fps": TARGET_FPS,
            "min_pixels": MIN_PIXELS,
            "max_pixels": MAX_PIXELS,
        },
        media_io_kwargs={"video": {"num_frames": MAX_FRAMES, "fps": TARGET_FPS}},
        disable_log_stats=False,  # needed so RequestOutput.metrics is populated
    )
    engine_load_s = time.perf_counter() - t_load_start
    print(f"Engine load: {engine_load_s:.1f} s")
    sp = SamplingParams(temperature=0.0, max_tokens=1)

    nextqa_rows = filter_to_real_files(load_samples(SAMPLES_DIR / "nextqa.jsonl"))
    mvbench_rows = filter_to_real_files(load_samples(SAMPLES_DIR / "mvbench.jsonl"))

    results = {
        "model": MODEL_ID,
        "target_fps": TARGET_FPS,
        "max_frames": MAX_FRAMES,
        "max_tokens": 1,
        "engine_load_s": engine_load_s,
        "nextqa": run_dataset(llm, sp, nextqa_rows, run_dir / "nextqa.jsonl"),
        "mvbench": run_dataset(llm, sp, mvbench_rows, run_dir / "mvbench.jsonl"),
    }
    with open(run_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {run_dir / 'results.json'}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
