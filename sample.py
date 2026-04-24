"""Pick N examples from each dataset for a vLLM offline run.

NExTQA: N random rows from config=MC / split=test (up to 8564 available).
MVBench: stratified N//18 per subtask × 18 usable subtasks.

Two MVBench subtasks are skipped up front because their videos aren't usable
via the llm.chat + file:// path:
  - fine_grained_pose (NTU RGB+D, not redistributed on HF)
  - episodic_reasoning (TVQA frames are pre-extracted directories, not videos)

CLI:
    uv run python sample.py --n 100  --out-dir samples/small
    uv run python sample.py --n 1000 --out-dir samples/large
"""

import argparse
import json
import os
import random
from pathlib import Path

ROOT = Path(__file__).parent
CACHE_DIR = ROOT / "hf_cache"
CACHE_DIR.mkdir(exist_ok=True)
os.environ["HF_HOME"] = str(CACHE_DIR)
os.environ["HF_DATASETS_CACHE"] = str(CACHE_DIR / "datasets")
os.environ["HF_HUB_CACHE"] = str(CACHE_DIR / "hub")

from datasets import get_dataset_config_names, load_dataset

SEED = 0
SKIP_SUBTASKS = {"fine_grained_pose", "episodic_reasoning"}


def sample_nextqa(n: int, out_dir: Path):
    ds = load_dataset("lmms-lab/NExTQA", "MC", split="test")
    rng = random.Random(SEED)
    n = min(n, len(ds))
    idxs = rng.sample(range(len(ds)), n)
    rows = []
    for i in idxs:
        ex = ds[i]
        candidates = [ex["a0"], ex["a1"], ex["a2"], ex["a3"], ex["a4"]]
        gold_idx = int(ex["answer"])
        rows.append({
            "dataset": "nextqa",
            "row_index": i,
            "qid": ex["qid"],
            "qtype": ex["type"],
            "video": f"{ex['video']}.mp4",
            "frame_count": ex["frame_count"],
            "width": ex["width"],
            "height": ex["height"],
            "question": ex["question"],
            "candidates": candidates,
            "gold_index": gold_idx,
            "gold_letter": "ABCDE"[gold_idx],
            "answer": candidates[gold_idx],
        })
    out = out_dir / "nextqa.jsonl"
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"nextqa: wrote {len(rows)} rows -> {out}")


def sample_mvbench(n: int, out_dir: Path):
    subtasks = [s for s in get_dataset_config_names("OpenGVLab/MVBench")
                if s not in SKIP_SUBTASKS]
    per_sub = n // len(subtasks)
    rng = random.Random(SEED)
    rows = []
    for sub in subtasks:
        ds = load_dataset("OpenGVLab/MVBench", sub, split="train")
        k = min(per_sub, len(ds))
        idxs = rng.sample(range(len(ds)), k)
        for i in idxs:
            ex = ds[i]
            candidates = list(ex["candidates"])
            gold_idx = candidates.index(ex["answer"])
            row = {
                "dataset": "mvbench",
                "subtask": sub,
                "row_index": i,
                "video": ex["video"],
                "question": ex["question"],
                "candidates": candidates,
                "gold_index": gold_idx,
                "gold_letter": "ABCDE"[gold_idx],
                "answer": ex["answer"],
            }
            for k_opt in ("start", "end", "accurate_start", "accurate_end"):
                if k_opt in ex:
                    row[k_opt] = ex[k_opt]
            rows.append(row)

    out = out_dir / "mvbench.jsonl"
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"mvbench: wrote {len(rows)} rows -> {out}  ({per_sub}/subtask × {len(subtasks)} subtasks)")
    print(f"  skipped subtasks: {sorted(SKIP_SUBTASKS)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100, help="target examples per dataset")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "samples" / "small",
                    help="directory to write {nextqa,mvbench}.jsonl")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    sample_nextqa(args.n, args.out_dir)
    print()
    sample_mvbench(args.n, args.out_dir)


if __name__ == "__main__":
    main()
