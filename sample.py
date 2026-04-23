"""Pick 100 examples from each dataset for the vLLM scoping run.

NExTQA: 100 random rows from config=MC / split=test.
MVBench: stratified 5 per subtask × 20 subtasks = 100 rows (train split).

Writes samples/{nextqa,mvbench}.jsonl with the raw metadata we need downstream
(video filename, question, candidates, gold answer, + subtask for MVBench).
"""

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
N_PER_DATASET = 100
OUT_DIR = ROOT / "samples"
OUT_DIR.mkdir(exist_ok=True)


def sample_nextqa():
    ds = load_dataset("lmms-lab/NExTQA", "MC", split="test")
    rng = random.Random(SEED)
    idxs = rng.sample(range(len(ds)), N_PER_DATASET)
    rows = []
    for i in idxs:
        ex = ds[i]
        candidates = [ex["a0"], ex["a1"], ex["a2"], ex["a3"], ex["a4"]]
        gold_idx = int(ex["answer"])  # 0..4
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
    out = OUT_DIR / "nextqa.jsonl"
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"nextqa: wrote {len(rows)} rows -> {out}")
    print(f"  sample keys: {list(rows[0].keys())}")
    print(f"  first row: {rows[0]}")


def sample_mvbench():
    subtasks = get_dataset_config_names("OpenGVLab/MVBench")
    # Skip fine_grained_pose — its videos come from NTU RGB+D which is NOT
    # redistributed on HF (README flags this). Keep the other 19 subtasks.
    skip = {"fine_grained_pose"}
    per_sub = N_PER_DATASET // len(subtasks)  # floor
    extra = N_PER_DATASET - per_sub * len(subtasks)  # 100 = 5*20, so extra=0

    rng = random.Random(SEED)
    rows = []
    dropped_subtasks = []
    for sub in subtasks:
        if sub in skip:
            dropped_subtasks.append(sub)
            continue
        ds = load_dataset("OpenGVLab/MVBench", sub, split="train")
        idxs = rng.sample(range(len(ds)), per_sub)
        for i in idxs:
            ex = ds[i]
            candidates = list(ex["candidates"])
            # In MVBench, `answer` is the gold answer text; find its index.
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
            # optional temporal fields on a few subtasks
            for k in ("start", "end", "accurate_start", "accurate_end"):
                if k in ex:
                    row[k] = ex[k]
            rows.append(row)

    out = OUT_DIR / "mvbench.jsonl"
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"mvbench: wrote {len(rows)} rows -> {out}")
    print(f"  dropped subtasks (NTU-licensed, not on HF): {dropped_subtasks}")
    print(f"  first row: {rows[0]}")


def main():
    sample_nextqa()
    print()
    sample_mvbench()


if __name__ == "__main__":
    main()
