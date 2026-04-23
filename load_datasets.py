"""Download lmms-lab/NExTQA and OpenGVLab/MVBench and print basic stats.

Per CLAUDE.md: no silent try/except, fail loudly on any real error.
"""

import os
from pathlib import Path
from collections import Counter

CACHE_DIR = Path(__file__).parent / "hf_cache"
CACHE_DIR.mkdir(exist_ok=True)
os.environ["HF_HOME"] = str(CACHE_DIR)
os.environ["HF_DATASETS_CACHE"] = str(CACHE_DIR / "datasets")
os.environ["HF_HUB_CACHE"] = str(CACHE_DIR / "hub")

from datasets import get_dataset_config_names, load_dataset


def describe_split(split_name, ds):
    print(f"    split {split_name!r}: {len(ds):,} examples")
    print(f"      features: {list(ds.features.keys())}")
    sample = ds[0]
    print(f"      sample[0] keys -> values (truncated):")
    for k, v in sample.items():
        s = repr(v)
        if len(s) > 160:
            s = s[:157] + "..."
        print(f"        {k}: {s}")


def summarize_dataset(repo_id, label):
    print("=" * 80)
    print(f"{label}: {repo_id}")
    print("=" * 80)

    configs = get_dataset_config_names(repo_id)
    print(f"  configs ({len(configs)}): {configs}")

    # MVBench has ~20 sub-task configs; NExTQA typically has just 'default' / 'MC' / 'OE'.
    # Summarise each config: splits, sizes, features, one sample row.
    per_config_sizes = {}
    for cfg in configs:
        print(f"\n  [config={cfg}]")
        ds = load_dataset(repo_id, cfg)
        split_sizes = {s: len(ds[s]) for s in ds.keys()}
        per_config_sizes[cfg] = split_sizes
        print(f"    splits: {split_sizes}")

        first_split = next(iter(ds.keys()))
        describe_split(first_split, ds[first_split])

    # Aggregate totals across configs/splits so the user can eyeball scale.
    total = sum(sz for sizes in per_config_sizes.values() for sz in sizes.values())
    print(f"\n  TOTAL examples across all configs/splits: {total:,}")
    return per_config_sizes


def main():
    print(f"HF cache dir: {CACHE_DIR}")
    print()

    nextqa_sizes = summarize_dataset("lmms-lab/NExTQA", "NExTQA (lmms-lab)")
    print()
    mvbench_sizes = summarize_dataset("OpenGVLab/MVBench", "MVBench (OpenGVLab)")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("\nNExTQA splits per config:")
    for cfg, sizes in nextqa_sizes.items():
        print(f"  {cfg}: {sizes}")

    print("\nMVBench examples per config (sub-task):")
    flat = Counter()
    for cfg, sizes in mvbench_sizes.items():
        n = sum(sizes.values())
        flat[cfg] = n
        print(f"  {cfg}: {n} ({sizes})")
    print(f"\nMVBench grand total: {sum(flat.values()):,}")


if __name__ == "__main__":
    main()
