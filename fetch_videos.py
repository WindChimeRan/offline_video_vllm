"""Download and extract the video archives for NExTQA and MVBench.

The raw QA parquets (from load_dataset) don't include the video files; both
datasets ship videos as zipped archives in the HF repo. This script:

  1. Reads samples/{nextqa,mvbench}.jsonl to know which video filenames are
     actually needed.
  2. Downloads the NExTQA videos.zip (one archive) and the MVBench video/*.zip
     archives (11 archives; ~17 GB total).
  3. Extracts each zip into videos/{nextqa,mvbench}/ preserving subfolder
     structure (MVBench JSONs reference paths like 'test_humor/foo.mp4').
  4. Verifies every sampled filename exists on disk; raises if any are missing.

Per CLAUDE.md: no silent try/except. If an archive is missing or a sampled
video can't be found, the script raises loudly so we can fix the mapping.
"""

import json
import os
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).parent
CACHE_DIR = ROOT / "hf_cache"
CACHE_DIR.mkdir(exist_ok=True)
os.environ["HF_HOME"] = str(CACHE_DIR)
os.environ["HF_HUB_CACHE"] = str(CACHE_DIR / "hub")

from huggingface_hub import hf_hub_download

VIDEOS_DIR = ROOT / "videos"
NEXTQA_DIR = VIDEOS_DIR / "nextqa"
MVBENCH_DIR = VIDEOS_DIR / "mvbench"
SAMPLES_DIR = ROOT / "samples"

MVBENCH_ARCHIVES = [
    "video/FunQA_test.zip",
    "video/Moments_in_Time_Raw.zip",
    "video/clevrer.zip",
    "video/data0613.zip",
    "video/perception.zip",
    "video/scene_qa.zip",
    "video/ssv2_video.zip",
    "video/sta.zip",
    "video/star.zip",
    "video/tvqa.zip",
    "video/vlnqa.zip",
]


def download_and_extract(repo_id: str, path_in_repo: str, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"  downloading {repo_id}:{path_in_repo} ...", flush=True)
    local = hf_hub_download(repo_id, path_in_repo, repo_type="dataset")
    size_gb = Path(local).stat().st_size / 1e9
    print(f"    -> {local} ({size_gb:.2f} GB); extracting ...", flush=True)
    with zipfile.ZipFile(local) as zf:
        zf.extractall(out_dir)
    print(f"    extracted to {out_dir}", flush=True)


def needed_videos(jsonl_path: Path) -> set[str]:
    names = set()
    with open(jsonl_path) as f:
        for line in f:
            r = json.loads(line)
            names.add(r["video"])
    return names


def find_video(root: Path, rel_name: str) -> Path | None:
    # First try the literal relative path
    p = root / rel_name
    if p.exists():
        return p
    # Fall back to a basename search (some archives nest inside a top folder
    # like 'clevrer/video_00001.mp4' while the JSON gives 'video_00001.mp4')
    base = Path(rel_name).name
    matches = list(root.rglob(base))
    if matches:
        return matches[0]
    return None


def verify(root: Path, needed: set[str], label: str) -> list[str]:
    missing = []
    for name in needed:
        if find_video(root, name) is None:
            missing.append(name)
    print(f"  {label}: {len(needed) - len(missing)}/{len(needed)} videos found on disk")
    if missing:
        print(f"    first 5 missing: {missing[:5]}")
    return missing


def main():
    print("=" * 80)
    print("NExTQA: downloading videos.zip (~6.5 GB) ...")
    print("=" * 80)
    download_and_extract("lmms-lab/NExTQA", "videos.zip", NEXTQA_DIR)

    print()
    print("=" * 80)
    print("MVBench: downloading 11 video archives (~17 GB total) ...")
    print("=" * 80)
    for archive in MVBENCH_ARCHIVES:
        download_and_extract("OpenGVLab/MVBench", archive, MVBENCH_DIR)

    print()
    print("=" * 80)
    print("Verifying sampled video filenames exist on disk ...")
    print("=" * 80)
    missing_nextqa = verify(NEXTQA_DIR, needed_videos(SAMPLES_DIR / "nextqa.jsonl"), "nextqa")
    missing_mvbench = verify(MVBENCH_DIR, needed_videos(SAMPLES_DIR / "mvbench.jsonl"), "mvbench")

    if missing_nextqa or missing_mvbench:
        raise RuntimeError(
            f"Missing videos after download: nextqa={len(missing_nextqa)}, "
            f"mvbench={len(missing_mvbench)}. See stdout for examples."
        )

    total_gb = sum(p.stat().st_size for p in VIDEOS_DIR.rglob("*") if p.is_file()) / 1e9
    print(f"\nAll sampled videos present. Total extracted size: {total_gb:.2f} GB")


if __name__ == "__main__":
    main()
