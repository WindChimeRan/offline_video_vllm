"""Aggregate runs/*_<model>_<loader>/results.json into the 3x2 matrix table.

Picks the latest run dir per (model, loader). Emits a markdown table to stdout
for LARGE_SPEED_LOG.md / README.md / the PR draft.

Usage: .venv/bin/python aggregate_matrix.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).parent
RUNS = ROOT / "runs"
LOADERS = ["opencv", "faithful", "keyframes"]
MODELS = ["qwen2.5", "qwen3"]
LOADER_LABEL = {
    "opencv": "opencv (uniform-32, default)",
    "faithful": "faithful (fps=2, qwen2_vl/qwen3_vl)",
    "keyframes": "pyav_keyframes (kf-16, ours)",
}


def latest(model: str, loader: str) -> dict | None:
    cands = sorted(RUNS.glob(f"*_{model}_{loader}"))
    for d in reversed(cands):
        rj = d / "results.json"
        if rj.exists():
            return json.loads(rj.read_text())
    return None


def pct(x) -> str:
    return f"{x * 100:.1f}%" if isinstance(x, (int, float)) else "—"


def f2(x) -> str:
    return f"{x:.2f}" if isinstance(x, (int, float)) else "—"


def main() -> None:
    print("| Model | Loader | NExTQA | MVBench | NExTQA req/s | MVBench req/s | wall (s) |")
    print("|---|---|---:|---:|---:|---:|---:|")
    for model in MODELS:
        for loader in LOADERS:
            r = latest(model, loader)
            if not r:
                print(f"| {model} | {LOADER_LABEL[loader]} | — | — | — | — | (no run) |")
                continue
            nq = r.get("nextqa", {})
            mv = r.get("mvbench", {})
            wall = (nq.get("wall_time_s") or 0) + (mv.get("wall_time_s") or 0)
            print(
                f"| {model} | {LOADER_LABEL[loader]} | {pct(nq.get('accuracy'))} | "
                f"{pct(mv.get('accuracy'))} | {f2(nq.get('requests_per_sec'))} | "
                f"{f2(mv.get('requests_per_sec'))} | {wall:.0f} |"
            )
    # MVBench per-subtask delta for the keyframe vs faithful cell (the lossy cost)
    for model in MODELS:
        faith, kf = latest(model, "faithful"), latest(model, "keyframes")
        if not (faith and kf):
            continue
        fa = (faith.get("mvbench") or {}).get("per_subtask_accuracy") or {}
        ka = (kf.get("mvbench") or {}).get("per_subtask_accuracy") or {}
        common = sorted(set(fa) & set(ka), key=lambda s: (ka[s] - fa[s]))
        if not common:
            continue
        print(f"\n### {model}: keyframes − faithful, MVBench per-subtask (worst first)")
        for s in common[:8]:
            print(f"- {s}: {(ka[s] - fa[s]) * 100:+.1f} pt  ({pct(fa[s])} → {pct(ka[s])})")


if __name__ == "__main__":
    main()
