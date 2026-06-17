"""Synthetic decode-speed microbench: lossless full-decode vs keyframe-only.

Backs the speed claim in vllm-project/vllm#45203 — the `pyav_keyframes` loader's
decode cost is near-constant in clip length, while a lossless loader that returns
uniformly-sampled frames pays decode cost that grows with the clip.

Both columns run on the SAME synthetic H.264 clips (640x360, yuv420p, moving
content, fixed GOP), extract `num_frames` frames, best-of-N, single process,
CPU only:

  * keyframes  — the *real* shipping loader (`pyav_keyframes`, registered by
    importing pyav_keyframe_backend). One demux pass for keyframe PTS, then
    seek+decode only those.
  * lossless   — an in-file reference: decode the stream and keep `num_frames`
    evenly-spaced frames. This is the work any loader pays to return arbitrary
    (non-keyframe) uniformly-sampled frames. (The pinned vllm 0.19.1 venv has no
    `pyav` registry loader to call directly, hence the reference impl.)

Run:
    python upstream_bench/bench_real_loader.py                      # PR clip configs
    python upstream_bench/bench_real_loader.py --num-frames 16 --runs 3
"""

from __future__ import annotations

import argparse
import sys
import time
from io import BytesIO
from pathlib import Path

import av
import numpy as np

# The real loader lives in the repo root; importing it registers "pyav_keyframes".
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pyav_keyframe_backend  # noqa: E402,F401  (side-effect import: registers the loader)

from vllm.multimodal.video import VIDEO_LOADER_REGISTRY  # noqa: E402

# (label, duration_s, gop_s) — the four clips from the PR's speed table.
DEFAULT_CONFIGS = [
    ("30s, GOP 2s", 30, 2),
    ("120s, GOP 2s", 120, 2),
    ("120s, GOP 10s", 120, 10),
    ("600s, GOP 4s", 600, 4),
]


def make_clip(
    duration_s: int, gop_s: int, fps: int = 30, width: int = 640, height: int = 360
) -> bytes:
    """Encode a synthetic H.264 clip with a keyframe every ``gop_s`` seconds.

    The content is a sliding bright block so P-frames carry real residual,
    keeping decode cost representative rather than trivially compressible.
    """
    keyint = gop_s * fps
    n_frames = duration_s * fps
    buf = BytesIO()
    with av.open(buf, mode="w", format="mp4") as container:
        stream = container.add_stream("h264", rate=fps)
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        stream.codec_context.gop_size = keyint
        stream.codec_context.max_b_frames = 0
        stream.codec_context.options = {
            "x264-params": f"scenecut=0:keyint={keyint}:min-keyint={keyint}"
        }
        for i in range(n_frames):
            img = np.zeros((height, width, 3), dtype=np.uint8)
            x = (i * 7) % (width - 40)
            img[:, x : x + 40] = 255  # moving block -> nonzero inter-frame residual
            frame = av.VideoFrame.from_ndarray(img, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():  # flush the encoder
            container.mux(packet)
    return buf.getvalue()


def decode_lossless_uniform(data: bytes, num_frames: int) -> np.ndarray:
    """Reference lossless baseline: decode the whole stream, keep num_frames
    evenly-spaced frames. Decode work is O(total_frames) -> grows with clip
    length."""
    with av.open(BytesIO(data)) as container:
        stream = container.streams.video[0]
        frames = [f.to_ndarray(format="rgb24") for f in container.decode(stream)]
    if not frames:
        raise ValueError("no frames decoded from synthetic clip")
    idx = np.round(np.linspace(0, len(frames) - 1, num_frames)).astype(int)
    return np.stack([frames[i] for i in idx])


def decode_keyframes(data: bytes, num_frames: int) -> np.ndarray:
    """The real shipping loader (`pyav_keyframes`)."""
    loader = VIDEO_LOADER_REGISTRY.load("pyav_keyframes")
    frames, _ = loader.load_bytes(data, num_frames=num_frames)
    return frames


def best_ms(fn, data: bytes, num_frames: int, runs: int) -> float:
    """Best-of-`runs` wall time in ms; assert the strategy returned the budget."""
    best = float("inf")
    for _ in range(runs):
        t0 = time.perf_counter()
        out = fn(data, num_frames)
        dt = time.perf_counter() - t0
        assert out.shape[0] == num_frames, (
            f"{fn.__name__} returned {out.shape[0]} frames, expected {num_frames}"
        )
        best = min(best, dt)
    return best * 1000.0


def main() -> None:
    ap = argparse.ArgumentParser(description="keyframe vs lossless decode microbench")
    ap.add_argument("--num-frames", type=int, default=16)
    ap.add_argument("--runs", type=int, default=3, help="best-of-N timing")
    args = ap.parse_args()

    print(f"num_frames={args.num_frames}  best-of-{args.runs}  CPU-only  640x360\n")
    header = (
        f"{'clip':>16} | {'lossless (ms)':>14} | "
        f"{'keyframes (ms)':>15} | {'speedup':>8}"
    )
    print(header)
    print("-" * len(header))
    for label, dur, gop in DEFAULT_CONFIGS:
        data = make_clip(dur, gop)
        loss = best_ms(decode_lossless_uniform, data, args.num_frames, args.runs)
        kf = best_ms(decode_keyframes, data, args.num_frames, args.runs)
        print(f"{label:>16} | {loss:>14.1f} | {kf:>15.1f} | {loss / kf:>7.1f}x")


if __name__ == "__main__":
    main()
