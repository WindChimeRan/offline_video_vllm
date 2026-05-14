"""Custom PyAV-based vLLM video loaders for the 0.19.x API.

Registers three loaders against ``vllm.multimodal.video.VIDEO_LOADER_REGISTRY``:

* ``pyav_seek`` — uniform ``num_frames`` indices like the default ``opencv``
  backend, but decoded via PyAV's seek-from-keyframe path. Same algorithm
  vLLM 0.20.x ships as a built-in mixin; we port it here for 0.19.1.
* ``pyav_keyframes`` — set ``skip_frame='NONKEY'`` on the codec context,
  iterate every keyframe, uniformly pick ``num_frames`` of them. If the
  bitstream has fewer keyframes than requested, fall back to ``pyav_seek``
  so the model still gets ``num_frames`` frames.
* ``pyav_keyframes_v2`` — smart keyframe sampling: one demux-only pass to
  enumerate keyframe PTS (no decode), then seek+decode only the K_target
  we'll keep. Decode work is O(min(K_total, num_frames)) regardless of
  clip length. When K_total < num_frames we *oversample* (duplicate
  keyframes) instead of falling back to non-keyframe decode, so this
  loader never pays B/P decode cost — a hard contract for the
  lossy-acceleration use case. Metadata reports the true source-frame
  index of each returned (possibly duplicated) keyframe.

Both loaders subclass ``OpenCVVideoBackend`` solely to reuse its
``compute_frames_index_to_sample`` (uniform-linspace policy). Decode is
PyAV all the way down — the cv2 capture is never opened.

Importing this module is what runs registration; do that once at process
start in ``infer.py``.
"""

from __future__ import annotations

from io import BytesIO
from typing import Any, ClassVar

import av
import numpy as np
import numpy.typing as npt

from vllm.logger import init_logger
from vllm.multimodal.video import (
    VIDEO_LOADER_REGISTRY,
    OpenCVVideoBackend,
    VideoSourceMetadata,
    VideoTargetMetadata,
)

logger = init_logger(__name__)


def _pyav_metadata(container: "av.container.InputContainer") -> VideoSourceMetadata:
    if not container.streams.video:
        raise ValueError("No video streams found in container")
    stream = container.streams.video[0]
    total_frames = stream.frames or 0
    fps = float(stream.average_rate) if stream.average_rate else 0.0
    duration = float(stream.duration * stream.time_base) if stream.duration else 0.0
    # Some containers (WebM, fragmented MP4) don't advertise frame count.
    if total_frames == 0 and duration > 0 and fps > 0:
        total_frames = int(duration * fps)
    return VideoSourceMetadata(total_frames, fps, duration)


def _decode_seek(
    container: "av.container.InputContainer",
    frame_indices: list[int],
    fps: float,
    duration: float,
) -> tuple[npt.NDArray, list[int]]:
    """Per-target seek + decode-forward-to-exact-target.

    For each target index we seek to its pts (PyAV snaps to the keyframe
    at-or-before that pts under default flags) and then iterate decode()
    until ``frame.pts >= target_pts``. That step is what makes this loader
    return the *target* frame instead of the preceding keyframe — vllm
    0.20.x's built-in PyAV mixin omits the forward-decode and instead
    takes ``next(decode())``, which returns the keyframe. On clips with
    long GOPs (much of MVBench) that costs >10 pt MVBench accuracy;
    measured 2026-05-13 in LARGE_SPEED_LOG.md.
    """
    stream = container.streams.video[0]
    # SLICE parallelizes within a single frame without the one-frame-per-thread
    # latency of FRAME threading.
    stream.thread_type = "SLICE"
    time_base = stream.time_base
    frame_interval = 1.0 / fps if fps > 0 else 0.1
    max_ts = max(0.0, duration - frame_interval) if duration > 0 else float("inf")

    frames_list: list[npt.NDArray] = []
    valid: list[int] = []
    for idx in frame_indices:
        ts = min(idx / fps, max_ts) if fps > 0 else 0.0
        target_pts = int(ts / time_base)
        container.seek(target_pts, stream=stream)
        chosen = None
        for frame in container.decode(video=0):
            if frame.pts is None:
                continue
            chosen = frame  # keep best-so-far so end-of-stream still returns something
            if frame.pts >= target_pts:
                break
        if chosen is not None:
            frames_list.append(chosen.to_ndarray(format="rgb24"))
            valid.append(idx)

    if not frames_list:
        return np.empty((0, 0, 0, 3), dtype=np.uint8), valid
    return np.stack(frames_list), valid


@VIDEO_LOADER_REGISTRY.register("pyav_seek")
class PyAVSeekBackend(OpenCVVideoBackend):
    """Same uniform sampling as ``opencv``, decoded via PyAV seek."""

    _backend_name: ClassVar[str] = "pyav_seek"

    @classmethod
    def load_bytes(
        cls,
        data: bytes,
        num_frames: int = -1,
        fps: int = -1,
        max_duration: int = 300,
        frame_recovery: bool = False,
        **kwargs,
    ) -> tuple[npt.NDArray, dict[str, Any]]:
        if frame_recovery:
            raise ValueError(
                "frame_recovery is cv2-only; not supported by pyav_seek."
            )
        with av.open(BytesIO(data)) as container:
            source = _pyav_metadata(container)
            target = VideoTargetMetadata(
                num_frames=num_frames, fps=fps, max_duration=max_duration
            )
            frame_idx = cls.compute_frames_index_to_sample(
                source=source, target=target
            )
            frames, valid = _decode_seek(
                container, frame_idx, source.original_fps, source.duration
            )
        return frames, cls.create_hf_metadata(
            source=source,
            video_backend=cls._backend_name,
            valid_frame_indices=valid,
        )


@VIDEO_LOADER_REGISTRY.register("pyav_keyframes")
class PyAVKeyframeBackend(OpenCVVideoBackend):
    """Keyframe-only sampling via PyAV ``skip_frame='NONKEY'``.

    On clips with at least ``num_frames`` keyframes: pick ``num_frames``
    of them uniformly by position. On clips with fewer keyframes (typical
    of short MVBench clips with long GOP): fall back to ``pyav_seek`` so
    the HF processor still receives ``num_frames`` frames. The label
    written into ``metadata['video_backend']`` distinguishes the two
    paths (``pyav_keyframes`` vs ``pyav_keyframes_fallback``).
    """

    _backend_name: ClassVar[str] = "pyav_keyframes"

    @classmethod
    def _decode_all_keyframes(
        cls,
        data: bytes,
    ) -> tuple[npt.NDArray, list[int], VideoSourceMetadata]:
        """Iterate the bitstream once with skip_frame='NONKEY'."""
        frames_list: list[npt.NDArray] = []
        indices: list[int] = []
        with av.open(BytesIO(data)) as container:
            source = _pyav_metadata(container)
            fps = source.original_fps if source.original_fps > 0 else 30.0
            stream = container.streams.video[0]
            stream.thread_type = "SLICE"
            # MUST be set before iterating decode().
            stream.codec_context.skip_frame = "NONKEY"
            time_base = stream.time_base

            for frame in container.decode(stream):
                ts = float(frame.pts * time_base) if frame.pts is not None else 0.0
                indices.append(int(round(ts * fps)))
                frames_list.append(frame.to_ndarray(format="rgb24"))

        if not frames_list:
            return np.empty((0, 0, 0, 3), dtype=np.uint8), indices, source
        return np.stack(frames_list), indices, source

    @classmethod
    def load_bytes(
        cls,
        data: bytes,
        num_frames: int = -1,
        fps: int = -1,
        max_duration: int = 300,
        frame_recovery: bool = False,
        **kwargs,
    ) -> tuple[npt.NDArray, dict[str, Any]]:
        if frame_recovery:
            raise ValueError(
                "frame_recovery is cv2-only; not supported by pyav_keyframes."
            )

        keyframes, kf_indices, source = cls._decode_all_keyframes(data)
        n_kf = len(kf_indices)

        if num_frames < 0:
            return keyframes, cls.create_hf_metadata(
                source=source,
                video_backend=cls._backend_name,
                valid_frame_indices=kf_indices,
            )

        if n_kf >= num_frames:
            pick = np.linspace(0, n_kf - 1, num_frames, dtype=int)
            frames = keyframes[pick]
            valid = [kf_indices[i] for i in pick]
            return frames, cls.create_hf_metadata(
                source=source,
                video_backend=cls._backend_name,
                valid_frame_indices=valid,
            )

        logger.info(
            "pyav_keyframes: %d keyframes < %d requested; falling back to "
            "uniform PyAV seek-based sampling.",
            n_kf,
            num_frames,
        )
        frames, meta = PyAVSeekBackend.load_bytes(
            data,
            num_frames=num_frames,
            fps=fps,
            max_duration=max_duration,
            frame_recovery=False,
            **kwargs,
        )
        meta["video_backend"] = f"{cls._backend_name}_fallback"
        return frames, meta


@VIDEO_LOADER_REGISTRY.register("pyav_keyframes_v2")
class PyAVKeyframeBackendV2(OpenCVVideoBackend):
    """Pure keyframe-only sampling for lossy video acceleration.

    Strategy:
      Pass 1 — walk the demuxer to enumerate keyframe PTS values. No
      decode; PyAV reads packet headers only. Cost is linear in packet
      count, microseconds per 100 packets.
      Pass 2 — for each unique keyframe we'll keep, seek + decode that
      single keyframe. Duplicates (when K_total < num_frames) share the
      same decoded ndarray rather than re-decoding.

    Decode work is O(min(K_total, num_frames)) I-frame decodes per clip,
    regardless of clip length or GOP. No B/P decode work *ever* — this is
    the hard contract that makes this loader the lossy-acceleration play.

    When K_total < num_frames, we oversample (duplicate keyframes) instead
    of falling back to non-keyframe decode. The HF processor still gets
    exactly num_frames frames. ``frames_indices`` reports the actual
    source-frame index of each returned (possibly duplicated) keyframe —
    so Qwen2.5-VL's temporal positional encoding sees identical positions
    for identical content, and the attention layer can collapse the
    duplicates naturally. Equivalent to "give the model K unique frames
    at correct times" with constant N.

    Caveat: returned frames sit on GOP boundaries, not uniform stride.
    For motion-dense MVBench subtasks (action_*, moving_*) this trades
    accuracy for speed (~9 pt MVBench drop measured previously). For
    NExTQA-style scene QA accuracy is preserved.
    """

    _backend_name: ClassVar[str] = "pyav_keyframes_v2"

    @classmethod
    def load_bytes(
        cls,
        data: bytes,
        num_frames: int = -1,
        fps: int = -1,
        max_duration: int = 300,
        frame_recovery: bool = False,
        **kwargs,
    ) -> tuple[npt.NDArray, dict[str, Any]]:
        if frame_recovery:
            raise ValueError(
                "frame_recovery is cv2-only; not supported by pyav_keyframes_v2."
            )

        with av.open(BytesIO(data)) as container:
            source = _pyav_metadata(container)
            stream = container.streams.video[0]
            stream.thread_type = "SLICE"
            time_base = stream.time_base
            src_fps = source.original_fps if source.original_fps > 0 else 30.0

            # Pass 1: enumerate keyframe PTS without decoding.
            kf_pts: list[int] = []
            for packet in container.demux(stream):
                # End-of-stream sentinel is a packet with pts=None; skip.
                if packet.pts is None:
                    continue
                if packet.is_keyframe:
                    kf_pts.append(packet.pts)
            n_kf = len(kf_pts)

            if n_kf == 0:
                # No keyframes anywhere — bitstream is broken or has no
                # IDR frames. We have nothing to return; fail loudly.
                raise ValueError(
                    "pyav_keyframes_v2: no keyframes found in bitstream"
                )

            # Build the list of target PTS (length == num_frames, with
            # duplicates when n_kf < num_frames). For num_frames < 0
            # return every keyframe once.
            if num_frames < 0:
                target_pts = kf_pts
            else:
                # Use round-not-truncate so n_kf=2 / num_frames=16 gives
                # [0]*8 + [1]*8 (balanced) instead of [0]*15 + [1]*1
                # (the linspace+int-truncate skew). For n_kf >> num_frames
                # the difference is sub-frame and matches OpenCV's policy.
                picks = (
                    np.round(np.linspace(0, n_kf - 1, num_frames))
                    .astype(int)
                    .tolist()
                )
                target_pts = [kf_pts[i] for i in picks]

            # Pass 2: decode each *unique* target keyframe once, then
            # duplicate the ndarray reference for any repeated picks.
            unique_pts = list(dict.fromkeys(target_pts))  # preserve order
            pts_to_frame: dict[int, npt.NDArray] = {}
            for tpts in unique_pts:
                container.seek(tpts, stream=stream)
                # After seek the decoder state is reset; next decoded frame
                # IS the keyframe at-or-before tpts. We selected tpts from
                # the keyframe-pts list, so we get exactly that keyframe.
                frame = next(container.decode(stream), None)
                if frame is None:
                    continue
                pts_to_frame[tpts] = frame.to_ndarray(format="rgb24")

            frames_list: list[npt.NDArray] = []
            valid_indices: list[int] = []
            for tpts in target_pts:
                if tpts not in pts_to_frame:
                    continue
                frames_list.append(pts_to_frame[tpts])
                ts = float(tpts * time_base)
                valid_indices.append(int(round(ts * src_fps)))

        if not frames_list:
            arr = np.empty((0, 0, 0, 3), dtype=np.uint8)
        else:
            arr = np.stack(frames_list)
        return arr, cls.create_hf_metadata(
            source=source,
            video_backend=cls._backend_name,
            valid_frame_indices=valid_indices,
        )
