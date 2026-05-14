"""Custom PyAV-based vLLM video loader for the 0.19.x API.

Registers ``pyav_keyframes_v2`` against
``vllm.multimodal.video.VIDEO_LOADER_REGISTRY``:

* Single demux pass to enumerate keyframe PTS (no decode), then seek+decode
  only the K_unique keyframes we keep. Decode cost is ``O(K_unique)``
  I-frame decodes per clip regardless of clip length.
* When ``K_total < num_frames`` we oversample (duplicate keyframes) rather
  than fall back to non-keyframe decode — a hard contract for the
  lossy-acceleration path: no B/P decode work *ever*.
* Metadata reports the true source-frame index of each returned (possibly
  duplicated) keyframe so Qwen2.5-VL's temporal positional encoding stays
  honest.

Importing this module is what runs registration; do that once at process
start in ``infer.py``.
"""

from __future__ import annotations

from io import BytesIO
from typing import Any, ClassVar

import av
import numpy as np
import numpy.typing as npt

from vllm.multimodal.video import (
    VIDEO_LOADER_REGISTRY,
    VideoLoader,
    VideoSourceMetadata,
)


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


@VIDEO_LOADER_REGISTRY.register("pyav_keyframes_v2")
class PyAVKeyframeBackendV2(VideoLoader):
    """Pure keyframe-only sampling for lossy video acceleration.

    Strategy:
      Pass 1 — walk the demuxer to enumerate keyframe PTS values. No
      decode; PyAV reads packet headers only. Cost is linear in packet
      count, microseconds per 100 packets.
      Pass 2 — for each unique keyframe we'll keep, seek + decode that
      single keyframe. Duplicates (when K_total < num_frames) share the
      same decoded ndarray rather than re-decoding.

    Decode work is ``O(K_unique)`` I-frame decodes per clip, where
    ``K_unique = len(set(picks)) <= min(K_total, num_frames)``. No B/P
    decode work *ever* — this is the hard contract that makes this loader
    the lossy-acceleration play.

    When ``K_total < num_frames``, oversample (duplicate keyframes) rather
    than fall back to non-keyframe decode. The HF processor still gets
    exactly ``num_frames`` frames. ``frames_indices`` reports the true
    source-frame index of each returned (possibly duplicated) keyframe —
    so Qwen2.5-VL's temporal positional encoding sees identical positions
    for identical content, and the attention layer can collapse the
    duplicates naturally.

    Caveat: returned frames sit on GOP boundaries, not uniform stride.
    Trades accuracy for speed on motion-dense subtasks; preserves
    scene/identity QA where temporal coverage isn't the signal.
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
    ) -> tuple[npt.NDArray, dict[str, Any]]:
        # frame_recovery is a cv2-only forward-scan flag; accepting it silently
        # would hide caller mistakes. Reject loudly.
        if frame_recovery:
            raise ValueError(
                f"frame_recovery is cv2-only; not supported by {cls._backend_name}."
            )

        with av.open(BytesIO(data)) as container:
            source = _pyav_metadata(container)
            stream = container.streams.video[0]
            stream.thread_type = "SLICE"
            time_base = stream.time_base
            src_fps = source.original_fps if source.original_fps > 0 else 30.0

            kf_pts: list[int] = []
            for packet in container.demux(stream):
                # End-of-stream sentinel is a packet with pts=None; skip.
                if packet.pts is None:
                    continue
                if packet.is_keyframe:
                    kf_pts.append(packet.pts)
            n_kf = len(kf_pts)

            if n_kf == 0:
                raise ValueError(
                    "pyav_keyframes_v2: no keyframes found in bitstream"
                )

            if num_frames < 0:
                target_pts = kf_pts
            else:
                # Round-not-truncate so n_kf=2 / num_frames=16 gives [0]*8 +
                # [1]*8 (balanced) rather than [0]*15 + [1]*1 (the
                # linspace+int-truncate skew). For n_kf >> num_frames the two
                # differ by sub-frame and match OpenCV's policy.
                picks = (
                    np.round(np.linspace(0, n_kf - 1, num_frames))
                    .astype(int)
                    .tolist()
                )
                target_pts = [kf_pts[i] for i in picks]

            unique_pts = list(dict.fromkeys(target_pts))  # preserve order
            pts_to_frame: dict[int, npt.NDArray] = {}
            for tpts in unique_pts:
                container.seek(tpts, stream=stream)
                # After seek the decoder state is reset; next decoded frame
                # IS the keyframe at-or-before tpts. We selected tpts from
                # the keyframe-pts list, so we get exactly that keyframe.
                frame = next(container.decode(stream), None)
                if frame is None:
                    # Demuxer said this PTS is a keyframe but the decoder
                    # can't produce a frame after seeking to it — that's a
                    # corrupted bitstream, not something to silently absorb.
                    raise ValueError(
                        f"pyav_keyframes_v2: keyframe pts={tpts} demuxed "
                        "but no frame decoded; bitstream may be corrupt"
                    )
                pts_to_frame[tpts] = frame.to_ndarray(format="rgb24")

            frames_list = [pts_to_frame[tpts] for tpts in target_pts]
            valid_indices = [
                int(round(float(tpts * time_base) * src_fps))
                for tpts in target_pts
            ]

        return np.stack(frames_list), cls.create_hf_metadata(
            source=source,
            video_backend=cls._backend_name,
            valid_frame_indices=valid_indices,
        )
