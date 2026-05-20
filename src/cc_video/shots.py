"""Shot detection — PySceneDetect AdaptiveDetector with content-aware threshold.

A "shot" is a continuous camera take. Detecting shots gives semantic units instead
of arbitrary time slices.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Shot:
    index: int
    start_seconds: float
    end_seconds: float
    start_frame: int
    end_frame: int

    @property
    def duration(self) -> float:
        return max(0.0, self.end_seconds - self.start_seconds)

    @property
    def middle_seconds(self) -> float:
        return (self.start_seconds + self.end_seconds) / 2


def detect_shots(
    video_path: str | Path,
    adaptive_threshold: float = 3.0,
    min_scene_len_seconds: float = 0.6,
) -> list[Shot]:
    """Run AdaptiveDetector. Falls back to a single shot if scenedetect missing."""
    try:
        from scenedetect import AdaptiveDetector, open_video, SceneManager
    except ImportError:
        return _single_shot_fallback(video_path)

    video = open_video(str(Path(video_path).resolve()))
    fps = video.frame_rate or 25.0
    sm = SceneManager()
    sm.add_detector(
        AdaptiveDetector(
            adaptive_threshold=adaptive_threshold,
            min_scene_len=max(1, int(min_scene_len_seconds * fps)),
        )
    )
    sm.detect_scenes(video=video, show_progress=False)
    scenes = sm.get_scene_list()

    if not scenes:
        return _single_shot_fallback(video_path)

    shots: list[Shot] = []
    for i, (start, end) in enumerate(scenes):
        shots.append(Shot(
            index=i,
            start_seconds=float(start.get_seconds()),
            end_seconds=float(end.get_seconds()),
            start_frame=int(start.get_frames()),
            end_frame=int(end.get_frames()),
        ))
    return shots


def _single_shot_fallback(video_path: str | Path) -> list[Shot]:
    """When scenedetect unavailable or no cuts detected: treat whole video as 1 shot."""
    from .probe import get_metadata
    meta = get_metadata(video_path)
    dur = meta["duration_seconds"]
    fps = meta.get("fps") or 25.0
    return [Shot(0, 0.0, dur, 0, int(dur * fps))]


def merge_micro_shots(shots: list[Shot], min_duration: float = 0.6) -> list[Shot]:
    """Glue shots shorter than min_duration into their neighbor.

    Some videos have rapid cuts (montage, music video) producing 30+ shots in 5s.
    Merging reduces per-shot processing overhead while preserving boundaries.
    """
    if len(shots) <= 1:
        return shots
    merged: list[Shot] = [shots[0]]
    for s in shots[1:]:
        last = merged[-1]
        if s.duration < min_duration:
            merged[-1] = Shot(
                index=last.index,
                start_seconds=last.start_seconds,
                end_seconds=s.end_seconds,
                start_frame=last.start_frame,
                end_frame=s.end_frame,
            )
        else:
            merged.append(s)
    for i, s in enumerate(merged):
        merged[i] = Shot(i, s.start_seconds, s.end_seconds, s.start_frame, s.end_frame)
    return merged
