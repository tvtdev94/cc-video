"""Per-shot keyframe extraction.

Default: middle frame of each shot (single representative). Cheap and gives one
semantically meaningful image per shot. For shots > 8s, picks start + middle + end.

Deep mode: extract candidate frames at 2 fps within shot, run CLIP embeddings,
k-means cluster, pick frame closest to each centroid. Gives diverse coverage of
shots that pan/zoom/transition internally.
"""
from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .shots import Shot


@dataclass
class Keyframe:
    shot_index: int
    timestamp_seconds: float
    path: str
    role: str  # "middle" | "start" | "end" | "cluster-{n}"


def extract_keyframes(
    video_path: str | Path,
    shots: list[Shot],
    out_dir: Path,
    resolution: int = 720,
    multi_keyframe_threshold: float = 8.0,
    deep: bool = False,
) -> list[Keyframe]:
    """Pick keyframes per shot and seek them out with ffmpeg.

    deep=True triggers CLIP-cluster mode (requires open_clip + sklearn extra).
    """
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg not on PATH.")
    out_dir.mkdir(parents=True, exist_ok=True)

    if deep:
        try:
            return _extract_clip_cluster(video_path, shots, out_dir, resolution)
        except ImportError:
            pass  # fall back to simple mode

    keyframes: list[Keyframe] = []
    for shot in shots:
        picks: list[tuple[float, str]] = []
        if shot.duration >= multi_keyframe_threshold:
            picks.append((shot.start_seconds + 0.2, "start"))
            picks.append((shot.middle_seconds, "middle"))
            picks.append((max(shot.start_seconds, shot.end_seconds - 0.4), "end"))
        else:
            picks.append((shot.middle_seconds, "middle"))

        for ts, role in picks:
            out_path = out_dir / f"shot{shot.index:04d}_{role}.jpg"
            _seek_frame(video_path, ts, out_path, resolution)
            if out_path.exists():
                keyframes.append(Keyframe(shot.index, ts, str(out_path), role))
    return keyframes


def _seek_frame(video_path: str | Path, t: float, out: Path, resolution: int) -> None:
    """Fast keyframe-snap seek then accurate seek for precision."""
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-ss", f"{max(0.0, t - 1.0):.3f}",
        "-i", str(Path(video_path).resolve()),
        "-ss", f"{min(1.0, t):.3f}",
        "-frames:v", "1",
        "-vf", f"scale={resolution}:-2",
        "-q:v", "3",
        str(out),
    ]
    subprocess.run(cmd, capture_output=True, text=True)


def _extract_clip_cluster(
    video_path: str | Path, shots: list[Shot], out_dir: Path, resolution: int,
) -> list[Keyframe]:
    """Deep mode: sample @ 2 fps within each shot, CLIP encode, KMeans cluster.

    Picks N=ceil(duration/4) centroids per shot, capped at 5.
    """
    import math
    import numpy as np
    import torch
    import open_clip
    from sklearn.cluster import KMeans
    from PIL import Image

    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k"
    )
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    keyframes: list[Keyframe] = []
    tmp = out_dir / "_candidates"
    tmp.mkdir(exist_ok=True)

    for shot in shots:
        n_target = min(5, max(1, math.ceil(shot.duration / 4.0)))
        candidate_paths: list[tuple[float, Path]] = []
        step = max(0.5, shot.duration / max(1, n_target * 4))
        t = shot.start_seconds
        idx = 0
        while t < shot.end_seconds and idx < 24:
            p = tmp / f"shot{shot.index:04d}_cand{idx:02d}.jpg"
            _seek_frame(video_path, t, p, resolution)
            if p.exists():
                candidate_paths.append((t, p))
            t += step
            idx += 1

        if not candidate_paths:
            continue
        if len(candidate_paths) <= n_target:
            for i, (ts, p) in enumerate(candidate_paths):
                final = out_dir / f"shot{shot.index:04d}_cluster-{i}.jpg"
                p.rename(final)
                keyframes.append(Keyframe(shot.index, ts, str(final), f"cluster-{i}"))
            continue

        with torch.no_grad():
            imgs = torch.stack([
                preprocess(Image.open(p).convert("RGB")) for _, p in candidate_paths
            ]).to(device)
            feats = model.encode_image(imgs).cpu().numpy()
            feats /= (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-9)

        km = KMeans(n_clusters=n_target, n_init=4, random_state=0).fit(feats)
        for cluster_id in range(n_target):
            cluster_mask = km.labels_ == cluster_id
            if not cluster_mask.any():
                continue
            dists = np.linalg.norm(feats[cluster_mask] - km.cluster_centers_[cluster_id], axis=1)
            pick = np.where(cluster_mask)[0][int(dists.argmin())]
            ts, src = candidate_paths[pick]
            final = out_dir / f"shot{shot.index:04d}_cluster-{cluster_id}.jpg"
            src.rename(final)
            keyframes.append(Keyframe(shot.index, ts, str(final), f"cluster-{cluster_id}"))
    return keyframes
