"""OCR on keyframes — extract on-screen text (terminal, IDE, slides, lower-thirds).

PaddleOCR is preferred (handles compressed video frames better than Tesseract per
benchmark). Falls back gracefully if not installed.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class OCRLine:
    text: str
    confidence: float
    bbox: tuple[int, int, int, int]  # x1,y1,x2,y2


@dataclass
class OCRResult:
    keyframe_path: str
    lines: list[OCRLine]

    @property
    def text(self) -> str:
        return "\n".join(line.text for line in self.lines)


def ocr_keyframes(
    keyframe_paths: list[str],
    lang: str = "en",
    min_confidence: float = 0.6,
) -> dict[str, OCRResult]:
    """Returns {path: OCRResult}. Silently returns empty dict if PaddleOCR missing or fails."""
    try:
        from paddleocr import PaddleOCR
    except ImportError:
        return {}

    try:
        # PaddleOCR 3.x API (enable_mkldnn=False works around regression
        # in paddlepaddle 3.3.x oneDNN PIR conversion path —
        # see github.com/PaddlePaddle/Paddle/issues/77340)
        ocr = PaddleOCR(use_textline_orientation=True, lang=lang, enable_mkldnn=False)
        api_version = 3
    except (TypeError, ValueError):
        try:
            # PaddleOCR 2.x API
            ocr = PaddleOCR(use_angle_cls=True, lang=lang, show_log=False)
            api_version = 2
        except Exception as e:
            print(f"[cc-video] OCR init failed: {e}")
            return {}
    except Exception as e:
        print(f"[cc-video] OCR init failed: {e}")
        return {}

    import time
    from . import log_util as logu
    results: dict[str, OCRResult] = {}
    n = len(keyframe_paths)
    t0 = time.time()
    for i, path in enumerate(keyframe_paths, 1):
        lines: list[OCRLine] = []
        try:
            if api_version == 3:
                lines = _run_predict_v3(ocr, path, min_confidence)
            else:
                lines = _run_ocr_v2(ocr, path, min_confidence)
        except Exception:
            pass  # per-frame failures just yield empty
        results[path] = OCRResult(path, _dedupe_lines(lines))
        if i == 1 or i % 10 == 0 or i == n:
            elapsed = time.time() - t0
            rate = i / elapsed if elapsed > 0 else 0
            eta = (n - i) / rate if rate > 0 else 0
            logu.progress(i, n, "frames", rate=rate, eta_seconds=eta)
    return results


def _run_predict_v3(ocr, path: str, min_confidence: float) -> list[OCRLine]:
    raw = ocr.predict(path)
    lines: list[OCRLine] = []
    for item in raw or []:
        texts = item.get("rec_texts") or []
        scores = item.get("rec_scores") or []
        polys = item.get("rec_polys") or item.get("dt_polys") or [None] * len(texts)
        for text, conf, poly in zip(texts, scores, polys):
            if conf is None or conf < min_confidence or not text.strip():
                continue
            if poly is not None and len(poly) >= 1:
                xs = [int(p[0]) for p in poly]
                ys = [int(p[1]) for p in poly]
                bbox = (min(xs), min(ys), max(xs), max(ys))
            else:
                bbox = (0, 0, 0, 0)
            lines.append(OCRLine(text=text.strip(), confidence=float(conf), bbox=bbox))
    return lines


def _run_ocr_v2(ocr, path: str, min_confidence: float) -> list[OCRLine]:
    raw = ocr.ocr(path, cls=True)
    lines: list[OCRLine] = []
    for page in raw or []:
        for entry in page or []:
            box, (text, conf) = entry[0], entry[1]
            if conf < min_confidence or not text.strip():
                continue
            xs = [int(p[0]) for p in box]
            ys = [int(p[1]) for p in box]
            lines.append(OCRLine(
                text=text.strip(), confidence=float(conf),
                bbox=(min(xs), min(ys), max(xs), max(ys)),
            ))
    return lines


def _dedupe_lines(lines: list[OCRLine]) -> list[OCRLine]:
    """Remove duplicate text reads within the same frame (PaddleOCR sometimes
    double-reads stacked text)."""
    seen: set[str] = set()
    out: list[OCRLine] = []
    for line in lines:
        key = line.text.lower().strip()
        if key in seen:
            continue
        seen.add(key)
        out.append(line)
    return out


def collapse_consecutive_shot_ocr(
    shot_to_ocr: dict[int, list[OCRResult]],
) -> dict[int, str]:
    """For each shot, merge OCR from its keyframes into one deduped text block.

    Multiple keyframes in the same shot often re-read the same screen text — we
    only need the union.
    """
    merged: dict[int, str] = {}
    for shot_idx, results in shot_to_ocr.items():
        seen: set[str] = set()
        out: list[str] = []
        for r in results:
            for line in r.lines:
                key = line.text.lower().strip()
                if key not in seen:
                    seen.add(key)
                    out.append(line.text)
        merged[shot_idx] = "\n".join(out)
    return merged
