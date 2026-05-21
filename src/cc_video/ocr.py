"""OCR on keyframes — extract on-screen text (terminal, IDE, slides, lower-thirds).

PaddleOCR is preferred (handles compressed video frames better than Tesseract per
benchmark). Falls back gracefully if not installed.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable


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
    on_result: Callable[[OCRResult], None] | None = None,
    skip_paths: set[str] | None = None,
    max_workers: int = 4,
    downsample_max_width: int = 720,
) -> dict[str, OCRResult]:
    """OCR each keyframe in parallel. Silently returns {} if PaddleOCR missing or fails.

    Performance levers:
      - downsample_max_width: shrink large frames before OCR (default 720px wide)
      - max_workers: parallel threads (PaddleOCR is thread-safe for predict calls)
      - skip_paths: skip already-processed paths (for resume from checkpoint)
      - on_result: callback invoked after each completed frame (for incremental save)
    """
    try:
        from paddleocr import PaddleOCR
    except ImportError:
        return {}

    rec_model = _rec_model_for(lang)
    try:
        # PaddleOCR 3.x API. Mobile det+rec models are ~5-10x faster than
        # the default server models on CPU. Setting an explicit det model
        # name causes the `lang` arg to be ignored, so we also pin a matching
        # mobile rec model (heavy PP-OCRv5_server_rec is the default otherwise).
        # Document orientation/unwarping are useful for scanned paper but
        # waste compute on screencast frames.
        # enable_mkldnn=False works around a regression in paddlepaddle 3.3.x
        # oneDNN PIR conversion path (see Paddle issue #77340).
        ocr_kwargs: dict = dict(
            text_detection_model_name="PP-OCRv5_mobile_det",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            enable_mkldnn=False,
        )
        if rec_model:
            ocr_kwargs["text_recognition_model_name"] = rec_model
        else:
            ocr_kwargs["lang"] = lang
        ocr = PaddleOCR(**ocr_kwargs)
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
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from . import log_util as logu

    todo = [p for p in keyframe_paths if not skip_paths or p not in skip_paths]
    if not todo:
        return {}

    results: dict[str, OCRResult] = {}
    n = len(todo)
    t0 = time.time()
    done = 0

    def _process(path: str) -> OCRResult:
        try:
            input_obj = _prep_image(path, downsample_max_width)
            if api_version == 3:
                lines = _run_predict_v3(ocr, input_obj, min_confidence)
            else:
                lines = _run_ocr_v2(ocr, path, min_confidence)
        except Exception:
            lines = []
        return OCRResult(path, _dedupe_lines(lines))

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_process, p): p for p in todo}
        for fut in as_completed(futures):
            res = fut.result()
            results[res.keyframe_path] = res
            if on_result is not None:
                on_result(res)
            done += 1
            if done == 1 or done % 10 == 0 or done == n:
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                eta = (n - done) / rate if rate > 0 else 0
                logu.progress(done, n, "frames", rate=rate, eta_seconds=eta)
    return results


_LATIN_LANGS = {"en", "vi", "fr", "de", "es", "pt", "it", "tr", "nl", "ro",
                "pl", "cs", "sv", "da", "no", "fi", "id", "ms", "tl"}


def _rec_model_for(lang: str) -> str | None:
    """Map language code to a PP-OCRv5 mobile rec model name. None = let PaddleOCR pick."""
    if lang in _LATIN_LANGS:
        return "latin_PP-OCRv5_mobile_rec"
    if lang in ("ch", "chinese_cht"):
        return "PP-OCRv5_mobile_rec"
    return None


def _prep_image(path: str, max_width: int):
    """Open and downsample for OCR. Returns a numpy array or original path on fail."""
    try:
        from PIL import Image
        import numpy as np
        img = Image.open(path).convert("RGB")
        if img.width > max_width:
            ratio = max_width / img.width
            img = img.resize((max_width, int(img.height * ratio)), Image.LANCZOS)
        return np.array(img)
    except Exception:
        return path  # fall back: let PaddleOCR open it itself


def _run_predict_v3(ocr, image_or_path, min_confidence: float) -> list[OCRLine]:
    raw = ocr.predict(image_or_path)
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
