"""Node 3, stage 3: pixel metrics (docs/pipeline/03-grading-node.md §3).

Each top candidate is rendered once with its text removed, so we see what
sits behind each text box. Contrast and busyness are read from that
render; the missing-asset placeholder share comes from geometry.
"""

from pathlib import Path

import numpy as np

from engine.pipeline.geometry import overlap_area
from engine.pipeline.render import render_to_image
from engine.pipeline.types import PriorityPlan
from renderer.converter.colorutil import parse_rgba

PIXEL_WEIGHTS = {"text_contrast": 0.6, "busyness": 0.25, "placeholder": 0.15}
CONTRAST_TARGET = 4.5      # WCAG AA for normal text
BACKDROP_PERCENTILES = (10, 90)
SOBEL_SCALE = 64.0         # mean gradient (0-255 grey) at which busyness bottoms out


def _linear(channel):
    c = np.asarray(channel, dtype=float) / 255.0
    return np.where(c <= 0.03928, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _luminance(rgb) -> np.ndarray:
    """WCAG relative luminance; works on one colour or an H x W x 3 array."""
    lin = _linear(rgb)
    return 0.2126 * lin[..., 0] + 0.7152 * lin[..., 1] + 0.0722 * lin[..., 2]


def _ratio(l1: float, l2: float) -> float:
    return (max(l1, l2) + 0.05) / (min(l1, l2) + 0.05)


def contrast_ratio(rgb_a, rgb_b) -> float:
    return _ratio(float(_luminance(rgb_a)), float(_luminance(rgb_b)))


def _sobel(grey: np.ndarray) -> np.ndarray:
    mag = np.zeros_like(grey)
    p = grey
    gx = (p[:-2, 2:] + 2 * p[1:-1, 2:] + p[2:, 2:]) - (p[:-2, :-2] + 2 * p[1:-1, :-2] + p[2:, :-2])
    gy = (p[2:, :-2] + 2 * p[2:, 1:-1] + p[2:, 2:]) - (p[:-2, :-2] + 2 * p[:-2, 1:-1] + p[:-2, 2:])
    mag[1:-1, 1:-1] = np.hypot(gx, gy)
    return mag


def _weighted(pairs) -> float:
    total = sum(w for w, _ in pairs)
    return sum(w * s for w, s in pairs) / total if total else 1.0


def pixel_metrics(design: dict, plan: PriorityPlan, assets_dir: str | Path | None = None) -> dict[str, float]:
    width, height = design["canvas"]["width"], design["canvas"]["height"]
    els = {e["id"]: e for e in design["elements"]}
    backdrop = dict(design, elements=[e for e in design["elements"] if e["type"] != "text"])
    rgb = np.asarray(render_to_image(backdrop, assets_dir), dtype=float)
    scale = rgb.shape[1] / width
    lum = _luminance(rgb)
    sobel = _sobel(rgb @ np.array([0.299, 0.587, 0.114]))

    contrast, busy = [], []
    for unit in plan.ranked_units():
        texts = [els[i] for i in unit.element_ids if i in els and els[i]["type"] == "text"]
        c_scores, b_scores = [], []
        for t in texts:
            x0, y0 = max(0, int(t["x"] * scale)), max(0, int(t["y"] * scale))
            x1 = min(rgb.shape[1], int((t["x"] + t["width"]) * scale) + 1)
            y1 = min(rgb.shape[0], int((t["y"] + t["height"]) * scale) + 1)
            if x1 <= x0 or y1 <= y0:
                continue
            region = rgb[y0:y1, x0:x1].reshape(-1, 3)
            r, g, b, alpha = parse_rgba(t.get("color"), default=(0, 0, 0, 1.0))
            alpha *= 1.0 if t.get("opacity") is None else t["opacity"]
            shown = alpha * np.array([r, g, b], dtype=float) + (1 - alpha) * np.median(region, axis=0)
            text_lum = float(_luminance(shown))
            # Light text is at risk from the light end of its backdrop, dark text from the dark end: take the worse.
            ratio = min(_ratio(text_lum, float(np.percentile(lum[y0:y1, x0:x1], p))) for p in BACKDROP_PERCENTILES)
            c_scores.append(min(1.0, ratio / CONTRAST_TARGET))
            b_scores.append(1.0 - min(1.0, float(sobel[y0:y1, x0:x1].mean()) / SOBEL_SCALE))
        if c_scores:
            weight = plan.weight(unit.unit_id)
            contrast.append((weight, sum(c_scores) / len(c_scores)))
            busy.append((weight, sum(b_scores) / len(b_scores)))

    missing = sum(
        overlap_area((els[i]["x"], els[i]["y"], els[i]["width"], els[i]["height"]), (0, 0, width, height))
        for u in plan.units if "missing_asset" in u.flags for i in u.element_ids if i in els)
    return {
        "text_contrast": _weighted(contrast),
        "busyness": _weighted(busy),
        "placeholder": 1.0 - min(1.0, missing / (width * height)),
    }


def pixel_total(m: dict[str, float], weights: dict[str, float] | None = None) -> float:
    """`weights` None means the hand-set PIXEL_WEIGHTS."""
    return sum(weight * m[key] for key, weight in (PIXEL_WEIGHTS if weights is None else weights).items())
