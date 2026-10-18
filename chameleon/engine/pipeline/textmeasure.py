"""Measure text the way the official renderer will wrap and draw it.

Line breaks come from renderer/converter/wrap.py with the same font lookup the
converter uses (convert.py `_append_text`). Vertical extent follows
cr-renderer's `text_utils.render_text`: line i starts at
i * font_size * line_height, and each line's glyphs reach from its top down to
the font's descent, so the last line is taller than one line pitch.
"""

import math
from pathlib import Path

import skia  # type: ignore
from cr_renderer.fonts import FONT_WEIGHTS

from renderer.converter.brand_fonts import load_brand_font_manager
from renderer.converter.wrap import char_widths, wrap_text

BOLD_WEIGHTS = {"bold", "semibold", "extrabold", "black"}
_DEFAULT_LINE_HEIGHT = 1.2


# The three helpers below mirror the private ones in renderer/converter/convert.py.
def _normalize_weight(weight: str | None) -> str:
    if not weight:
        return "regular"
    weight = weight.lower()
    return weight if weight in FONT_WEIGHTS else "regular"


def _is_italic(value) -> bool:
    if isinstance(value, bool):
        return value
    return value == "italic"


def _font_style(is_bold: bool, is_italic: bool) -> str:
    if is_bold and is_italic:
        return "bolditalic"
    if is_bold:
        return "bold"
    if is_italic:
        return "italic"
    return "regular"


class TextMeasurer:
    def __init__(self, fonts_dir: str | Path = "brand_kit/fonts"):
        self._fonts = load_brand_font_manager(str(fonts_dir))
        self._width_cache: dict[tuple, list[float]] = {}
        self._glyph_height_cache: dict[tuple, float] = {}

    def _font(self, el: dict) -> tuple[str, str, str]:
        weight = _normalize_weight(el.get("font_weight"))
        italic = _is_italic(el.get("font_style") or el.get("italic"))
        return el.get("font_family", "Montserrat"), weight, _font_style(weight in BOLD_WEIGHTS, italic)

    def _char_widths(self, el: dict, font_size: float) -> list[float]:
        family, weight, style = self._font(el)
        text = str(el.get("content", ""))
        key = (family, weight, style, float(font_size), text)
        if key not in self._width_cache:
            self._width_cache[key] = char_widths(self._fonts, family, weight, style, float(font_size), text)
        return self._width_cache[key]

    def _glyph_height(self, el: dict, font_size: float) -> float:
        """Descent minus ascent of the font cr-renderer draws with: it picks
        the face from bold/italic alone and truncates the size to an int."""
        family, _, style = self._font(el)
        weight = "bold" if style in ("bold", "bolditalic") else "regular"
        key = (family, weight, style, int(font_size))
        if key not in self._glyph_height_cache:
            typeface = skia.Typeface.MakeFromData(self._fonts.lookup(family, weight, style))
            metrics = skia.Font(typeface, int(font_size)).getMetrics()
            self._glyph_height_cache[key] = metrics.fDescent - metrics.fAscent
        return self._glyph_height_cache[key]

    def wrap(self, el: dict, font_size: float, width: float) -> str:
        return wrap_text(str(el.get("content", "")), self._char_widths(el, font_size), float(width))

    def line_count(self, el: dict, font_size: float, width: float) -> int:
        wrapped = self.wrap(el, font_size, width)
        return wrapped.count("\n") + 1 if wrapped else 0

    def block_height(self, el: dict, font_size: float, width: float) -> float:
        lines = self.line_count(el, font_size, width)
        if lines == 0:
            return 0.0
        line_height = float(el.get("line_height", _DEFAULT_LINE_HEIGHT)) or 1.0
        return (lines - 1) * font_size * line_height + self._glyph_height(el, font_size)

    def _spans_width(self, widths: list[float], spans: list[list[int]], letter_spacing: float) -> list[float]:
        return [sum(widths[i] for i in span) + letter_spacing * max(0, len(span) - 1) for span in spans]

    def line_widths(self, el: dict, font_size: float, width: float) -> list[float]:
        """Drawn width of each wrapped line, letter spacing included (the
        converter wraps without it, the renderer draws with it)."""
        wrapped = self.wrap(el, font_size, width)
        spans, current = [], []
        for i, c in enumerate(wrapped):
            if c == "\n":
                spans.append(current)
                current = []
            else:
                current.append(i)
        spans.append(current)
        letter_spacing = float(el.get("letter_spacing", 0) or 0)
        return self._spans_width(self._char_widths(el, font_size), spans, letter_spacing)

    def longest_word_width(self, el: dict, font_size: float) -> float:
        text = str(el.get("content", ""))
        words, current = [], []
        for i, c in enumerate(text):
            if c in (" ", "\n"):
                if current:
                    words.append(current)
                current = []
            else:
                current.append(i)
        if current:
            words.append(current)
        letter_spacing = float(el.get("letter_spacing", 0) or 0)
        return max(self._spans_width(self._char_widths(el, font_size), words, letter_spacing), default=0.0)

    def fits(self, el: dict, font_size: float, width: float, height: float) -> bool:
        if self.block_height(el, font_size, width) > height + 1:
            return False
        if max(self.line_widths(el, font_size, width), default=0.0) > width + 0.5:
            return False
        max_lines = el.get("max_lines")
        return max_lines is None or self.line_count(el, font_size, width) <= max_lines

    def fit_font_size(self, el: dict, width: float, height: float, lo: float, hi: float) -> float | None:
        """Largest multiple of 0.5 in [lo, hi] that fits, or None if even `lo`
        doesn't. The search keeps `fits(good)` and `not fits(bad)`, so the
        answer always sits on a fit/no-fit boundary even where wrapping makes
        fit non-monotonic in size."""
        if not self.fits(el, lo, width, height):
            return None
        grid = [k / 2 for k in range(math.ceil(lo * 2), math.floor(hi * 2) + 1)]
        if not grid:
            return lo
        if self.fits(el, grid[-1], width, height):
            return grid[-1]
        good, bad = -1, len(grid) - 1
        while bad - good > 1:
            mid = (good + bad) // 2
            if self.fits(el, grid[mid], width, height):
                good = mid
            else:
                bad = mid
        return grid[good] if good >= 0 else lo
