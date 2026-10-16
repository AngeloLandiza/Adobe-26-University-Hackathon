import numpy as np
import pytest
from engine.pipeline.render import render_to_image
from tests.conftest import element


def test_short_headline_one_line(measurer, l1):
    assert measurer.line_count(element(l1, "headline"), 60, 800) == 1


def test_narrow_box_wraps(measurer, l1):
    assert measurer.line_count(element(l1, "headline"), 60, 300) >= 2


def test_block_height_adds_one_line_pitch_per_line(measurer, l1):
    # cr-renderer places line i at i * font_size * line_height (default 1.2).
    el = element(l1, "headline")
    n = measurer.line_count(el, 60, 300)
    assert n >= 2
    one, many = measurer.block_height(el, 60, 800), measurer.block_height(el, 60, 300)
    assert many - one == pytest.approx((n - 1) * 60 * 1.2)
    assert one >= 60
    taller = dict(el, line_height=1.5)
    assert measurer.block_height(taller, 60, 300) - measurer.block_height(taller, 60, 800) == pytest.approx((n - 1) * 90)


def test_fit_font_size_is_maximal(measurer, l3):
    el = element(l3, "headline")
    f = measurer.fit_font_size(el, 380, 70, lo=6, hi=60)
    assert f is not None and measurer.fits(el, f, 380, 70)
    assert f == 60 or not measurer.fits(el, f + 0.5, 380, 70)


def test_fit_font_size_none_when_floor_fails(measurer, l3):
    assert measurer.fit_font_size(element(l3, "headline"), 100, 10, lo=14, hi=40) is None


def test_fits_respects_max_lines(measurer, l1):
    el = element(l1, "headline")
    assert measurer.fits(el, 60, 300, 1000)
    assert not measurer.fits(dict(el, max_lines=1), 60, 300, 1000)


def test_letter_spacing_counts_toward_line_width(measurer, l1):
    # The converter wraps without letter spacing but the renderer draws with it.
    el = element(l1, "headline")
    assert measurer.fits(el, 60, 800, 200)
    assert not measurer.fits(dict(el, letter_spacing=30), 60, 800, 200)


@pytest.mark.parametrize("task,el_id,box", [
    ("l3", "headline", (100, 10, 380, 70)),  # long serif copy, wraps to several lines
    ("l1", "cta", (20, 10, 700, 50)),        # one sans line limited by height: descenders must fit
])
def test_rendered_text_stays_inside_box(measurer, request, task, el_id, box):
    x, y, w, h = box
    text = dict(element(request.getfixturevalue(task), el_id), x=x, y=y, width=w, height=h,
                z_index=1, color="rgba(255,255,255,1)")
    text["font_size"] = measurer.fit_font_size(text, w, h, lo=6, hi=60)
    text["height"] = measurer.block_height(text, text["font_size"], w)
    bg = {"id": "bg", "type": "shape", "shape_type": "rectangle", "fill_color": "rgba(0,0,0,1)",
          "x": 0, "y": 0, "width": 728, "height": 90, "z_index": 0}
    img = render_to_image({"canvas": {"width": 728, "height": 90}, "elements": [bg, text]})
    s = img.width / 728
    ys, xs = np.nonzero(np.asarray(img.convert("L")) > 128)
    assert xs.min() >= (x - 2) * s and xs.max() <= (x + w + 2) * s
    assert ys.min() >= (y - 2) * s and ys.max() <= (y + text["height"] + 2) * s
