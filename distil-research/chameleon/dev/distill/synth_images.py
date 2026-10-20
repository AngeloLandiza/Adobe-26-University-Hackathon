"""Draw the synthetic stand-in images for the research corpus.

    cd distil-research/chameleon && python -m dev.distill.synth_images

Writes training-data/corpus/assets/syn_*.png and its README. Every image is
drawn with PIL/numpy from code below (gradients, blurred blobs, patterns,
simple product silhouettes, wordmarks set in the brand-kit fonts) with a
fixed seed, so a rerun gives the same pixels. No generative model and no
downloads are involved.

These are RESEARCH TRAINING DATA ONLY: they give the corpus more than the
brand kit's three images (other aspect ratios, colours and busy textures), so
anything learned from it isn't tuned to one bottle and one gradient. The
engine never draws, edits or ships them; it only resizes images it is given.
"""

import io
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from dev.distill.assets import SYNTH_ASSETS_DIR

FONTS = Path(__file__).resolve().parents[2] / "brand_kit" / "fonts"
SEED = 2026


def _hex(c: str) -> tuple[int, int, int]:
    c = c.lstrip("#")
    return tuple(int(c[i:i + 2], 16) for i in (0, 2, 4))


def _font(name: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(FONTS / name), size)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

def gradient(w: int, h: int, stops: list[tuple[float, str]], angle: float = 90) -> np.ndarray:
    """H x W x 3 float array: a linear multi-stop gradient. angle 90 = top to
    bottom, 0 = left to right."""
    yy, xx = np.mgrid[0:h, 0:w].astype(float)
    rad = math.radians(angle)
    t = (xx / max(w - 1, 1)) * math.cos(rad) + (yy / max(h - 1, 1)) * math.sin(rad)
    t = (t - t.min()) / max(t.max() - t.min(), 1e-9)
    pos = np.array([p for p, _ in stops])
    cols = np.array([_hex(c) for _, c in stops], dtype=float)
    out = np.zeros((h, w, 3))
    for ch in range(3):
        out[..., ch] = np.interp(t, pos, cols[:, ch])
    return out


def radial(w: int, h: int, cx: float, cy: float, r: float, inner: str, outer: str) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w].astype(float)
    t = np.clip(np.hypot(xx - cx * w, yy - cy * h) / (r * max(w, h)), 0, 1)[..., None]
    return (1 - t) * np.array(_hex(inner), float) + t * np.array(_hex(outer), float)


def blobs(base: np.ndarray, rng, n: int, colors: list[str], alpha: tuple[float, float], size: tuple[float, float],
          blur: float) -> np.ndarray:
    """Soft out-of-focus light blobs (bokeh-like) blended over `base`."""
    h, w, _ = base.shape
    out = base.copy()
    for _ in range(n):
        layer = Image.new("L", (w, h), 0)
        d = ImageDraw.Draw(layer)
        r = rng.uniform(*size) * max(w, h)
        cx, cy = rng.uniform(-0.1, 1.1) * w, rng.uniform(-0.1, 1.1) * h
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=255)
        mask = np.asarray(layer.filter(ImageFilter.GaussianBlur(blur * r)), float)[..., None] / 255
        a = rng.uniform(*alpha)
        col = np.array(_hex(colors[int(rng.integers(len(colors)))]), float)
        out = out * (1 - a * mask) + col * (a * mask)
    return out


def _write_png(img: Image.Image, name: str) -> Image.Image:
    """Save only if the bytes changed (rewriting identical files makes
    cloud-synced folders leave conflict copies)."""
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    path = SYNTH_ASSETS_DIR / name
    if not path.exists() or path.read_bytes() != buf.getvalue():
        path.write_bytes(buf.getvalue())
    return img


def save_rgb(arr: np.ndarray, name: str) -> Image.Image:
    return _write_png(Image.fromarray(np.clip(arr, 0, 255).round().astype(np.uint8), "RGB"), name)


def save_rgba(img: Image.Image, name: str) -> Image.Image:
    return _write_png(img, name)


def shadow(img: Image.Image, offset: tuple[int, int], blur: int, alpha: int = 90) -> Image.Image:
    """`img` with a soft drop shadow under its opaque pixels (same canvas)."""
    a = img.getchannel("A")
    sh = Image.new("RGBA", img.size, (0, 0, 0, 0))
    sh.putalpha(a.point(lambda v: v * alpha // 255))
    sh = sh.filter(ImageFilter.GaussianBlur(blur))
    moved = Image.new("RGBA", img.size, (0, 0, 0, 0))
    moved.paste(sh, offset)
    return Image.alpha_composite(moved, img)


def centered_text(d: ImageDraw.ImageDraw, xy, text: str, font, fill, spacing: int = 0):
    """Draw `text` centred on xy, with optional letter spacing (px)."""
    if not spacing:
        d.text(xy, text, font=font, fill=fill, anchor="mm")
        return
    widths = [font.getlength(c) for c in text]
    total = sum(widths) + spacing * (len(text) - 1)
    x = xy[0] - total / 2
    for c, cw in zip(text, widths):
        d.text((x, xy[1]), c, font=font, fill=fill, anchor="lm")
        x += cw + spacing


# ---------------------------------------------------------------------------
# The images. Each returns (file name, one-line description).
# ---------------------------------------------------------------------------

def backgrounds(rng) -> list[tuple[str, str]]:
    out = []
    a = gradient(1200, 1500, [(0, "#0B1630"), (0.55, "#123A55"), (1, "#1C5A63")])
    save_rgb(blobs(a, rng, 9, ["#3E8CA8", "#7FC4C9", "#F2B680"], (0.10, 0.25), (0.06, 0.16), 0.45),
             "syn_bg_harbor_night.png")
    out.append(("syn_bg_harbor_night.png", "dark: navy-to-teal vertical gradient with soft out-of-focus light blobs"))

    a = radial(1500, 1200, 0.3, 0.35, 0.95, "#2E5B3E", "#0E2418")
    save_rgb(blobs(a, rng, 8, ["#6FA36B", "#C9D9A0"], (0.08, 0.2), (0.05, 0.14), 0.5), "syn_bg_forest_mist.png")
    out.append(("syn_bg_forest_mist.png", "dark: deep-green radial gradient with pale soft blobs"))

    a = gradient(1200, 1200, [(0, "#1E120C"), (0.5, "#3A2418"), (1, "#5A3520")], angle=60)
    save_rgb(blobs(a, rng, 6, ["#C9772F", "#E8B27A"], (0.10, 0.22), (0.08, 0.2), 0.5), "syn_bg_roast_ember.png")
    out.append(("syn_bg_roast_ember.png", "dark: espresso-brown diagonal gradient with a warm ember glow"))

    a = radial(1200, 1600, 0.7, 0.25, 0.9, "#4A1C7A", "#130825")
    save_rgb(blobs(a, rng, 8, ["#B03AD8", "#FF4FA3", "#6C5CE7"], (0.10, 0.22), (0.05, 0.15), 0.45),
             "syn_bg_violet_glow.png")
    out.append(("syn_bg_violet_glow.png", "dark: deep-violet radial glow with magenta and blue blobs"))

    a = gradient(1500, 1000, [(0, "#FFE4D2"), (0.5, "#FBD3D6"), (1, "#FCE9DD")], angle=20)
    save_rgb(blobs(a, rng, 7, ["#FFFFFF", "#F7B8A8", "#FFD9A8"], (0.15, 0.35), (0.08, 0.2), 0.5),
             "syn_bg_peach_haze.png")
    out.append(("syn_bg_peach_haze.png", "light: peach-to-pink haze with soft white and apricot blobs"))

    a = gradient(1200, 1500, [(0, "#F4F9FD"), (0.6, "#DDEEF8"), (1, "#C7E2F2")])
    save_rgb(blobs(a, rng, 6, ["#FFFFFF", "#A9D3EE"], (0.15, 0.3), (0.08, 0.2), 0.5), "syn_bg_glacier.png")
    out.append(("syn_bg_glacier.png", "light: icy white-to-pale-blue vertical gradient with soft blobs"))

    a = gradient(1600, 900, [(0, "#FFF9E3"), (0.45, "#FFEFC2"), (1, "#FFE0A6")], angle=0)
    save_rgb(blobs(a, rng, 6, ["#FFFFFF", "#FFD27A"], (0.15, 0.3), (0.08, 0.22), 0.5), "syn_bg_sunlit_cream.png")
    out.append(("syn_bg_sunlit_cream.png", "light: warm cream-to-butter horizontal gradient (wide 16:9)"))

    a = gradient(1200, 1200, [(0, "#F3F4F6"), (1, "#DADDE2")], angle=75)
    save_rgb(blobs(a, rng, 6, ["#FFFFFF", "#C4C9D1"], (0.15, 0.35), (0.08, 0.2), 0.5), "syn_bg_fog.png")
    out.append(("syn_bg_fog.png", "light: cool grey fog gradient with soft blobs"))
    return out


def busy(rng) -> list[tuple[str, str]]:
    out = []
    # Confetti: hundreds of small rotated chips on a dark ground.
    img = Image.new("RGB", (1200, 1200), _hex("#1A1530"))
    d = ImageDraw.Draw(img)
    cols = ["#FF4FA3", "#FFD23F", "#3BCEAC", "#4D9DE0", "#FF7A00", "#FFFFFF", "#C6F432"]
    for _ in range(900):
        x, y = rng.uniform(0, 1200), rng.uniform(0, 1200)
        w, h, ang = rng.uniform(10, 30), rng.uniform(5, 14), rng.uniform(0, math.pi)
        c, s = math.cos(ang), math.sin(ang)
        pts = [(x + dx * c - dy * s, y + dx * s + dy * c) for dx, dy in
               ((-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2))]
        col = cols[int(rng.integers(len(cols)))]
        if rng.random() < 0.3:
            r = rng.uniform(4, 9)
            d.ellipse([x - r, y - r, x + r, y + r], fill=col)
        else:
            d.polygon(pts, fill=col)
    _write_png(img, "syn_busy_confetti.png")
    out.append(("syn_busy_confetti.png", "BUSY: dense multicolour confetti chips on dark violet"))

    # Diagonal stripes in five colours.
    w = h = 1200
    yy, xx = np.mgrid[0:h, 0:w]
    band = ((xx + yy) // 46) % 5
    pal = np.array([_hex(c) for c in ("#E63946", "#F1FAEE", "#A8DADC", "#457B9D", "#1D3557")], float)
    save_rgb(pal[band], "syn_busy_stripes.png")
    out.append(("syn_busy_stripes.png", "BUSY: high-contrast diagonal stripes in five colours"))

    # Terrazzo: irregular stone chips on a pale ground.
    img = Image.new("RGB", (1500, 1000), _hex("#EFE8DE"))
    d = ImageDraw.Draw(img)
    cols = ["#C8553D", "#2A9D8F", "#264653", "#E9C46A", "#8D99AE", "#F4A261"]
    for _ in range(700):
        x, y, r = rng.uniform(0, 1500), rng.uniform(0, 1000), rng.uniform(6, 26)
        n = int(rng.integers(4, 8))
        pts = [(x + r * rng.uniform(0.5, 1.1) * math.cos(2 * math.pi * k / n),
                y + r * rng.uniform(0.5, 1.1) * math.sin(2 * math.pi * k / n)) for k in range(n)]
        d.polygon(pts, fill=cols[int(rng.integers(len(cols)))])
    _write_png(img, "syn_busy_terrazzo.png")
    out.append(("syn_busy_terrazzo.png", "BUSY: terrazzo stone chips on a pale ground (3:2)"))

    # Mosaic tiles with per-tile noise.
    tiles = rng.integers(0, 6, size=(25, 25))
    pal = np.array([_hex(c) for c in ("#FF6B6B", "#FFD93D", "#6BCB77", "#4D96FF", "#2B2D42", "#F8F9FA")], float)
    arr = np.kron(pal[tiles], np.ones((40, 40, 1)))
    arr += np.kron(rng.normal(0, 22, size=(250, 250)), np.ones((4, 4)))[..., None]  # 4px grain
    save_rgb(arr, "syn_busy_mosaic.png")
    out.append(("syn_busy_mosaic.png", "BUSY: random colour mosaic tiles with pixel noise"))
    return out


def products(rng) -> list[tuple[str, str]]:
    out = []
    # Tall cold-brew bottle (2:5).
    img = Image.new("RGBA", (400, 1000), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([60, 330, 340, 960], 70, fill=_hex("#2B1A12"))
    d.polygon([(150, 160), (250, 160), (340, 360), (60, 360)], fill=_hex("#2B1A12"))
    d.rectangle([150, 70, 250, 180], fill=_hex("#2B1A12"))
    d.rounded_rectangle([140, 30, 260, 90], 12, fill=_hex("#C8553D"))
    d.rectangle([60, 520, 340, 800], fill=_hex("#F1E3CF"))
    d.rectangle([60, 560, 340, 590], fill=_hex("#C8553D"))
    centered_text(d, (200, 680), "COLD", _font("SolsticeSerif-Bold.ttf", 58), _hex("#2B1A12"))
    centered_text(d, (200, 750), "BREW", _font("SolsticeSans-Bold.ttf", 34), _hex("#2B1A12"), spacing=8)
    shine = Image.new("RGBA", img.size, (0, 0, 0, 0))
    ImageDraw.Draw(shine).rounded_rectangle([82, 380, 104, 900], 11, fill=(255, 255, 255, 45))
    img = Image.alpha_composite(img, shine)
    save_rgba(shadow(img, (6, 10), 12), "syn_prod_tall_bottle.png")
    out.append(("syn_prod_tall_bottle.png", "product cut-out, tall 2:5: dark cold-brew bottle with a cream label"))

    # Wide shoe box (5:3).
    img = Image.new("RGBA", (1000, 600), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.polygon([(80, 200), (820, 200), (940, 120), (200, 120)], fill=_hex("#FF7A00"))
    d.polygon([(820, 200), (940, 120), (940, 480), (820, 560)], fill=_hex("#B85600"))
    d.rectangle([80, 200, 820, 560], fill=_hex("#FF8F1F"))
    d.rectangle([80, 200, 820, 280], fill=_hex("#1B0F2E"))
    d.rectangle([80, 470, 820, 500], fill=_hex("#1B0F2E"))
    centered_text(d, (450, 380), "VOLTSTRIDE", _font("SolsticeSans-BoldItalic.ttf", 84), _hex("#1B0F2E"))
    save_rgba(shadow(img, (10, 14), 14), "syn_prod_wide_box.png")
    out.append(("syn_prod_wide_box.png", "product cut-out, wide 5:3: orange sneaker box in three-quarter view"))

    # Round coffee tin seen from above (1:1, round).
    img = Image.new("RGBA", (800, 800), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.ellipse([40, 40, 760, 760], fill=_hex("#B5651D"))
    d.ellipse([80, 80, 720, 720], fill=_hex("#3B2A20"))
    d.ellipse([170, 170, 630, 630], fill=_hex("#EADBC8"))
    d.ellipse([190, 190, 610, 610], outline=_hex("#B5651D"), width=6)
    centered_text(d, (400, 360), "Q", _font("SolsticeSerif-Bold.ttf", 190), _hex("#3B2A20"))
    centered_text(d, (400, 500), "HOUSE BLEND", _font("SolsticeSans-Bold.ttf", 36), _hex("#3B2A20"), spacing=6)
    for k in range(24):
        a = 2 * math.pi * k / 24
        d.ellipse([400 + 340 * math.cos(a) - 7, 400 + 340 * math.sin(a) - 7,
                   400 + 340 * math.cos(a) + 7, 400 + 340 * math.sin(a) + 7], fill=_hex("#EADBC8"))
    save_rgba(shadow(img, (8, 12), 14), "syn_prod_round_tin.png")
    out.append(("syn_prod_round_tin.png", "product cut-out, round 1:1: coffee tin seen from above"))

    # Square art book (1:1).
    img = Image.new("RGBA", (800, 800), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rectangle([90, 70, 730, 730], fill=_hex("#E9E2D4"))  # page block
    d.rectangle([70, 60, 710, 720], fill=_hex("#264653"))
    d.rectangle([70, 60, 100, 720], fill=_hex("#1B323B"))
    d.ellipse([330, 160, 610, 440], fill=_hex("#E9C46A"))
    d.rectangle([160, 300, 420, 560], fill=_hex("#E76F51"))
    centered_text(d, (400, 625), "THE QUIET HOURS", _font("SolsticeSerif-Bold.ttf", 48), _hex("#F4F1EA"))
    save_rgba(shadow(img, (10, 12), 14), "syn_prod_square_book.png")
    out.append(("syn_prod_square_book.png", "product cut-out, square 1:1: art-book cover with abstract shapes"))

    # Sneaker silhouette (2:1).
    img = Image.new("RGBA", (1200, 600), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    sole = [(80, 470), (1120, 470), (1140, 520), (1100, 545), (110, 545), (70, 515)]
    upper = [(110, 470), (120, 330), (210, 250), (380, 200), (520, 120), (640, 110), (700, 200),
             (860, 280), (1040, 330), (1120, 400), (1120, 470)]
    d.polygon(upper, fill=_hex("#7B2FF7"))
    d.polygon(sole, fill=_hex("#F5F0FF"))
    d.line([(150, 470), (1110, 470)], fill=_hex("#C6F432"), width=16)
    d.polygon([(420, 300), (760, 230), (980, 360), (760, 330)], fill=_hex("#C6F432"))
    for k in range(5):
        d.line([(560 + 30 * k, 160 + 22 * k), (650 + 30 * k, 190 + 22 * k)], fill=_hex("#F5F0FF"), width=10)
    save_rgba(shadow(img, (8, 16), 16), "syn_prod_sneaker.png")
    out.append(("syn_prod_sneaker.png", "product cut-out, wide 2:1: violet-and-lime sneaker in side view"))

    # Phone with a fitness app screen (1:2).
    img = Image.new("RGBA", (500, 1000), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([20, 20, 480, 980], 70, fill=_hex("#1A1A1A"))
    d.rounded_rectangle([44, 50, 456, 950], 52, fill=_hex("#FFF8D6"))
    d.rounded_rectangle([190, 66, 310, 92], 13, fill=_hex("#1A1A1A"))
    d.arc([110, 170, 390, 450], 135, 405, fill=_hex("#E5DFC4"), width=34)
    d.arc([110, 170, 390, 450], 135, 330, fill=_hex("#FF7A00"), width=34)
    centered_text(d, (250, 310), "7.2", _font("SolsticeSans-Bold.ttf", 86), _hex("#222222"))
    for k, (wd, col) in enumerate(((300, "#3BB273"), (220, "#FFD23F"), (340, "#FF7A00"))):
        d.rounded_rectangle([90, 540 + 110 * k, 90 + wd, 600 + 110 * k], 30, fill=_hex(col))
    save_rgba(shadow(img, (8, 12), 14), "syn_prod_phone.png")
    out.append(("syn_prod_phone.png", "product cut-out, tall 1:2: phone showing a fitness-app ring and bars"))

    # Stand-up coffee pouch (7:9).
    img = Image.new("RGBA", (700, 900), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.polygon([(110, 80), (590, 80), (640, 840), (60, 840)], fill=_hex("#EADBC8"))
    d.rectangle([110, 80, 590, 130], fill=_hex("#C9B79F"))
    d.rectangle([90, 420, 610, 640], fill=_hex("#3B2A20"))
    centered_text(d, (350, 500), "QUILLBROOK", _font("SolsticeSerif-Bold.ttf", 66), _hex("#EADBC8"))
    centered_text(d, (350, 580), "SINGLE ORIGIN", _font("SolsticeSans-Bold.ttf", 30), _hex("#E0A458"), spacing=6)
    d.ellipse([290, 220, 410, 340], fill=_hex("#B5651D"))
    save_rgba(shadow(img, (8, 12), 14), "syn_prod_coffee_bag.png")
    out.append(("syn_prod_coffee_bag.png", "product cut-out, portrait 7:9: stand-up coffee pouch"))
    return out


def logos(rng) -> list[tuple[str, str]]:
    out = []
    for name, col in (("syn_logo_blocks_light.png", (250, 248, 242, 255)), ("syn_logo_blocks_dark.png", (34, 30, 40, 255))):
        img = Image.new("RGBA", (1000, 250), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        d.ellipse([20, 45, 180, 205], fill=col)
        d.ellipse([62, 87, 138, 163], fill=(0, 0, 0, 0))
        x = 230
        for k, wd in enumerate((150, 110, 190, 90, 160)):  # wordmark-ish letter blocks
            top = 75 if k % 2 else 60
            d.rounded_rectangle([x, top, x + wd, 190], 18, fill=col)
            x += wd + 26
        save_rgba(img, name)
        tone = "light (for dark backgrounds)" if "light" in name else "dark (for light backgrounds)"
        out.append((name, f"logo, wide 4:1, {tone}: ring mark plus abstract wordmark-like blocks, no text"))

    img = Image.new("RGBA", (600, 600), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.ellipse([10, 10, 590, 590], fill=_hex("#EADBC8"))
    d.ellipse([40, 40, 560, 560], fill=_hex("#3B2A20"))
    d.ellipse([70, 70, 530, 530], outline=_hex("#E0A458"), width=6)
    centered_text(d, (300, 260), "Q", _font("SolsticeSerif-Bold.ttf", 230), _hex("#EADBC8"))
    centered_text(d, (300, 420), "QUILLBROOK", _font("SolsticeSans-Bold.ttf", 44), _hex("#E0A458"), spacing=4)
    save_rgba(img, "syn_logo_quillbrook_badge.png")
    out.append(("syn_logo_quillbrook_badge.png", "logo, round 1:1: coffee-roaster badge for the invented brand Quillbrook"))

    img = Image.new("RGBA", (1000, 200), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.polygon([(40, 20), (120, 20), (90, 90), (140, 90), (50, 185), (70, 110), (20, 110)], fill=_hex("#C6F432"))
    d.text((170, 100), "VOLTSTRIDE", font=_font("SolsticeSans-BoldItalic.ttf", 128), fill=_hex("#F5F0FF"),
           anchor="lm")
    save_rgba(img, "syn_logo_voltstride_light.png")
    out.append(("syn_logo_voltstride_light.png", "logo, wide 5:1, light: bolt mark + italic wordmark for the invented "
                                                 "sneaker brand Voltstride"))

    img = Image.new("RGBA", (900, 600), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    ink = _hex("#2A1B3D")
    d.ellipse([430, 10, 470, 50], outline=ink, width=8)  # a small lantern: ring, cap, glowing window, base
    d.polygon([(400, 70), (500, 70), (480, 45), (420, 45)], fill=ink)
    d.rectangle([410, 70, 490, 180], fill=ink)
    d.rounded_rectangle([428, 88, 472, 162], 10, fill=_hex("#E9B44C"))
    d.rectangle([395, 180, 505, 200], fill=ink)
    centered_text(d, (450, 320), "INKWELL &", _font("SolsticeSerif-Bold.ttf", 110), ink)
    centered_text(d, (450, 440), "LANTERN", _font("SolsticeSerif-Bold.ttf", 110), ink)
    d.rectangle([250, 510, 650, 518], fill=ink)
    centered_text(d, (450, 560), "BOOKS", _font("SolsticeSans-Bold.ttf", 40), ink, spacing=14)
    save_rgba(img, "syn_logo_inkwell_stacked.png")
    out.append(("syn_logo_inkwell_stacked.png", "logo, stacked 3:2, dark ink: lantern mark over a two-line serif "
                                                "wordmark for the invented bookstore Inkwell & Lantern"))

    img = Image.new("RGBA", (900, 400), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([20, 80, 260, 320], 60, fill=_hex("#FF7A00"))
    d.line([(60, 210), (110, 210), (140, 140), (180, 270), (205, 200), (230, 200)], fill=_hex("#FFF8D6"),
           width=18, joint="curve")
    d.text((290, 200), "pacewell", font=_font("SolsticeSans-Bold.ttf", 120), fill=_hex("#222222"), anchor="lm")
    save_rgba(img, "syn_logo_pacewell.png")
    out.append(("syn_logo_pacewell.png", "logo, wide 9:4, dark: app-icon tile with a pulse line + wordmark for "
                                         "the invented fitness app Pacewell"))
    return out


README = """# corpus/assets: synthetic stand-in images (research only)

**What it is:** {n} PNG images drawn by code (`chameleon/dev/distill/synth_images.py`) for the research corpus.
They exist so the corpus isn't limited to the brand kit's three images (one bottle, one gradient, one logo):
backgrounds of other colours and tones, *busy* textures (the rubric says text must never sit on a busy
image, so the corpus needs examples), product cut-outs with transparency in several proportions, and logo
marks in several aspect ratios.

**How they were made:** pure PIL/numpy drawing with a fixed seed (`SEED = {seed}`): linear and radial
gradients, Gaussian-blurred ellipses, random chips and stripes, simple polygons for the products, and
wordmarks set in the brand kit's own Solstice fonts. No generative AI model and no downloads.
`python -m dev.distill.synth_images` (from `distil-research/chameleon/`) redraws them all identically.

**Where they are used:** only by corpus ads (`origin` `augmented` or `offbrand`) via `asset_id: "syn_..."`.
Every file is prefixed `syn_` so it can never collide with a brand-kit asset name;
`dev/distill/assets.py` merges this folder with `chameleon/brand_kit/assets/` for each run.

**Not shipped:** these are research training data. The engine never draws, edits or outputs them (the
challenge disqualifies regenerating image content; these are *inputs* standing in for a real brand's
photos, and they never leave `distil-research/`). The brand names on them (Quillbrook, Voltstride,
Inkwell & Lantern, Pacewell) are invented.

| file | size | what it is |
|---|---|---|
{rows}
"""


def main() -> None:
    SYNTH_ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    made = backgrounds(rng) + busy(rng) + products(rng) + logos(rng)
    for stale in SYNTH_ASSETS_DIR.glob("syn_*"):
        if stale.name not in {name for name, _ in made}:
            stale.unlink()
    rows = []
    for name, desc in made:
        with Image.open(SYNTH_ASSETS_DIR / name) as im:
            rows.append(f"| `{name}` | {im.width}x{im.height} {im.mode} | {desc} |")
    readme = README.format(n=len(made), seed=SEED, rows="\n".join(rows))
    if not (SYNTH_ASSETS_DIR / "README.md").exists() or (SYNTH_ASSETS_DIR / "README.md").read_text() != readme:
        (SYNTH_ASSETS_DIR / "README.md").write_text(readme)
    print(f"wrote {len(made)} images to {SYNTH_ASSETS_DIR}")


if __name__ == "__main__":
    main()
