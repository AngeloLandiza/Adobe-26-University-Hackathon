"""Draw the challenge corpus's own synthetic images (prefix `chl_`).

    cd distil-research/chameleon && python -m dev.distill.challenge_images

Same method as synth_images.py (PIL/numpy drawing, fixed seed, no generative
model, no downloads), but deliberately different STYLES and ASPECTS from the
`syn_` set, so an approach tuned on the corpus meets imagery it has never
seen: layered "photo-like" landscapes with film grain, a night skyline, grainy
monochrome concrete, a halftone and a Bauhaus pattern, a 4:1 panorama, and
product cut-outs and logos for five invented brands (Lumen & Fig, Obsidian
Audio, Fizzbang, Monograph, Meridian Air). Evaluation data only; never shipped.
Written next to the `syn_` images in training-data/corpus/assets/ (the `chl_`
prefix can't collide; synth_images.py only manages `syn_*` files).
"""

import math

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from dev.distill.assets import SYNTH_ASSETS_DIR
from dev.distill.synth_images import _font, _hex, _write_png, centered_text, gradient, radial, save_rgba, shadow

SEED = 4242
PREFIX = "chl_"


def _grain(arr: np.ndarray, rng, amount: float = 6.0, cell: int = 2) -> np.ndarray:
    """Soft film grain: blocky noise, `cell` px per grain."""
    h, w = arr.shape[:2]
    noise = rng.normal(0, amount, size=(-(-h // cell), -(-w // cell)))
    noise = np.kron(noise, np.ones((cell, cell)))[:h, :w]
    return arr + noise[..., None]


def _save_rgb(arr: np.ndarray, name: str) -> Image.Image:
    return _write_png(Image.fromarray(np.clip(arr, 0, 255).round().astype(np.uint8), "RGB"), name)


def _ridge(w: int, h: int, rng, base: float, amp: float, waves: int) -> np.ndarray:
    """A smooth random skyline height (in px from the top) for each column."""
    x = np.linspace(0, 1, w)
    y = np.full(w, base * h)
    for k in range(1, waves + 1):
        y += amp * h / k * np.sin(2 * math.pi * (k * x + rng.uniform(0, 1)))
    return y


def _layer(arr: np.ndarray, ridge: np.ndarray, color: str, bottom: float | None = None) -> np.ndarray:
    """Paint everything below `ridge` (and above `bottom`, if given) with `color`,
    edges softened by a slight blur."""
    h, w, _ = arr.shape
    rows = np.arange(h)[:, None]
    mask = (rows >= ridge[None, :]) & (rows < (h if bottom is None else bottom))
    mask = mask.astype(np.uint8) * 255
    mask = np.asarray(Image.fromarray(mask).filter(ImageFilter.GaussianBlur(1.2)), float)[..., None] / 255
    return arr * (1 - mask) + np.array(_hex(color), float) * mask


# ---------------------------------------------------------------------------
# Photo-like scenes and patterns
# ---------------------------------------------------------------------------

def scenes(rng) -> list[tuple[str, str]]:
    out = []
    # Dunes at golden hour (3:2): sky gradient, low sun, four layered dune ridges, grain.
    w, h = 1500, 1000
    a = gradient(w, h, [(0, "#F6C9A0"), (0.45, "#F3A979"), (1, "#E9875C")])
    yy, xx = np.mgrid[0:h, 0:w]
    sun = np.clip(1 - np.hypot(xx - 0.68 * w, yy - 0.42 * h) / 120, 0, 1)[..., None]
    a = a * (1 - sun) + np.array(_hex("#FFF1D6"), float) * sun
    for base, amp, col in ((0.50, 0.03, "#D9784E"), (0.60, 0.04, "#C4643F"), (0.72, 0.05, "#A9512F"),
                           (0.85, 0.04, "#8A3F24")):
        a = _layer(a, _ridge(w, h, rng, base, amp, 3), col)
    _save_rgb(_grain(a, rng, 3.5, 3), f"{PREFIX}photo_dunes.png")
    out.append((f"{PREFIX}photo_dunes.png", "photo-like, 3:2, warm: dunes at golden hour, low sun, film grain"))

    # Night skyline (16:9): navy sky, stars, three rows of towers with lit windows, a reflection band.
    w, h = 1600, 900
    img = Image.fromarray(gradient(w, h, [(0, "#060B1E"), (0.6, "#14204A"), (1, "#24346B")]).astype(np.uint8))
    d = ImageDraw.Draw(img)
    for _ in range(140):
        x, y = rng.uniform(0, w), rng.uniform(0, h * 0.45)
        r = rng.uniform(0.6, 1.8)
        d.ellipse([x - r, y - r, x + r, y + r], fill=(220, 230, 255))
    for row, (top, col, lit) in enumerate(((0.38, "#1B2650", 0.10), (0.48, "#121A3A", 0.18), (0.60, "#0A0F24", 0.28))):
        x = -20
        while x < w:
            bw = rng.uniform(50, 130)
            bh = rng.uniform(0.15, 0.45) * h if row < 2 else rng.uniform(0.1, 0.3) * h
            y0 = top * h + rng.uniform(-0.12, 0.12) * h
            d.rectangle([x, y0, x + bw, h], fill=_hex(col))
            for wy in np.arange(y0 + 10, h - 10, 16):
                for wx in np.arange(x + 8, x + bw - 8, 14):
                    if rng.random() < lit:
                        d.rectangle([wx, wy, wx + 6, wy + 8], fill=(255, 214, 140) if rng.random() < 0.8 else (140, 230, 255))
            x += bw + rng.uniform(2, 14)
    _write_png(img.filter(ImageFilter.GaussianBlur(0.6)), f"{PREFIX}photo_city_night.png")
    out.append((f"{PREFIX}photo_city_night.png", "photo-like, 16:9, dark: night skyline, lit windows, stars"))

    # Monochrome concrete architecture (4:5): grey planes with hard shadows, grain.
    w, h = 1200, 1500
    img = Image.fromarray(gradient(w, h, [(0, "#D9D9D9"), (1, "#A6A6A6")], angle=80).astype(np.uint8))
    d = ImageDraw.Draw(img)
    d.polygon([(0, 260), (760, 120), (760, 1500), (0, 1500)], fill=(176, 176, 176))
    d.polygon([(760, 120), (1200, 330), (1200, 1500), (760, 1500)], fill=(120, 120, 120))
    d.polygon([(0, 780), (760, 700), (760, 760), (0, 850)], fill=(92, 92, 92))
    d.polygon([(760, 700), (1200, 820), (1200, 880), (760, 760)], fill=(62, 62, 62))
    d.polygon([(160, 900), (420, 870), (420, 1500), (160, 1500)], fill=(140, 140, 140))
    d.polygon([(420, 870), (560, 900), (560, 1500), (420, 1500)], fill=(96, 96, 96))
    arr = _grain(np.asarray(img, float), rng, 5, 3)
    _save_rgb(arr, f"{PREFIX}photo_concrete_mono.png")
    out.append((f"{PREFIX}photo_concrete_mono.png", "photo-like, 4:5, monochrome: concrete building planes and "
                                                    "hard shadows, grain"))

    # Botanical (4:5): big soft leaves on blush.
    w, h = 1200, 1500
    img = Image.new("RGB", (w, h), _hex("#EFD9CC"))
    for k in range(16):
        layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        ld = ImageDraw.Draw(layer)
        cx, cy = rng.uniform(-0.1, 1.1) * w, rng.uniform(-0.1, 1.1) * h
        L, W = rng.uniform(380, 700), rng.uniform(120, 220)
        ang = rng.uniform(0, math.pi)
        pts = []
        for t in np.linspace(0, 1, 40):
            r = math.sin(math.pi * t) * W
            pts.append((t * L - L / 2, r))
        pts += [(x, -y) for x, y in reversed(pts)]
        c, s = math.cos(ang), math.sin(ang)
        poly = [(cx + x * c - y * s, cy + x * s + y * c) for x, y in pts]
        col = ["#6F8A4E", "#57703C", "#8DA66A", "#455A2E"][k % 4]
        ld.polygon(poly, fill=_hex(col) + (230,))
        ld.line([(cx - L / 2 * c, cy - L / 2 * s), (cx + L / 2 * c, cy + L / 2 * s)], fill=_hex("#3A4A26") + (200,), width=5)
        img = Image.alpha_composite(img.convert("RGBA"), layer.filter(ImageFilter.GaussianBlur(2 if k < 8 else 0.5))).convert("RGB")
    _save_rgb(_grain(np.asarray(img, float), rng, 3, 3), f"{PREFIX}photo_botanical.png")
    out.append((f"{PREFIX}photo_botanical.png", "photo-like, 4:5, light: overlapping olive leaves on blush, grain"))

    # Coastal panorama (4:1): sky with soft clouds, a green headland on the left, blue sea with glitter.
    w, h = 2000, 500
    horizon = 0.58 * h
    a = gradient(w, h, [(0, "#7FC1EE"), (0.57, "#E3F3FB"), (0.58, "#2A86CF"), (1, "#0B4A8A")])
    yy, xx = np.mgrid[0:h, 0:w]
    for _ in range(7):
        cx, cy, rx, ry = rng.uniform(0.3, 1.0) * w, rng.uniform(0.12, 0.38) * h, rng.uniform(90, 220), rng.uniform(18, 34)
        m = np.clip(1 - np.hypot((xx - cx) / rx, (yy - cy) / ry), 0, 1)[..., None] ** 0.6 * 0.7
        a = a * (1 - m) + 255 * m
    x = np.linspace(0, 1, w)
    hump = np.clip(1 - (x / 0.38) ** 2, 0, None) ** 0.7
    a = _layer(a, horizon - 0.34 * h * hump + 0.02 * h * np.sin(40 * x), "#3E7A4C", horizon)
    a = _layer(a, horizon - 0.22 * h * np.clip(1 - (x / 0.26) ** 2, 0, None) ** 0.8, "#2C5E3A", horizon)
    img = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(img)
    for _ in range(260):
        gx, gy = rng.uniform(0.42, 1.0) * w, rng.uniform(horizon + 8, 0.95 * h)
        d.line([(gx, gy), (gx + rng.uniform(8, 26), gy)], fill=(235, 248, 255), width=2)
    _save_rgb(_grain(np.asarray(img, float), rng, 3, 3), f"{PREFIX}photo_coast_panorama.png")
    out.append((f"{PREFIX}photo_coast_panorama.png", "photo-like, 4:1 panorama: sky with clouds, green headland, "
                                                     "blue sea with glitter"))

    # Halftone (1:1): saturated pink dots on yellow, dot size sweeping across.
    w = h = 1200
    img = Image.new("RGB", (w, h), _hex("#FFE600"))
    d = ImageDraw.Draw(img)
    step = 30
    for gy in range(0, h + step, step):
        for gx in range(0, w + step, step):
            off = step / 2 if (gy // step) % 2 else 0
            r = 2 + 12 * (gx + gy) / (w + h)
            d.ellipse([gx + off - r, gy - r, gx + off + r, gy + r], fill=_hex("#FF2E88"))
    _write_png(img, f"{PREFIX}pattern_halftone.png")
    out.append((f"{PREFIX}pattern_halftone.png", "BUSY pattern, 1:1, saturated: pink halftone dots on yellow"))

    # Bauhaus (3:1): flat circles, quarter discs and bars, saturated.
    w, h = 1500, 500
    img = Image.new("RGB", (w, h), _hex("#1F3BFF"))
    d = ImageDraw.Draw(img)
    cols = ["#FF2E88", "#FFE600", "#FFFFFF", "#14002E", "#00D1B2"]
    for i in range(12):
        x0 = i * 125
        kind = i % 4
        col = _hex(cols[int(rng.integers(len(cols)))])
        if kind == 0:
            d.ellipse([x0 + 10, 60, x0 + 240, 290], fill=col)
        elif kind == 1:
            d.pieslice([x0 - 125, 250, x0 + 125, 500], 270, 360, fill=col)
        elif kind == 2:
            d.rectangle([x0 + 30, 0, x0 + 70, 500], fill=col)
        else:
            d.polygon([(x0, 500), (x0 + 125, 300), (x0 + 250, 500)], fill=col)
    _write_png(img, f"{PREFIX}pattern_bauhaus.png")
    out.append((f"{PREFIX}pattern_bauhaus.png", "BUSY pattern, 3:1, saturated: flat Bauhaus circles, bars and triangles"))
    return out


# ---------------------------------------------------------------------------
# Products
# ---------------------------------------------------------------------------

def products(rng) -> list[tuple[str, str]]:
    out = []
    # Over-ear headphones (1:1.1).
    img = Image.new("RGBA", (800, 880), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.arc([110, 40, 690, 620], 180, 360, fill=_hex("#2A2E36"), width=60)
    d.arc([130, 60, 670, 600], 190, 350, fill=_hex("#C9CED6"), width=10)
    for x0 in (60, 520):
        d.rounded_rectangle([x0 + 40, 300, x0 + 100, 380], 18, fill=_hex("#2A2E36"))
        d.rounded_rectangle([x0, 360, x0 + 220, 780], 100, fill=_hex("#16181D"))
        d.rounded_rectangle([x0 + 30, 400, x0 + 190, 740], 80, fill=_hex("#2A2E36"))
    d.ellipse([565, 520, 655, 610], fill=_hex("#34E1F2"))
    save_rgba(shadow(img, (8, 14), 16), f"{PREFIX}prod_headphones.png")
    out.append((f"{PREFIX}prod_headphones.png", "product cut-out, 10:11: black over-ear headphones, cyan accent"))

    # Earbuds charging case, open (3:2).
    img = Image.new("RGBA", (900, 600), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([120, 260, 780, 540], 120, fill=_hex("#1C1F26"))
    d.rounded_rectangle([120, 120, 780, 300], 90, fill=_hex("#2A2E36"))
    d.rectangle([120, 260, 780, 290], fill=_hex("#0C0D10"))
    for cx in (330, 570):
        d.ellipse([cx - 70, 170, cx + 70, 310], fill=_hex("#E9ECF1"))
        d.ellipse([cx - 30, 210, cx + 30, 270], fill=_hex("#34E1F2"))
    d.ellipse([435, 470, 465, 500], fill=_hex("#34E1F2"))
    save_rgba(shadow(img, (8, 12), 14), f"{PREFIX}prod_earbuds_case.png")
    out.append((f"{PREFIX}prod_earbuds_case.png", "product cut-out, 3:2: open earbuds case, two white buds"))

    # Serum dropper bottle (1:3).
    img = Image.new("RGBA", (300, 900), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([110, 20, 190, 230], 30, fill=_hex("#2E2A26"))
    d.rectangle([90, 220, 210, 300], fill=_hex("#B4532A"))
    d.rounded_rectangle([40, 290, 260, 880], 40, fill=_hex("#D8875E"))
    d.rounded_rectangle([60, 470, 240, 760], 10, fill=_hex("#F7F1EA"))
    centered_text(d, (150, 560), "L&F", _font("SolsticeSerif-Bold.ttf", 54), _hex("#2E2A26"))
    centered_text(d, (150, 650), "SERUM", _font("SolsticeSans-Bold.ttf", 26), _hex("#B4532A"), spacing=6)
    shine = Image.new("RGBA", img.size, (0, 0, 0, 0))
    ImageDraw.Draw(shine).rounded_rectangle([58, 320, 76, 850], 9, fill=(255, 255, 255, 60))
    save_rgba(shadow(Image.alpha_composite(img, shine), (6, 10), 12), f"{PREFIX}prod_serum.png")
    out.append((f"{PREFIX}prod_serum.png", "product cut-out, tall 1:3: amber serum dropper bottle with a paper label"))

    # Squat cream jar (3:2).
    img = Image.new("RGBA", (900, 600), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([130, 80, 770, 230], 40, fill=_hex("#5A6B3A"))
    d.rounded_rectangle([110, 210, 790, 560], 60, fill=_hex("#F2E6DA"))
    d.rectangle([110, 330, 790, 440], fill=_hex("#EBC9B9"))
    centered_text(d, (450, 385), "LUMEN & FIG", _font("SolsticeSerif-Bold.ttf", 56), _hex("#2E2A26"))
    save_rgba(shadow(img, (8, 12), 14), f"{PREFIX}prod_cream_jar.png")
    out.append((f"{PREFIX}prod_cream_jar.png", "product cut-out, wide 3:2: squat cream jar with an olive lid"))

    # Wristwatch, upright (5:8), monochrome.
    img = Image.new("RGBA", (500, 800), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([170, 0, 330, 800], 30, fill=_hex("#2B2B2B"))
    for y in range(20, 800, 40):
        d.line([(180, y), (320, y)], fill=_hex("#1A1A1A"), width=3)
    d.ellipse([60, 210, 440, 590], fill=_hex("#BDBDBD"))
    d.ellipse([85, 235, 415, 565], fill=_hex("#F5F5F5"))
    for k in range(12):
        a = 2 * math.pi * k / 12
        r0, r1 = (130, 160) if k % 3 else (115, 160)
        d.line([(250 + r0 * math.sin(a), 400 - r0 * math.cos(a)), (250 + r1 * math.sin(a), 400 - r1 * math.cos(a))],
               fill=_hex("#111111"), width=6 if k % 3 == 0 else 3)
    d.line([(250, 400), (250, 290)], fill=_hex("#111111"), width=8)
    d.line([(250, 400), (330, 440)], fill=_hex("#111111"), width=6)
    d.rectangle([440, 380, 470, 420], fill=_hex("#BDBDBD"))
    save_rgba(shadow(img, (8, 12), 14), f"{PREFIX}prod_watch.png")
    out.append((f"{PREFIX}prod_watch.png", "product cut-out, 5:8, monochrome: wristwatch with strap, white dial"))

    # Soda can (5:9).
    img = Image.new("RGBA", (500, 900), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([70, 40, 430, 120], 30, fill=_hex("#D0D3DA"))
    d.rounded_rectangle([50, 90, 450, 860], 50, fill=_hex("#FF2E88"))
    d.rectangle([50, 300, 450, 620], fill=_hex("#FFE600"))
    d.polygon([(50, 620), (450, 520), (450, 600), (50, 700)], fill=_hex("#1F3BFF"))
    centered_text(d, (250, 400), "FIZZ", _font("SolsticeSans-BoldItalic.ttf", 120), _hex("#14002E"))
    centered_text(d, (250, 510), "BANG", _font("SolsticeSans-BoldItalic.ttf", 84), _hex("#FF2E88"))
    shine = Image.new("RGBA", img.size, (0, 0, 0, 0))
    ImageDraw.Draw(shine).rounded_rectangle([90, 130, 120, 820], 15, fill=(255, 255, 255, 70))
    save_rgba(shadow(Image.alpha_composite(img, shine), (8, 12), 14), f"{PREFIX}prod_soda_can.png")
    out.append((f"{PREFIX}prod_soda_can.png", "product cut-out, tall 5:9, saturated: pink-and-yellow soda can"))

    # Game controller (8:5).
    img = Image.new("RGBA", (960, 600), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([140, 120, 820, 400], 140, fill=_hex("#1F3BFF"))
    d.ellipse([60, 200, 380, 560], fill=_hex("#1F3BFF"))
    d.ellipse([580, 200, 900, 560], fill=_hex("#1F3BFF"))
    d.rectangle([215, 230, 245, 350], fill=_hex("#FFFFFF"))
    d.rectangle([170, 275, 290, 305], fill=_hex("#FFFFFF"))
    for (cx, cy, col) in ((690, 230, "#FFE600"), (750, 290, "#FF2E88"), (630, 290, "#00D1B2"), (690, 350, "#FFFFFF")):
        d.ellipse([cx - 26, cy - 26, cx + 26, cy + 26], fill=_hex(col))
    d.ellipse([360, 340, 450, 430], fill=_hex("#14002E"))
    d.ellipse([510, 340, 600, 430], fill=_hex("#14002E"))
    save_rgba(shadow(img, (8, 12), 14), f"{PREFIX}prod_controller.png")
    out.append((f"{PREFIX}prod_controller.png", "product cut-out, wide 8:5, saturated: electric-blue game controller"))

    # Hard-shell suitcase (3:4).
    img = Image.new("RGBA", (600, 800), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([230, 20, 370, 120], 30, outline=_hex("#1B2440"), width=24)
    d.rectangle([270, 100, 330, 140], fill=_hex("#1B2440"))
    d.rounded_rectangle([80, 130, 520, 740], 50, fill=_hex("#FF5A4E"))
    for x in (190, 300, 410):
        d.rectangle([x - 8, 150, x + 8, 720], fill=_hex("#E04A3F"))
    for x in (130, 470):
        d.ellipse([x - 30, 730, x + 30, 790], fill=_hex("#1B2440"))
    save_rgba(shadow(img, (8, 12), 14), f"{PREFIX}prod_suitcase.png")
    out.append((f"{PREFIX}prod_suitcase.png", "product cut-out, 3:4: coral hard-shell suitcase"))

    # Sunglasses (3:1).
    img = Image.new("RGBA", (900, 300), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    for x0 in (60, 500):
        d.rounded_rectangle([x0, 60, x0 + 340, 260], 90, fill=_hex("#1B2440"))
        d.rounded_rectangle([x0 + 20, 80, x0 + 320, 240], 75, fill=_hex("#2F5D9E"))
        d.ellipse([x0 + 60, 100, x0 + 140, 150], fill=(255, 255, 255, 110))
    d.arc([380, 60, 520, 160], 200, 340, fill=_hex("#1B2440"), width=22)
    d.rectangle([0, 80, 70, 100], fill=_hex("#1B2440"))
    d.rectangle([830, 80, 900, 100], fill=_hex("#1B2440"))
    save_rgba(shadow(img, (6, 10), 12), f"{PREFIX}prod_sunglasses.png")
    out.append((f"{PREFIX}prod_sunglasses.png", "product cut-out, very wide 3:1: navy sunglasses with blue lenses"))
    return out


# ---------------------------------------------------------------------------
# Logos
# ---------------------------------------------------------------------------

def logos(rng) -> list[tuple[str, str]]:
    out = []
    # Lumen & Fig: a fig mark + serif wordmark, dark ink (4:1).
    img = Image.new("RGBA", (1000, 250), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    ink = _hex("#2E2A26")
    d.ellipse([20, 60, 170, 230], fill=_hex("#B4532A"))
    d.polygon([(95, 20), (120, 70), (70, 70)], fill=_hex("#5A6B3A"))
    d.text((200, 135), "Lumen & Fig", font=_font("SolsticeSerif-Bold.ttf", 120), fill=ink, anchor="lm")
    save_rgba(img, f"{PREFIX}logo_lumen_fig.png")
    out.append((f"{PREFIX}logo_lumen_fig.png", "logo, wide 4:1, dark ink: fig mark + serif wordmark (Lumen & Fig)"))

    # Obsidian Audio: light wordmark with a cyan wave mark (5:1).
    img = Image.new("RGBA", (1000, 200), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    for k, hgt in enumerate((40, 90, 140, 90, 40)):
        x = 20 + 26 * k
        d.rounded_rectangle([x, 100 - hgt / 2, x + 14, 100 + hgt / 2], 7, fill=_hex("#34E1F2"))
    d.text((170, 100), "OBSIDIAN", font=_font("SolsticeSans-Bold.ttf", 112), fill=_hex("#F2F4F7"), anchor="lm")
    save_rgba(img, f"{PREFIX}logo_obsidian_light.png")
    out.append((f"{PREFIX}logo_obsidian_light.png", "logo, wide 5:1, light: sound-wave bars + wordmark (Obsidian Audio)"))

    # Fizzbang: a self-contained sticker logo on its own yellow lozenge (2:1), works on any background.
    img = Image.new("RGBA", (800, 400), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([10, 40, 790, 360], 160, fill=_hex("#14002E"))
    d.rounded_rectangle([30, 60, 770, 340], 140, fill=_hex("#FFE600"))
    centered_text(d, (400, 205), "FIZZBANG!", _font("SolsticeSans-BoldItalic.ttf", 130), _hex("#FF2E88"))
    save_rgba(img, f"{PREFIX}logo_fizzbang.png")
    out.append((f"{PREFIX}logo_fizzbang.png", "logo, 2:1, saturated: pink italic wordmark on a yellow lozenge "
                                              "with a dark rim (Fizzbang)"))

    # Monograph: black square monogram + stacked wordmark (1:1), and a white version.
    for name, fg in (("logo_monograph_black.png", (17, 17, 17, 255)), ("logo_monograph_white.png", (255, 255, 255, 255))):
        img = Image.new("RGBA", (600, 600), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        d.rectangle([150, 40, 450, 340], outline=fg, width=16)
        centered_text(d, (300, 196), "M", _font("SolsticeSerif-Bold.ttf", 230), fg)
        centered_text(d, (300, 440), "MONOGRAPH", _font("SolsticeSans-Bold.ttf", 66), fg, spacing=10)
        centered_text(d, (300, 530), "GALLERY", _font("SolsticeSans-Regular.ttf", 44), fg, spacing=18)
        save_rgba(img, PREFIX + name)
        tone = "black (for light backgrounds)" if "black" in name else "white (for dark backgrounds)"
        out.append((PREFIX + name, f"logo, square 1:1, {tone}: framed M monogram over a spaced wordmark (Monograph)"))

    # Meridian Air: white swoosh-globe + wordmark (3:1), and a cobalt version.
    for name, fg in (("logo_meridian_white.png", (255, 255, 255, 255)), ("logo_meridian_cobalt.png", _hex("#0038CC") + (255,))):
        img = Image.new("RGBA", (900, 300), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        d.ellipse([20, 40, 240, 260], outline=fg, width=18)
        d.arc([-40, 90, 300, 330], 200, 330, fill=_hex("#FF5A4E") + (255,), width=18)
        d.line([(130, 40), (130, 260)], fill=fg, width=10)
        d.text((280, 150), "Meridian", font=_font("SolsticeSans-Bold.ttf", 130), fill=fg, anchor="lm")
        save_rgba(img, PREFIX + name)
        tone = "white (for dark backgrounds)" if "white" in name else "cobalt (for light backgrounds)"
        out.append((PREFIX + name, f"logo, wide 3:1, {tone}: globe mark with a coral arc + wordmark (Meridian Air)"))
    return out


def draw_all() -> list[tuple[str, str]]:
    SYNTH_ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    made = scenes(rng) + products(rng) + logos(rng)
    for stale in SYNTH_ASSETS_DIR.glob(f"{PREFIX}*"):
        if stale.name not in {n for n, _ in made}:
            stale.unlink()
    return made


def main() -> None:
    made = draw_all()
    print(f"wrote {len(made)} {PREFIX} images to {SYNTH_ASSETS_DIR}")


if __name__ == "__main__":
    main()
