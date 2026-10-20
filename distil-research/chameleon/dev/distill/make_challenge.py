"""Write the challenge ads: 40 new designs nothing in the loop has been tuned
or selected on, to measure which approach GENERALISES.

    cd distil-research/chameleon && python -m dev.distill.challenge_images   # the chl_ images (once)
    cd distil-research/chameleon && python -m dev.distill.make_challenge

Writes training-data/corpus/challenge/<task_id>.json (split "challenge",
origin "challenge", brand "challenge") and the review sheet
training-data/corpus/review.nosync/sheet_challenge.png. The folder is separate
from corpus/designs/, so `load_corpus()` never returns these ads unless asked.

What makes them different from the corpus: five new invented brands with their
own palettes (Lumen & Fig skincare, light; Obsidian Audio headphones, dark;
Fizzbang soda and arcade, saturated; Monograph gallery and watches,
monochrome; Meridian Air travel, saturated), only the new `chl_` images
(challenge_images.py), structures the corpus doesn't have (two products, quote-
led, price-led, event with a date/time block, list-led, logo-only centrepiece,
a 1-element ad, 20+ element ads, very long legal text, rotated stickers,
full-bleed photo with a text panel), and mostly source sizes the corpus never
used (600x900, 1200x300, 1500x500, 1456x180, 2400x600, 300x1050, ...).

Every ad is explicit data, built with make_offbrand's helpers (text box heights
measured by the engine's TextMeasurer), and must pass `quality_issues`
(make_corpus.check strict + pixel_check); a failing ad is reported and not
written. EVALUATION ONLY: never used for tuning or selection.
"""

import json
import re

from dev.distill import make_corpus as mc
from dev.distill.assets import merged_assets_dir
from dev.distill.make_offbrand import Col, bg, c, img, label_pill, t
from dev.distill.paths import CHALLENGE_DIR

REVIEW_SHEET = mc.REVIEW_DIR / "sheet_challenge.png"

LF = dict(paper="#F7F1EA", blush="#EBC9B9", terra="#B4532A", olive="#5A6B3A", ink="#2E2A26", white="#FFFFFF")
OA = dict(black="#0C0D10", graphite="#1C1F26", slate="#2A2E36", cyan="#34E1F2", silver="#C9CED6", white="#F2F4F7")
FB = dict(pink="#FF2E88", blue="#1F3BFF", yellow="#FFE600", deep="#14002E", white="#FFFFFF", mint="#00D1B2",
          grape="#9B5CFF")
MG = dict(black="#111111", white="#FFFFFF", g1="#F2F2F2", g2="#D9D9D9", g3="#8A8A8A", g4="#3A3A3A")
MA = dict(cobalt="#0038CC", navy="#0B1F4D", coral="#FF5A4E", sky="#DDEFFF", white="#FFFFFF", sand="#FFF4E6")

BRANDS = {"lumen": "light", "obsidian": "dark", "fizzbang": "saturated", "monograph": "monochrome",
          "meridian": "saturated"}

# Overlaps a designer made on purpose (none yet: stickers sit in free space).
INTENDED_OVERLAPS: dict[str, set[tuple[str, str]]] = {}


def ad(task_id, brand, w, h, elements, notes, features) -> dict:
    rec = mc.ad(task_id, "challenge", w, h, elements, notes, features)
    rec.update(origin="challenge", brand="challenge", challenge_brand=brand, palette=BRANDS[brand])
    return rec


def sticker(prefix, cx, cy, d, label, size, *, fill, color, rotation, kind="ellipse", w=None) -> list[dict]:
    """A rotated round (or rectangular) sticker centred on (cx, cy) with a
    centred label, grouped."""
    w = w or d
    box = mc.shape(f"{prefix}_bg", round(cx - w / 2), round(cy - d / 2), w, d, fill, kind=kind, z=6,
                   group=prefix, rotation=rotation, radius=None if kind == "ellipse" else 12)
    lab = mc.text(f"{prefix}_label", label, round(cx - w / 2 + w * 0.1), 0, round(w * 0.8), 10, size, bold=True,
                  color=color, align="center", z=7, group=prefix, rotation=rotation)
    from dev.distill.augment import _needed_height
    lab["height"] = _needed_height(lab, size)
    lab["y"] = round(cy - lab["height"] / 2 + size * 0.08)
    return [box, lab]


def legal_at_bottom(id, content, x, w, size, bottom, color, align="left") -> dict:
    el = t(id, content, x, 0, w, size, color=color, align=align)
    el["y"] = bottom - el["height"]
    return el


# ---------------------------------------------------------------------------
# Lumen & Fig (botanical skincare, LIGHT)
# ---------------------------------------------------------------------------

def lumen() -> list[dict]:
    ads, P = [], LF
    A = ads.append
    LOGO = "chl_logo_lumen_fig.png"

    # L1 600x900: full-bleed botanical photo, product on the photo, paper panel with logo + copy
    col = Col(64, 0, 472)
    col.image("logo", LOGO, 22, h=40)
    col.text("kicker", "NEW NIGHT SERUM", 14, 8, bold=True, color=c(P["terra"]), letter_spacing=3)
    col.text("headline", "Wake up to softer skin", 40, 12, serif=True, bold=True, color=c(P["ink"]))
    col.text("subhead", "Fig-seed oil and bakuchiol, cold-pressed in small batches.", 17, 22,
             color=c(P["ink"], 0.85))
    col.pill("cta", "Shop the Serum", 16, fill=c(P["terra"]), color=c(P["paper"]))
    col.center_in(520, 872)
    A(ad("chl01_lf_600x900_photo_panel", "lumen", 600, 900, [
        bg(600, 900, "chl_photo_botanical.png"),
        img("product", "chl_prod_serum.png", 250, 70, h=400),
        mc.shape("panel", 32, 500, 536, 372, c(P["paper"]), radius=6, z=1),
        *col.els,
    ], "Full-bleed leafy photo; serum on the photo; logo and copy on an opaque paper panel.",
        ["portrait-2:3", "light", "full-bleed-panel", "photo-bg", "panel", "groups"]))

    # L2 1200x300: two products side by side, price in copy
    A(ad("chl02_lf_1200x300_duo", "lumen", 1200, 300, [
        bg(1200, 300, c(P["paper"])),
        mc.ellipse("deco_disc", 790, -60, 420, 420, c(P["blush"]), 1.0),
        img("logo", LOGO, 48, 36, h=44),
        t("headline", "The evening duo", 50, 112, 520, 46, serif=True, bold=True, color=c(P["ink"])),
        t("subhead", "Night Serum and Night Cream, together for $64.", 50, 178, 520, 19, color=c(P["ink"], 0.85)),
        *label_pill("cta", 600, 122, "Shop the Duo", 18, fill=c(P["terra"]), color=c(P["paper"]), h=56, pad=28),
        img("product1", "chl_prod_serum.png", 840, 26, h=246),
        img("product2", "chl_prod_cream_jar.png", 938, 112, w=240),
    ], "Wide 4:1 banner selling two products together (tall serum + squat jar), price in the subhead.",
        ["banner-4:1", "light", "two-products", "solid-bg", "deco-overlap", "groups"]))

    # L3 1000x1000: list-led evening ritual
    els = [bg(1000, 1000, c(P["paper"])),
           mc.ellipse("deco_sun", 560, 600, 560, 560, c(P["blush"]), 0.7),
           img("logo", LOGO, 80, 64, h=50)]
    head = t("headline", "Your four-step evening ritual", 80, 150, 760, 58, serif=True, bold=True, color=c(P["ink"]))
    els.append(head)
    steps = ["Cleanse with the Fig Milk wash", "Tone with rose and white tea", "Press in two drops of Night Serum",
             "Seal it in with Night Cream"]
    y = head["y"] + head["height"] + 50
    for i, s in enumerate(steps, 1):
        g = f"step{i}"
        els += [mc.shape(f"step{i}_dot", 80, y, 56, 56, c(P["olive"]), kind="ellipse", z=2, group=g),
                mc.text(f"step{i}_num", str(i), 94, y + 10, 28, 36, 26, bold=True, color=c(P["paper"]), align="center",
                        z=3, group=g),
                t(f"step{i}_text", s, 160, y + 10, 560, 27, color=c(P["ink"]), group=g)]
        y += 92
    els += mc.pill("cta", 80, 830, 330, 80, "Build Your Ritual", 24, fill=c(P["terra"]), color=c(P["paper"]))
    els.append(img("product", "chl_prod_cream_jar.png", 600, 740, w=340))
    A(ad("chl03_lf_1000x1000_ritual_list", "lumen", 1000, 1000, els,
         "List-led square: four numbered steps (dot + number + line, grouped), jar bottom right.",
         ["square", "light", "list-led", "text-groups", "solid-bg", "deco-overlap"]))

    # L4 250x250: tiny square, rotated NEW sticker
    col = Col(16, 150, 218)
    col.text("headline", "Night Cream", 26, 2, serif=True, bold=True, color=c(P["ink"]))
    col.text("subhead", "Whipped fig and shea, $38", 12, 8, color=c(P["ink"], 0.85))
    col.pill("cta", "Shop Now", 12, h=30, fill=c(P["terra"]), color=c(P["paper"]))
    col.center_in(146, 240)
    A(ad("chl04_lf_250x250_sticker", "lumen", 250, 250, [
        bg(250, 250, c(P["blush"])),
        img("product", "chl_prod_cream_jar.png", 96, 22, w=138),
        *sticker("sticker", 50, 54, 64, "NEW", 15, fill=c(P["terra"]), color=c(P["paper"]), rotation=-12),
        *col.els,
    ], "Tiny 250x250: jar, a rotated round NEW sticker in the corner, headline + price + button.",
        ["small-square", "light", "rotated-sticker", "rotated", "solid-bg", "groups"]))

    # L5 1500x500: quote-led testimonial
    col = Col(210, 0, 840)
    col.text("quote", "My skin finally looks like it slept eight hours.", 52, 22, serif=True, italic=True,
             color=c(P["ink"]))
    col.text("attribution", "MAYA R., VERIFIED BUYER", 18, 30, bold=True, color=c(P["terra"]), letter_spacing=3)
    col.pill("cta", "Read 2,400 Reviews", 18, fill=c(P["ink"]), color=c(P["paper"]))
    col.center_in(30, 400)
    qm = t("quote_mark", "“", 80, 0, 110, 200, serif=True, bold=True, color=c(P["terra"]))
    qm["y"] = col.els[0]["y"] - 40
    A(ad("chl05_lf_1500x500_quote", "lumen", 1500, 500, [
        bg(1500, 500, c(P["paper"])),
        mc.ellipse("deco_disc", 1100, 40, 420, 420, c(P["blush"]), 0.8),
        qm, *col.els,
        img("product", "chl_prod_serum.png", 1250, 50, h=400),
        img("logo", LOGO, 210, 420, h=40),
    ], "Quote-led 3:1: oversized quote mark, italic serif testimonial, attribution, button, product right.",
        ["banner-3:1", "light", "quote-led", "italic", "solid-bg", "deco-overlap", "groups"]))

    # L6 500x1000: price-led set
    col = Col(40, 0, 420, "center")
    col.text("kicker", "THE NIGHT SET", 18, 4, bold=True, color=c(P["terra"]), letter_spacing=4)
    col.text("price", "$48", 140, 0, serif=True, bold=True, color=c(P["ink"]))
    col.text("was", "Worth $72 when bought separately", 18, 18, color=c(P["ink"], 0.75))
    col.text("subhead", "Night Serum, Night Cream and the Fig Milk wash.", 20, 30, color=c(P["ink"]))
    col.pill("cta", "Get the Set", 22, fill=c(P["terra"]), color=c(P["paper"]))
    col.center_in(470, 960)
    A(ad("chl06_lf_500x1000_price", "lumen", 500, 1000, [
        bg(500, 1000, c(P["blush"])),
        mc.ellipse("deco_disc", 50, 120, 400, 400, c(P["paper"]), 0.9),
        img("logo", LOGO, 170, 44, h=40),
        img("product1", "chl_prod_serum.png", 120, 140, h=300),
        img("product2", "chl_prod_cream_jar.png", 236, 306, w=200),
        *col.els,
    ], "Price-led 1:2: a giant serif price, 'worth' line, two products on a pale disc.",
        ["tall-1:2", "light", "price-led", "two-products", "solid-bg", "deco-overlap", "groups"]))

    # L7 1350x1080: very long legal
    col = Col(620, 70, 660)
    col.image("logo", LOGO, 40, h=56)
    col.text("kicker", "CLINICALLY TESTED", 18, 10, bold=True, color=c(P["terra"]), letter_spacing=4)
    col.text("headline", "93% saw calmer skin in 28 days", 62, 20, serif=True, bold=True, color=c(P["ink"]))
    col.text("subhead", "Night Serum, tested on sensitive skin by an independent lab.", 24, 30, color=c(P["ink"], 0.85))
    col.pill("cta", "See the Study", 22, fill=c(P["terra"]), color=c(P["paper"]))
    legal = legal_at_bottom(
        "legal", "Results from an independent 28-day consumer study of 112 women aged 25 to 60 with self-declared "
        "sensitive skin, using Night Serum once a day; individual results vary. Percentages are of participants "
        "who agreed in a self-assessment questionnaire. Patch test before first use: apply a little behind the ear "
        "and wait 48 hours. Stop using it if irritation occurs. For external use only; avoid the eye area. Keep out "
        "of reach of children. Store below 25 C away from direct sunlight and use within six months of opening. "
        "Free delivery on orders over $40 to mainland addresses only. Lumen & Fig Ltd., 2026.",
        620, 660, 15, 1030, c(P["ink"], 0.7))
    col.center_in(40, legal["y"] - 40)
    A(ad("chl07_lf_1350x1080_long_legal", "lumen", 1350, 1080, [
        bg(1350, 1080, c(P["paper"])),
        mc.image("photo", "chl_photo_botanical.png", 0, 0, 540, 1080, fit="cover", z=1),
        img("product", "chl_prod_cream_jar.png", 70, 700, w=400),
        *col.els, legal,
    ], "Landscape 5:4: half-width photo with the jar on it, claim headline, and an 8-line legal block.",
        ["landscape-5:4", "light", "long-legal", "partial-photo", "solid-bg", "groups"]))

    # L8 640x640: logo-only centrepiece
    A(ad("chl08_lf_640x640_logo_centre", "lumen", 640, 640, [
        bg(640, 640, c(P["paper"])),
        mc.ellipse("deco_disc", 50, 50, 540, 540, c(P["blush"]), 0.8),
        img("logo", LOGO, 130, 256, w=380),
        t("tagline", "Skincare, slowly.", 120, 390, 400, 26, serif=True, italic=True, color=c(P["ink"]), align="center"),
    ], "Logo-only centrepiece: the wordmark large on a soft disc, one italic tagline (4 elements).",
        ["square-small", "light", "logo-centrepiece", "solid-bg", "deco-overlap", "no-cta"]))
    return ads


# ---------------------------------------------------------------------------
# Obsidian Audio (headphones, DARK)
# ---------------------------------------------------------------------------

def obsidian() -> list[dict]:
    ads, P = [], OA
    A = ads.append
    LOGO = "chl_logo_obsidian_light.png"

    # A1 970x250 billboard
    A(ad("chl09_oa_970x250_billboard", "obsidian", 970, 250, [
        bg(970, 250, c(P["graphite"])),
        mc.ellipse("deco_glow", 10, 10, 230, 230, c(P["cyan"]), 0.12),
        img("product", "chl_prod_headphones.png", 30, 22, h=206),
        img("logo", LOGO, 270, 36, h=36),
        t("headline", "Silence, engineered.", 270, 90, 420, 40, bold=True, color=c(P["white"])),
        t("subhead", "Adaptive noise cancelling. 40-hour battery.", 270, 150, 420, 18, color=c(P["silver"])),
        t("price", "$349", 730, 54, 200, 52, bold=True, color=c(P["cyan"]), align="center"),
        *mc.pill("cta", 740, 140, 180, 54, "Pre-order", 20, fill=c(P["cyan"]), color=c(P["black"])),
    ], "970x250 billboard: product left, logo + headline + spec line, price and button right.",
        ["billboard", "dark", "price", "solid-bg", "deco-overlap", "groups"]))

    # A2 1600x900: full-bleed night skyline + panel
    col = Col(140, 0, 580)
    col.image("logo", LOGO, 36, h=48)
    col.text("kicker", "OBSIDIAN ONE", 20, 10, bold=True, color=c(P["cyan"]), letter_spacing=5)
    col.text("headline", "Hear the city. Or don't.", 68, 20, bold=True, color=c(P["white"]))
    col.text("subhead", "One switch between full transparency and total quiet.", 26, 34, color=c(P["silver"]))
    col.pill("cta", "Try a Pair", 26, fill=c(P["cyan"]), color=c(P["black"]))
    col.center_in(180, 720)
    A(ad("chl10_oa_1600x900_city_panel", "obsidian", 1600, 900, [
        bg(1600, 900, "chl_photo_city_night.png"),
        mc.shape("panel", 80, 140, 700, 620, c(P["graphite"]), radius=24, z=1),
        img("product", "chl_prod_headphones.png", 960, 150, h=620),
        *col.els,
    ], "Full-bleed night skyline photo; copy on an opaque graphite panel; headphones on the photo.",
        ["landscape-16:9", "dark", "full-bleed-panel", "photo-bg", "panel", "groups"]))

    # A3 160x600 skyscraper
    col = Col(12, 0, 136, "center")
    col.text("headline", "Pocket-size silence", 22, 10, bold=True, color=c(P["white"]))
    col.text("subhead", "Obsidian Buds, $179", 14, color=c(P["silver"]))
    col.center_in(250, 420)
    A(ad("chl11_oa_160x600_buds", "obsidian", 160, 600, [
        bg(160, 600, c(P["black"])),
        img("logo", LOGO, 15, 36, w=130),
        img("product", "chl_prod_earbuds_case.png", 10, 110, w=140),
        *col.els,
        *mc.pill("cta", 20, 480, 120, 44, "Shop", 16, fill=c(P["cyan"]), color=c(P["black"])),
    ], "160x600 skyscraper: logo, wide earbuds case, two lines, button.",
        ["skyscraper", "dark", "solid-bg", "groups"]))

    # A4 1080x566: two products with labels
    col = Col(60, 0, 440)
    col.text("kicker", "PICK YOUR SOUND", 16, 8, bold=True, color=c(P["cyan"]), letter_spacing=4)
    col.text("headline", "Over-ear or in-ear, the same silence", 44, 16, bold=True, color=c(P["white"]))
    col.text("subhead", "Both cancel 38 dB of noise. Both last all week.", 20, 26, color=c(P["silver"]))
    col.pill("cta", "Compare Models", 18, fill=c(P["cyan"]), color=c(P["black"]))
    col.center_in(120, 520)
    A(ad("chl12_oa_1080x566_two_products", "obsidian", 1080, 566, [
        bg(1080, 566, c(P["graphite"])),
        mc.ellipse("deco_left", 530, 330, 300, 120, c(P["slate"]), 1.0),
        mc.ellipse("deco_right", 815, 380, 250, 90, c(P["slate"]), 1.0),
        img("logo", LOGO, 60, 44, h=36),
        img("product1", "chl_prod_headphones.png", 545, 110, h=300),
        img("product2", "chl_prod_earbuds_case.png", 830, 277, w=220),
        t("label1", "Obsidian One, $349", 540, 440, 270, 18, bold=True, color=c(P["silver"]), align="center"),
        t("label2", "Obsidian Buds, $179", 822, 440, 240, 18, bold=True, color=c(P["silver"]), align="center"),
        *col.els,
    ], "Two products on pedestals, each with its own name + price label; copy left.",
        ["landscape-1.91:1", "dark", "two-products", "solid-bg", "deco-overlap", "groups"]))

    # A5 900x1600: spec sheet, 20+ elements
    els = [bg(900, 1600, c(P["black"])),
           mc.ellipse("deco_glow", 150, 150, 600, 600, c(P["cyan"]), 0.1),
           img("logo", LOGO, 300, 70, w=300),
           img("product", "chl_prod_headphones.png", 205, 190, h=540)]
    col = Col(80, 790, 740, "center")
    col.text("headline", "Obsidian One", 76, 8, bold=True, color=c(P["white"]))
    col.text("subhead", "Every number, measured in our own lab.", 24, color=c(P["silver"]))
    els += col.els
    els.append(mc.shape("rule", 80, 990, 740, 2, c(P["slate"]), z=1))
    specs = [("40 h", "battery life"), ("-38 dB", "noise reduction"), ("250 g", "weight"),
             ("10 min", "charge for 5 hours"), ("6 mics", "for clear calls"), ("BT 5.4", "multipoint")]
    for i, (val, lab) in enumerate(specs):
        x, y = 80 + (i % 2) * 380, 1030 + (i // 2) * 120
        g = f"spec{i + 1}"
        els += [t(f"{g}_value", val, x, y, 360, 44, bold=True, color=c(P["cyan"]), group=g),
                t(f"{g}_label", lab, x, y + 58, 360, 18, color=c(P["silver"]), group=g)]
    els += mc.pill("cta", 280, 1400, 340, 80, "Pre-order, $349", 26, fill=c(P["cyan"]), color=c(P["black"]))
    els.append(legal_at_bottom("legal", "Battery life measured with noise cancelling on at 50% volume.", 80, 740, 14,
                               1560, c(P["silver"], 0.8), align="center"))
    A(ad("chl13_oa_900x1600_spec_sheet", "obsidian", 900, 1600, els,
         "Story-shaped spec sheet: product, headline, a rule and a 2x3 grid of grouped spec values (22 elements).",
         ["story-9:16", "dark", "20-plus-elements", "text-groups", "solid-bg", "deco-overlap", "legal"]))

    # A6 600x600: quote-led review
    col = Col(60, 0, 480)
    col.text("quote_mark", "“", 150, 0, serif=True, bold=True, color=c(P["cyan"]))
    col.text("quote", "The quietest room you will ever wear.", 44, 26, bold=True, italic=True, color=c(P["white"]))
    col.text("attribution", "SOUND & STEREO REVIEW, 2026", 14, 8, bold=True, color=c(P["cyan"]), letter_spacing=2)
    col.text("rating", "Rated 5 out of 5", 16, color=c(P["silver"]))
    col.center_in(20, 500)
    A(ad("chl14_oa_600x600_quote", "obsidian", 600, 600, [
        bg(600, 600, c(P["black"])), *col.els, img("logo", LOGO, 60, 528, h=36),
    ], "Quote-led square: big cyan quote mark, bold italic review line, source, rating, logo (6 elements).",
        ["square-small", "dark", "quote-led", "italic", "solid-bg", "no-image-product", "no-groups"]))

    # A7 468x60 strip
    A(ad("chl15_oa_468x60_strip", "obsidian", 468, 60, [
        bg(468, 60, c(P["graphite"])),
        img("logo", LOGO, 12, 16, h=28),
        t("headline", "Silence, engineered", 172, 17, 184, 17, bold=True, color=c(P["white"])),
        *mc.pill("cta", 370, 12, 86, 36, "Shop", 14, fill=c(P["cyan"]), color=c(P["black"])),
    ], "468x60 strip: logo, one line, button (5 elements).", ["strip-8:1", "dark", "solid-bg", "groups"]))

    # A8 1200x400: event with a date/time block
    date = Col(60, 0, 200, "center")
    date.text("date_month", "NOV", 30, 0, bold=True, color=c(P["black"]), letter_spacing=6, group="date")
    date.text("date_day", "14", 110, 0, bold=True, color=c(P["black"]), group="date")
    date.text("date_time", "FRI, 7:30 PM", 18, bold=True, color=c(P["black"]), group="date")
    date.center_in(60, 340)
    col = Col(310, 0, 520)
    col.text("kicker", "LISTENING SESSION", 16, 8, bold=True, color=c(P["cyan"]), letter_spacing=4)
    col.text("headline", "Obsidian One, heard live", 48, 12, bold=True, color=c(P["white"]))
    col.text("subhead", "Hi-fi demo room, 12 Arden Street. Free entry, RSVP needed.", 20, 22, color=c(P["silver"]))
    col.pill("cta", "RSVP Free", 18, fill=c(P["cyan"]), color=c(P["black"]))
    col.center_in(40, 360)
    A(ad("chl16_oa_1200x400_event", "obsidian", 1200, 400, [
        bg(1200, 400, c(P["black"])),
        mc.shape("date_card", 60, 60, 200, 280, c(P["cyan"]), radius=18, z=1, group="date"),
        *date.els, *col.els,
        mc.ellipse("deco_glow", 860, 40, 320, 320, c(P["slate"]), 1.0),
        img("product", "chl_prod_headphones.png", 890, 50, h=300),
    ], "3:1 event ad: a grouped date/time block (month, day, weekday + time on a cyan card), copy, product.",
        ["banner-3:1", "dark", "event-date", "text-groups", "solid-bg", "deco-overlap", "no-logo"]))
    return ads


# ---------------------------------------------------------------------------
# Fizzbang (soda + arcade, SATURATED)
# ---------------------------------------------------------------------------

def fizzbang() -> list[dict]:
    ads, P = [], FB
    A = ads.append
    LOGO = "chl_logo_fizzbang.png"

    # F1 800x800: product with two rotated stickers
    A(ad("chl17_fb_800x800_stickers", "fizzbang", 800, 800, [
        bg(800, 800, c(P["blue"])),
        mc.ellipse("deco_sun", 160, 190, 480, 480, c(P["yellow"]), 1.0),
        t("headline", "Cherry Bang is here", 60, 40, 680, 62, bold=True, italic=True, color=c(P["white"]),
          align="center"),
        img("product", "chl_prod_soda_can.png", 270, 170, h=480),
        *sticker("sticker1", 648, 262, 160, "NEW FLAVOUR", 22, fill=c(P["pink"]), color=c(P["white"]),
                 rotation=12),
        *sticker("sticker2", 140, 560, 76, "ZERO SUGAR", 22, fill=c(P["white"]), color=c(P["deep"]),
                 rotation=-10, kind="rectangle", w=190),
        img("logo", LOGO, 40, 690, w=160),
        *mc.pill("cta", 270, 686, 260, 80, "Grab a Can", 26, fill=c(P["yellow"]), color=c(P["deep"])),
    ], "Square with two rotated stickers (round + rectangular) around a can on a yellow disc.",
        ["square", "saturated", "rotated-sticker", "rotated", "solid-bg", "deco-overlap", "groups"]))

    # F2 300x1050: price-led tower
    col = Col(20, 0, 260, "center")
    col.text("kicker", "2 FOR", 34, 0, bold=True, color=c(P["white"]), letter_spacing=4)
    col.text("price", "$3", 150, 0, bold=True, italic=True, color=c(P["deep"]))
    col.text("subhead", "Every flavour, all week long.", 20, 26, color=c(P["white"]))
    col.pill("cta", "Find a Store", 18, fill=c(P["deep"]), color=c(P["yellow"]))
    col.center_in(640, 1010)
    A(ad("chl18_fb_300x1050_price", "fizzbang", 300, 1050, [
        bg(300, 1050, c(P["pink"])),
        img("logo", LOGO, 40, 36, w=220),
        img("product", "chl_prod_soda_can.png", 37, 180, h=420) | {"x": 33},
        *col.els,
    ], "Very tall 2:7 tower: logo, can, then a huge price block.",
        ["very-tall-2:7", "saturated", "price-led", "solid-bg", "groups"]))

    # F3 2000x1000: arcade night event with a date/time block
    date = Col(120, 0, 360, "center")
    date.text("date_weekday", "SAT", 40, 0, bold=True, color=c(P["deep"]), letter_spacing=8, group="date")
    date.text("date_day", "22", 200, 0, bold=True, color=c(P["deep"]), group="date")
    date.text("date_month", "NOV", 48, 10, bold=True, color=c(P["deep"]), letter_spacing=8, group="date")
    date.text("date_time", "8 PM - 1 AM", 30, bold=True, color=c(P["deep"]), group="date")
    date.center_in(120, 720)
    col = Col(560, 0, 760)
    col.text("kicker", "FIZZBANG ARCADE NIGHT", 28, 14, bold=True, color=c(P["pink"]), letter_spacing=5)
    col.text("headline", "Free play, free cans, one very loud night", 76, 24, bold=True, italic=True,
             color=c(P["white"]))
    col.text("subhead", "Pier 9 Arcade: 40 retro cabinets, DJ from 10 pm.", 30, 40, color=c(P["white"], 0.85))
    col.pill("cta", "Get a Free Ticket", 30, fill=c(P["pink"]), color=c(P["white"]))
    col.center_in(120, 760)
    A(ad("chl19_fb_2000x1000_arcade_event", "fizzbang", 2000, 1000, [
        bg(2000, 1000, c(P["deep"])),
        mc.image("art", "chl_pattern_bauhaus.png", 0, 830, 2000, 170, fit="cover", z=1),
        mc.shape("date_card", 120, 120, 360, 600, c(P["yellow"]), radius=28, z=1, group="date"),
        *date.els, *col.els,
        img("logo", LOGO, 1620, 70, w=300),
        img("product", "chl_prod_controller.png", 1400, 300, w=540),
    ], "2:1 event poster: a big grouped date/time card, headline, controller, Bauhaus band along the bottom.",
        ["landscape-2:1", "saturated", "event-date", "text-groups", "solid-bg", "partial-pattern", "groups"]))

    # F4 1456x180: two products in a thin strip
    A(ad("chl20_fb_1456x180_two_products", "fizzbang", 1456, 180, [
        bg(1456, 180, c(P["yellow"])),
        img("product1", "chl_prod_soda_can.png", 36, 15, h=150),
        img("product2", "chl_prod_controller.png", 130, 34, w=200),
        t("headline", "Snack. Play. Repeat.", 372, 30, 620, 44, bold=True, italic=True, color=c(P["deep"])),
        t("subhead", "A free Fizzbang with every arcade card top-up.", 372, 96, 620, 20, color=c(P["deep"])),
        img("logo", LOGO, 1010, 50, w=160),
        *mc.pill("cta", 1196, 54, 230, 72, "Top Up Now", 22, fill=c(P["blue"]), color=c(P["white"])),
    ], "Very thin 8:1 strip carrying two products, headline, logo and button.",
        ["strip-8:1", "saturated", "two-products", "solid-bg", "groups"]))

    # F5 600x900: list-led flavours
    els = [bg(600, 900, c(P["blue"])), img("logo", LOGO, 180, 30, w=240),
           t("headline", "Five flavours. Zero sugar.", 40, 170, 520, 42, bold=True, italic=True, color=c(P["white"]),
             align="center")]
    flavours = [("Cherry Bang", P["pink"]), ("Lime Zap", P["mint"]), ("Mango Boom", P["yellow"]),
                ("Grape Pop", P["grape"]), ("Cola Crash", P["white"])]
    for i, (name, col_) in enumerate(flavours, 1):
        y = 300 + 66 * (i - 1)
        g = f"flavour{i}"
        els += [mc.shape(f"flavour{i}_dot", 60, y + 6, 32, 32, c(col_), kind="ellipse", z=2, group=g),
                t(f"flavour{i}_name", name, 110, y, 240, 28, bold=True, color=c(P["white"]), group=g)]
    els.append(img("product", "chl_prod_soda_can.png", 370, 290, h=380))
    els += mc.pill("cta", 160, 730, 280, 80, "Pick Your Flavour", 24, fill=c(P["yellow"]), color=c(P["deep"]))
    A(ad("chl21_fb_600x900_flavour_list", "fizzbang", 600, 900, els,
         "List-led 2:3: five flavours as colour dot + name rows (grouped), can on the right.",
         ["portrait-2:3", "saturated", "list-led", "text-groups", "solid-bg"]))

    # F6 240x400 small
    col = Col(14, 0, 212, "center")
    col.text("headline", "Cherry Bang", 30, 4, bold=True, italic=True, color=c(P["white"]))
    col.text("subhead", "Zero sugar, all bang.", 15, 12, color=c(P["white"]))
    col.pill("cta", "Try It", 15, h=40, fill=c(P["deep"]), color=c(P["yellow"]))
    col.center_in(236, 392)
    A(ad("chl22_fb_240x400_small", "fizzbang", 240, 400, [
        bg(240, 400, c(P["pink"])),
        mc.ellipse("deco_disc", 30, 20, 180, 180, c(P["yellow"]), 1.0),
        img("product", "chl_prod_soda_can.png", 62, 14, h=214),
        *col.els,
    ], "240x400: can on a yellow disc, headline, line, button.", ["tall-3:5", "saturated", "solid-bg", "groups",
                                                                   "deco-overlap", "no-logo"]))

    # F7 1500x500: full-bleed Bauhaus art with a white panel
    col = Col(500, 0, 500, "center")
    col.image("logo", LOGO, 16, w=220)
    col.text("headline", "Pop art, now in a can", 46, 20, bold=True, italic=True, color=c(P["deep"]))
    col.pill("cta", "Shop the Art Cans", 22, fill=c(P["pink"]), color=c(P["white"]))
    col.center_in(80, 420)
    A(ad("chl23_fb_1500x500_art_panel", "fizzbang", 1500, 500, [
        bg(1500, 500, "chl_pattern_bauhaus.png"),
        mc.shape("panel", 470, 60, 560, 380, c(P["white"]), radius=0, z=1),
        *col.els,
    ], "Full-bleed BUSY Bauhaus pattern; logo and copy on a square white panel in the centre.",
        ["banner-3:1", "saturated", "full-bleed-panel", "busy-bg", "panel", "groups", "no-product"]))

    # F8 400x400: logo-only centrepiece on halftone
    A(ad("chl24_fb_400x400_logo_centre", "fizzbang", 400, 400, [
        bg(400, 400, "chl_pattern_halftone.png"),
        img("logo", LOGO, 30, 100, w=340),
        *label_pill("tag", 0, 300, "SINCE 2026", 16, fill=c(P["deep"]), color=c(P["white"]), letter_spacing=3),
    ], "Logo-only centrepiece on a BUSY halftone: the logo's own lozenge carries it, one small tag pill.",
        ["square-small", "saturated", "logo-centrepiece", "busy-bg", "groups", "no-cta"]))
    tag = ads[-1]["source"]["elements"][2:]
    dx = round((400 - tag[0]["width"]) / 2) - tag[0]["x"]
    for e in tag:
        e["x"] += dx
    return ads


# ---------------------------------------------------------------------------
# Monograph (gallery + watches, MONOCHROME)
# ---------------------------------------------------------------------------

def monograph() -> list[dict]:
    ads, P = [], MG
    A = ads.append
    LOGO_B, LOGO_W = "chl_logo_monograph_black.png", "chl_logo_monograph_white.png"

    # M1 1200x1500: ONE element
    head = t("headline", "Form follows silence.", 100, 0, 1000, 150, serif=True, bold=True, color=c(P["black"]),
             align="center")
    head["y"] = round((1500 - head["height"]) / 2)
    A(ad("chl25_mg_1200x1500_one_element", "monograph", 1200, 1500, [head],
         "One element: a single serif headline on the bare (white) canvas. No background, no image, no button.",
         ["portrait-4:5", "monochrome", "one-element", "text-only", "no-bg", "no-logo", "no-cta"]))

    # M2 600x900: exhibition poster with a date/time block
    col = Col(36, 456, 528)
    col.text("kicker", "EXHIBITION", 14, 6, bold=True, color=c(P["g4"]), letter_spacing=5)
    col.text("headline", "Concrete Poetry", 54, 18, serif=True, bold=True, color=c(P["black"]))
    els = [bg(600, 900, c(P["white"])),
           mc.image("photo", "chl_photo_concrete_mono.png", 0, 0, 600, 420, fit="cover", z=1), *col.els]
    y = col.y
    els += [mc.shape("date_card", 36, y, 528, 116, c(P["black"]), z=1, group="date"),
            t("date_range", "06.12 - 28.02", 60, y + 18, 480, 34, bold=True, color=c(P["white"]), group="date")]
    els.append(t("date_hours", "Tuesday to Sunday, 10:00 - 18:00", 60, els[-1]["y"] + els[-1]["height"] + 4, 480, 18,
                 color=c(P["g2"]), group="date"))
    els += [t("address", "Monograph Gallery, 4 Kiln Lane", 36, y + 140, 400, 16, color=c(P["g4"])),
            img("logo", LOGO_B, 474, 774, w=90)]
    A(ad("chl26_mg_600x900_exhibition", "monograph", 600, 900, els,
         "Exhibition poster: photo top, serif title, a black date/time card (grouped), address, small logo.",
         ["portrait-2:3", "monochrome", "event-date", "partial-photo", "text-groups", "solid-bg"]))

    # M3 1200x300: quote-led
    col = Col(150, 0, 780)
    col.text("quote", "A building should be quiet enough to hear the light move.", 38, 18, serif=True, italic=True,
             color=c(P["white"]))
    col.text("attribution", "AIKO TANABE, ARCHITECT", 14, bold=True, color=c(P["g3"]), letter_spacing=4)
    col.center_in(0, 300)
    qm = t("quote_mark", "“", 40, 0, 90, 150, serif=True, bold=True, color=c(P["white"]))
    qm["y"] = col.els[0]["y"] - 30
    A(ad("chl27_mg_1200x300_quote", "monograph", 1200, 300, [
        bg(1200, 300, c(P["black"])), qm, *col.els, img("logo", LOGO_W, 1010, 70, w=160),
    ], "Quote-led 4:1 on black: big quote mark, two-line italic serif quote, attribution, square logo right.",
        ["banner-4:1", "monochrome", "quote-led", "italic", "solid-bg", "no-groups", "no-cta"]))

    # M4 160x600: watch, price-led
    col = Col(10, 0, 140, "center")
    col.text("headline", "Field Watch No. 2", 18, 6, serif=True, bold=True, color=c(P["black"]))
    col.text("price", "$420", 40, 2, bold=True, color=c(P["black"]))
    col.text("note", "Free engraving", 12, 14, color=c(P["g4"]))
    col.pill("cta", "Shop", 14, h=38, fill=c(P["black"]), color=c(P["white"]))
    col.center_in(340, 586)
    A(ad("chl28_mg_160x600_watch_price", "monograph", 160, 600, [
        bg(160, 600, c(P["g1"])),
        img("logo", LOGO_B, 40, 20, w=80),
        img("product", "chl_prod_watch.png", 25, 120, w=110),
        *col.els,
    ], "160x600: square logo, watch, name, big price, button.", ["skyscraper", "monochrome", "price-led",
                                                                  "solid-bg", "groups"]))

    # M5 1500x500: full-bleed concrete photo + white panel
    col = Col(860, 0, 520)
    col.image("logo", LOGO_B, 14, w=90)
    col.text("kicker", "NOW OPEN", 16, 6, bold=True, color=c(P["g4"]), letter_spacing=5)
    col.text("headline", "The Kiln Lane building", 46, 12, serif=True, bold=True, color=c(P["black"]))
    col.text("subhead", "Six floors of architecture, photography and print. Free entry.", 19, 18,
             color=c(P["g4"]))
    col.pill("cta", "Plan a Visit", 18, fill=c(P["black"]), color=c(P["white"]))
    col.center_in(50, 450)
    A(ad("chl29_mg_1500x500_photo_panel", "monograph", 1500, 500, [
        bg(1500, 500, "chl_photo_concrete_mono.png"),
        mc.shape("panel", 820, 30, 600, 440, c(P["white"]), z=1),
        *col.els,
    ], "Full-bleed grainy concrete photo; logo and copy on a white panel on the right.",
        ["banner-3:1", "monochrome", "full-bleed-panel", "photo-bg", "panel", "groups", "no-product"]))

    # M6 1000x1500: very long legal
    col = Col(80, 790, 840, "center")
    col.text("headline", "Built to outlast the warranty", 58, 14, serif=True, bold=True, color=c(P["black"]))
    col.text("subhead", "Hand-assembled movement, sapphire crystal, water resistant to 100 m.", 22, 14,
             color=c(P["g4"]))
    col.text("price", "$420, engraving included", 26, 22, bold=True, color=c(P["black"]))
    col.pill("cta", "Book a Fitting", 22, fill=c(P["black"]), color=c(P["white"]))
    col.center_in(760, 1320)
    legal = legal_at_bottom(
        "legal", "Warranty: five years from the date of purchase against defects in materials and workmanship, for "
        "the original purchaser only, on presentation of the dated receipt. It does not cover the strap, the "
        "crystal, normal wear, accidents, water damage beyond the stated rating, or any repair made outside a "
        "Monograph service centre. Water resistance is tested at assembly and can decrease over time; have the "
        "seals checked every two years. Engraving is free on full-price watches, takes up to ten working days, and "
        "engraved watches cannot be returned. Prices include VAT. Fittings by appointment at 4 Kiln Lane, Tuesday "
        "to Sunday. Monograph Editions Ltd., 2026.", 80, 840, 14, 1460, c(P["g4"]), align="center")
    A(ad("chl30_mg_1000x1500_long_legal", "monograph", 1000, 1500, [
        bg(1000, 1500, c(P["white"])),
        img("logo", LOGO_B, 60, 50, w=110),
        img("product", "chl_prod_watch.png", 0, 110, h=620) | {"x": 306},
        *col.els, legal,
    ], "2:3 watch ad on white: product, serif headline, price, button and a 9-line warranty legal block.",
        ["portrait-2:3", "monochrome", "long-legal", "solid-bg", "groups"]))

    # M7 970x250: list-led talks
    els = [bg(970, 250, c(P["black"])), img("logo", LOGO_W, 30, 45, w=160),
           t("headline", "Three talks this week", 220, 60, 250, 30, serif=True, bold=True, color=c(P["white"]))]
    els.append(t("subhead", "All free, all at 19:00.", 220, els[-1]["y"] + els[-1]["height"] + 8, 250, 16,
                 color=c(P["g2"])))
    talks = [("TUE", "Light as a building material"), ("WED", "Drawing by hand in 2026"), ("THU", "The archive, opened")]
    for i, (day, title) in enumerate(talks, 1):
        y = 50 + 58 * (i - 1)
        g = f"talk{i}"
        els += [t(f"talk{i}_day", day, 510, y + 4, 60, 16, bold=True, color=c(P["g3"]), letter_spacing=2, group=g),
                t(f"talk{i}_title", title, 580, y, 360, 21, serif=True, color=c(P["white"]), group=g)]
        if i < 3:
            els.append(mc.shape(f"rule{i}", 510, y + 44, 430, 1, c(P["g4"]), z=1))
    A(ad("chl31_mg_970x250_talks_list", "monograph", 970, 250, els,
         "List-led billboard: logo, title, then three day + talk rows (grouped) with hairline rules.",
         ["billboard", "monochrome", "list-led", "text-groups", "solid-bg", "no-cta"]))

    # M8 1200x1600: programme, 20+ elements
    els = [bg(1200, 1600, c(P["g1"])),
           img("logo", LOGO_B, 80, 80, w=150),
           mc.image("photo", "chl_photo_concrete_mono.png", 760, 80, 360, 360, fit="cover", z=1)]
    col = Col(80, 270, 640)
    col.text("kicker", "WINTER PROGRAMME", 20, 10, bold=True, color=c(P["g4"]), letter_spacing=5)
    col.text("headline", "Six weeks of open studios", 70, 0, serif=True, bold=True, color=c(P["black"]))
    els += col.els
    weeks = [("02 DEC", "Printmaking with Ines Varga", "18:00"), ("09 DEC", "Model making, 1:50", "18:30"),
             ("16 DEC", "Photographing concrete", "18:00"), ("06 JAN", "Letterpress evening", "19:00"),
             ("13 JAN", "Drawing the city at night", "18:00"), ("20 JAN", "Bookbinding for beginners", "18:30")]
    y0 = 560
    for i, (d, title, time) in enumerate(weeks, 1):
        y = y0 + 110 * (i - 1)
        g = f"week{i}"
        els += [t(f"week{i}_date", d, 80, y + 6, 180, 24, bold=True, color=c(P["black"]), letter_spacing=2, group=g),
                t(f"week{i}_title", title, 280, y, 660, 30, serif=True, color=c(P["black"]), group=g),
                t(f"week{i}_time", time, 980, y + 6, 140, 24, color=c(P["g4"]), align="right", group=g)]
    els += mc.pill("cta", 80, 1260, 360, 88, "Book a Studio Place", 26, fill=c(P["black"]), color=c(P["white"]))
    els.append(legal_at_bottom("legal", "Places are free and limited to 12 per evening. Materials provided.", 80, 1040,
                               18, 1530, c(P["g4"])))
    A(ad("chl32_mg_1200x1600_programme", "monograph", 1200, 1600, els,
         "3:4 programme: logo, small photo, title, six date/title/time rows (grouped, 18 texts), button, legal "
         "(25 elements).", ["portrait-3:4", "monochrome", "20-plus-elements", "list-led", "event-date",
                             "text-groups", "solid-bg"]))
    return ads


# ---------------------------------------------------------------------------
# Meridian Air (travel, SATURATED cobalt + coral)
# ---------------------------------------------------------------------------

def meridian() -> list[dict]:
    ads, P = [], MA
    A = ads.append
    LOGO_W, LOGO_C = "chl_logo_meridian_white.png", "chl_logo_meridian_cobalt.png"

    # R1 1200x300: price-led on a panorama
    col = Col(40, 0, 400)
    col.image("logo", LOGO_W, 16, h=54)
    col.text("headline", "Lisbon from $89", 46, 4, bold=True, color=c(P["white"]))
    col.text("subhead", "One way, March to May. Book by Sunday.", 17, 14, color=c(P["white"]))
    col.pill("cta", "Book Now", 17, h=46, fill=c(P["coral"]), color=c(P["navy"]))
    col.center_in(0, 300)
    A(ad("chl33_ma_1200x300_price", "meridian", 1200, 300, [
        bg(1200, 300, "chl_photo_coast_panorama.png"),
        mc.shape("panel", 0, 0, 480, 300, c(P["cobalt"]), z=1),
        *col.els,
    ], "4:1 price-led banner: cobalt copy panel on the left of a coastal panorama photo.",
        ["banner-4:1", "saturated", "price-led", "photo-bg", "panel", "groups", "no-product"]))

    # R2 600x900: dunes photo, card panel on top
    col = Col(76, 0, 448)
    col.image("logo", LOGO_C, 18, h=50)
    col.text("headline", "Marrakech, three hours away", 38, 12, bold=True, color=c(P["navy"]))
    col.text("subhead", "Direct flights twice a day from June.", 18, 22, color=c(P["navy"], 0.85))
    col.pill("cta", "See Fares", 17, fill=c(P["cobalt"]), color=c(P["white"]))
    col.center_in(40, 380)
    A(ad("chl34_ma_600x900_photo_card", "meridian", 600, 900, [
        bg(600, 900, "chl_photo_dunes.png"),
        mc.shape("panel", 40, 40, 520, 340, c(P["white"]), radius=20, z=1),
        *col.els,
    ], "Full-bleed dunes photo with a white card panel at the TOP holding logo and copy.",
        ["portrait-2:3", "saturated", "full-bleed-panel", "photo-bg", "panel", "groups", "no-product"]))

    # R3 970x250: two products
    pack = Col(30, 82, 560)
    pack.text("headline", "Pack light, fly far", 38, 2, bold=True, color=c(P["navy"]))
    pack.text("subhead", "A carry-on bag is included in every fare this summer.", 17, 12, color=c(P["navy"], 0.85))
    pack.pill("cta", "See Summer Fares", 15, h=42, fill=c(P["coral"]), color=c(P["navy"]))
    pack.center_in(78, 238)
    A(ad("chl35_ma_970x250_pack_light", "meridian", 970, 250, [
        bg(970, 250, c(P["sand"])),
        mc.ellipse("deco_disc", 640, 10, 240, 240, c(P["sky"]), 1.0),
        img("logo", LOGO_C, 30, 22, h=46),
        *pack.els,
        img("product1", "chl_prod_suitcase.png", 650, 22, h=206),
        img("product2", "chl_prod_sunglasses.png", 812, 170, w=140),
    ], "970x250 with two products (suitcase + sunglasses) on a soft disc; copy and button left.",
        ["billboard", "saturated", "two-products", "solid-bg", "deco-overlap", "groups"]))

    # R4 160x600: price-led skyscraper
    col = Col(10, 0, 140, "center")
    col.text("headline", "New York", 26, 4, bold=True, color=c(P["white"]))
    col.text("from", "from", 16, 0, color=c(P["white"]))
    col.text("price", "$199", 46, 0, bold=True, color=c(P["white"]))
    col.center_in(96, 330)
    A(ad("chl36_ma_160x600_price", "meridian", 160, 600, [
        bg(160, 600, c(P["cobalt"])),
        img("logo", LOGO_W, 10, 30, w=140),
        *col.els,
        img("product", "chl_prod_suitcase.png", 22, 350, h=158),
        *mc.pill("cta", 20, 530, 120, 44, "Book", 16, fill=c(P["coral"]), color=c(P["navy"])),
    ], "160x600 price-led: logo, city, 'from', a big price, suitcase, button.",
        ["skyscraper", "saturated", "price-led", "solid-bg", "groups"]))

    # R5 1500x500: list of three destination cards with prices
    els = [bg(1500, 500, c(P["navy"])), img("logo", LOGO_W, 60, 60, h=70)]
    col = Col(60, 170, 560)
    col.text("headline", "Spring fares are out", 56, 16, bold=True, color=c(P["white"]))
    col.text("subhead", "Forty routes, all under $150 one way.", 22, 30, color=c(P["white"], 0.85))
    col.pill("cta", "See All 40 Routes", 20, fill=c(P["coral"]), color=c(P["navy"]))
    els += col.els
    for i, (city, price) in enumerate([("Lisbon", "$89"), ("Marrakech", "$119"), ("Reykjavik", "$139")], 1):
        x, g = 700 + 260 * (i - 1), f"card{i}"
        els += [mc.shape(f"card{i}_bg", x, 70, 236, 360, c(P["white"]), radius=18, z=1, group=g),
                t(f"card{i}_city", city, x + 24, 110, 188, 28, bold=True, color=c(P["navy"]), group=g),
                t(f"card{i}_from", "from", x + 24, 230, 188, 16, color=c(P["navy"], 0.8), group=g),
                t(f"card{i}_price", price, x + 24, 258, 188, 60, bold=True, color=c(P["cobalt"]), group=g)]
    A(ad("chl37_ma_1500x500_destinations", "meridian", 1500, 500, els,
         "3:1 list of three destination price cards (card + city + from + price, grouped), copy left (17 elements).",
         ["banner-3:1", "saturated", "list-led", "price-led", "cards", "text-groups", "solid-bg"]))

    # R6 1080x566: very long legal
    col = Col(60, 40, 640)
    col.image("logo", LOGO_C, 20, h=60)
    col.text("headline", "Fly now, pay in four", 52, 12, bold=True, color=c(P["navy"]))
    col.text("subhead", "Split any fare into four interest-free payments.", 21, 22, color=c(P["navy"], 0.85))
    col.pill("cta", "How It Works", 18, fill=c(P["cobalt"]), color=c(P["white"]))
    legal = legal_at_bottom(
        "legal", "Pay in four is offered by Meridian Pay Ltd. to customers aged 18 or over, subject to an eligibility "
        "check that does not affect your credit score. The first payment is taken at booking and the rest every two "
        "weeks. Missed payments may incur a fee of up to $8 each, capped at 25% of the fare, and may limit future "
        "use. Fares are per person, one way, include taxes and one carry-on bag, and are subject to availability. "
        "Changes and refunds follow the fare rules of the ticket bought. Meridian Air, 2026.",
        60, 960, 13, 540, c(P["navy"], 0.8))
    col.center_in(20, legal["y"] - 30)
    A(ad("chl38_ma_1080x566_long_legal", "meridian", 1080, 566, [
        bg(1080, 566, c(P["sky"])),
        mc.ellipse("deco_disc", 760, 20, 300, 300, c(P["white"]), 0.8),
        img("product", "chl_prod_suitcase.png", 800, 40, h=290),
        *col.els, legal,
    ], "1.91:1 link-share ad: logo, headline, button, suitcase, and a 5-line finance legal block across the bottom.",
        ["landscape-1.91:1", "saturated", "long-legal", "solid-bg", "deco-overlap", "groups"]))

    # R7 2400x600: rotated sticker on a panorama
    col = Col(120, 0, 760)
    col.image("logo", LOGO_W, 24, h=84)
    col.text("headline", "The big spring sale", 76, 12, bold=True, color=c(P["white"]))
    col.text("subhead", "Up to 40% off 60 routes. Ends Sunday at midnight.", 28, 30, color=c(P["white"]))
    col.pill("cta", "Find a Fare", 28, fill=c(P["coral"]), color=c(P["navy"]))
    col.center_in(60, 540)
    A(ad("chl39_ma_2400x600_sale_sticker", "meridian", 2400, 600, [
        bg(2400, 600, "chl_photo_coast_panorama.png"),
        mc.shape("panel", 60, 50, 880, 500, c(P["cobalt"]), radius=28, z=1),
        *col.els,
        *sticker("sticker", 1130, 190, 250, "UP TO 40% OFF", 36, fill=c(P["coral"]), color=c(P["navy"]),
                 rotation=-10),
        img("product", "chl_prod_sunglasses.png", 1750, 400, w=480),
    ], "4:1 sale banner: cobalt panel on a panorama, a rotated round sale sticker, sunglasses on the sea.",
        ["banner-4:1", "saturated", "rotated-sticker", "rotated", "photo-bg", "panel", "groups"]))

    # R8 800x1200: fare table, 20+ elements
    els = [bg(800, 1200, c(P["white"])),
           mc.shape("band", 0, 0, 800, 340, c(P["cobalt"]), z=1),
           img("logo", LOGO_W, 60, 50, h=70),
           t("headline", "Weekend fares from Lisbon", 60, 150, 680, 54, bold=True, color=c(P["white"]))]
    routes = [("Paris", "Fri to Sun", "$79"), ("Rome", "Fri to Mon", "$95"), ("Berlin", "Sat to Sun", "$88"),
              ("Madrid", "Fri to Sun", "$49"), ("London", "Thu to Sun", "$99"), ("Vienna", "Fri to Sun", "$109")]
    for i, (city, days, price) in enumerate(routes, 1):
        y, g = 380 + 98 * (i - 1), f"route{i}"
        els += [t(f"route{i}_city", city, 60, y, 300, 30, bold=True, color=c(P["navy"]), group=g),
                t(f"route{i}_days", days, 60, y + 42, 300, 17, color=c(P["navy"], 0.8), group=g),
                t(f"route{i}_price", price, 520, y + 4, 220, 40, bold=True, color=c(P["cobalt"]), align="right",
                  group=g)]
    els += mc.pill("cta", 60, 990, 340, 84, "Book a Weekend", 26, fill=c(P["coral"]), color=c(P["navy"]))
    els.append(legal_at_bottom("legal", "Fares per person, return, including taxes. Limited seats.", 60, 680, 16,
                               1160, c(P["navy"], 0.8)))
    A(ad("chl40_ma_800x1200_fare_table", "meridian", 800, 1200, els,
         "2:3 fare table: cobalt header band with logo + headline, six city/days/price rows (grouped), button, "
         "legal (25 elements).", ["portrait-2:3", "saturated", "20-plus-elements", "list-led", "price-led",
                                  "text-groups", "solid-bg"]))
    return ads


# ---------------------------------------------------------------------------
# Checks and output
# ---------------------------------------------------------------------------

def challenge_ads() -> list[dict]:
    return lumen() + obsidian() + fizzbang() + monograph() + meridian()


def quality_issues(rec: dict, assets) -> list[str]:
    """Everything make_corpus's strict layout check and pixel check flag, minus
    overlaps listed as intended."""
    allowed = INTENDED_OVERLAPS.get(rec["task_id"], set())
    issues = []
    for w in mc.check(rec, strict=True):
        m = re.fullmatch(r"overlap: (\S+) x (\S+)", w)
        if m and ((m[1], m[2]) in allowed or (m[2], m[1]) in allowed):
            continue
        issues.append(w)
    return issues + mc.pixel_check(rec, assets)


def sync(records: list[dict]) -> None:
    """Write records into CHALLENGE_DIR; rewrite only changed files, delete stale ones."""
    CHALLENGE_DIR.mkdir(parents=True, exist_ok=True)
    keep = {f"{r['task_id']}.json": json.dumps(r, indent=2, ensure_ascii=False) + "\n" for r in records}
    for path in CHALLENGE_DIR.glob("*.json"):
        if path.name not in keep:
            path.unlink()
    for name, text in keep.items():
        path = CHALLENGE_DIR / name
        if not path.exists() or path.read_text() != text:
            path.write_text(text)


def main() -> None:
    records = challenge_ads()
    keep, problems = [], 0
    with merged_assets_dir() as assets:
        for rec in records:
            issues = quality_issues(rec, assets)
            for w in issues:
                print(f"[drop] {rec['task_id']}: {w}")
            problems += len(issues)
            if not issues:
                keep.append(rec)
        sync(keep)
        mc.REVIEW_DIR.mkdir(parents=True, exist_ok=True)
        mc.contact_sheet(keep, REVIEW_SHEET, assets_dir=assets)
    print(f"wrote {len(keep)} of {len(records)} challenge ads to {CHALLENGE_DIR}; {problems} problems; "
          f"review sheet: {REVIEW_SHEET}")


if __name__ == "__main__":
    main()
