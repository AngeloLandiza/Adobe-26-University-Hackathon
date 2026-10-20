"""Small synthetic designs for edge cases the three samples don't cover
(found by the whole-branch review). Builders fill the schema's required fields."""


def text(id, content, x, y, w, h, size, **kw):
    return {"id": id, "type": "text", "content": content, "font_family": "Solstice Sans", "font_size": size,
            "color": "rgba(0,0,0,1)", "x": x, "y": y, "width": w, "height": h, **kw}


def shape(id, x, y, w, h, **kw):
    return {"id": id, "type": "shape", "shape_type": "rectangle", "fill_color": "rgba(255,255,255,1)",
            "x": x, "y": y, "width": w, "height": h, **kw}


def image(id, asset, x, y, w, h, **kw):
    return {"id": id, "type": "image", "asset_id": asset, "x": x, "y": y, "width": w, "height": h, **kw}


def design(*elements, width=1080, height=1080):
    return {"canvas": {"width": width, "height": height}, "elements": list(elements)}


# A flat colour rectangle as the background, one big headline on it.
BG_SHAPE_HEADLINE = design(
    shape("bg", 0, 0, 1080, 1080, z_index=0, fill_color="rgba(250,240,220,1)"),
    text("headline", "Summer sale starts now", 90, 340, 900, 400, 120, z_index=1, text_align="center"))

# The headline sits on a white card; a short disclaimer sits below the card.
HEADLINE_CARD = design(
    image("photo", "solstice_bg_gradient.png", 0, 0, 1080, 1080, fit="cover", z_index=0),
    image("product", "solstice_bottle.png", 390, 80, 300, 420, z_index=1),
    shape("card", 100, 560, 880, 300, z_index=1),
    text("headline", "Chase the light", 140, 600, 800, 220, 72, z_index=2, text_align="center"),
    text("disclaimer", "Please enjoy responsibly.", 340, 960, 400, 30, 16, z_index=1, text_align="center"))

# A schema-legal zero-height line divider between two texts.
LINE_DIVIDER = design(
    text("headline", "Summer sale", 90, 300, 900, 140, 100, z_index=1),
    shape("divider", 240, 500, 600, 0, shape_type="line", stroke_width=4, stroke_color="rgba(0,0,0,1)", z_index=1),
    text("sub", "Everything must go this weekend only", 90, 560, 900, 80, 40, z_index=1))

# A translucent white panel behind two lines of black copy on a dark photo.
PANEL_BEHIND_COPY = design(
    image("photo", "solstice_bg_gradient.png", 0, 0, 1080, 1080, fit="cover", z_index=0),
    shape("panel", 100, 600, 880, 380, fill_color="rgba(255,255,255,0.85)", z_index=1),
    text("headline", "Chase the light", 140, 630, 800, 140, 90, z_index=2),
    text("sub", "Botanical sparkling infusions, crafted for golden hour.", 140, 800, 800, 120, 36, z_index=2))

# A faint 280 px "SUMMER" behind the bottle: the biggest font, but not the headline
# (dev/samples/dev_sample_08_ghost_word.json, cut down).
GHOST_WORD = design(
    image("bg", "solstice_bg_gradient.png", 0, 0, 1080, 1080, z_index=0),
    text("ghost_word", "SUMMER", -60, 240, 1200, 340, 280, z_index=1, opacity=0.08),
    image("product", "solstice_bottle.png", 370, 150, 340, 560, z_index=3),
    image("logo", "solstice_logo_lockup.png", 60, 50, 240, 76, z_index=4),
    text("headline", "Chase the Light", 90, 740, 900, 90, 72, z_index=4),
    text("subhead", "Botanical sparkling infusions, crafted for golden hour.", 140, 836, 800, 40, 26, z_index=4),
    shape("cta_bg", 365, 910, 350, 84, z_index=3, group_id="cta"),
    text("cta_label", "Shop the Summer Line", 365, 932, 350, 40, 26, z_index=4, group_id="cta"))
