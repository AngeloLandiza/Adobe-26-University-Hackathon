"""What node 1 sends a hosted LLM: one fixed system prompt, the units as compact
JSON, and a response schema built from this design's unit ids so the model
can't invent ids. Claude and Gemini get identical inputs, which keeps the
provider benchmark fair."""

import json

from engine.pipeline.types import ROLES

SYSTEM_PROMPT = """\
You label the parts of a marketing graphic so a layout engine can resize it to other formats.

Each unit is one element, or a few elements that move together (a button shape and its label, for example). For each unit you get its type, position and size as fractions of the canvas, stacking order, opacity, rotation, text and font size, image file name, and shape. File names can hint at content but may be generic. Background units are listed by id only.

Give every unit exactly one role:
- background: a full-bleed image or shape behind everything
- decoration: shapes or flourishes that carry no information
- headline: the main message
- subheadline: supporting copy
- cta: the call to action, as a button or as plain text
- logo: the brand mark
- hero_image: the product or main visual
- badge: a short promotional label or ribbon
- secondary_image: any other image, including watermarks and icons
- legal: fine print and disclaimers
- other_text: any other text

Then rank every unit that is not background or decoration, most important first. The core message, what is being sold, how to act on it, and who it is from come first; supporting copy next; badges, secondary images and fine print last. On a very small format only the top few units stay readable, so put first what a viewer must not miss.

Before the roles, write a summary of what the graphic advertises in 30 words or fewer."""


def _round_floats(value, places: int = 3):
    if isinstance(value, float):
        return round(value, places)
    if isinstance(value, list):
        return [_round_floats(v, places) for v in value]
    if isinstance(value, dict):
        return {k: _round_floats(v, places) for k, v in value.items()}
    return value


def build_user_message(features: list[dict]) -> str:
    """Compact, key-sorted JSON so identical designs produce identical prompts."""
    background = [f["unit_id"] for f in features if "background" in f["flags"]]
    units = [_round_floats(f) for f in features if "background" not in f["flags"]]
    return json.dumps({"background_unit_ids": background, "units": units}, sort_keys=True, separators=(",", ":"))


def response_schema(unit_ids: list[str]) -> dict:
    ids = list(unit_ids)
    return {
        "type": "object",
        "properties": {
            "summary": {"type": "string"},
            "roles": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "unit_id": {"type": "string", "enum": ids},
                        "role": {"type": "string", "enum": list(ROLES)},
                    },
                    "required": ["unit_id", "role"],
                    "additionalProperties": False,
                },
            },
            "ranking": {"type": "array", "items": {"type": "string", "enum": ids}},
        },
        "required": ["summary", "roles", "ranking"],
        "additionalProperties": False,
    }
