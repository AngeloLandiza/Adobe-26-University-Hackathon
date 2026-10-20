"""Per-unit features shared by the LLM prompt and the heuristic ranker
(docs/pipeline/01-priority-node.md §3). Positions are normalised to the source
canvas; `aspect` is width/height in pixels. Attributes a unit doesn't have
(e.g. text features of an image) are None."""

from engine.pipeline.geometry import element_box, overlap_area, union_box
from engine.pipeline.textmeasure import BOLD_WEIGHTS
from engine.pipeline.types import Unit

TEXT_PREVIEW_CHARS = 160
FULL_CANVAS = 0.9
MAX_OVERLAPS = 3


def _font_size(members: list[dict]) -> float | None:
    sizes = [m.get("font_size") for m in members if m["type"] == "text" and m.get("font_size") is not None]
    return max(sizes) if sizes else None


def unit_features(source: dict, units: list[Unit]) -> list[dict]:
    width, height = source["canvas"]["width"], source["canvas"]["height"]
    by_id = {el["id"]: el for el in source["elements"]}
    members_of = {u.unit_id: [by_id[i] for i in u.element_ids] for u in units}
    boxes = {u.unit_id: union_box([element_box(m) for m in members_of[u.unit_id]]) for u in units}
    background = {u.unit_id for u in units if u.role == "background"}
    sizes = {u.unit_id: _font_size(members_of[u.unit_id]) for u in units}
    distinct_sizes = sorted({s for s in sizes.values() if s is not None}, reverse=True)

    features = []
    for unit in units:
        members = members_of[unit.unit_id]
        texts = [m for m in members if m["type"] == "text"]
        images = [m for m in members if m["type"] == "image"]
        shapes = [m for m in members if m["type"] == "shape"]
        box = boxes[unit.unit_id]
        x, y, w, h = box
        area_pct = overlap_area(box, (0, 0, width, height)) / (width * height)

        content = " ".join(str(t.get("content", "")) for t in texts) if texts else None
        letters = [c for c in content if c.isalpha()] if content else []
        font_size = sizes[unit.unit_id]

        overlaps = sorted(
            ((overlap_area(box, boxes[o.unit_id]), o.unit_id) for o in units
             if o is not unit and o.unit_id not in background),
            key=lambda pair: (-pair[0], pair[1]),
        )

        features.append({
            "unit_id": unit.unit_id,
            "kind": unit.kind,
            "flags": list(unit.flags),
            "types": [m["type"] for m in members],
            "bbox": [x / width, y / height, w / width, h / height],
            "area_pct": area_pct,
            "aspect": w / h if h else None,
            "center": [(x + w / 2) / width, (y + h / 2) / height],
            "z_index": max(m.get("z_index") or 0 for m in members),
            "opacity": max(1 if m.get("opacity") is None else m["opacity"] for m in members),
            "rotation": next((m["rotation"] for m in members if m.get("rotation")), 0),
            "is_full_canvas": area_pct >= FULL_CANVAS,
            "text": content[:TEXT_PREVIEW_CHARS] if content is not None else None,
            "text_len": len(content) if content is not None else None,
            "font_size": font_size,
            "font_size_rank": distinct_sizes.index(font_size) + 1 if font_size is not None else None,
            "bold": any((t.get("font_weight") or "regular").lower() in BOLD_WEIGHTS for t in texts) if texts else None,
            "all_caps": (bool(letters) and all(c.isupper() for c in letters)) if texts else None,
            "word_count": len(content.split()) if content is not None else None,
            "asset_id": images[0].get("asset_id") if images else None,
            "fit": images[0].get("fit", "contain") if images else None,
            "missing_asset": ("missing_asset" in unit.flags) if images else None,
            "shape_type": shapes[0].get("shape_type") if shapes else None,
            "has_label": bool(texts) if shapes else None,
            "overlaps": [uid for area, uid in overlaps if area > 0][:MAX_OVERLAPS],
        })
    return features
