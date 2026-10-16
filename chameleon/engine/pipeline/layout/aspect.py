"""Target-shape classes that decide which templates run
(docs/pipeline/02-resize-engine.md §1)."""


def aspect_class(width: float, height: float) -> str:
    r = width / height
    if r < 0.55:
        return "very_tall"
    if r < 0.85:
        return "tall"
    if r < 1.25:
        return "square"
    if r < 2.2:
        return "landscape"
    if r <= 5:
        return "wide"
    return "strip"
