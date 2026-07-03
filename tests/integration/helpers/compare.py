"""Tolerant comparison of detected mask rectangles against golden rectangles.

OCR coordinates drift by a few pixels between runs and model versions, so a
golden region is not required to match a single mask exactly. What matters for
deidentification is that the sensitive region is hidden: a golden region counts
as covered when the union of all detected masks covers at least ``threshold`` of
its area. This makes coverage many-to-many on purpose. One large mask may cover
several golden regions, and several masks may jointly cover one golden region.

For deidentification, a false negative (a golden region left unmasked) is the
critical failure; a false positive (a mask not overlapping any golden region) is
reported but less severe.
"""

from dataclasses import dataclass, field

Rect = tuple[int, int, int, int]


def parse_rect(rect: str) -> Rect:
    """Parse an ``"x y width height"`` string into a tuple of ints."""
    x, y, w, h = (int(v) for v in rect.split())
    return x, y, w, h


def _clip(golden: Rect, rect: Rect) -> tuple[int, int, int, int] | None:
    """Intersection of ``rect`` with ``golden`` as ``(x1, y1, x2, y2)``, or None."""
    gx, gy, gw, gh = golden
    rx, ry, rw, rh = rect
    x1 = max(gx, rx)
    y1 = max(gy, ry)
    x2 = min(gx + gw, rx + rw)
    y2 = min(gy + gh, ry + rh)
    if x2 > x1 and y2 > y1:
        return x1, y1, x2, y2
    return None


def covered_fraction(golden: Rect, detected: list[Rect]) -> float:
    """Fraction of ``golden``'s area covered by the union of ``detected`` rects.

    The detected rectangles are clipped to ``golden`` and their union area is
    computed by coordinate compression, so overlapping masks are not
    double-counted.
    """
    golden_area = golden[2] * golden[3]
    if golden_area <= 0:
        return 0.0

    clipped = [c for c in (_clip(golden, r) for r in detected) if c is not None]
    if not clipped:
        return 0.0

    xs = sorted({x for r in clipped for x in (r[0], r[2])})
    ys = sorted({y for r in clipped for y in (r[1], r[3])})

    covered = 0
    for i in range(len(xs) - 1):
        x_lo, x_hi = xs[i], xs[i + 1]
        for j in range(len(ys) - 1):
            y_lo, y_hi = ys[j], ys[j + 1]
            for rx1, ry1, rx2, ry2 in clipped:
                if rx1 <= x_lo and rx2 >= x_hi and ry1 <= y_lo and ry2 >= y_hi:
                    covered += (x_hi - x_lo) * (y_hi - y_lo)
                    break
    return covered / golden_area


@dataclass
class ComparisonResult:
    matched: list[Rect] = field(default_factory=list)
    false_negatives: list[tuple[Rect, float]] = field(default_factory=list)
    false_positives: list[Rect] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """No missed golden region (false negatives are the critical failure)."""
        return not self.false_negatives


def compare(
    golden: list[str], detected: list[str], threshold: float = 0.9
) -> ComparisonResult:
    """Check each golden region is covered by the union of the detected masks.

    A golden region is matched when at least ``threshold`` of its area is covered
    by the detected masks combined, regardless of how many masks contribute. A
    detected mask that does not overlap any golden region is a false positive.
    """
    golden_rects = [parse_rect(r) for r in golden]
    detected_rects = [parse_rect(r) for r in detected]

    result = ComparisonResult()
    for g in golden_rects:
        cov = covered_fraction(g, detected_rects)
        if cov >= threshold:
            result.matched.append(g)
        else:
            result.false_negatives.append((g, cov))

    result.false_positives = [
        d for d in detected_rects if all(_clip(g, d) is None for g in golden_rects)
    ]
    return result
