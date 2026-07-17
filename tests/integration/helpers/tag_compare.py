"""Set comparison of reported sensitive tags against a golden set.

Reporting a tag that is not in the golden is a false positive; it
is surfaced as a warning only.
"""

from dataclasses import dataclass, field


@dataclass
class TagComparison:
    matched: list[str] = field(default_factory=list)
    false_negatives: list[str] = field(default_factory=list)
    false_positives: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """No golden tag went missing"""
        return not self.false_negatives


def compare(golden: list[str], detected: list[str]) -> TagComparison:
    """Compare reported tags against the golden set, ignoring order.

    A golden tag missing from ``detected`` is a false negative; a detected tag
    absent from ``golden`` is a false positive. Results are sorted for stable,
    readable assertion messages.
    """
    golden_set = set(golden)
    detected_set = set(detected)
    return TagComparison(
        matched=sorted(golden_set & detected_set),
        false_negatives=sorted(golden_set - detected_set),
        false_positives=sorted(detected_set - golden_set),
    )
