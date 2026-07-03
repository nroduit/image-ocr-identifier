# Integration Tests

Per-golden union coverage
For each golden region, covered_fraction computes the fraction of its area covered by the union of all detected masks (clipped + coordinate-compression so overlaps aren't double-counted).
Covered if coverage >= threshold (0.5), regardless of how many masks contribute.
False positive = a detected mask overlapping no golden region (warning only, unchanged severity).

The 0.5 threshold means "at least 50% of the golden area is masked"