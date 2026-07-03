from typing import Protocol

import numpy as np


class OcrBackend(Protocol):
    def detect(self, image: np.ndarray) -> tuple[list[str], list[list[int]]]:
        """Run detection and recognition on an image.

        Returns ``(texts, boxes)`` where each box is
        ``[x_min, y_min, x_max, y_max]`` expressed in the coordinate space of
        ``image`` (any backend-internal padding must already be reversed).
        """
        ...
