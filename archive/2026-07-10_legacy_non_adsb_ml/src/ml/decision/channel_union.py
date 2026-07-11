"""Boolean union helpers for independent alarm channels."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np


def union_onsets(channels: Iterable[np.ndarray]) -> np.ndarray:
    """Combine aligned channel-onset masks into one operator notification mask."""

    masks = [np.asarray(channel, dtype=bool) for channel in channels]
    if not masks:
        raise ValueError("At least one channel onset mask is required")
    lengths = {mask.shape for mask in masks}
    if len(lengths) != 1 or any(mask.ndim != 1 for mask in masks):
        raise ValueError("Channel onset masks must be one-dimensional and aligned")
    return np.logical_or.reduce(masks)
