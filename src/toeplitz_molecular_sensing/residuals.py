from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DetectedEvent:
    position: int
    score: float
    members: tuple[int, ...]


def residual_magnitude(observed: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    """Euclidean multichannel residual magnitude at every signal position."""
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if observed.shape != predicted.shape or observed.ndim != 2:
        raise ValueError("observed and predicted must have the same (N, C) shape")
    return np.linalg.norm(observed - predicted, axis=1)


def smooth_residual(residual: np.ndarray, window: int = 3) -> np.ndarray:
    """Centered moving average with edge-value padding, as in the manuscript."""
    residual = np.asarray(residual, dtype=float)
    if residual.ndim != 1:
        raise ValueError("residual must be one-dimensional")
    if window < 1 or window % 2 == 0:
        raise ValueError("window must be a positive odd integer")
    half = window // 2
    padded = np.pad(residual, (half, half), mode="edge")
    return np.convolve(padded, np.ones(window) / window, mode="valid")


def select_events(
    smoothed: np.ndarray,
    n_candidates: int,
    radius: int,
    max_events: int,
    eligible: np.ndarray | None = None,
) -> list[DetectedEvent]:
    """Rank residual positions, cluster them, and retain the strongest clusters."""
    smoothed = np.asarray(smoothed, dtype=float)
    if smoothed.ndim != 1:
        raise ValueError("smoothed residual must be one-dimensional")
    if n_candidates < 1 or radius < 0 or max_events < 1:
        raise ValueError("invalid event-selection parameters")
    positions = np.arange(smoothed.size) if eligible is None else np.asarray(eligible, dtype=int)
    if positions.ndim != 1 or np.any((positions < 0) | (positions >= smoothed.size)):
        raise ValueError("eligible positions are invalid")
    ranked = positions[np.argsort(smoothed[positions])[::-1][:n_candidates]]

    clusters: list[list[int]] = []
    centers: list[int] = []
    for pos in ranked:
        target = None
        for idx, center in enumerate(centers):
            if abs(int(pos) - center) <= radius:
                target = idx
                break
        if target is None:
            clusters.append([int(pos)])
            centers.append(int(pos))
        else:
            clusters[target].append(int(pos))
            centers[target] = int(max(clusters[target], key=lambda p: smoothed[p]))

    events = [
        DetectedEvent(center, float(smoothed[center]), tuple(sorted(members)))
        for center, members in zip(centers, clusters)
    ]
    events.sort(key=lambda event: event.score, reverse=True)
    return events[:max_events]
