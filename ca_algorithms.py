"""Cellular automata style wildfire spread comparison models."""

from __future__ import annotations

import numpy as np


class PersistenceCAModel:
    """Probabilistic CA-style persistence model.

    Uses a convex combination of current cell activation and neighborhood pressure.
    """

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        return None

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        cur = np.clip(x[:, 0], 0, 1)
        neigh = np.clip(x[:, 1], 0, 1)
        return np.clip(0.8 * cur + 0.2 * neigh, 0, 1)


class DeterministicCAModel:
    """Deterministic CA rule converted to confidence-like scores.

    Rule:
      - Active cell persists with high probability.
      - Inactive cell ignites if neighborhood pressure exceeds threshold.
    """

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        return None

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        cur = np.clip(x[:, 0], 0, 1)
        neigh = np.clip(x[:, 1], 0, 1)

        persists = cur >= 0.4
        ignites = neigh >= 0.5
        state = np.where(persists | ignites, 1.0, 0.0)

        # Keep deterministic behavior while allowing ROC ranking.
        return np.where(state > 0, np.maximum(cur, neigh), 0.0)
