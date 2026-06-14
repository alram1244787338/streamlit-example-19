"""Core spiral computation and display preparation.

Separates the computational layer from the presentation layer,
providing downsampling for large point counts to maintain
interactive responsiveness.
"""

import numpy as np
import pandas as pd

# Points above this threshold are downsampled before rendering.
# Below this value, all points are sent to the chart as-is so that
# fine detail is preserved for typical usage (e.g. ~1000 points).
DISPLAY_THRESHOLD = 3000


def compute_spiral(n_points: int, n_turns: int, seed: int = 42) -> pd.DataFrame:
    """Compute spiral geometry with full fidelity.

    Parameters
    ----------
    n_points : int
        Total number of data points along the spiral.
    n_turns : int
        Number of complete turns the spiral makes.
    seed : int, optional
        Seed for reproducible random values used for point sizes.

    Returns
    -------
    pd.DataFrame
        Columns: x, y, idx, rand.  Always contains exactly *n_points* rows.
    """
    indices = np.linspace(0, 1, n_points)
    theta = 2 * np.pi * n_turns * indices
    radius = indices

    x = radius * np.cos(theta)
    y = radius * np.sin(theta)

    rng = np.random.default_rng(seed)

    return pd.DataFrame({
        "x": x,
        "y": y,
        "idx": indices,
        "rand": rng.normal(size=n_points),
    })


def prepare_for_render(
    df: pd.DataFrame,
    max_display: int = DISPLAY_THRESHOLD,
) -> pd.DataFrame:
    """Prepare a DataFrame for chart rendering.

    When the full dataset exceeds *max_display* rows, systematic
    (evenly-spaced) sampling is applied so the spiral shape is
    faithfully preserved while keeping the chart responsive.

    For small datasets (below the threshold) the data is returned
    unchanged — no detail is lost.

    Parameters
    ----------
    df : pd.DataFrame
        Full spiral DataFrame as returned by :func:`compute_spiral`.
    max_display : int, optional
        Maximum number of rows to keep for display.

    Returns
    -------
    pd.DataFrame
        A (possibly downsampled) **copy** suitable for rendering.
    """
    if len(df) <= max_display:
        return df.copy()

    # Systematic sampling keeps points evenly distributed along the
    # spiral, preserving its visual structure far better than random
    # sampling would.  np.linspace guarantees that both the first
    # (idx=0) and last (idx=1) points are always included.
    positions = np.linspace(0, len(df) - 1, max_display, dtype=int)
    result = df.iloc[positions].copy()

    return result
