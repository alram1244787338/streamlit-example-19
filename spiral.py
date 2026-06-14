"""Spiral computation layer.

This module is deliberately free of any Streamlit / Altair imports so the
spiral math and the display-downsampling logic can be unit-tested without a
running Streamlit context.

Two responsibilities, kept separate on purpose:

* ``compute_spiral`` is the *computation layer*: it always produces the full,
  honest, full-resolution spiral for the requested number of points. It is
  deterministic (seeded RNG) so the same parameters always yield the same
  figure -- no more flicker when a slider is nudged.
* ``downsample_for_display`` is the *display layer*: it caps how many points
  are actually handed to the (SVG-based) chart so the browser stays responsive
  at high point counts, while preserving the shape of the spiral.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Above this many points, SVG point-rendering in the browser starts to stutter.
# Anything at or below it (e.g. the common ~1000 range) is drawn in full detail.
MAX_DISPLAY_POINTS = 2000

# Fixed seed so the "rand" size channel is reproducible across reruns.
DEFAULT_SEED = 0


@dataclass(frozen=True)
class DisplayResult:
    """Outcome of preparing data for the chart.

    Attributes
    ----------
    data:
        The DataFrame that should actually be rendered.
    displayed_points:
        Number of rows in ``data`` (i.e. points the chart will draw).
    computed_points:
        Number of points in the full-resolution source data.
    downsampled:
        Whether the display data was thinned relative to the source.
    """

    data: pd.DataFrame
    displayed_points: int
    computed_points: int
    downsampled: bool


def compute_spiral(
    num_points: int, num_turns: int, seed: int = DEFAULT_SEED
) -> pd.DataFrame:
    """Build the full-resolution spiral (the computation layer).

    The result is deterministic for a given ``(num_points, num_turns, seed)``
    so repeated reruns produce an identical figure.
    """
    if num_points < 1:
        raise ValueError("num_points must be >= 1")
    if num_turns < 1:
        raise ValueError("num_turns must be >= 1")

    indices = np.linspace(0, 1, num_points)
    theta = 2 * np.pi * num_turns * indices
    radius = indices

    x = radius * np.cos(theta)
    y = radius * np.sin(theta)

    rng = np.random.default_rng(seed)
    rand = rng.standard_normal(num_points)

    return pd.DataFrame({"x": x, "y": y, "idx": indices, "rand": rand})


def downsample_for_display(
    df: pd.DataFrame, max_points: int = MAX_DISPLAY_POINTS
) -> DisplayResult:
    """Cap render density while preserving the spiral's shape (display layer).

    Points are selected at evenly spaced positions along the curve. Because the
    spiral is a parametric path, evenly thinning the samples traces the very
    same spiral -- just with fewer dots -- rather than cropping or distorting it.
    """
    if max_points < 1:
        raise ValueError("max_points must be >= 1")

    computed = len(df)
    if computed <= max_points:
        return DisplayResult(
            data=df.reset_index(drop=True),
            displayed_points=computed,
            computed_points=computed,
            downsampled=False,
        )

    selection = np.unique(np.linspace(0, computed - 1, max_points).astype(int))
    display_df = df.iloc[selection].reset_index(drop=True)
    return DisplayResult(
        data=display_df,
        displayed_points=len(display_df),
        computed_points=computed,
        downsampled=True,
    )


def prepare_spiral(
    num_points: int,
    num_turns: int,
    max_points: int = MAX_DISPLAY_POINTS,
    seed: int = DEFAULT_SEED,
) -> DisplayResult:
    """Convenience wrapper: compute the full spiral, then thin it for display."""
    full_df = compute_spiral(num_points, num_turns, seed=seed)
    return downsample_for_display(full_df, max_points=max_points)
