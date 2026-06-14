"""Spiral demo core: deterministic data generation and chart construction.

Deliberately free of Streamlit so the logic is unit-testable without a running
Streamlit server. ``streamlit_app.py`` wires these helpers into the page and
adds caching plus the cache/stable indicator.
"""
from __future__ import annotations

from dataclasses import dataclass

import altair as alt
import numpy as np
import pandas as pd

# Default RNG seed. With a fixed seed the random size signal is fully
# reproducible, so the same parameters always render identical point sizes.
DEFAULT_SEED = 0

# Slider bounds (mirrored by the Streamlit sliders so the app and the
# validation share a single source of truth).
MIN_POINTS = 1
MAX_POINTS = 10_000
MIN_TURNS = 1
MAX_TURNS = 300

# The raw normal draw is clipped to this many standard deviations before being
# used as the size signal. Keeps a stray outlier from squashing every other
# point down to near-invisible.
SIZE_CLIP = 2.5

# Final rendered point-size range. The minimum is large enough that the
# smallest point is always visible; the maximum is capped so no point becomes
# absurdly large.
MIN_POINT_SIZE = 18
MAX_POINT_SIZE = 180


@dataclass(frozen=True)
class SpiralParams:
    """Spiral inputs. ``clamped`` keeps them inside the supported ranges."""

    num_points: int
    num_turns: int
    seed: int = DEFAULT_SEED

    def clamped(self) -> "SpiralParams":
        return SpiralParams(
            num_points=int(np.clip(self.num_points, MIN_POINTS, MAX_POINTS)),
            num_turns=int(np.clip(self.num_turns, MIN_TURNS, MAX_TURNS)),
            seed=self.seed,
        )


def generate_spiral_data(num_points, num_turns, seed=DEFAULT_SEED):
    """Build the spiral DataFrame as a pure function of the parameters.

    Determinism is the whole point: identical ``(num_points, num_turns, seed)``
    always yields an identical DataFrame, so a plain Streamlit rerun can never
    reshuffle the point sizes. A dedicated ``numpy`` Generator is used instead
    of the global ``np.random`` state so nothing else in the process can
    perturb the result. Out-of-range parameters are clamped, so extreme slider
    values never produce a broken or empty chart.
    """
    params = SpiralParams(num_points, num_turns, seed).clamped()
    n = params.num_points

    rng = np.random.default_rng(params.seed)

    indices = np.linspace(0, 1, n)
    theta = 2 * np.pi * params.num_turns * indices
    radius = indices

    x = radius * np.cos(theta)
    y = radius * np.sin(theta)

    # Clip the raw normal draw so the size signal stays in a stable, bounded
    # band -> no near-invisible and no runaway points, regardless of n.
    raw = rng.standard_normal(n)
    size_signal = np.clip(raw, -SIZE_CLIP, SIZE_CLIP)

    return pd.DataFrame(
        {
            "x": x,
            "y": y,
            "idx": indices,
            "size_signal": size_signal,
        }
    )


def build_spiral_chart(df, height=700, width=700):
    """Construct the Altair spiral chart (pure: no Streamlit calls).

    The size scale uses a fixed domain/range with ``clamp=True`` so point
    sizes are absolute and consistent across parameter changes rather than
    being rescaled to each sample's own min/max.
    """
    size_scale = alt.Scale(
        domain=[-SIZE_CLIP, SIZE_CLIP],
        range=[MIN_POINT_SIZE, MAX_POINT_SIZE],
        clamp=True,
    )
    return (
        alt.Chart(df, height=height, width=width)
        .mark_point(filled=True)
        .encode(
            x=alt.X("x", axis=None),
            y=alt.Y("y", axis=None),
            color=alt.Color("idx", legend=None, scale=alt.Scale()),
            size=alt.Size("size_signal", legend=None, scale=size_scale),
        )
    )
