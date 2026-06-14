"""Data generation for spiral scatter plot.

Uses @st.cache_data with a deterministic seed so that the same
(num_points, num_turns) pair always produces identical data,
eliminating random jitter on Streamlit reruns.
"""

import hashlib

import numpy as np
import pandas as pd
import streamlit as st


def _make_seed(num_points: int, num_turns: int) -> int:
    """Derive a stable integer seed from the two parameters."""
    raw = f"spiral-{num_points}-{num_turns}"
    return int(hashlib.md5(raw.encode()).hexdigest()[:8], 16)


@st.cache_data(show_spinner=False)
def generate_spiral_data(num_points: int, num_turns: int) -> pd.DataFrame:
    """Generate spiral data with deterministic, bounded point sizes.

    Returns a DataFrame with columns: x, y, idx, size.
    - idx: normalised position along the spiral [0, 1], used for colour.
    - size: uniform random in [0.3, 1.0], used for marker size.
    """
    rng = np.random.default_rng(_make_seed(num_points, num_turns))

    indices = np.linspace(0, 1, num_points)
    theta = 2 * np.pi * num_turns * indices
    radius = indices

    x = radius * np.cos(theta)
    y = radius * np.sin(theta)

    # Uniform random sizes — avoids the "invisible or huge" problem
    # that np.random.randn caused.
    raw_sizes = rng.uniform(0.3, 1.0, num_points)

    return pd.DataFrame({
        "x": x,
        "y": y,
        "idx": indices,
        "size": raw_sizes,
    })
