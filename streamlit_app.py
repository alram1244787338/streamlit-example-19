import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

"""
# Welcome to Streamlit!

Edit `/streamlit_app.py` to customize this app to your heart's desire :heart:.
If you have any questions, checkout our [documentation](https://docs.streamlit.io) and [community
forums](https://discuss.streamlit.io).

In the meantime, below is an example of what you can do with just a few lines of code:
"""


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def generate_spiral_data(num_points: int, num_turns: int, seed: int = 42) -> pd.DataFrame:
    """Generate an Archimedean spiral with per-point visual attributes.

    Parameters
    ----------
    num_points : int
        Total number of points along the spiral.
    num_turns : int
        How many full 360-degree turns the spiral makes.
    seed : int
        Random seed so that point sizes stay stable across re-renders.

    Returns
    -------
    pd.DataFrame
        Columns: x, y, progress, point_size
    """
    rng = np.random.default_rng(seed)

    progress = np.linspace(0.0, 1.0, num_points)
    theta = 2.0 * np.pi * num_turns * progress
    radius = progress

    x = radius * np.cos(theta)
    y = radius * np.sin(theta)

    # Uniform [0, 1] gives a stable, always-positive size metric.
    point_size = rng.random(num_points)

    return pd.DataFrame({
        "x": x,
        "y": y,
        "progress": progress,
        "point_size": point_size,
    })


# ---------------------------------------------------------------------------
# Chart rendering
# ---------------------------------------------------------------------------

def create_spiral_chart(df: pd.DataFrame) -> alt.Chart:
    """Render the spiral DataFrame as an Altair scatter chart.

    The encoding uses:
    * **x / y** – Cartesian position on the spiral plane (axes hidden).
    * **progress** – how far along the spiral a point sits (0 → 1),
      mapped to colour so the spiral direction is visible.
    * **point_size** – a uniform random weight in [0, 1] that controls
      marker radius, giving each point a slightly different visual
      emphasis without extreme outliers.
    """
    return (
        alt.Chart(df, height=700, width=700)
        .mark_point(filled=True)
        .encode(
            x=alt.X("x", axis=None, title=None),
            y=alt.Y("y", axis=None, title=None),
            color=alt.Color(
                "progress:Q",
                legend=alt.Legend(title="Progress along spiral"),
                scale=alt.Scale(scheme="turbo"),
            ),
            size=alt.Size(
                "point_size:Q",
                legend=alt.Legend(title="Point size (random weight)"),
                scale=alt.Scale(domain=[0.0, 1.0], range=[10, 150]),
            ),
        )
        .properties(title="Archimedean Spiral")
    )


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

num_points = st.slider("Number of points in spiral", 1, 10000, 1100)
num_turns = st.slider("Number of turns in spiral", 1, 300, 31)

df = generate_spiral_data(num_points, num_turns)
chart = create_spiral_chart(df)
st.altair_chart(chart)

with st.expander("About the data fields"):
    st.markdown(
        """
| Field | Range | Used for | Description |
|---|---|---|---|
| **x, y** | varies | position | Cartesian coordinates of each point on the spiral. |
| **progress** | 0 → 1 | colour | How far along the spiral the point sits; 0 is the centre, 1 is the outermost tip. |
| **point_size** | 0 → 1 | marker size | A uniform random weight that gives each point a slightly different visual emphasis. Always non-negative so no points collapse to invisible. |
"""
    )
