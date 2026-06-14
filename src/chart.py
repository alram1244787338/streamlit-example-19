"""Altair chart configuration for the spiral scatter plot."""

import altair as alt
import pandas as pd


def build_spiral_chart(df: pd.DataFrame) -> alt.Chart:
    """Build a filled-point spiral chart with colour gradient and sized markers.

    The size encoding uses the 'size' column (range [0.3, 1.0]) and maps
    it to a pixel range of [10, 60], keeping markers visible but bounded
    regardless of num_points.
    """
    return (
        alt.Chart(df, height=700, width=700)
        .mark_point(filled=True)
        .encode(
            x=alt.X("x", axis=None),
            y=alt.Y("y", axis=None),
            color=alt.Color("idx", legend=None, scale=alt.Scale()),
            size=alt.Size(
                "size",
                legend=None,
                scale=alt.Scale(domain=[0.3, 1.0], range=[10, 60]),
            ),
        )
    )
