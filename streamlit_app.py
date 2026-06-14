import altair as alt
import numpy as np
import pandas as pd

# The spiral can hold up to 10,000 points (see the slider). Altair otherwise
# refuses to embed more than 5,000 rows inline; lift that cap since this data is
# small (a few numeric columns) and intentionally dense.
alt.data_transformers.disable_max_rows()

# Size encoding is pinned to a fixed domain/range so a given ``magnitude`` always
# maps to the same point size, regardless of how many points are drawn or how the
# random sample happened to come out. This is what keeps the size distribution
# stable across reruns and across small/single-turn/many-turn spirals.
SIZE_DOMAIN = (0.0, 1.0)
SIZE_RANGE = (15, 150)

# Single source of truth for what each column means. Rendered on the page so the
# visual encodings are self-explanatory, and asserted in the tests so the on-page
# explanation can never drift away from the actual data.
COLUMN_DOCS = {
    "x": "Horizontal position — where the point sits on the spiral.",
    "y": "Vertical position — where the point sits on the spiral.",
    "progress": (
        "How far along the spiral each point is (0 = center, 1 = outer edge). "
        "Drives the **color**."
    ),
    "magnitude": (
        "A per-point magnitude in the range [0, 1]. Drives the **size**, on a "
        "fixed scale so points stay comparable across reruns."
    ),
}


def build_spiral_data(num_points, num_turns, rng=None):
    """Build the spiral point cloud as a tidy dataframe.

    Columns are named for what they represent (see ``COLUMN_DOCS``):
    ``x``/``y`` are position, ``progress`` is the 0..1 position along the spiral
    (used for color), and ``magnitude`` is a non-negative, bounded per-point
    weight in [0, 1) (used for size).

    ``rng`` lets callers (notably the tests) inject a seeded generator for
    deterministic output; the app leaves it ``None`` for fresh randomness.
    """
    if rng is None:
        rng = np.random.default_rng()

    progress = np.linspace(0, 1, num_points)
    theta = 2 * np.pi * num_turns * progress
    radius = progress

    return pd.DataFrame({
        "x": radius * np.cos(theta),
        "y": radius * np.sin(theta),
        "progress": progress,
        # Uniform [0, 1): non-negative and bounded, so it is meaningful as a
        # "size" and pairs with the fixed SIZE_DOMAIN below.
        "magnitude": rng.random(num_points),
    })


def build_spiral_chart(df):
    """Render the spiral dataframe as an Altair point chart.

    Color encodes ``progress`` and size encodes ``magnitude`` on a fixed
    [0, 1] -> SIZE_RANGE scale, so the size mapping does not depend on the
    realized sample.
    """
    return (
        alt.Chart(df, height=700, width=700)
        .mark_point(filled=True)
        .encode(
            x=alt.X("x", axis=None),
            y=alt.Y("y", axis=None),
            color=alt.Color("progress", legend=None),
            size=alt.Size(
                "magnitude",
                legend=None,
                scale=alt.Scale(domain=list(SIZE_DOMAIN), range=list(SIZE_RANGE)),
            ),
        )
    )


def main():
    import streamlit as st

    st.markdown(
        """
# Welcome to Streamlit!

Edit `/streamlit_app.py` to customize this app to your heart's desire :heart:.
If you have any questions, checkout our [documentation](https://docs.streamlit.io) and [community
forums](https://discuss.streamlit.io).

In the meantime, below is an example of what you can do with just a few lines of code:
"""
    )

    num_points = st.slider("Number of points in spiral", 1, 10000, 1100)
    num_turns = st.slider("Number of turns in spiral", 1, 300, 31)

    df = build_spiral_data(num_points, num_turns)
    st.altair_chart(build_spiral_chart(df))

    st.markdown("#### What each field means")
    for column, description in COLUMN_DOCS.items():
        st.markdown(f"- **`{column}`** — {description}")


if __name__ == "__main__":
    main()
