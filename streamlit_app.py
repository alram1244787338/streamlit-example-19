import altair as alt
import streamlit as st

from spiral import MAX_DISPLAY_POINTS, compute_spiral, downsample_for_display

"""
# Welcome to Streamlit!

Edit `/streamlit_app.py` to customize this app to your heart's desire :heart:.
If you have any questions, checkout our [documentation](https://docs.streamlit.io) and [community
forums](https://discuss.streamlit.io).

In the meantime, below is an example of what you can do with just a few lines of code:
"""


@st.cache_data(show_spinner=False)
def cached_spiral(num_points: int, num_turns: int):
    """Cache the (deterministic) full-resolution computation layer.

    Re-dragging a slider back to a value already seen returns instantly instead
    of recomputing, which keeps the app from feeling stuck under heavy use.
    """
    return compute_spiral(num_points, num_turns)


def build_chart(data):
    """Build the spiral chart. Visually identical to the original demo."""
    return (
        alt.Chart(data, height=700, width=700)
        .mark_point(filled=True)
        .encode(
            x=alt.X("x", axis=None),
            y=alt.Y("y", axis=None),
            color=alt.Color("idx", legend=None, scale=alt.Scale()),
            size=alt.Size("rand", legend=None, scale=alt.Scale(range=[1, 150])),
        )
    )


num_points = st.slider("Number of points in spiral", 1, 10000, 1100)
num_turns = st.slider("Number of turns in spiral", 1, 300, 31)

# Status keeps the user informed at every stage so a busy app is never mistaken
# for a hung one.
with st.status("Building spiral…", expanded=False) as status:
    status.update(label=f"Computing {num_points:,} points…")
    full_df = cached_spiral(num_points, num_turns)

    result = downsample_for_display(full_df, MAX_DISPLAY_POINTS)

    if result.downsampled:
        status.update(
            label=(
                f"Rendering {result.displayed_points:,} of "
                f"{result.computed_points:,} points…"
            )
        )
    else:
        status.update(label=f"Rendering {result.displayed_points:,} points…")

    chart = build_chart(result.data)
    status.update(label="Spiral ready", state="complete")

st.altair_chart(chart)

# Honest accounting of compute vs. display, so the on-screen result always
# matches the chosen parameters -- even when the display layer thins the data.
if result.downsampled:
    st.caption(
        f"Computed **{result.computed_points:,}** points · "
        f"displaying **{result.displayed_points:,}** "
        f"(thinned to keep rendering smooth). "
        f"Reduce the point count below {MAX_DISPLAY_POINTS:,} to draw every point."
    )
else:
    st.caption(
        f"Computed and displaying all **{result.displayed_points:,}** points "
        f"(full detail)."
    )
