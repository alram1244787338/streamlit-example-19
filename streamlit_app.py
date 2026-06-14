import altair as alt
import streamlit as st

from spiral_core import compute_spiral, prepare_for_render, DISPLAY_THRESHOLD

"""
# Welcome to Streamlit!

Edit `/streamlit_app.py` to customize this app to your heart's desire :heart:.
If you have any questions, checkout our [documentation](https://docs.streamlit.io) and [community
forums](https://discuss.streamlit.io).

In the meantime, below is an example of what you can do with just a few lines of code:
"""

# ---------------------------------------------------------------------------
# Sidebar controls — wrapped in a form so the chart only re-renders
# when the user explicitly clicks "Generate chart", avoiding constant
# recomputation while dragging sliders.
# ---------------------------------------------------------------------------
with st.sidebar:
    with st.form("params"):
        num_points = st.slider(
            "Number of points in spiral", 1, 10000, 1100,
            help="Total data points computed. Values above "
                 f"{DISPLAY_THRESHOLD:,} are downsampled for display.",
        )
        num_turns = st.slider("Number of turns in spiral", 1, 300, 31)
        submitted = st.form_submit_button("Generate chart")

# ---------------------------------------------------------------------------
# Compute & render
# ---------------------------------------------------------------------------
with st.spinner("Generating spiral…"):
    full_df = compute_spiral(num_points, num_turns)
    render_df = prepare_for_render(full_df)

requested = len(full_df)
rendered = len(render_df)

# Status line — makes it obvious whether downsampling is in effect,
# so the user never wonders if the page is stuck or why the chart
# looks slightly different at high point counts.
if rendered < requested:
    st.info(
        f"Computed **{requested:,}** points; displaying **{rendered:,}** "
        f"(downsampled above {DISPLAY_THRESHOLD:,} for responsiveness). "
        "The spiral shape is preserved."
    )
else:
    st.caption(f"Displaying all **{rendered:,}** points.")

# ---------------------------------------------------------------------------
# Altair chart — identical encoding to the original demo
# ---------------------------------------------------------------------------
chart = (
    alt.Chart(render_df, height=700, width=700)
    .mark_point(filled=True)
    .encode(
        x=alt.X("x", axis=None),
        y=alt.Y("y", axis=None),
        color=alt.Color("idx", legend=None, scale=alt.Scale()),
        size=alt.Size("rand", legend=None, scale=alt.Scale(range=[1, 150])),
    )
)
st.altair_chart(chart)
