# Streamlit spiral demo.
#
# Data generation and chart construction live in spiral.py (pure + testable).
# This module owns the page only: sliders, caching, and the cache/stable
# indicator. Add new UI here; add new data/chart behavior in spiral.py.
from datetime import datetime

import streamlit as st

from spiral import (
    MAX_POINTS,
    MAX_TURNS,
    MIN_POINTS,
    MIN_TURNS,
    build_spiral_chart,
    generate_spiral_data,
)

"""
# Welcome to Streamlit!

Edit `/streamlit_app.py` to customize this app to your heart's desire :heart:.
If you have any questions, checkout our [documentation](https://docs.streamlit.io) and [community
forums](https://discuss.streamlit.io).

In the meantime, below is an example of what you can do with just a few lines of code:
"""


@st.cache_data(show_spinner=False)
def load_spiral_data(num_points, num_turns):
    """Cache the spiral data per (num_points, num_turns).

    Streamlit only runs this body when the arguments change, so a bare rerun
    reuses the cached frame untouched -- sizes stay put. ``generated_at``
    records when the frame was actually computed: it is identical across cache
    hits and only advances on a cache miss, which powers the indicator below.
    """
    df = generate_spiral_data(num_points, num_turns)
    return df, datetime.now()


num_points = st.slider("Number of points in spiral", MIN_POINTS, MAX_POINTS, 1100)
num_turns = st.slider("Number of turns in spiral", MIN_TURNS, MAX_TURNS, 31)

df, generated_at = load_spiral_data(num_points, num_turns)

# Compare against the previous run to tell the user whether they are looking at
# freshly regenerated data or the stable cached frame.
signature = (num_points, num_turns)
params_changed = st.session_state.get("spiral_signature") != signature
st.session_state["spiral_signature"] = signature

if params_changed:
    st.success(
        f"Regenerated data for {num_points} points / {num_turns} turns "
        f"(generated {generated_at:%H:%M:%S})."
    )
else:
    st.info(
        f"Stable cached data — same parameters, point sizes unchanged "
        f"(generated {generated_at:%H:%M:%S}; timestamp stays frozen on rerun)."
    )

st.altair_chart(build_spiral_chart(df))
