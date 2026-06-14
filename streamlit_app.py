import time

import streamlit as st

from src.chart import build_spiral_chart
from src.data_generator import generate_spiral_data

st.title("Spiral Scatter Plot")

# ── Sidebar controls ────────────────────────────────────────────────
num_points = st.sidebar.slider("Number of points in spiral", 1, 10000, 1100)
num_turns = st.sidebar.slider("Number of turns in spiral", 1, 300, 31)

# ── Data generation (cached by parameter pair) ──────────────────────
# A flag set *inside* the cached function would not survive the rerun
# boundary cleanly, so we use a wall-clock proxy: if the call returns
# in < 5 ms it was a cache hit; otherwise it was freshly computed.
_t0 = time.perf_counter()
df = generate_spiral_data(num_points, num_turns)
_elapsed_ms = (time.perf_counter() - _t0) * 1000

_cache_hit = _elapsed_ms < 5
_prev_key = st.session_state.get("_last_param_key")
_curr_key = (num_points, num_turns)
if _prev_key != _curr_key:
    st.session_state["_last_param_key"] = _curr_key
    st.session_state["_was_regenerated"] = True
else:
    st.session_state["_was_regenerated"] = False

# ── Status indicator ────────────────────────────────────────────────
if st.session_state.get("_was_regenerated"):
    st.info("Parameters changed — data regenerated.")
elif _cache_hit:
    st.success("Using cached data (same parameters, no regeneration).")
else:
    st.info("Data loaded.")

st.caption(
    f"Points: **{num_points}** · Turns: **{num_turns}** · "
    f"Elapsed: {_elapsed_ms:.1f} ms"
)

# ── Chart ────────────────────────────────────────────────────────────
chart = build_spiral_chart(df)
st.altair_chart(chart)
