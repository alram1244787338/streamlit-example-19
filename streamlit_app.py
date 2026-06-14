import hashlib

import streamlit as st

from src.chart import build_spiral_chart
from src.data_generator import generate_spiral_data

st.title("Spiral Scatter Plot")

# ── Sidebar controls ────────────────────────────────────────────────
num_points = st.sidebar.slider("Number of points in spiral", 1, 10000, 1100)
num_turns = st.sidebar.slider("Number of turns in spiral", 1, 300, 31)

# ── Parameter-change detection via session_state ────────────────────
# session_state persists across Streamlit reruns within the same user
# session, so comparing the current (num_points, num_turns) tuple to the
# previously stored value gives a reliable "did the user actually move a
# slider?" signal — without any timing heuristics.
_curr_key = (num_points, num_turns)
_prev_key = st.session_state.get("_last_param_key")

if _prev_key != _curr_key:
    st.session_state["_last_param_key"] = _curr_key
    _regenerated = True
else:
    _regenerated = False

# ── Data generation (cached by parameter pair) ──────────────────────
df = generate_spiral_data(num_points, num_turns)

# ── Data fingerprint ────────────────────────────────────────────────
# A short, deterministic hash of the DataFrame content.  Users can
# visually confirm "same params → same fingerprint → same picture"
# without having to eyeball the scatter plot.
_fingerprint = hashlib.sha256(
    df[["x", "y", "size"]].values.tobytes()
).hexdigest()[:8]

# ── Status indicator ────────────────────────────────────────────────
if _regenerated:
    st.info("Parameters changed — data regenerated.")
else:
    st.success("Using cached data (same parameters, no regeneration).")

st.caption(
    f"Points: **{num_points}** · Turns: **{num_turns}** · "
    f"Fingerprint: `{_fingerprint}`"
)

# ── Chart ────────────────────────────────────────────────────────────
chart = build_spiral_chart(df)
st.altair_chart(chart)
