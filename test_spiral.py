"""Validation for the spiral demo's data generation.

Focus: prove that identical parameters never re-jitter, that changing
parameters does update the data, and that extreme slider values stay sane.
The pure helpers in ``spiral.py`` need no Streamlit runtime to test.
"""
import numpy as np
import pandas as pd
import pytest

from spiral import (
    MAX_POINTS,
    MAX_TURNS,
    MIN_POINT_SIZE,
    MAX_POINT_SIZE,
    SIZE_CLIP,
    build_spiral_chart,
    generate_spiral_data,
)


def test_same_params_are_deterministic():
    # Two independent calls (i.e. two reruns) must be byte-for-byte identical:
    # no random jitter in point sizes or anything else.
    first = generate_spiral_data(1100, 31)
    second = generate_spiral_data(1100, 31)
    pd.testing.assert_frame_equal(first, second)


def test_changing_num_points_updates_data():
    a = generate_spiral_data(1100, 31)
    b = generate_spiral_data(1200, 31)
    assert len(a) == 1100
    assert len(b) == 1200
    assert not a.equals(b)


def test_changing_num_turns_updates_geometry_but_not_sizes():
    a = generate_spiral_data(1100, 31)
    b = generate_spiral_data(1100, 60)
    # Same seed + same length -> the size signal is stable across turn changes.
    np.testing.assert_array_equal(
        a["size_signal"].to_numpy(), b["size_signal"].to_numpy()
    )
    # ...but the spiral geometry must actually move.
    assert not np.allclose(a["x"].to_numpy(), b["x"].to_numpy())


@pytest.mark.parametrize("num_turns", [1, 31, MAX_TURNS])
@pytest.mark.parametrize("num_points", [1, 2, 10, 1000, MAX_POINTS])
def test_edge_cases_finite_and_bounded(num_points, num_turns):
    df = generate_spiral_data(num_points, num_turns)
    assert len(df) == num_points
    # No NaN/inf anywhere -> no display glitches.
    assert np.isfinite(df.to_numpy(dtype=float)).all()
    # Size signal stays bounded -> never invisible, never absurd after scaling.
    assert df["size_signal"].abs().max() <= SIZE_CLIP + 1e-9
    # Spiral stays inside the unit disc regardless of turns.
    assert df["x"].abs().max() <= 1 + 1e-9
    assert df["y"].abs().max() <= 1 + 1e-9


def test_out_of_range_params_are_clamped():
    assert len(generate_spiral_data(0, 0)) == 1
    assert len(generate_spiral_data(99999, 9999)) == MAX_POINTS


def test_single_point_renders_at_center():
    df = generate_spiral_data(1, 50)
    assert len(df) == 1
    assert df["x"].iloc[0] == 0.0
    assert df["y"].iloc[0] == 0.0
    assert np.isfinite(df["size_signal"].iloc[0])


def test_distinct_seeds_diverge():
    a = generate_spiral_data(500, 20, seed=0)
    b = generate_spiral_data(500, 20, seed=1)
    assert not np.array_equal(
        a["size_signal"].to_numpy(), b["size_signal"].to_numpy()
    )


def test_chart_uses_fixed_bounded_size_scale():
    chart = build_spiral_chart(generate_spiral_data(50, 10))
    spec = chart.to_dict()  # raises if the Altair spec is invalid
    size_scale = spec["encoding"]["size"]["scale"]
    assert size_scale["range"] == [MIN_POINT_SIZE, MAX_POINT_SIZE]
    assert size_scale["domain"] == [-SIZE_CLIP, SIZE_CLIP]
