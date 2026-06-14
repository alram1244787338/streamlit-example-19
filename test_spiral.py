"""Validation for the spiral demo.

Covers two parameter tiers explicitly:

* a *common* range (~1000 points) -- proves the fix did not coarsen the
  everyday experience: every point is still drawn.
* a *high-stress* range (10000 points, 300 turns) -- proves the demo degrades
  gracefully (display density is capped so the chart can actually render and
  stay interactive) without the figure going wrong.

The tests import only the pure ``spiral`` module, so no Streamlit runtime is
needed. Run with ``pytest`` or directly: ``python test_spiral.py``.
"""

from __future__ import annotations

import time

import numpy as np

from spiral import (
    DEFAULT_SEED,
    MAX_DISPLAY_POINTS,
    compute_spiral,
    downsample_for_display,
    prepare_spiral,
)

COMMON_POINTS = 1100
COMMON_TURNS = 31
STRESS_POINTS = 10000
STRESS_TURNS = 300


def _spiral_xy(idx: np.ndarray, num_turns: int):
    theta = 2 * np.pi * num_turns * idx
    return idx * np.cos(theta), idx * np.sin(theta)


# --- common range: detail must be preserved -------------------------------- #

def test_common_range_keeps_full_detail():
    result = prepare_spiral(COMMON_POINTS, COMMON_TURNS)
    assert not result.downsampled
    assert result.computed_points == COMMON_POINTS
    assert result.displayed_points == COMMON_POINTS
    assert len(result.data) == COMMON_POINTS


def test_common_range_is_genuine_spiral():
    result = prepare_spiral(COMMON_POINTS, COMMON_TURNS)
    data = result.data
    # radius == idx, and x/y reconstruct from the parametric definition.
    exp_x, exp_y = _spiral_xy(data["idx"].to_numpy(), COMMON_TURNS)
    assert np.allclose(data["x"].to_numpy(), exp_x)
    assert np.allclose(data["y"].to_numpy(), exp_y)


# --- high-stress range: graceful degradation ------------------------------- #

def test_stress_range_caps_display_density():
    result = prepare_spiral(STRESS_POINTS, STRESS_TURNS)
    # Computation layer stays honest: the full data is really produced.
    assert result.computed_points == STRESS_POINTS
    # Display layer is capped so the browser can render it.
    assert result.downsampled
    assert result.displayed_points <= MAX_DISPLAY_POINTS
    assert len(result.data) == result.displayed_points


def test_stress_range_preserves_shape_after_thinning():
    result = prepare_spiral(STRESS_POINTS, STRESS_TURNS)
    data = result.data
    # The thinned points still lie exactly on the same spiral curve.
    exp_x, exp_y = _spiral_xy(data["idx"].to_numpy(), STRESS_TURNS)
    assert np.allclose(data["x"].to_numpy(), exp_x)
    assert np.allclose(data["y"].to_numpy(), exp_y)
    # Full radial span is retained (not cropped): first ~0, last ~1.
    assert data["idx"].iloc[0] == 0.0
    assert abs(data["idx"].iloc[-1] - 1.0) < 1e-9


def test_stress_path_is_fast_enough():
    # Guards the *server-side* main path against hanging / accidental blow-ups.
    start = time.perf_counter()
    result = prepare_spiral(STRESS_POINTS, STRESS_TURNS)
    elapsed = time.perf_counter() - start
    assert result.displayed_points <= MAX_DISPLAY_POINTS
    assert elapsed < 2.0, f"prepare_spiral too slow: {elapsed:.3f}s"


# --- honesty: reported counts match what is rendered ----------------------- #

def test_reported_counts_match_data():
    for n in (1, 500, COMMON_POINTS, 2000, 5000, STRESS_POINTS):
        result = prepare_spiral(n, COMMON_TURNS)
        assert result.computed_points == n
        assert result.displayed_points == len(result.data)
        assert result.downsampled == (n > MAX_DISPLAY_POINTS)


# --- determinism: no flicker, reproducible figure -------------------------- #

def test_compute_is_deterministic():
    a = compute_spiral(COMMON_POINTS, COMMON_TURNS, seed=DEFAULT_SEED)
    b = compute_spiral(COMMON_POINTS, COMMON_TURNS, seed=DEFAULT_SEED)
    assert a.equals(b)
    # The "rand" size channel is what used to re-roll every rerun.
    assert np.array_equal(a["rand"].to_numpy(), b["rand"].to_numpy())


# --- edges ----------------------------------------------------------------- #

def test_minimum_points():
    result = prepare_spiral(1, 1)
    assert result.computed_points == 1
    assert result.displayed_points == 1
    assert not result.downsampled


def test_just_above_threshold_downsamples():
    result = prepare_spiral(MAX_DISPLAY_POINTS + 1, COMMON_TURNS)
    assert result.downsampled
    assert result.displayed_points <= MAX_DISPLAY_POINTS


def test_at_threshold_keeps_all():
    result = prepare_spiral(MAX_DISPLAY_POINTS, COMMON_TURNS)
    assert not result.downsampled
    assert result.displayed_points == MAX_DISPLAY_POINTS


def test_invalid_inputs_raise():
    for bad in ((0, 10), (10, 0), (-5, 10)):
        try:
            compute_spiral(*bad)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {bad}")


def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failures = 0
    for fn in tests:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001 - surface any failure
            failures += 1
            print(f"FAIL {fn.__name__}: {exc}")
        else:
            print(f"PASS {fn.__name__}")

    # Print the concrete high-stress numbers as visible evidence.
    stress = prepare_spiral(STRESS_POINTS, STRESS_TURNS)
    start = time.perf_counter()
    prepare_spiral(STRESS_POINTS, STRESS_TURNS)
    elapsed = time.perf_counter() - start
    print(
        f"\nhigh-stress: computed {stress.computed_points:,} -> "
        f"displayed {stress.displayed_points:,} "
        f"(downsampled={stress.downsampled}) in {elapsed * 1000:.2f} ms"
    )
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return failures


if __name__ == "__main__":
    raise SystemExit(1 if _run_all() else 0)
