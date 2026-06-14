"""Tests for spiral_core module.

Covers two tiers:
  - Typical parameters (<= 3000 points): verifies full-fidelity output.
  - High-pressure parameters (up to 10000 points): verifies downsampling,
    performance bounds, and end-to-end stability.
"""

import time

import numpy as np
import pandas as pd
import pytest

from spiral_core import (
    DISPLAY_THRESHOLD,
    compute_spiral,
    prepare_for_render,
)

EXPECTED_COLUMNS = {"x", "y", "idx", "rand"}


# ---------------------------------------------------------------------------
# compute_spiral
# ---------------------------------------------------------------------------
class TestComputeSpiral:
    """Core geometry and data-shape guarantees."""

    def test_columns_present(self):
        df = compute_spiral(100, 5)
        assert EXPECTED_COLUMNS.issubset(df.columns)

    def test_row_count_matches_request(self):
        for n in (1, 50, 1000, 2999, 3000, 5000, 10000):
            assert len(compute_spiral(n, 10)) == n

    def test_spiral_starts_at_origin(self):
        df = compute_spiral(200, 10)
        assert abs(df["x"].iloc[0]) < 1e-12
        assert abs(df["y"].iloc[0]) < 1e-12

    def test_idx_range_zero_to_one(self):
        df = compute_spiral(500, 10)
        assert df["idx"].iloc[0] == pytest.approx(0.0)
        assert df["idx"].iloc[-1] == pytest.approx(1.0)

    def test_single_turn_unit_circle_endpoint(self):
        """After exactly one full turn the spiral should return near (1, 0)."""
        df = compute_spiral(1000, 1)
        assert df["x"].iloc[-1] == pytest.approx(1.0, abs=0.01)
        assert df["y"].iloc[-1] == pytest.approx(0.0, abs=0.01)

    def test_seed_reproducibility(self):
        a = compute_spiral(200, 5, seed=42)
        b = compute_spiral(200, 5, seed=42)
        pd.testing.assert_frame_equal(a, b)

    def test_different_seeds_differ(self):
        a = compute_spiral(200, 5, seed=1)
        b = compute_spiral(200, 5, seed=2)
        # Geometry columns are identical; only 'rand' should differ.
        pd.testing.assert_series_equal(a["x"], b["x"])
        pd.testing.assert_series_equal(a["y"], b["y"])
        assert not (a["rand"].values == b["rand"].values).all()

    def test_high_pressure_10000(self):
        """10 000 points is the slider maximum; must not raise."""
        df = compute_spiral(10000, 300)
        assert len(df) == 10000

    def test_one_point(self):
        df = compute_spiral(1, 1)
        assert len(df) == 1

    def test_no_nan_or_inf(self):
        df = compute_spiral(10000, 300)
        assert not df.isnull().any().any()
        assert np.all(np.isfinite(df[["x", "y", "idx"]].values))


# ---------------------------------------------------------------------------
# prepare_for_render
# ---------------------------------------------------------------------------
class TestPrepareForRender:
    """Downsampling logic."""

    @pytest.mark.parametrize("n", [1, 100, 1000, 2999, 3000])
    def test_no_downsample_under_threshold(self, n):
        df = compute_spiral(n, 10)
        result = prepare_for_render(df)
        assert len(result) == n

    def test_downsample_above_threshold(self):
        df = compute_spiral(10000, 30)
        result = prepare_for_render(df)
        assert len(result) <= DISPLAY_THRESHOLD

    def test_downsampled_result_is_copy(self):
        df = compute_spiral(10000, 10)
        result = prepare_for_render(df)
        original_first_x = df["x"].iloc[0]
        result.loc[result.index[0], "x"] = 999.0
        assert df["x"].iloc[0] == original_first_x

    def test_columns_preserved_after_downsample(self):
        df = compute_spiral(10000, 30)
        result = prepare_for_render(df)
        assert set(result.columns) == EXPECTED_COLUMNS

    def test_idx_monotonic_after_downsample(self):
        df = compute_spiral(8000, 50)
        result = prepare_for_render(df)
        diffs = result["idx"].diff().dropna()
        assert (diffs > 0).all(), "idx must remain strictly increasing"

    def test_custom_max_display(self):
        df = compute_spiral(5000, 10)
        result = prepare_for_render(df, max_display=500)
        assert len(result) <= 500

    def test_exact_threshold_no_downsample(self):
        """Exactly at threshold should NOT trigger downsampling."""
        df = compute_spiral(DISPLAY_THRESHOLD, 10)
        result = prepare_for_render(df)
        assert len(result) == DISPLAY_THRESHOLD

    def test_just_above_threshold_downsamples(self):
        df = compute_spiral(DISPLAY_THRESHOLD + 1, 10)
        result = prepare_for_render(df)
        assert len(result) <= DISPLAY_THRESHOLD

    def test_returns_copy_for_small_df(self):
        df = compute_spiral(100, 5)
        result = prepare_for_render(df)
        original_first_x = df["x"].iloc[0]
        result.loc[result.index[0], "x"] = 999.0
        assert df["x"].iloc[0] == original_first_x


# ---------------------------------------------------------------------------
# End-to-end integration
# ---------------------------------------------------------------------------
class TestEndToEnd:
    """Full pipeline: compute → render-prep."""

    def test_typical_usage(self):
        """Default slider values: 1100 points, 31 turns."""
        df = compute_spiral(1100, 31)
        result = prepare_for_render(df)
        assert len(result) == 1100  # no downsampling

    def test_max_points_max_turns(self):
        """Absolute worst case: 10000 points, 300 turns."""
        df = compute_spiral(10000, 300)
        result = prepare_for_render(df)
        assert 0 < len(result) <= DISPLAY_THRESHOLD
        assert not result.isnull().any().any()

    def test_boundary_just_over_threshold(self):
        df = compute_spiral(3001, 50)
        result = prepare_for_render(df)
        assert len(result) <= DISPLAY_THRESHOLD

    def test_spiral_shape_preserved_after_downsample(self):
        """The first and last points must survive downsampling."""
        df = compute_spiral(10000, 10)
        result = prepare_for_render(df)
        # First point
        assert result["x"].iloc[0] == pytest.approx(df["x"].iloc[0], abs=1e-9)
        assert result["y"].iloc[0] == pytest.approx(df["y"].iloc[0], abs=1e-9)
        # idx should still span the full [0, 1] range
        assert result["idx"].iloc[0] == pytest.approx(0.0)
        assert result["idx"].iloc[-1] == pytest.approx(1.0, abs=0.01)


# ---------------------------------------------------------------------------
# Performance
# ---------------------------------------------------------------------------
class TestPerformance:
    """Wall-clock bounds to catch severe regressions."""

    def test_compute_10000_under_2s(self):
        start = time.perf_counter()
        compute_spiral(10000, 300)
        assert time.perf_counter() - start < 2.0

    def test_prepare_10000_under_1s(self):
        df = compute_spiral(10000, 300)
        start = time.perf_counter()
        prepare_for_render(df)
        assert time.perf_counter() - start < 1.0

    def test_full_pipeline_10000_under_3s(self):
        start = time.perf_counter()
        df = compute_spiral(10000, 300)
        prepare_for_render(df)
        assert time.perf_counter() - start < 3.0

    def test_downsampled_size_bounded(self):
        """Downsampled output should be meaningfully smaller than input."""
        df = compute_spiral(10000, 100)
        result = prepare_for_render(df)
        assert len(result) <= DISPLAY_THRESHOLD
        assert len(result) < len(df)
