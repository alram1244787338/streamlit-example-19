"""Regression tests for the spiral chart app.

Covers:
* Data generation correctness and stability
* Point-size distribution sanity (no negative values, bounded range)
* Edge cases (1 point, single turn, many turns)
* Chart encoding references the right fields
* Source-level check that the field-description table matches the data
"""

import sys
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import pytest

# Allow importing the app module without installing it.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from streamlit_app import create_spiral_chart, generate_spiral_data  # noqa: E402

# ---------------------------------------------------------------------------
# generate_spiral_data
# ---------------------------------------------------------------------------


class TestGenerateSpiralData:
    """Tests for the data-generation helper."""

    def test_columns(self):
        df = generate_spiral_data(100, 5)
        assert list(df.columns) == ["x", "y", "progress", "point_size"]

    def test_shape(self):
        df = generate_spiral_data(200, 10)
        assert len(df) == 200

    def test_progress_range(self):
        df = generate_spiral_data(500, 3)
        assert df["progress"].min() == pytest.approx(0.0)
        assert df["progress"].max() == pytest.approx(1.0)

    def test_point_size_non_negative(self):
        """The old code used randn() which produces negatives; this must not."""
        df = generate_spiral_data(10_000, 31)
        assert (df["point_size"] >= 0.0).all(), "point_size must be >= 0"

    def test_point_size_bounded(self):
        df = generate_spiral_data(10_000, 31)
        assert (df["point_size"] <= 1.0).all(), "point_size must be <= 1"

    def test_point_size_has_variance(self):
        """Sizes should not all be the same constant."""
        df = generate_spiral_data(500, 5)
        assert df["point_size"].std() > 0.05, "point_size should vary"

    def test_point_size_distribution_reasonable(self):
        """Check the distribution isn't heavily skewed to near-zero or near-one.

        With a uniform [0, 1] draw, the mean should be ~0.5 and the fraction
        of points below 0.1 should be roughly 10 %.
        """
        df = generate_spiral_data(5_000, 10)
        assert 0.3 < df["point_size"].mean() < 0.7
        frac_tiny = (df["point_size"] < 0.1).mean()
        assert frac_tiny < 0.20, f"Too many near-invisible points: {frac_tiny:.1%}"

    def test_deterministic_with_same_seed(self):
        a = generate_spiral_data(100, 5, seed=7)
        b = generate_spiral_data(100, 5, seed=7)
        pd.testing.assert_frame_equal(a, b)

    def test_different_seed_gives_different_sizes(self):
        a = generate_spiral_data(100, 5, seed=1)
        b = generate_spiral_data(100, 5, seed=2)
        assert not np.allclose(a["point_size"], b["point_size"])

    # -- edge cases ----------------------------------------------------------

    def test_single_point(self):
        df = generate_spiral_data(1, 1)
        assert len(df) == 1
        assert df["progress"].iloc[0] == pytest.approx(0.0)
        assert 0.0 <= df["point_size"].iloc[0] <= 1.0

    def test_single_turn(self):
        df = generate_spiral_data(100, 1)
        assert len(df) == 100

    def test_many_turns(self):
        df = generate_spiral_data(100, 300)
        assert len(df) == 100
        # Spiral should still span [0, 1] progress
        assert df["progress"].max() == pytest.approx(1.0)

    def test_large_point_count(self):
        df = generate_spiral_data(10_000, 50)
        assert len(df) == 10_000


# ---------------------------------------------------------------------------
# create_spiral_chart
# ---------------------------------------------------------------------------


class TestCreateSpiralChart:
    """Tests for the chart-rendering helper."""

    @pytest.fixture()
    def sample_df(self):
        return generate_spiral_data(50, 3)

    def test_returns_altair_chart(self, sample_df):
        chart = create_spiral_chart(sample_df)
        assert isinstance(chart, alt.Chart)

    def test_chart_encodes_progress_as_color(self, sample_df):
        chart = create_spiral_chart(sample_df)
        encoding = chart.encoding
        assert encoding.color.field == "progress" or "progress" in str(
            encoding.color
        )

    def test_chart_encodes_point_size(self, sample_df):
        chart = create_spiral_chart(sample_df)
        encoding = chart.encoding
        assert encoding.size.field == "point_size" or "point_size" in str(
            encoding.size
        )

    def test_chart_has_legend_for_color(self, sample_df):
        chart = create_spiral_chart(sample_df)
        # legend should not be None (we want a visible legend)
        assert chart.encoding.color.legend is not None

    def test_chart_has_legend_for_size(self, sample_df):
        chart = create_spiral_chart(sample_df)
        assert chart.encoding.size.legend is not None

    def test_chart_has_title(self, sample_df):
        chart = create_spiral_chart(sample_df)
        assert chart.title is not None and len(chart.title) > 0


# ---------------------------------------------------------------------------
# Source-level sanity check for the field-description table
# ---------------------------------------------------------------------------


class TestFieldDescriptions:
    """Ensure the source file documents every DataFrame column."""

    def test_all_columns_documented(self):
        source = Path(__file__).resolve().parent.parent / "streamlit_app.py"
        text = source.read_text()
        for col in ("x", "y", "progress", "point_size"):
            assert col in text, f"Column '{col}' not mentioned in source"
