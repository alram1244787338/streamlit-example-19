"""Tests for spiral data generation and chart building.

Verifies:
- Same parameters → identical data (no random jitter).
- Changed parameters → different data.
- Edge cases (num_points=1, num_points=10000, num_turns=300).
- Size column stays within [0.3, 1.0].
- Chart object is built correctly.
"""

import sys
from pathlib import Path

# Allow tests to import src without installing the package.
_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

# Mock streamlit.cache_data so the decorator becomes a no-op at test time.
# Without this, pytest would need a live Streamlit runtime.
import streamlit  # noqa: E402

streamlit.cache_data = lambda show_spinner=False: lambda fn: fn  # type: ignore[attr-defined]

from src.data_generator import generate_spiral_data, _make_seed  # noqa: E402
from src.chart import build_spiral_chart  # noqa: E402


# ── Stability ────────────────────────────────────────────────────────

def test_same_params_produce_identical_data():
    """The core bug fix: repeated calls must not jitter."""
    df1 = generate_spiral_data(500, 10)
    df2 = generate_spiral_data(500, 10)
    assert df1.equals(df2), "Data changed between identical calls"


def test_different_params_produce_different_data():
    """Changing a parameter must actually regenerate data."""
    df1 = generate_spiral_data(500, 10)
    df2 = generate_spiral_data(600, 10)
    assert not df1.equals(df2), "Different num_points gave same data"

    df3 = generate_spiral_data(500, 20)
    assert not df1.equals(df3), "Different num_turns gave same data"


# ── Seed determinism ────────────────────────────────────────────────

def test_seed_is_deterministic():
    assert _make_seed(100, 5) == _make_seed(100, 5)
    assert _make_seed(100, 5) != _make_seed(100, 6)


# ── Edge cases ──────────────────────────────────────────────────────

def test_single_point():
    df = generate_spiral_data(1, 1)
    assert len(df) == 1
    assert df["size"].iloc[0] >= 0.3
    assert df["size"].iloc[0] <= 1.0


def test_max_points():
    df = generate_spiral_data(10000, 31)
    assert len(df) == 10000


def test_max_turns():
    df = generate_spiral_data(100, 300)
    assert len(df) == 100


# ── Size bounds ─────────────────────────────────────────────────────

def test_sizes_within_expected_range():
    df = generate_spiral_data(1000, 31)
    assert df["size"].min() >= 0.3
    assert df["size"].max() <= 1.0


def test_required_columns_present():
    df = generate_spiral_data(10, 1)
    assert set(df.columns) == {"x", "y", "idx", "size"}


# ── Chart ───────────────────────────────────────────────────────────

def test_chart_builds_without_error():
    df = generate_spiral_data(100, 5)
    chart = build_spiral_chart(df)
    # Altair charts expose a to_dict() method; if the config is broken
    # this will raise.
    spec = chart.to_dict()
    assert spec["mark"]["type"] == "point"
    assert spec["mark"]["filled"] is True


# ── Supplementary: data quality ─────────────────────────────────────

def test_sizes_are_not_all_identical():
    """RNG should produce variation, not a constant column."""
    df = generate_spiral_data(200, 5)
    assert df["size"].nunique() > 1, "All sizes are identical"


def test_no_nan_in_generated_data():
    df = generate_spiral_data(1000, 31)
    assert not df.isnull().any().any(), "DataFrame contains NaN values"


def test_dataframe_row_count_matches_num_points():
    for n in (1, 50, 5000):
        df = generate_spiral_data(n, 10)
        assert len(df) == n


def test_spiral_starts_at_origin():
    """idx=0 → radius=0 → (x, y) = (0, 0)."""
    df = generate_spiral_data(100, 5)
    first = df.iloc[0]
    assert first["x"] == 0.0
    assert first["y"] == 0.0


# ── Supplementary: stability stress test ────────────────────────────

def test_ten_consecutive_calls_are_identical():
    """Hammer the function 10 times — all results must match."""
    results = [generate_spiral_data(300, 15) for _ in range(10)]
    for i in range(1, len(results)):
        assert results[i].equals(results[0]), f"Call {i} diverged"


# ── Supplementary: chart encoding ───────────────────────────────────

def test_chart_size_encoding_domain():
    """Size scale should be pinned to [0.3, 1.0] → [10, 60] px."""
    df = generate_spiral_data(100, 5)
    spec = build_spiral_chart(df).to_dict()
    size_encoding = spec["encoding"]["size"]
    assert size_encoding["scale"]["domain"] == [0.3, 1.0]
    assert size_encoding["scale"]["range"] == [10, 60]
