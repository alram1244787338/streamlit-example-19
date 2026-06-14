"""Regression tests for the spiral demo's data prep and size encoding.

Runnable two ways:
    .venv/bin/python -m pytest test_streamlit_app.py
    .venv/bin/python test_streamlit_app.py        # no pytest needed

The point of these tests is to lock in the two things that were previously
broken/unclear:

1. Point *size* now comes from a bounded, non-negative ``magnitude`` and is
   rendered on a FIXED [0, 1] scale, so the size mapping is stable across reruns
   and across small / single-turn / many-turn spirals (it no longer depends on
   the realized random sample's min/max).
2. The on-page field explanation (``COLUMN_DOCS``) stays in lock-step with the
   columns the data actually produces.
"""

import numpy as np

from streamlit_app import (
    COLUMN_DOCS,
    SIZE_DOMAIN,
    SIZE_RANGE,
    build_spiral_chart,
    build_spiral_data,
)

# (num_points, num_turns): tiny, single-turn, and large/many-turn cases.
SCENARIOS = [
    (1, 1),
    (2, 1),
    (3, 1),
    (5, 1),
    (50, 1),
    (1100, 31),
    (10000, 300),
]


def _size_scale(df):
    scale = build_spiral_chart(df).to_dict()["encoding"]["size"]["scale"]
    return scale["domain"], scale["range"]


def test_columns_match_docs():
    # The page explanation must describe exactly the columns we produce — no
    # undocumented columns, no docs for columns that don't exist.
    for num_points, num_turns in SCENARIOS:
        df = build_spiral_data(num_points, num_turns, rng=np.random.default_rng(0))
        assert set(df.columns) == set(COLUMN_DOCS), (
            f"columns {set(df.columns)} != docs {set(COLUMN_DOCS)} "
            f"for {num_points} points"
        )


def test_magnitude_is_bounded_and_non_negative():
    # The size source must be a sane, bounded magnitude — never negative like the
    # old ``np.random.randn`` column.
    for num_points, num_turns in SCENARIOS:
        df = build_spiral_data(num_points, num_turns, rng=np.random.default_rng(1))
        mag = df["magnitude"].to_numpy()
        assert mag.min() >= 0.0, f"magnitude went negative for {num_points} points"
        assert mag.max() <= 1.0, f"magnitude exceeded 1 for {num_points} points"


def test_progress_spans_zero_to_one():
    for num_points, num_turns in SCENARIOS:
        df = build_spiral_data(num_points, num_turns, rng=np.random.default_rng(2))
        prog = df["progress"].to_numpy()
        assert len(df) == num_points
        assert prog.min() >= 0.0 and prog.max() <= 1.0


def test_size_scale_is_fixed_not_data_derived():
    # Core stability guarantee: the size scale domain/range is pinned to the
    # configured constants for EVERY scenario, so it never shifts with the data.
    for num_points, num_turns in SCENARIOS:
        df = build_spiral_data(num_points, num_turns, rng=np.random.default_rng(3))
        domain, rng_ = _size_scale(df)
        assert domain == list(SIZE_DOMAIN), f"size domain moved for {num_points} points"
        assert rng_ == list(SIZE_RANGE), f"size range moved for {num_points} points"


def test_size_mapping_stable_across_random_samples():
    # Regression for the original bug: two different random samples produce
    # different raw magnitude ranges, yet the size *mapping* (scale domain/range)
    # stays identical -> a given magnitude always renders at the same size.
    df_a = build_spiral_data(500, 12, rng=np.random.default_rng(10))
    df_b = build_spiral_data(500, 12, rng=np.random.default_rng(99))

    # The underlying samples really are different...
    assert not np.allclose(
        np.sort(df_a["magnitude"].to_numpy()),
        np.sort(df_b["magnitude"].to_numpy()),
    ), "sanity check failed: the two samples should differ"

    # ...but the size encoding does not move between them.
    assert _size_scale(df_a) == _size_scale(df_b)


def test_data_is_reproducible_for_a_given_seed():
    a = build_spiral_data(64, 7, rng=np.random.default_rng(42))
    b = build_spiral_data(64, 7, rng=np.random.default_rng(42))
    for col in COLUMN_DOCS:
        assert np.array_equal(a[col].to_numpy(), b[col].to_numpy()), col


def test_single_point_does_not_blow_up():
    df = build_spiral_data(1, 1, rng=np.random.default_rng(7))
    assert len(df) == 1
    # First point sits at the spiral's center.
    assert df["x"].iloc[0] == 0.0 and df["y"].iloc[0] == 0.0
    assert df["progress"].iloc[0] == 0.0
    # Size scale is still well defined for a single point.
    assert _size_scale(df) == (list(SIZE_DOMAIN), list(SIZE_RANGE))


if __name__ == "__main__":
    failures = 0
    for _name, _fn in sorted(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            try:
                _fn()
                print(f"PASS {_name}")
            except AssertionError as exc:
                failures += 1
                print(f"FAIL {_name}: {exc}")
    print(f"\n{'OK' if not failures else 'FAILED'} — {failures} failure(s)")
    raise SystemExit(1 if failures else 0)
