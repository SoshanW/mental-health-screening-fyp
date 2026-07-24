"""Tests for the 2-NN clusterability diagnostic (torch-free; numpy + sklearn)."""

from __future__ import annotations

import numpy as np
import pytest

from src.noise.clusterability import (
    ClusterabilityConfig,
    report_to_frame,
    run_clusterability_diagnostic,
)


def _clustered_vs_scattered() -> tuple[np.ndarray, np.ndarray]:
    """Build a dataset where condition A is clusterable and B is not.

    ``A`` posts sit in a tight blob around one direction, so an A post's two
    nearest neighbours are almost always other A posts (high agreement). ``B``
    posts are few and each perturbed in its own direction near the dense A blob, so
    a B post's nearest neighbours are A posts, not other B posts (low agreement).
    This mirrors the phase-predominance intuition: the rare condition's neighbours
    belong to the dominant one.
    """
    rng = np.random.default_rng(0)
    dim = 16
    a_base = np.zeros(dim, dtype=np.float32)
    a_base[0] = 1.0
    a = a_base + 0.01 * rng.standard_normal((120, dim)).astype(np.float32)

    # B points near the A blob but scattered from each other via unique directions.
    b = a_base + 0.02 * rng.standard_normal((12, dim)).astype(np.float32)
    for i in range(len(b)):
        b[i, 1 + (i % (dim - 1))] += 0.05  # push each B a little, uniquely

    features = np.vstack([a, b])
    labels = np.array(["A"] * len(a) + ["B"] * len(b))
    return features, labels


def test_depression_like_condition_scores_higher_than_rare_one() -> None:
    features, labels = _clustered_vs_scattered()
    config = ClusterabilityConfig(sample_size=len(labels), n_rounds=5, seed=1)
    report = run_clusterability_diagnostic(features, labels, config)

    assert report.agreement["A"] > report.agreement["B"]
    assert report.agreement["A"] > 0.9  # clusterable condition
    assert report.agreement["B"] < 0.5  # rare, non-clusterable condition


def test_pattern_counts_partition_the_centers() -> None:
    features, labels = _clustered_vs_scattered()
    report = run_clusterability_diagnostic(
        features, labels, ClusterabilityConfig(sample_size=len(labels), n_rounds=4, seed=2)
    )
    for cond, counts in report.pattern_counts.items():
        total = counts["both_match"] + counts["one_match"] + counts["none_match"]
        assert total == report.center_counts[cond]
        assert all(v >= 0 for v in counts.values())


def test_self_is_not_counted_as_its_own_neighbour() -> None:
    # Two well-separated singleton-ish clusters: if self leaked in, a lone-label
    # point would spuriously "match" itself. Here each label has >=3 members so the
    # 2 true neighbours share the label; agreement must be exactly 1.0, and it would
    # be < 1.0 (or NaN-prone) if self displaced a real neighbour incorrectly.
    dim = 8
    g1 = np.tile(np.eye(dim, dtype=np.float32)[0], (5, 1)) + 1e-3
    g2 = np.tile(np.eye(dim, dtype=np.float32)[1], (5, 1)) + 1e-3
    features = np.vstack([g1, g2])
    labels = np.array(["x"] * 5 + ["y"] * 5)
    report = run_clusterability_diagnostic(
        features, labels, ClusterabilityConfig(sample_size=10, n_rounds=3, seed=0)
    )
    assert report.agreement["x"] == pytest.approx(1.0)
    assert report.agreement["y"] == pytest.approx(1.0)


def test_determinism_under_fixed_seed() -> None:
    features, labels = _clustered_vs_scattered()
    cfg = ClusterabilityConfig(sample_size=80, n_rounds=6, seed=7)
    r1 = run_clusterability_diagnostic(features, labels, cfg)
    r2 = run_clusterability_diagnostic(features, labels, cfg)
    assert r1.agreement == r2.agreement
    assert r1.pattern_counts == r2.pattern_counts


def test_length_mismatch_raises() -> None:
    features = np.zeros((5, 4), dtype=np.float32)
    labels = np.array(["a", "b", "c"])
    with pytest.raises(ValueError, match="length mismatch"):
        run_clusterability_diagnostic(features, labels)


def test_report_frame_has_one_row_per_condition() -> None:
    features, labels = _clustered_vs_scattered()
    report = run_clusterability_diagnostic(
        features, labels, ClusterabilityConfig(sample_size=100, n_rounds=3, seed=3)
    )
    frame = report_to_frame(report)
    assert set(frame["condition"]) == {"A", "B"}
    assert len(frame) == 2
