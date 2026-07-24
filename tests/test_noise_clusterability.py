"""Tests for the 2-NN clusterability diagnostic (torch-free; numpy + sklearn)."""

from __future__ import annotations

import numpy as np
import pytest

from src.noise.clusterability import (
    ClusterabilityConfig,
    compare_reports,
    neighbor_distribution_to_frame,
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


def _common_and_rare_separated(common_n: int = 300, rare_n: int = 15):
    """A big common cluster and a small, well-separated rare cluster.

    In the FULL dataset a rare post has plenty of same-class neighbours (high
    agreement). In a small sampled E it is neighbour-starved (D-036), so within_e
    agreement for the rare class is much lower.
    """
    rng = np.random.default_rng(4)
    dim = 16
    e0 = np.eye(dim, dtype=np.float32)[0]
    e1 = np.eye(dim, dtype=np.float32)[1]
    common = e0 + 0.01 * rng.standard_normal((common_n, dim)).astype(np.float32)
    rare = e1 + 0.01 * rng.standard_normal((rare_n, dim)).astype(np.float32)
    features = np.vstack([common, rare])
    labels = np.array(["common"] * common_n + ["rare"] * rare_n)
    return features, labels


def test_within_e_agreement_is_lower_than_full_for_rare_class() -> None:
    features, labels = _common_and_rare_separated()
    # Small E so the rare class is neighbour-starved inside E.
    within = run_clusterability_diagnostic(
        features, labels,
        ClusterabilityConfig(scope="within_e", sample_size=40, n_rounds=20, seed=0),
    )
    full = run_clusterability_diagnostic(
        features, labels,
        ClusterabilityConfig(scope="full_dataset", sample_size=40, n_rounds=20, seed=0),
    )
    assert within.scope == "within_e"
    assert full.scope == "full_dataset"
    # Full has the whole same-class pool available; within_e does not.
    assert full.per_neighbor_agreement["rare"] > 0.9
    assert within.per_neighbor_agreement["rare"] < 0.6
    assert full.per_neighbor_agreement["rare"] > within.per_neighbor_agreement["rare"]


def test_candidate_counts_match_pool_sizes() -> None:
    features, labels = _common_and_rare_separated(common_n=300, rare_n=15)
    e = 40
    report = run_clusterability_diagnostic(
        features, labels,
        ClusterabilityConfig(scope="within_e", sample_size=e, n_rounds=30, seed=1),
    )
    # Full pool = exact class counts.
    assert report.candidates_full["rare"] == 15
    assert report.candidates_full["common"] == 300
    # In-E pool averages |E| * chance (uniform sampling without replacement).
    n = len(labels)
    assert report.candidates_in_e["rare"] == pytest.approx(e * 15 / n, abs=1.5)
    assert report.candidates_in_e["common"] == pytest.approx(e * 300 / n, abs=2.0)


def test_scope_validation_and_min_sample() -> None:
    with pytest.raises(ValueError, match="scope must be one of"):
        ClusterabilityConfig(scope="whole_thing")
    features, labels = _common_and_rare_separated(common_n=10, rare_n=5)
    with pytest.raises(ValueError, match="within_e"):
        run_clusterability_diagnostic(
            features, labels, ClusterabilityConfig(scope="within_e", sample_size=2)
        )


def test_within_e_equals_full_when_e_covers_everything() -> None:
    # When sample_size >= n every round samples the whole set, so the two scopes
    # must agree exactly (the fitted pool is identical).
    features, labels = _clustered_vs_scattered()
    cfg_kwargs = dict(sample_size=len(labels), n_rounds=3, seed=7)
    within = run_clusterability_diagnostic(
        features, labels, ClusterabilityConfig(scope="within_e", **cfg_kwargs)
    )
    full = run_clusterability_diagnostic(
        features, labels, ClusterabilityConfig(scope="full_dataset", **cfg_kwargs)
    )
    for c in within.per_neighbor_agreement:
        assert within.per_neighbor_agreement[c] == pytest.approx(
            full.per_neighbor_agreement[c]
        )


def test_report_frame_has_one_row_per_condition() -> None:
    features, labels = _clustered_vs_scattered()
    report = run_clusterability_diagnostic(
        features, labels, ClusterabilityConfig(sample_size=100, n_rounds=3, seed=3)
    )
    frame = report_to_frame(report)
    assert set(frame["condition"]) == {"A", "B"}
    assert len(frame) == 2


def test_chance_sums_to_one_and_matches_base_rates() -> None:
    features, labels = _clustered_vs_scattered()
    report = run_clusterability_diagnostic(
        features, labels, ClusterabilityConfig(sample_size=len(labels), n_rounds=2, seed=0)
    )
    assert sum(report.chance.values()) == pytest.approx(1.0)
    n = len(labels)
    assert report.chance["A"] == pytest.approx((labels == "A").sum() / n)


def _common_and_rare_tight_clusters() -> tuple[np.ndarray, np.ndarray]:
    """A common class and a rare class, each a tight, well-separated blob.

    Both are clusterable (per-neighbour agreement ~1.0), but the rare class has a
    much lower base rate, so it should show a much higher lift. This mirrors the
    D-034 finding: bipolar is rare yet distinctive, hence the highest lift.
    """
    rng = np.random.default_rng(3)
    dim = 16
    e0 = np.eye(dim, dtype=np.float32)[0]
    e1 = np.eye(dim, dtype=np.float32)[1]
    common = e0 + 0.01 * rng.standard_normal((150, dim)).astype(np.float32)
    rare = e1 + 0.01 * rng.standard_normal((12, dim)).astype(np.float32)
    features = np.vstack([common, rare])
    labels = np.array(["common"] * len(common) + ["rare"] * len(rare))
    return features, labels


def test_lift_equals_agreement_over_chance() -> None:
    features, labels = _common_and_rare_tight_clusters()
    report = run_clusterability_diagnostic(
        features, labels, ClusterabilityConfig(sample_size=len(labels), n_rounds=3, seed=1)
    )
    for c in report.lift:
        assert report.lift[c] == pytest.approx(
            report.per_neighbor_agreement[c] / report.chance[c]
        )
    # Both classes are clusterable, but the rare one is far more distinctive
    # relative to its base rate (the D-034 bipolar phenomenon).
    assert report.per_neighbor_agreement["rare"] > 0.9
    assert report.lift["rare"] > report.lift["common"]


def test_distribution_diagonal_matches_per_neighbor_agreement() -> None:
    # With sample_size >= n and replace=False, every round samples all centres, so
    # the pooled diagonal equals the per-round-averaged per-neighbour agreement.
    features, labels = _clustered_vs_scattered()
    report = run_clusterability_diagnostic(
        features, labels, ClusterabilityConfig(sample_size=len(labels), n_rounds=2, seed=2)
    )
    dist = neighbor_distribution_to_frame(report)
    for c in report.per_neighbor_agreement:
        assert dist.loc[f"centre_{c}", f"neigh_{c}"] == pytest.approx(
            report.per_neighbor_agreement[c]
        )
    # Each centre row of the distribution is a proper distribution.
    for c in report.neighbor_label_distribution:
        assert sum(report.neighbor_label_distribution[c].values()) == pytest.approx(1.0)


def test_compare_reports_delta_and_zero_self_delta() -> None:
    features, labels = _clustered_vs_scattered()
    cfg = ClusterabilityConfig(sample_size=len(labels), n_rounds=3, seed=5)
    ft = run_clusterability_diagnostic(features, labels, cfg, extractor="finetuned")

    # Same features vs itself: delta is exactly zero.
    same = compare_reports(ft, ft, name_a="finetuned", name_b="base")
    assert same["delta_a_minus_b"].abs().max() == pytest.approx(0.0)

    # Degrade B by shuffling features so neighbours no longer share labels; A's
    # agreement should exceed the degraded run's, giving a positive delta.
    rng = np.random.default_rng(9)
    scrambled = features[rng.permutation(len(features))]
    base = run_clusterability_diagnostic(scrambled, labels, cfg, extractor="base")
    cmp = compare_reports(ft, base)
    assert "agreement_finetuned" in cmp.columns
    assert "agreement_base" in cmp.columns
    a_row = cmp.loc[cmp["condition"] == "A", "delta_a_minus_b"].iloc[0]
    assert a_row > 0
