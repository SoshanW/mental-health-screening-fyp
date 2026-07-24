"""2-NN clusterability diagnostic for the condition-dependent noise model (C1).

DECISIONS.md open item 2. This measures, per condition, how often a post's two
nearest neighbours in MentalBERT embedding space carry the *same noisy proxy
label* as the post. It is the empirical test of the clusterability assumption HOC
(Zhu, Song and Liu, 2021) relies on: if neighbours in feature space do not share a
post's label, the estimator has no signal to recover a transition matrix from.

Why this licenses C2 (see DECISIONS.md D-030): HOC assumes a post's nearest
neighbours share its *true* class. A truly-bipolar person writing during a
depressive phase writes like someone with depression, so their neighbours in
MentalBERT space are depression posts. The clinical fact that creates the label
noise (phase predominance, diagnostic delay) is the same fact that breaks
clusterability. We expect high agreement for depression and noticeably lower
agreement for bipolar.

This statistic is descriptive, not confirmatory. It is confounded between the
noise level and clusterability failure and cannot separate them, so it functions
as a ONE-SIDED test: high agreement (e.g. for depression) would falsify the
clusterability-failure argument, whereas low agreement (e.g. for bipolar) is only
consistent with that argument and does not establish it. The primary evidence for
C1's failure claim is the seed-instability of the noise estimators themselves (the
next diagnostic stage). See DECISIONS.md open item 2 (amended 2026-07-24).

DECISION (distance metric): neighbours are ranked by cosine, exactly as in Zhu,
Song and Liu (2021). We L2-normalise the features and use scikit-learn's
``metric="cosine"``. Verified 2026-07-24 that cosine-distance ordering (1 - cos
similarity) is identical to ranking by negative cosine similarity, and that the
query point's own row is returned first (distance 0) so it can be dropped as
"self". Euclidean is deliberately NOT substituted.

DECISION (no full pairwise matrix): at ~112k training posts a full 112k x 112k
distance matrix is ~50 GB and never materialised. We follow HOC's Monte-Carlo
approach: sample ``sample_size`` centre indices per round over ``n_rounds`` rounds
and average the per-condition agreement across rounds. Each centre's 2-NN are
found exactly against the full set via a fitted brute-force cosine index (queries
are chunked internally by scikit-learn), so only the *centres* are sampled, not
the neighbour pool.

Measured against NOISY labels, not true labels: Zhu, Song and Liu (2021, Table 3)
report roughly 78 to 88 percent feasible 2-NN tuples on noisy CIFAR-10 measured
against TRUE labels. This diagnostic scores agreement against the noisy proxy
labels, so the numbers are NOT directly comparable; the reported output says so.

Torch-free: numpy + scikit-learn only, so it runs locally on the cached embeddings
brought back from Colab.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

from ..modeling.config import ArtifactPaths
from .embeddings import default_embeddings_dir, load_embeddings

#: The caveat printed and returned with every report, so no reader mistakes this
#: for a true-label clusterability measurement, nor treats it as confirmatory.
#: DECISION (one-sided, descriptive): see DECISIONS.md open item 2 (amended
#: 2026-07-24). The statistic cannot separate noise level from clusterability
#: failure, so it is reported as a one-sided falsification test, not as evidence
#: that clusterability has failed.
NOISY_LABEL_CAVEAT: str = (
    "Agreement is measured against NOISY proxy labels, not true labels. "
    "Zhu, Song and Liu (2021, Table 3) report ~78-88% feasible 2-NN tuples on "
    "noisy CIFAR-10 against TRUE labels; those numbers are NOT directly comparable "
    "to these. This statistic is DESCRIPTIVE, not confirmatory: it is confounded "
    "between the noise level and clusterability failure and cannot separate them. "
    "It is a ONE-SIDED test: high agreement would falsify the clusterability-failure "
    "argument, but low agreement is only consistent with it and does not establish "
    "it. Seed-instability of the noise estimators (the next diagnostic stage) is the "
    "primary evidence. See DECISIONS.md open item 2 (amended 2026-07-24)."
)


@dataclass(frozen=True)
class ClusterabilityConfig:
    """Knobs for the 2-NN clusterability diagnostic (injected, not global).

    Attributes:
        n_neighbors: Neighbours per post, excluding self. 2 reproduces Zhu, Song
            and Liu (2021).
        sample_size: ``|E|`` centre indices sampled per round.
        n_rounds: ``G`` rounds averaged over.
        seed: RNG seed for the centre sampling.
        metric: scikit-learn neighbour metric. ``"cosine"`` reproduces the paper;
            do not change to a Euclidean metric.
    """

    n_neighbors: int = 2
    sample_size: int = 15000
    n_rounds: int = 20
    seed: int = 42
    metric: str = "cosine"

    def __post_init__(self) -> None:
        if self.n_neighbors < 1:
            raise ValueError(f"n_neighbors must be >= 1; got {self.n_neighbors}.")
        if self.sample_size < 1 or self.n_rounds < 1:
            raise ValueError("sample_size and n_rounds must be >= 1.")


@dataclass(frozen=True)
class ClusterabilityReport:
    """Per-condition results of the diagnostic.

    Attributes:
        agreement: condition -> mean over rounds of the fraction of that
            condition's centres whose ALL ``n_neighbors`` neighbours share its
            noisy label (the headline "both neighbours match" fraction at k=2).
        agreement_std: condition -> std of that per-round fraction (a
            stability signal; large values mean the estimate is round-sensitive).
        pattern_counts: condition -> ``{"both_match","one_match","none_match"}``
            centre counts pooled across all rounds. At ``n_neighbors=2`` these are
            the full 3-way agreement pattern; if ``n_neighbors>2``, ``one_match``
            aggregates all partial-agreement centres.
        center_counts: condition -> total centres of that condition summed across
            rounds (the denominator behind ``pattern_counts``).
        config: the config used.
        n_posts: number of posts the index was fitted on.
        caveat: :data:`NOISY_LABEL_CAVEAT`.
    """

    agreement: dict[str, float]
    agreement_std: dict[str, float]
    pattern_counts: dict[str, dict[str, int]]
    center_counts: dict[str, int]
    config: ClusterabilityConfig
    n_posts: int
    caveat: str = NOISY_LABEL_CAVEAT


def _two_nn_excluding_self(
    index: NearestNeighbors, query: np.ndarray, centers: np.ndarray, k: int
) -> np.ndarray:
    """Return the ``(len(centers), k)`` neighbour indices, self removed.

    ``index`` is fitted on the full set; ``query`` is the normalised centre rows.
    The fitted set contains each centre, so its own row comes back first at
    distance 0. We ask for ``k + 1`` neighbours and drop the self row per centre,
    which is robust even if an exact-duplicate embedding displaces self from
    position 0.
    """
    _, idx = index.kneighbors(query, n_neighbors=k + 1)
    self_mask = idx == centers[:, None]
    keep = ~self_mask
    # Take the first k kept (distance-sorted) columns per row. Each row has at most
    # one self, so >= k kept columns remain among the k + 1 returned.
    order_ok = np.cumsum(keep, axis=1) <= k
    take = keep & order_ok
    return idx[take].reshape(len(centers), k)


def run_clusterability_diagnostic(
    features: np.ndarray,
    labels: np.ndarray | pd.Series,
    config: ClusterabilityConfig = ClusterabilityConfig(),
) -> ClusterabilityReport:
    """Run the per-condition 2-NN noisy-label agreement diagnostic.

    Args:
        features: ``(n, d)`` embedding matrix (any float dtype; upcast to float32).
        labels: length-``n`` noisy proxy labels aligned to ``features`` rows.
        config: sampling / neighbour knobs.

    Raises:
        ValueError: on a features/labels length mismatch or too few posts.
    """
    labels = np.asarray(labels)
    n = int(features.shape[0])
    if n != len(labels):
        raise ValueError(f"features/labels length mismatch: {n} vs {len(labels)}.")
    if n <= config.n_neighbors:
        raise ValueError(
            f"Need more than n_neighbors={config.n_neighbors} posts; got {n}."
        )

    k = config.n_neighbors
    # L2-normalise then use cosine so ordering matches negative cosine similarity.
    x_norm = normalize(features).astype(np.float32, copy=False)
    index = NearestNeighbors(
        n_neighbors=k + 1, metric=config.metric, algorithm="brute", n_jobs=-1
    )
    index.fit(x_norm)

    rng = np.random.default_rng(config.seed)
    sample_size = min(config.sample_size, n)
    conditions = sorted({str(c) for c in labels.tolist()})

    per_round_frac: dict[str, list[float]] = {c: [] for c in conditions}
    pattern: dict[str, dict[str, int]] = {
        c: {"both_match": 0, "one_match": 0, "none_match": 0} for c in conditions
    }
    center_counts: dict[str, int] = {c: 0 for c in conditions}

    for _ in range(config.n_rounds):
        centers = rng.choice(n, size=sample_size, replace=False)
        neigh = _two_nn_excluding_self(index, x_norm[centers], centers, k)

        center_labels = labels[centers]
        neigh_labels = labels[neigh]
        n_match = (neigh_labels == center_labels[:, None]).sum(axis=1)  # 0..k per centre

        for c in conditions:
            sel = center_labels == c
            m = int(sel.sum())
            if m == 0:
                per_round_frac[c].append(np.nan)
                continue
            nm = n_match[sel]
            both = int((nm == k).sum())
            none = int((nm == 0).sum())
            one = m - both - none  # at k=2: exactly-one; at k>2: partial
            per_round_frac[c].append(both / m)
            pattern[c]["both_match"] += both
            pattern[c]["one_match"] += one
            pattern[c]["none_match"] += none
            center_counts[c] += m

    agreement = {c: float(np.nanmean(per_round_frac[c])) for c in conditions}
    agreement_std = {c: float(np.nanstd(per_round_frac[c])) for c in conditions}

    return ClusterabilityReport(
        agreement=agreement,
        agreement_std=agreement_std,
        pattern_counts=pattern,
        center_counts=center_counts,
        config=config,
        n_posts=n,
    )


def report_to_frame(report: ClusterabilityReport) -> pd.DataFrame:
    """Flatten a report into a per-condition DataFrame for CSV output / printing."""
    rows = []
    for cond in sorted(report.agreement):
        counts = report.pattern_counts[cond]
        total = report.center_counts[cond]
        rows.append(
            {
                "condition": cond,
                "n_centers_total": total,
                "both_match_agreement_mean": report.agreement[cond],
                "both_match_agreement_std": report.agreement_std[cond],
                "both_match": counts["both_match"],
                "one_match": counts["one_match"],
                "none_match": counts["none_match"],
                "one_match_frac": (counts["one_match"] / total) if total else float("nan"),
                "none_match_frac": (counts["none_match"] / total) if total else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def format_report(report: ClusterabilityReport) -> str:
    """Human-readable summary, sorted by agreement so the weakest condition is last."""
    frame = report_to_frame(report).sort_values(
        "both_match_agreement_mean", ascending=False
    )
    lines = [
        "=== 2-NN clusterability diagnostic (per condition) ===",
        f"posts indexed: {report.n_posts}   "
        f"sample_size={report.config.sample_size}  n_rounds={report.config.n_rounds}  "
        f"k={report.config.n_neighbors}  metric={report.config.metric}",
        "",
        frame.to_string(index=False, float_format=lambda v: f"{v:.4f}"),
        "",
        report.caveat,
    ]
    return "\n".join(lines)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Per-condition 2-NN clusterability diagnostic.")
    p.add_argument("--artifacts-root", type=Path, default=None, help="Models dir (default <repo>/Models).")
    p.add_argument("--embeddings-dir", type=Path, default=None,
                   help="Cache dir with features.npy + metadata.csv (default <Models>/embeddings/<split>).")
    p.add_argument("--split", type=str, default="train")
    p.add_argument("--sample-size", type=int, default=ClusterabilityConfig().sample_size)
    p.add_argument("--n-rounds", type=int, default=ClusterabilityConfig().n_rounds)
    p.add_argument("--n-neighbors", type=int, default=ClusterabilityConfig().n_neighbors)
    p.add_argument("--seed", type=int, default=ClusterabilityConfig().seed)
    p.add_argument("--out-csv", type=Path, default=None,
                   help="Where to write the per-condition report (default alongside the embeddings).")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.embeddings_dir is not None:
        cache_dir = args.embeddings_dir
    else:
        artifacts = ArtifactPaths(root=args.artifacts_root) if args.artifacts_root else ArtifactPaths.default()
        cache_dir = default_embeddings_dir(artifacts, args.split)

    features, metadata = load_embeddings(cache_dir)
    labels = metadata["condition"].to_numpy()

    config = ClusterabilityConfig(
        n_neighbors=args.n_neighbors,
        sample_size=args.sample_size,
        n_rounds=args.n_rounds,
        seed=args.seed,
    )
    report = run_clusterability_diagnostic(features, labels, config)

    print(format_report(report))

    out_csv = args.out_csv or (Path(cache_dir) / "clusterability_report.csv")
    report_to_frame(report).to_csv(out_csv, index=False)
    print(f"\nWrote per-condition report -> {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
