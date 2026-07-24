"""2-NN clusterability diagnostic for the condition-dependent noise model (C1).

DECISIONS.md open item 2, D-034 (the diagnostic falsified its own prediction) and
D-035 (base-embedding control). This measures, per condition, how often a post's
two nearest neighbours in MentalBERT embedding space carry the *same noisy proxy
label* as the post. It is the empirical test of the clusterability assumption HOC
(Zhu, Song and Liu, 2021) relies on: if neighbours in feature space do not share a
post's label, the estimator has no signal to recover a transition matrix from.

Why this was run (see DECISIONS.md D-030): HOC assumes a post's nearest neighbours
share its *true* class. A truly-bipolar person writing during a depressive phase
writes like someone with depression. The prediction was high agreement for
depression and low for bipolar.

D-034 records that the prediction was FALSIFIED: bipolar agreement came back high
(86.70%, top of Zhu et al.'s working range) with the highest lift over chance of
all four conditions. This statistic is descriptive, not confirmatory, for two
reasons logged in D-034: (1) the fine-tuned embeddings were optimised to separate
these same noisy labels, so agreement measured in that space is circular; (2) posts
are partitioned by their noisy label, so the truly-bipolar-posted-in-r/depression
cases the argument depends on are filed under depression by construction. It is a
ONE-SIDED test: high agreement falsifies the clusterability-failure argument; low
agreement is only consistent with it. Seed-instability of the noise estimators (the
next stage) is the load-bearing evidence.

DECISION (metrics reported): the headline is PER-NEIGHBOUR agreement (fraction of
neighbour slots sharing the centre's noisy label), which is the diagonal of the
row-normalised neighbour-label distribution and matches the D-034 table. ``chance``
is the class base rate (a random neighbour's probability of carrying that label);
``lift`` is agreement / chance, so a rare-but-distinctive class scores high. The
``both_match`` fraction (both 2-NN match) is retained as a secondary view.

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
are chunked internally by scikit-learn), so only the *centres* are sampled.

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
#: 2026-07-24) and D-034. The statistic cannot separate noise level from
#: clusterability failure, so it is reported as a one-sided falsification test.
NOISY_LABEL_CAVEAT: str = (
    "Agreement is measured against NOISY proxy labels, not true labels. "
    "Zhu, Song and Liu (2021, Table 3) report ~78-88% feasible 2-NN tuples on "
    "noisy CIFAR-10 against TRUE labels; those numbers are NOT directly comparable "
    "to these. This statistic is DESCRIPTIVE, not confirmatory: it is confounded "
    "between the noise level and clusterability failure and cannot separate them. "
    "It is a ONE-SIDED test: high agreement would falsify the clusterability-failure "
    "argument, but low agreement is only consistent with it and does not establish "
    "it. Seed-instability of the noise estimators (the next diagnostic stage) is the "
    "primary evidence. See DECISIONS.md D-034 and open item 2 (amended 2026-07-24)."
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
        per_neighbor_agreement: condition -> mean over rounds of the fraction of
            that condition's centres' neighbour slots that share its noisy label.
            This is the headline number and matches the D-034 table.
        per_neighbor_agreement_std: condition -> std of that per-round fraction (a
            stability signal; large values mean the estimate is round-sensitive).
        chance: condition -> class base rate (a random neighbour's probability of
            carrying that label), computed over the whole indexed set.
        lift: condition -> ``per_neighbor_agreement / chance``. High lift means the
            class is distinctive relative to how common it is.
        neighbor_label_distribution: centre condition -> {neighbour condition ->
            fraction of that centre's neighbour slots}, pooled across rounds and
            row-normalised. Its diagonal is ``per_neighbor_agreement`` (pooled).
        agreement: condition -> mean over rounds of the fraction of centres whose
            ALL ``n_neighbors`` neighbours match ("both neighbours match" at k=2).
            Secondary view, retained for continuity.
        agreement_std: std of the per-round ``agreement`` fraction.
        pattern_counts: condition -> ``{"both_match","one_match","none_match"}``
            centre counts pooled across rounds (the full 3-way pattern at k=2).
        center_counts: condition -> total centres of that condition across rounds.
        config: the config used.
        n_posts: number of posts the index was fitted on.
        extractor: label of the feature extractor these embeddings came from
            (e.g. "finetuned" or "base"); "unknown" if the cache did not record it.
        caveat: :data:`NOISY_LABEL_CAVEAT`.
    """

    per_neighbor_agreement: dict[str, float]
    per_neighbor_agreement_std: dict[str, float]
    chance: dict[str, float]
    lift: dict[str, float]
    neighbor_label_distribution: dict[str, dict[str, float]]
    agreement: dict[str, float]
    agreement_std: dict[str, float]
    pattern_counts: dict[str, dict[str, int]]
    center_counts: dict[str, int]
    config: ClusterabilityConfig
    n_posts: int
    extractor: str = "unknown"
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
    extractor: str = "unknown",
) -> ClusterabilityReport:
    """Run the per-condition 2-NN noisy-label agreement diagnostic.

    Args:
        features: ``(n, d)`` embedding matrix (any float dtype; upcast to float32).
        labels: length-``n`` noisy proxy labels aligned to ``features`` rows.
        config: sampling / neighbour knobs.
        extractor: label recorded on the report for provenance.

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

    # Class base rate = chance a random neighbour carries a given label.
    uniq, counts = np.unique(labels.astype(str), return_counts=True)
    chance = {str(u): float(c) / n for u, c in zip(uniq, counts)}
    conditions = sorted(chance)

    rng = np.random.default_rng(config.seed)
    sample_size = min(config.sample_size, n)

    both_round_frac: dict[str, list[float]] = {c: [] for c in conditions}
    perneigh_round_frac: dict[str, list[float]] = {c: [] for c in conditions}
    pattern: dict[str, dict[str, int]] = {
        c: {"both_match": 0, "one_match": 0, "none_match": 0} for c in conditions
    }
    center_counts: dict[str, int] = {c: 0 for c in conditions}
    # Pooled neighbour-label slot counts: neigh_counts[centre][neighbour] = slots.
    neigh_counts: dict[str, dict[str, int]] = {
        c: {c2: 0 for c2 in conditions} for c in conditions
    }
    slot_totals: dict[str, int] = {c: 0 for c in conditions}

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
                both_round_frac[c].append(np.nan)
                perneigh_round_frac[c].append(np.nan)
                continue
            nm = n_match[sel]
            both = int((nm == k).sum())
            none = int((nm == 0).sum())
            one = m - both - none  # at k=2: exactly-one; at k>2: partial
            matched_slots = int(nm.sum())
            total_slots = m * k

            both_round_frac[c].append(both / m)
            perneigh_round_frac[c].append(matched_slots / total_slots)
            pattern[c]["both_match"] += both
            pattern[c]["one_match"] += one
            pattern[c]["none_match"] += none
            center_counts[c] += m
            slot_totals[c] += total_slots

            # Per-neighbour label breakdown for the distribution matrix.
            c_neigh = neigh_labels[sel]
            for c2 in conditions:
                neigh_counts[c][c2] += int((c_neigh == c2).sum())

    per_neighbor_agreement = {
        c: float(np.nanmean(perneigh_round_frac[c])) for c in conditions
    }
    per_neighbor_agreement_std = {
        c: float(np.nanstd(perneigh_round_frac[c])) for c in conditions
    }
    agreement = {c: float(np.nanmean(both_round_frac[c])) for c in conditions}
    agreement_std = {c: float(np.nanstd(both_round_frac[c])) for c in conditions}
    lift = {
        c: (per_neighbor_agreement[c] / chance[c]) if chance[c] > 0 else float("nan")
        for c in conditions
    }
    neighbor_label_distribution = {
        c: {
            c2: (neigh_counts[c][c2] / slot_totals[c]) if slot_totals[c] else float("nan")
            for c2 in conditions
        }
        for c in conditions
    }

    return ClusterabilityReport(
        per_neighbor_agreement=per_neighbor_agreement,
        per_neighbor_agreement_std=per_neighbor_agreement_std,
        chance=chance,
        lift=lift,
        neighbor_label_distribution=neighbor_label_distribution,
        agreement=agreement,
        agreement_std=agreement_std,
        pattern_counts=pattern,
        center_counts=center_counts,
        config=config,
        n_posts=n,
        extractor=extractor,
    )


def report_to_frame(report: ClusterabilityReport) -> pd.DataFrame:
    """Flatten a report into a per-condition DataFrame for CSV output / printing."""
    rows = []
    for cond in sorted(report.per_neighbor_agreement):
        counts = report.pattern_counts[cond]
        total = report.center_counts[cond]
        rows.append(
            {
                "condition": cond,
                "n_centers_total": total,
                "per_neighbor_agreement": report.per_neighbor_agreement[cond],
                "per_neighbor_agreement_std": report.per_neighbor_agreement_std[cond],
                "chance": report.chance[cond],
                "lift_over_chance": report.lift[cond],
                "both_match_agreement": report.agreement[cond],
                "both_match": counts["both_match"],
                "one_match": counts["one_match"],
                "none_match": counts["none_match"],
            }
        )
    return pd.DataFrame(rows)


def neighbor_distribution_to_frame(report: ClusterabilityReport) -> pd.DataFrame:
    """Row-normalised neighbour-label distribution as a centre x neighbour matrix."""
    conds = sorted(report.neighbor_label_distribution)
    data = {
        f"neigh_{c2}": [report.neighbor_label_distribution[c][c2] for c in conds]
        for c2 in conds
    }
    frame = pd.DataFrame(data, index=[f"centre_{c}" for c in conds])
    return frame


def format_report(report: ClusterabilityReport) -> str:
    """Human-readable summary, sorted by agreement so the weakest condition is last."""
    frame = report_to_frame(report).sort_values("per_neighbor_agreement", ascending=False)
    dist = neighbor_distribution_to_frame(report)
    lines = [
        f"=== 2-NN clusterability diagnostic (per condition) [extractor: {report.extractor}] ===",
        f"posts indexed: {report.n_posts}   "
        f"sample_size={report.config.sample_size}  n_rounds={report.config.n_rounds}  "
        f"k={report.config.n_neighbors}  metric={report.config.metric}",
        "",
        frame.to_string(index=False, float_format=lambda v: f"{v:.4f}"),
        "",
        "neighbour-label distribution (rows = centre, cols = neighbour, row-normalised):",
        dist.to_string(float_format=lambda v: f"{v:.4f}"),
        "",
        report.caveat,
    ]
    return "\n".join(lines)


def compare_reports(
    report_a: ClusterabilityReport,
    report_b: ClusterabilityReport,
    name_a: str | None = None,
    name_b: str | None = None,
) -> pd.DataFrame:
    """Side-by-side per-condition comparison of two extractors (D-035 control).

    ``delta`` is ``agreement_a - agreement_b``: how much extractor A's agreement
    exceeds B's. For the D-035 control, A is the fine-tuned run and B the base run,
    so a large positive delta means the apparent clustering is largely a
    fine-tuning artifact.
    """
    name_a = name_a or report_a.extractor
    name_b = name_b or report_b.extractor
    conds = sorted(set(report_a.per_neighbor_agreement) | set(report_b.per_neighbor_agreement))
    rows = []
    for c in conds:
        a = report_a.per_neighbor_agreement.get(c, float("nan"))
        b = report_b.per_neighbor_agreement.get(c, float("nan"))
        rows.append(
            {
                "condition": c,
                f"agreement_{name_a}": a,
                f"agreement_{name_b}": b,
                "delta_a_minus_b": a - b,
                f"lift_{name_a}": report_a.lift.get(c, float("nan")),
                f"lift_{name_b}": report_b.lift.get(c, float("nan")),
                "chance": report_a.chance.get(c, report_b.chance.get(c, float("nan"))),
            }
        )
    return pd.DataFrame(rows)


def _run_from_dir(cache_dir: Path, config: ClusterabilityConfig) -> ClusterabilityReport:
    """Load a cached embedding set and run the diagnostic, stamping its extractor."""
    features, metadata = load_embeddings(cache_dir)
    labels = metadata["condition"].to_numpy()
    extractor = "unknown"
    if "extractor" in metadata.columns and len(metadata):
        extractor = str(metadata["extractor"].iloc[0])
    return run_clusterability_diagnostic(features, labels, config, extractor=extractor)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Per-condition 2-NN clusterability diagnostic.")
    p.add_argument("--artifacts-root", type=Path, default=None, help="Models dir (default <repo>/Models).")
    p.add_argument("--embeddings-dir", type=Path, default=None,
                   help="Cache dir with features.npy + metadata.csv (default <Models>/embeddings/<split>).")
    p.add_argument("--base-embeddings-dir", type=Path, default=None,
                   help="Second cache dir (base MentalBERT) for the D-035 side-by-side comparison.")
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
        cache_dir = Path(args.embeddings_dir)
    else:
        artifacts = ArtifactPaths(root=args.artifacts_root) if args.artifacts_root else ArtifactPaths.default()
        cache_dir = default_embeddings_dir(artifacts, args.split)

    config = ClusterabilityConfig(
        n_neighbors=args.n_neighbors,
        sample_size=args.sample_size,
        n_rounds=args.n_rounds,
        seed=args.seed,
    )

    report = _run_from_dir(cache_dir, config)
    print(format_report(report))

    out_csv = args.out_csv or (Path(cache_dir) / "clusterability_report.csv")
    report_to_frame(report).to_csv(out_csv, index=False)
    print(f"\nWrote per-condition report -> {out_csv}")

    # D-035 control: if a base-embedding dir is given, run it on the SAME protocol
    # (same config, same seed) and print the fine-tuned-vs-base comparison.
    if args.base_embeddings_dir is not None:
        base_report = _run_from_dir(Path(args.base_embeddings_dir), config)
        if base_report.extractor == report.extractor:
            print(
                f"\nWARNING: both runs report extractor '{report.extractor}'. "
                "The comparison may be against the same feature set."
            )
        print("\n" + "=" * 70)
        print("D-035 control: base-embedding run")
        print("=" * 70)
        print(format_report(base_report))

        comparison = compare_reports(report, base_report)
        print("\n=== fine-tuned vs base comparison (delta = fine-tuned - base) ===")
        print(comparison.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        cmp_csv = Path(cache_dir) / "clusterability_comparison.csv"
        comparison.to_csv(cmp_csv, index=False)
        print(f"\nWrote comparison -> {cmp_csv}")
        print(
            "\nD-035 interpretation rule (set in advance): base bipolar agreement "
            ">70% => drop the clusterability-failure argument; <40% => the fine-tuned "
            "number was substantially artifact. See DECISIONS.md D-035."
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
