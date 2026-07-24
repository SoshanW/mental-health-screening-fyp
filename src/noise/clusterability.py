"""2-NN clusterability diagnostic for the condition-dependent noise model (C1).

DECISIONS.md open item 2, D-034 (the diagnostic falsified its own prediction),
D-035 (base-embedding control) and D-036 (base result + neighbour-scope
correction). This measures, per condition, how often a post's two nearest
neighbours in MentalBERT embedding space carry the *same noisy proxy label* as the
post. It is the empirical test of the clusterability assumption HOC (Zhu, Song and
Liu, 2021) relies on: if neighbours in feature space do not share a post's label,
the estimator has no signal to recover a transition matrix from.

Why this was run (see D-030): HOC assumes a post's nearest neighbours share its
*true* class. D-034 records that the prediction (high depression, low bipolar) was
FALSIFIED on fine-tuned embeddings; D-035/D-036 then showed via a base-MentalBERT
control that the apparent structure was largely a fine-tuning artifact (bipolar
dropped from 86.7% fine-tuned to ~29% base). The statistic is descriptive and
ONE-SIDED: high agreement falsifies the clusterability-failure argument; low
agreement is only consistent with it. Seed-instability of the noise estimators
(Stage 2) is the load-bearing evidence.

DECISION (neighbour search scope, D-036): HOC's Algorithm 1 finds each centre's
2-NN WITHIN the sampled subset E (n1 = arg min over n' in E, n' != n), not over the
full dataset. Zhu, Song and Liu restrict it deliberately, to preserve the i.i.d.
property of the 3-tuples so the consensus estimates stay consistent (their E*_3
disjointness condition, Section 3.3). ``scope="within_e"`` is therefore the DEFAULT
because this diagnostic exists to characterise HOC's operating regime;
``scope="full_dataset"`` searches the whole corpus and so measures something
slightly easier than what HOC faces (D-036: +4 to +6pp on the rare conditions).
Both are reported; within-E is primary.

DECISION (metrics reported): the headline is PER-NEIGHBOUR agreement (fraction of
neighbour slots sharing the centre's noisy label), the diagonal of the
row-normalised neighbour-label distribution, matching the D-034/D-036 tables.
``chance`` is the class base rate; ``lift`` is agreement / chance. Per-condition
same-class candidate counts (in E vs the full corpus) are reported because the rare
conditions are neighbour-starved in exactly the regime HOC operates in (D-036); at
|E| = 15000 bipolar has ~566 same-class candidates versus ~4221 in the full data.

DECISION (distance metric): neighbours are ranked by cosine, as in Zhu, Song and
Liu (2021). We L2-normalise the features and use scikit-learn's ``metric="cosine"``.
Verified 2026-07-24 that cosine-distance ordering (1 - cos similarity) is identical
to ranking by negative cosine similarity, and the query point's own row is returned
first (distance 0) so it can be dropped as "self". Euclidean is NOT substituted.

DECISION (no full pairwise matrix): we follow HOC's Monte-Carlo approach, sampling
``sample_size`` centre indices per round over ``n_rounds`` rounds and averaging.
Under ``within_e`` a fresh brute-force cosine index is fitted on the sampled subset
each round; under ``full_dataset`` one index is fitted on the whole corpus and the
sampled centres query it. Either way no full n x n distance matrix is materialised.

On fp16 (D-036): the on-disk fp16 storage barely perturbs base-space neighbour
ordering (median 2nd-vs-3rd similarity gaps of ~0.002 to 0.003 versus an fp16 error
scale ~1e-3). Counterintuitively the fine-tuned space has much smaller gaps and is
the more fp16-vulnerable one, but it does not matter there because near-neighbours
almost always share a label. To bound the effect, extract the base features at
``--output-dtype float32`` into a separate cache and compare (see the module CLI and
the notebook). This diagnostic upcasts to float32 internally, so it adds no rounding
of its own beyond whatever dtype the cache was stored in.

Torch-free: numpy + scikit-learn only, so it runs locally on the cached embeddings.
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

#: Allowed neighbour search scopes.
SCOPES: tuple[str, ...] = ("within_e", "full_dataset")

#: The caveat printed and returned with every report, so no reader mistakes this
#: for a true-label clusterability measurement, nor treats it as confirmatory.
#: DECISION (one-sided, descriptive): see DECISIONS.md open item 2 (amended
#: 2026-07-24), D-034 and D-036.
NOISY_LABEL_CAVEAT: str = (
    "Agreement is measured against NOISY proxy labels, not true labels. "
    "Zhu, Song and Liu (2021, Table 3) report ~78-88% feasible 2-NN tuples on "
    "noisy CIFAR-10 against TRUE labels; those numbers are NOT directly comparable "
    "to these. This statistic is DESCRIPTIVE, not confirmatory: it is confounded "
    "between the noise level and clusterability failure and cannot separate them. "
    "It is a ONE-SIDED test: high agreement would falsify the clusterability-failure "
    "argument, but low agreement is only consistent with it and does not establish "
    "it. Seed-instability of the noise estimators (Stage 2) is the primary evidence. "
    "See DECISIONS.md D-034, D-036 and open item 2 (amended 2026-07-24)."
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
        scope: ``"within_e"`` (HOC's Algorithm 1: 2-NN within the sampled subset;
            DEFAULT) or ``"full_dataset"`` (2-NN over the whole corpus). See the
            module DECISION note and D-036.
    """

    n_neighbors: int = 2
    sample_size: int = 15000
    n_rounds: int = 20
    seed: int = 42
    metric: str = "cosine"
    scope: str = "within_e"

    def __post_init__(self) -> None:
        if self.n_neighbors < 1:
            raise ValueError(f"n_neighbors must be >= 1; got {self.n_neighbors}.")
        if self.sample_size < 1 or self.n_rounds < 1:
            raise ValueError("sample_size and n_rounds must be >= 1.")
        if self.scope not in SCOPES:
            raise ValueError(f"scope must be one of {SCOPES}; got {self.scope!r}.")


@dataclass(frozen=True)
class ClusterabilityReport:
    """Per-condition results of the diagnostic.

    Attributes:
        per_neighbor_agreement: condition -> mean over rounds of the fraction of
            that condition's centres' neighbour slots that share its noisy label.
            Headline number; matches the D-034/D-036 tables.
        per_neighbor_agreement_std: condition -> std of that per-round fraction.
        chance: condition -> class base rate over the whole indexed set.
        lift: condition -> ``per_neighbor_agreement / chance``.
        neighbor_label_distribution: centre condition -> {neighbour condition ->
            fraction of that centre's neighbour slots}, pooled across rounds and
            row-normalised. Its diagonal is ``per_neighbor_agreement`` (pooled).
        candidates_in_e: condition -> mean count of that class inside the sampled
            subset E per round (the same-class pool a centre can match against
            under ``within_e``; candidates = this minus the centre itself).
        candidates_full: condition -> count of that class in the whole corpus (the
            same-class pool under ``full_dataset``).
        agreement: condition -> "both neighbours match" fraction (secondary view).
        agreement_std: std of the per-round ``agreement`` fraction.
        pattern_counts: condition -> ``{"both_match","one_match","none_match"}``.
        center_counts: condition -> total centres of that condition across rounds.
        config: the config used.
        n_posts: number of posts the diagnostic saw.
        scope: the neighbour search scope actually used.
        extractor: label of the feature extractor ("finetuned"/"base"/"unknown").
        caveat: :data:`NOISY_LABEL_CAVEAT`.
    """

    per_neighbor_agreement: dict[str, float]
    per_neighbor_agreement_std: dict[str, float]
    chance: dict[str, float]
    lift: dict[str, float]
    neighbor_label_distribution: dict[str, dict[str, float]]
    candidates_in_e: dict[str, float]
    candidates_full: dict[str, int]
    agreement: dict[str, float]
    agreement_std: dict[str, float]
    pattern_counts: dict[str, dict[str, int]]
    center_counts: dict[str, int]
    config: ClusterabilityConfig
    n_posts: int
    scope: str
    extractor: str = "unknown"
    caveat: str = NOISY_LABEL_CAVEAT


def _two_nn_excluding_self(
    index: NearestNeighbors, query: np.ndarray, centers: np.ndarray, k: int
) -> np.ndarray:
    """Return the ``(len(centers), k)`` neighbour indices, self removed.

    ``index`` is fitted on the search pool; ``query`` is the normalised centre rows;
    ``centers`` are the ids (into the fitted pool) of those centres, used to drop the
    self row. We ask for ``k + 1`` neighbours and remove the (at most one) self entry
    per centre, robust even if an exact-duplicate embedding displaces self from
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
        config: sampling / neighbour / scope knobs.
        extractor: label recorded on the report for provenance.

    Raises:
        ValueError: on a features/labels length mismatch, too few posts, or (under
            ``within_e``) an effective sample size below ``n_neighbors + 1``.
    """
    labels = np.asarray(labels).astype(str)
    n = int(features.shape[0])
    if n != len(labels):
        raise ValueError(f"features/labels length mismatch: {n} vs {len(labels)}.")
    if n <= config.n_neighbors:
        raise ValueError(
            f"Need more than n_neighbors={config.n_neighbors} posts; got {n}."
        )

    k = config.n_neighbors
    sample_size = min(config.sample_size, n)
    if config.scope == "within_e" and sample_size < k + 1:
        raise ValueError(
            f"scope='within_e' needs sample_size >= n_neighbors+1={k + 1}; "
            f"effective sample_size is {sample_size}."
        )

    # L2-normalise then use cosine so ordering matches negative cosine similarity.
    x_norm = normalize(features).astype(np.float32, copy=False)

    uniq, counts = np.unique(labels, return_counts=True)
    chance = {str(u): float(c) / n for u, c in zip(uniq, counts)}
    candidates_full = {str(u): int(c) for u, c in zip(uniq, counts)}
    conditions = sorted(chance)

    # Under full_dataset the pool is the whole corpus, fitted once. Under within_e a
    # fresh index is fitted on the sampled subset each round.
    full_index: NearestNeighbors | None = None
    if config.scope == "full_dataset":
        full_index = NearestNeighbors(
            n_neighbors=k + 1, metric=config.metric, algorithm="brute", n_jobs=-1
        ).fit(x_norm)

    rng = np.random.default_rng(config.seed)

    both_round_frac: dict[str, list[float]] = {c: [] for c in conditions}
    perneigh_round_frac: dict[str, list[float]] = {c: [] for c in conditions}
    candidates_e_round: dict[str, list[int]] = {c: [] for c in conditions}
    pattern: dict[str, dict[str, int]] = {
        c: {"both_match": 0, "one_match": 0, "none_match": 0} for c in conditions
    }
    center_counts: dict[str, int] = {c: 0 for c in conditions}
    neigh_counts: dict[str, dict[str, int]] = {
        c: {c2: 0 for c2 in conditions} for c in conditions
    }
    slot_totals: dict[str, int] = {c: 0 for c in conditions}

    for _ in range(config.n_rounds):
        e_idx = rng.choice(n, size=sample_size, replace=False)
        e_labels = labels[e_idx]
        for c in conditions:
            candidates_e_round[c].append(int((e_labels == c).sum()))

        if config.scope == "within_e":
            # HOC Algorithm 1: 2-NN within the sampled subset E.
            sub = x_norm[e_idx]
            local_ids = np.arange(sample_size)
            local_index = NearestNeighbors(
                n_neighbors=k + 1, metric=config.metric, algorithm="brute", n_jobs=-1
            ).fit(sub)
            neigh_local = _two_nn_excluding_self(local_index, sub, local_ids, k)
            center_labels = e_labels
            neigh_labels = e_labels[neigh_local]
        else:
            # full_dataset: sampled centres search the whole corpus.
            assert full_index is not None
            neigh_global = _two_nn_excluding_self(full_index, x_norm[e_idx], e_idx, k)
            center_labels = e_labels
            neigh_labels = labels[neigh_global]

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
    candidates_in_e = {c: float(np.mean(candidates_e_round[c])) for c in conditions}

    return ClusterabilityReport(
        per_neighbor_agreement=per_neighbor_agreement,
        per_neighbor_agreement_std=per_neighbor_agreement_std,
        chance=chance,
        lift=lift,
        neighbor_label_distribution=neighbor_label_distribution,
        candidates_in_e=candidates_in_e,
        candidates_full=candidates_full,
        agreement=agreement,
        agreement_std=agreement_std,
        pattern_counts=pattern,
        center_counts=center_counts,
        config=config,
        n_posts=n,
        scope=config.scope,
        extractor=extractor,
    )


def report_to_frame(report: ClusterabilityReport) -> pd.DataFrame:
    """Flatten a report into a per-condition DataFrame for CSV output / printing."""
    rows = []
    for cond in sorted(report.per_neighbor_agreement):
        counts = report.pattern_counts[cond]
        rows.append(
            {
                "condition": cond,
                "n_centers_total": report.center_counts[cond],
                "per_neighbor_agreement": report.per_neighbor_agreement[cond],
                "per_neighbor_agreement_std": report.per_neighbor_agreement_std[cond],
                "chance": report.chance[cond],
                "lift_over_chance": report.lift[cond],
                "candidates_in_e": report.candidates_in_e[cond],
                "candidates_full": report.candidates_full[cond],
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
    return pd.DataFrame(data, index=[f"centre_{c}" for c in conds])


def format_report(report: ClusterabilityReport) -> str:
    """Human-readable summary, sorted by agreement so the weakest condition is last."""
    frame = report_to_frame(report).sort_values("per_neighbor_agreement", ascending=False)
    dist = neighbor_distribution_to_frame(report)
    lines = [
        f"=== 2-NN clusterability diagnostic [extractor: {report.extractor}  "
        f"scope: {report.scope}] ===",
        f"posts: {report.n_posts}   sample_size={report.config.sample_size}  "
        f"n_rounds={report.config.n_rounds}  k={report.config.n_neighbors}  "
        f"metric={report.config.metric}",
        "",
        frame.to_string(index=False, float_format=lambda v: f"{v:.4f}"),
        "",
        "neighbour-label distribution (rows = centre, cols = neighbour, row-normalised):",
        dist.to_string(float_format=lambda v: f"{v:.4f}"),
        "",
        "candidates_in_e = same-class pool inside E (candidates a centre can match "
        "against = this minus 1). D-036: the rare conditions are neighbour-starved "
        "in HOC's operating regime.",
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
    """Side-by-side per-condition comparison of two runs (extractor or scope).

    ``delta`` is ``agreement_a - agreement_b``. For the D-035 control, A is the
    fine-tuned run and B the base run, so a large positive delta means the apparent
    clustering is largely a fine-tuning artifact. Also used for the D-036 scope
    comparison (within_e vs full_dataset) and the fp16-vs-fp32 bounding check.
    """
    # Default labels use the extractor; the scope comparison passes explicit names
    # (within_e/full_dataset) since there both runs share an extractor.
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


def _run_from_dir(
    cache_dir: Path, config: ClusterabilityConfig
) -> ClusterabilityReport:
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
                   help="Second cache dir (e.g. base MentalBERT) for the D-035 side-by-side comparison.")
    p.add_argument("--split", type=str, default="train")
    p.add_argument("--scope", type=str, default=ClusterabilityConfig().scope, choices=SCOPES,
                   help="Neighbour search scope (D-036): within_e (HOC, default) or full_dataset.")
    p.add_argument("--also-full-dataset", action="store_true",
                   help="Additionally run scope=full_dataset on the primary cache and print the "
                        "within_e-vs-full comparison (D-036).")
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

    base_config = dict(
        n_neighbors=args.n_neighbors,
        sample_size=args.sample_size,
        n_rounds=args.n_rounds,
        seed=args.seed,
    )
    config = ClusterabilityConfig(scope=args.scope, **base_config)

    report = _run_from_dir(cache_dir, config)
    print(format_report(report))

    out_csv = args.out_csv or (Path(cache_dir) / f"clusterability_report_{args.scope}.csv")
    report_to_frame(report).to_csv(out_csv, index=False)
    print(f"\nWrote per-condition report -> {out_csv}")

    # D-036: within_e vs full_dataset on the same cache.
    if args.also_full_dataset and args.scope == "within_e":
        full_report = _run_from_dir(cache_dir, ClusterabilityConfig(scope="full_dataset", **base_config))
        print("\n" + "=" * 70)
        print("D-036 scope check: full_dataset run (same cache)")
        print("=" * 70)
        print(format_report(full_report))
        scope_cmp = compare_reports(report, full_report, name_a="within_e", name_b="full_dataset")
        print("\n=== within_e vs full_dataset (delta = within_e - full_dataset) ===")
        print(scope_cmp.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        scope_cmp.to_csv(Path(cache_dir) / "clusterability_scope_comparison.csv", index=False)

    # D-035 control: fine-tuned vs base, same scope.
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
        comparison.to_csv(Path(cache_dir) / "clusterability_comparison.csv", index=False)
        print(
            "\nD-035 interpretation rule (set in advance): base bipolar agreement "
            ">70% => drop the clusterability-failure argument; <40% => the fine-tuned "
            "number was substantially artifact. See DECISIONS.md D-035/D-036."
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
