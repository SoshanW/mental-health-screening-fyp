"""HOC transition-matrix estimator (Stage 2, C1).

A reimplementation of the HOC (High-Order Consensus) estimator of Zhu, Song and Liu
(2021, ICML, "Clusterability as an alternative to anchor points") in this repo's
conventions, run on the FINE-TUNED MentalBERT embeddings. Running it on the
fine-tuned features (not the base ones) is deliberate and recorded in DECISIONS.md
D-035: HOC's own protocol takes the extractor from a model trained to near-100%
training accuracy on the noisy labels, so the fine-tuned features are the fair
input. There is a pre-registered prediction for this run in D-036: read it before
interpreting any output.

Method (their Algorithm 1). Under 2-NN clusterability a post and its two nearest
neighbours share a true class i, and their noisy labels are conditionally
independent given i. With ``T[i][j] = P(noisy = j | true = i)`` (row-stochastic)
and prior ``p[i] = P(true = i)``, the r-th order consensus among the triple is:

    c1[j]        = sum_i p[i] T[i][j]
    c2[j1,j2]    = sum_i p[i] T[i][j1] T[i][j2]
    c3[j1,j2,j3] = sum_i p[i] T[i][j1] T[i][j2] T[i][j3]

We count the empirical c1/c2/c3 from 2-NN triples (over G rounds, |E| centres each),
then recover T and p by matching the model consensus to the empirical, exactly as
HOC does: T and p are softmax-reparameterised (``smt = Softmax(dim=1)`` over T rows,
``smp = Softmax(dim=0)`` over p, per their utils.py) and optimised with Adam.

DECISION (full consensus tensors, not HOC's cyclic-shift reduction): their code
reduces the 2nd/3rd-order consensus via a cyclic shift for large-K tractability,
which relies on the uniform-off-diagonal assumption D-018 flags as unsatisfiable at
these prevalences. At K = 4 the full tensors are 16 and 64 entries, so we match them
in full. This is the faithful objective of Algorithm 1 and avoids that assumption.
Correctness is checked by a synthetic recovery test (see tests/test_noise_hoc.py).

DECISION (within-E neighbour scope): 2-NN are found WITHIN the sampled subset E each
round, matching HOC's ``count_y`` (distCosine over the sampled features only) and the
correction in DECISIONS.md D-036. The neighbour helper is shared with
:mod:`src.noise.clusterability`.

DECISION (condition order): rows/columns of T follow ``sorted(unique(labels))`` =
(bipolar, depression, eating_disorder, schizophrenia), matching the order used by
the clusterability diagnostic and named in D-036. Note this is alphabetical, not the
``CONDITION_TO_ID`` enum order.

DECISION (no cleanup): the estimate is reported raw. It is NOT smoothed, clipped or
regularised. An implausible or seed-unstable matrix is the result (D-030/D-036).

Torch is imported lazily inside the optimiser, so the config, the consensus counting
and the data helpers stay importable without the deep-learning stack; only the Adam
optimisation needs torch.
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
from .clusterability import _two_nn_excluding_self
from .embeddings import default_embeddings_dir, load_embeddings

#: Observed noisy class proportions (D-036), for the prior-divergence diagnostic.
OBSERVED_NOISY_PROPORTIONS: dict[str, float] = {
    "bipolar": 0.038,
    "depression": 0.812,
    "eating_disorder": 0.096,
    "schizophrenia": 0.055,
}


@dataclass(frozen=True)
class HOCConfig:
    """Knobs for the HOC estimator (injected, not global).

    Attributes:
        n_rounds: ``G``, consensus-counting rounds (HOC Global default 50).
        sample_size: ``|E|`` centres sampled per round (HOC default 15000).
        max_iter: Adam steps for the T/p optimisation (HOC ``--max_iter`` 1500).
        lr: Adam learning rate (HOC default 0.1).
        metric: neighbour metric; ``"cosine"`` per HOC's ``distCosine``.
        order_weights: relative loss weights for (1st, 2nd, 3rd) order consensus.
            HOC's exact weights were not recoverable from the reference; equal
            weights are validated by the synthetic recovery test.
        device: torch device string, or None to auto-select.
    """

    n_rounds: int = 50
    sample_size: int = 15000
    max_iter: int = 1500
    lr: float = 0.1
    metric: str = "cosine"
    order_weights: tuple[float, float, float] = (1.0, 1.0, 1.0)
    device: str | None = None

    def __post_init__(self) -> None:
        if self.n_rounds < 1 or self.sample_size < 3 or self.max_iter < 1:
            raise ValueError("n_rounds>=1, sample_size>=3, max_iter>=1 required.")
        if len(self.order_weights) != 3 or any(w < 0 for w in self.order_weights):
            raise ValueError("order_weights must be 3 non-negative numbers.")


@dataclass(frozen=True)
class HOCResult:
    """Result of one HOC run (one seed).

    Attributes:
        conditions: class labels in row/column order of ``T`` and ``p``.
        T: ``(K, K)`` estimated transition matrix, ``T[i][j] = P(noisy=j|true=i)``.
        p: ``(K,)`` estimated true-class prior.
        noisy_marginal: ``(K,)`` observed noisy-label proportions in the data.
        seed: the RNG seed used.
        diagonal: ``T`` diagonal (``T[i][i]``).
        row_diagonally_dominant: condition -> whether ``T[i][i] > 0.5`` (HOC
            Assumption 2, per row).
        is_nonsingular: whether ``|det(T)|`` exceeds a small tolerance (Assumption 1).
        det: ``det(T)``.
        final_loss: consensus-matching loss at the last optimisation step.
    """

    conditions: list[str]
    T: np.ndarray
    p: np.ndarray
    noisy_marginal: np.ndarray
    seed: int
    diagonal: np.ndarray
    row_diagonally_dominant: dict[str, bool]
    is_nonsingular: bool
    det: float
    final_loss: float


def _count_consensus(
    x_norm: np.ndarray, labels_idx: np.ndarray, n_classes: int, config: HOCConfig, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Count empirical 1st/2nd/3rd-order consensus from within-E 2-NN triples.

    Returns ``(c1, c2, c3)`` normalised so each order sums to 1 (joint over its
    slots), pooled across ``n_rounds`` rounds.
    """
    n = len(labels_idx)
    sample_size = min(config.sample_size, n)
    if sample_size < 3:
        raise ValueError(f"Need >=3 points to form a 2-NN triple; got {sample_size}.")

    cnt1 = np.zeros(n_classes, dtype=np.float64)
    cnt2 = np.zeros((n_classes, n_classes), dtype=np.float64)
    cnt3 = np.zeros((n_classes, n_classes, n_classes), dtype=np.float64)

    for _ in range(config.n_rounds):
        e_idx = rng.choice(n, size=sample_size, replace=False)
        sub = x_norm[e_idx]
        sub_lab = labels_idx[e_idx]
        local_index = NearestNeighbors(
            n_neighbors=3, metric=config.metric, algorithm="brute", n_jobs=-1
        ).fit(sub)
        neigh = _two_nn_excluding_self(local_index, sub, np.arange(sample_size), 2)
        l0 = sub_lab
        l1 = sub_lab[neigh[:, 0]]  # 1st NN
        l2 = sub_lab[neigh[:, 1]]  # 2nd NN
        np.add.at(cnt1, l0, 1.0)
        np.add.at(cnt2, (l0, l1), 1.0)
        np.add.at(cnt3, (l0, l1, l2), 1.0)

    return cnt1 / cnt1.sum(), cnt2 / cnt2.sum(), cnt3 / cnt3.sum()


def _optimize_T_p(
    c1: np.ndarray, c2: np.ndarray, c3: np.ndarray, n_classes: int, config: HOCConfig
) -> tuple[np.ndarray, np.ndarray, float]:
    """Recover ``(T, p)`` matching the empirical consensus, via softmax + Adam.

    T is reparameterised row-softmax (``smt``, dim=1) so its rows are distributions;
    p is column-softmax (``smp``, dim=0) on the simplex, matching HOC's utils.py.
    """
    import torch

    device = config.device or ("cuda" if torch.cuda.is_available() else "cpu")
    c1t = torch.as_tensor(c1, dtype=torch.float64, device=device)
    c2t = torch.as_tensor(c2, dtype=torch.float64, device=device)
    c3t = torch.as_tensor(c3, dtype=torch.float64, device=device)

    # Init: a mild diagonal boost on T and p seeded from the empirical noisy
    # marginal (c1). The uniform/symmetric init lets two true-class rows collapse
    # into one another (a degenerate consensus-matching optimum); a mild diagonal
    # tie between true class i and noisy label i breaks that symmetry. It is mild
    # (row ~ softmax([2,0,...]) so the diagonal starts ~0.4 to 0.8, not pinned), so
    # the consensus objective is still free to move the estimate off the diagonal.
    t_init = np.full((n_classes, n_classes), 0.0)
    np.fill_diagonal(t_init, 2.0)
    t_logits = torch.tensor(t_init, dtype=torch.float64, device=device, requires_grad=True)
    p_logits = torch.tensor(
        np.log(np.clip(c1, 1e-6, None)), dtype=torch.float64, device=device, requires_grad=True
    )
    smt = torch.nn.Softmax(dim=1)
    smp = torch.nn.Softmax(dim=0)
    opt = torch.optim.Adam([t_logits, p_logits], lr=config.lr)
    w1, w2, w3 = config.order_weights

    loss_val = float("nan")
    for _ in range(config.max_iter):
        opt.zero_grad()
        t_mat = smt(t_logits)
        p_vec = smp(p_logits)
        m1 = torch.einsum("i,ij->j", p_vec, t_mat)
        m2 = torch.einsum("i,ij,ik->jk", p_vec, t_mat, t_mat)
        m3 = torch.einsum("i,ij,ik,il->jkl", p_vec, t_mat, t_mat, t_mat)
        loss = (
            w1 * torch.norm(c1t - m1)
            + w2 * torch.norm(c2t - m2)
            + w3 * torch.norm(c3t - m3)
        )
        loss.backward()
        opt.step()
        loss_val = float(loss.detach())

    with torch.no_grad():
        t_final = smt(t_logits).detach().cpu().numpy()
        p_final = smp(p_logits).detach().cpu().numpy()
    return t_final, p_final, loss_val


def estimate_hoc(
    features: np.ndarray,
    labels: np.ndarray | pd.Series,
    config: HOCConfig = HOCConfig(),
    seed: int = 0,
) -> HOCResult:
    """Run one HOC estimate (one seed) on ``features`` with noisy ``labels``."""
    labels = np.asarray(labels).astype(str)
    n = int(features.shape[0])
    if n != len(labels):
        raise ValueError(f"features/labels length mismatch: {n} vs {len(labels)}.")

    conditions = sorted(set(labels.tolist()))
    n_classes = len(conditions)
    idx = {c: i for i, c in enumerate(conditions)}
    labels_idx = np.array([idx[l] for l in labels])

    x_norm = normalize(features).astype(np.float32, copy=False)
    rng = np.random.default_rng(seed)
    c1, c2, c3 = _count_consensus(x_norm, labels_idx, n_classes, config, rng)
    t_mat, p_vec, loss = _optimize_T_p(c1, c2, c3, n_classes, config)

    noisy_marginal = np.array([(labels == c).mean() for c in conditions])
    diagonal = np.diag(t_mat).copy()
    row_dd = {conditions[i]: bool(t_mat[i, i] > 0.5) for i in range(n_classes)}
    det = float(np.linalg.det(t_mat))

    return HOCResult(
        conditions=conditions,
        T=t_mat,
        p=p_vec,
        noisy_marginal=noisy_marginal,
        seed=seed,
        diagonal=diagonal,
        row_diagonally_dominant=row_dd,
        is_nonsingular=abs(det) > 1e-8,
        det=det,
        final_loss=loss,
    )


def run_hoc_multiseed(
    features: np.ndarray,
    labels: np.ndarray | pd.Series,
    seeds: list[int],
    config: HOCConfig = HOCConfig(),
) -> list[HOCResult]:
    """Run HOC once per seed. Report the spread, never a single point estimate."""
    if len(seeds) < 1:
        raise ValueError("Provide at least one seed (>=5 recommended).")
    return [estimate_hoc(features, labels, config, seed=s) for s in seeds]


def aggregate_results(results: list[HOCResult]) -> dict[str, object]:
    """Mean and per-element std of T and p across seeds (the seed spread)."""
    if not results:
        raise ValueError("No results to aggregate.")
    conditions = results[0].conditions
    stack_t = np.stack([r.T for r in results])
    stack_p = np.stack([r.p for r in results])
    return {
        "conditions": conditions,
        "mean_T": stack_t.mean(axis=0),
        "std_T": stack_t.std(axis=0),
        "mean_p": stack_p.mean(axis=0),
        "std_p": stack_p.std(axis=0),
        "n_seeds": len(results),
    }


def check_pre_registered_prediction(
    mean_T: np.ndarray, std_T: np.ndarray, conditions: list[str]
) -> dict[str, object]:
    """Evaluate the D-036 pre-registered prediction against an aggregated T.

    Prediction (D-036): all four diagonals above ~0.85 (noise suppressed by
    fine-tuning). Falsifier: bipolar diagonal below ~0.7 with a dominant
    bipolar-to-depression off-diagonal, stable across seeds.
    """
    diag = np.diag(mean_T)
    prediction_holds = bool(np.all(diag >= 0.85))

    bip_falsifier = False
    bip_detail: dict[str, object] = {}
    if "bipolar" in conditions:
        b = conditions.index("bipolar")
        off = {conditions[j]: float(mean_T[b, j]) for j in range(len(conditions)) if j != b}
        dominant = max(off, key=off.get) if off else None
        bip_diag_std = float(std_T[b, b])
        bip_falsifier = bool(mean_T[b, b] < 0.7 and dominant == "depression")
        bip_detail = {
            "bipolar_diagonal": float(mean_T[b, b]),
            "bipolar_diagonal_std": bip_diag_std,
            "bipolar_offdiagonal": off,
            "dominant_offdiagonal": dominant,
        }

    return {
        "prediction_holds": prediction_holds,
        "falsifier_met": bip_falsifier,
        "diagonals": {conditions[i]: float(diag[i]) for i in range(len(conditions))},
        "bipolar": bip_detail,
    }


def make_clusterable_noisy_data(
    transition: np.ndarray,
    prior: np.ndarray,
    conditions: list[str],
    n: int,
    dim: int = 16,
    spread: float = 0.02,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate clusterable data from a known ``(T, p)`` for the recovery test.

    Each sample draws a true class from ``prior``, is placed tightly near that
    class's orthonormal centroid (so 2-NN share the true class: clusterability
    holds), and is given a noisy label from ``transition[true]``.
    """
    rng = np.random.default_rng(seed)
    n_classes = len(conditions)
    centroids = np.eye(dim, dtype=np.float32)[:n_classes]
    true = rng.choice(n_classes, size=n, p=prior)
    features = centroids[true] + spread * rng.standard_normal((n, dim)).astype(np.float32)
    noisy_idx = np.array([rng.choice(n_classes, p=transition[t]) for t in true])
    labels = np.array([conditions[j] for j in noisy_idx])
    return features, labels


def format_multiseed(results: list[HOCResult], agg: dict[str, object]) -> str:
    """Human-readable multi-seed report: bipolar row first, then mean+/-std T."""
    conditions = agg["conditions"]
    mean_t = np.asarray(agg["mean_T"])
    std_t = np.asarray(agg["std_T"])
    lines: list[str] = ["=== HOC transition matrix estimate (fine-tuned embeddings) ==="]
    lines.append(f"conditions (row=true, col=noisy): {conditions}")
    lines.append(f"seeds: {[r.seed for r in results]}")

    if "bipolar" in conditions:
        b = conditions.index("bipolar")
        lines.append("")
        lines.append("BIPOLAR ROW (true=bipolar -> noisy), mean +/- std across seeds:")
        for j, c in enumerate(conditions):
            lines.append(f"  -> {c:16s} {mean_t[b, j]:.4f} +/- {std_t[b, j]:.4f}")

    lines.append("")
    lines.append("mean T (row=true, col=noisy):")
    lines.append(pd.DataFrame(mean_t, index=[f"true_{c}" for c in conditions],
                              columns=[f"noisy_{c}" for c in conditions]).to_string(
                              float_format=lambda v: f"{v:.4f}"))
    lines.append("")
    lines.append("per-element std across seeds:")
    lines.append(pd.DataFrame(std_t, index=[f"true_{c}" for c in conditions],
                              columns=[f"noisy_{c}" for c in conditions]).to_string(
                              float_format=lambda v: f"{v:.4f}"))

    lines.append("")
    lines.append("per-seed diagnostics (Assumptions 1 and 2):")
    for r in results:
        bad_rows = [c for c, ok in r.row_diagonally_dominant.items() if not ok]
        lines.append(
            f"  seed {r.seed}: nonsingular={r.is_nonsingular} det={r.det:.4e}  "
            f"diag={np.round(r.diagonal, 3).tolist()}  "
            f"rows NOT diagonally dominant: {bad_rows or 'none'}  loss={r.final_loss:.4e}"
        )

    lines.append("")
    lines.append("estimated prior p vs observed noisy marginal (divergence is informative):")
    mean_p = np.asarray(agg["mean_p"])
    for i, c in enumerate(conditions):
        obs = OBSERVED_NOISY_PROPORTIONS.get(c, results[0].noisy_marginal[i])
        lines.append(f"  {c:16s} p={mean_p[i]:.4f}  noisy_marginal={results[0].noisy_marginal[i]:.4f}  (D-036 obs {obs})")

    pred = check_pre_registered_prediction(mean_t, std_t, conditions)
    lines.append("")
    lines.append("--- D-036 pre-registered prediction (read the entry; report raw result first) ---")
    lines.append(f"  prediction (all diagonals >= 0.85) holds: {pred['prediction_holds']}")
    lines.append(f"  falsifier (bipolar diag < 0.7 and dominant off-diagonal = depression) met: {pred['falsifier_met']}")
    lines.append("  If falsified: per D-036, drop the clusterability line of argument; the C1 "
                 "argument then rests on the point-estimate-vs-region and finite-sample-rate "
                 "points (D-018), which survive either outcome.")
    return "\n".join(lines)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="HOC transition-matrix estimator (Stage 2, C1).")
    p.add_argument("--artifacts-root", type=Path, default=None)
    p.add_argument("--embeddings-dir", type=Path, default=None,
                   help="FINE-TUNED embeddings cache (default <Models>/embeddings/<split>). "
                        "Do NOT point this at base embeddings (D-035).")
    p.add_argument("--split", type=str, default="train")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4],
                   help="RNG seeds; >=5 recommended (report the spread, D-030/D-036).")
    p.add_argument("--n-rounds", type=int, default=HOCConfig().n_rounds)
    p.add_argument("--sample-size", type=int, default=HOCConfig().sample_size)
    p.add_argument("--max-iter", type=int, default=HOCConfig().max_iter)
    p.add_argument("--lr", type=float, default=HOCConfig().lr)
    p.add_argument("--out-csv", type=Path, default=None)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.embeddings_dir is not None:
        cache_dir = Path(args.embeddings_dir)
    else:
        artifacts = ArtifactPaths(root=args.artifacts_root) if args.artifacts_root else ArtifactPaths.default()
        cache_dir = default_embeddings_dir(artifacts, args.split)

    features, metadata = load_embeddings(cache_dir)
    extractor = str(metadata["extractor"].iloc[0]) if "extractor" in metadata.columns and len(metadata) else "unknown"
    if extractor == "base":
        print("WARNING: these embeddings are the BASE extractor. D-035 says HOC must run "
              "on the FINE-TUNED embeddings; results on base features are not a fair test.")
    labels = metadata["condition"].to_numpy()

    config = HOCConfig(
        n_rounds=args.n_rounds,
        sample_size=args.sample_size,
        max_iter=args.max_iter,
        lr=args.lr,
    )
    results = run_hoc_multiseed(features, labels, args.seeds, config)
    agg = aggregate_results(results)
    print(format_multiseed(results, agg))

    out_csv = args.out_csv or (Path(cache_dir) / "hoc_mean_T.csv")
    conditions = agg["conditions"]
    pd.DataFrame(np.asarray(agg["mean_T"]), index=[f"true_{c}" for c in conditions],
                 columns=[f"noisy_{c}" for c in conditions]).to_csv(out_csv)
    print(f"\nWrote mean T -> {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
