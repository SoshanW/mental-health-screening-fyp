"""Noise-model diagnostics layer (C1).

Per DECISIONS.md D-030, C1 is a *diagnostic*, not an estimator: it runs the
available no-clean-data noise estimators on the real data and measures whether
they are usable, which is what licenses the clinical-elicitation route in C2. This
package currently implements DECISIONS.md open item 2, the 2-NN clusterability
diagnostic.

Public surface (light imports only). The extraction helpers in :mod:`embeddings`
lazily import torch/transformers *inside* their functions, so importing this
package needs only numpy/pandas/scikit-learn; the deep-learning stack is required
only when actually extracting embeddings (a step meant to run on Colab against the
Drive checkpoint).

This layer consumes the checkpoint/frames produced by :mod:`src.modeling` and
never the reverse (one-directional dependency).
"""

from __future__ import annotations

from .clusterability import (
    NOISY_LABEL_CAVEAT,
    ClusterabilityConfig,
    ClusterabilityReport,
    format_report,
    report_to_frame,
    run_clusterability_diagnostic,
)
from .embeddings import (
    EmbeddingConfig,
    build_metadata_frame,
    default_embeddings_dir,
    extract_pooled_embeddings,
    load_embeddings,
    save_embeddings,
)

__all__ = [
    "EmbeddingConfig",
    "extract_pooled_embeddings",
    "build_metadata_frame",
    "save_embeddings",
    "load_embeddings",
    "default_embeddings_dir",
    "ClusterabilityConfig",
    "ClusterabilityReport",
    "run_clusterability_diagnostic",
    "report_to_frame",
    "format_report",
    "NOISY_LABEL_CAVEAT",
]
