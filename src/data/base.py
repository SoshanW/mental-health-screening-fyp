"""The loader abstraction (Interface Segregation + Liskov Substitution).

Single responsibility: define the one small interface every dataset loader
implements -- ``load() -> DataFrame`` -- so that :mod:`src.data.combine` can treat
all sources uniformly. The interface is deliberately minimal: DAIC's extra output
columns are additive *data*, not extra interface methods.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class DatasetLoader(Protocol):
    """Structural interface for a dataset loader.

    Implementations must return a pandas DataFrame conforming to the canonical
    schema (see :func:`src.data.schema.validate_schema`). The ``source`` property
    names the dataset (used for reporting and for label harmonization keys).
    """

    @property
    def source(self) -> str:
        """The dataset's canonical source name (e.g. ``"swmh"``)."""
        ...

    def load(self) -> pd.DataFrame:
        """Read, harmonize, and return rows in the canonical schema."""
        ...
