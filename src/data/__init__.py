"""Data-ingestion layer for multi-condition mental-health screening.

Loads three datasets (SWMH, Low et al., DAIC-WoZ), harmonizes their labels onto a
single :class:`~src.data.schema.Condition` vocabulary, and emits clean pandas
DataFrames in one canonical schema for downstream training/calibration code.

Public surface:

* :class:`~src.data.schema.Condition`, :data:`~src.data.schema.POC_CONDITIONS`,
  :func:`~src.data.schema.validate_schema`
* :func:`~src.data.label_map.harmonize_label`
* :class:`~src.data.config.DataPaths`
* :class:`~src.data.base.DatasetLoader`
* loaders: :class:`~src.data.loaders.SWMHLoader`,
  :class:`~src.data.loaders.LowEtAlLoader`,
  :class:`~src.data.loaders.DAICWoZLoader`
* :func:`~src.data.combine.combine_sources`
"""

from __future__ import annotations

from .base import DatasetLoader
from .combine import combine_sources
from .config import DataPaths
from .label_map import harmonize_label
from .loaders import DAICWoZLoader, LowEtAlLoader, SWMHLoader
from .schema import (
    CANONICAL_COLUMNS,
    POC_CONDITIONS,
    Condition,
    validate_schema,
)

__all__ = [
    "Condition",
    "POC_CONDITIONS",
    "CANONICAL_COLUMNS",
    "validate_schema",
    "harmonize_label",
    "DataPaths",
    "DatasetLoader",
    "SWMHLoader",
    "LowEtAlLoader",
    "DAICWoZLoader",
    "combine_sources",
]
