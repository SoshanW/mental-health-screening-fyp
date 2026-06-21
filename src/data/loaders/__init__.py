"""Concrete dataset loaders.

Each module here implements one dataset behind the
:class:`~src.data.base.DatasetLoader` interface. Adding a future dataset means
adding one new module here -- no edits to existing loaders or to
:mod:`src.data.combine` (Open/Closed).
"""

from __future__ import annotations

from .daic_woz import DAICWoZLoader
from .low_et_al import LowEtAlLoader
from .swmh import SWMHLoader

__all__ = ["SWMHLoader", "LowEtAlLoader", "DAICWoZLoader"]
