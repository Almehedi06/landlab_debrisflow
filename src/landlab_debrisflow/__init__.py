"""Compatibility import path for the landlab_debrisflow repository."""

from __future__ import annotations

import importlib
import sys

from debris_landlab import __version__


for _submodule in ("components", "mmp"):
    sys.modules[f"{__name__}.{_submodule}"] = importlib.import_module(f"debris_landlab.{_submodule}")


__all__ = ["__version__"]
