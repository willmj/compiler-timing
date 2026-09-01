# Copyright 2025-2026 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Set `optimize_restickify.BEAM_WIDTH` with no other instrumentation.

`restickify_beam_timing` can also override the width, but it wraps
`BeamState.__init__` and `EdgeCostMap.cost`, which fire once per state and per
edge lookup and cost roughly 10% at large graph sizes. This module exists so a
width A/B can be timed on a clean arm.

`Frontier(BEAM_WIDTH)` reads the module global when the search starts, so
assigning it before the first compile is sufficient.

Enable with SPYRE_RESTICKIFY_BEAM_WIDTH=<int>.
"""

from __future__ import annotations

import os
import sys

_INSTALLED = False


def install() -> int | None:
    """Apply the width override. Returns the width applied, or None."""
    global _INSTALLED
    if _INSTALLED:
        return None
    raw = os.environ.get("SPYRE_RESTICKIFY_BEAM_WIDTH")
    if not raw:
        return None
    _INSTALLED = True
    width = int(raw)
    if width < 1:
        raise ValueError(f"SPYRE_RESTICKIFY_BEAM_WIDTH={width} must be >= 1")

    from torch_spyre._inductor import optimize_restickify as om

    om.BEAM_WIDTH = width
    print(f"restickify BEAM_WIDTH := {width}", file=sys.stderr)
    return width


__all__ = ["install"]
