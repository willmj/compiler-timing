"""Arm scratchpad_substage_timing without modifying the torch-spyre tree.

The recorder normally lives in-tree at
torch_spyre/_inductor/timing_recorder.py. Here it is aliased into that
namespace from this scratch dir instead, so the tree stays pristine and
the patch's own import path is exercised unchanged.

`import torch` must come first: torch's backend autoload imports
torch_spyre, so reaching torch_spyre._inductor before torch is loaded
deadlocks on a partially initialized module.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: F401  (must precede torch_spyre; see docstring)
import torch_spyre._inductor as _si

import timing_recorder as _tr

sys.modules["torch_spyre._inductor.timing_recorder"] = _tr
_si.timing_recorder = _tr

import pass_pipeline_timing as ppt  # noqa: F401  (import-time alias only)
import scratchpad_substage_timing as sst

if __name__ == "__main__":
    sys.exit(sst.main(sys.argv[1:]))
