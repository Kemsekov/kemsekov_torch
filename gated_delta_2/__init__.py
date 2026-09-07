"""Gated Delta Rule-2 (arXiv:2605.22791) mixing layers.

Three interchangeable implementations of the same recurrent operator:

* ``GatedDelta2``       -- token-by-token reference loop (``serial.py``)
* ``GatedDelta2Scan``   -- chunked parallel scan with manual backward
                           (``chunked.py`` + ``scan.py``) -- recommended
* ``GatedDelta2ScanDense`` -- per-token dense-map scan, for comparison only
                           (``dense.py``)

All classes share the same parameters and buffers, so state dicts are
interchangeable between them.
"""

from .serial import GatedDelta2
from .chunked import GatedDelta2Scan
from .dense import GatedDelta2ScanDense

__version__ = "0.1.0"

__all__ = ["GatedDelta2", "GatedDelta2Scan", "GatedDelta2ScanDense"]
