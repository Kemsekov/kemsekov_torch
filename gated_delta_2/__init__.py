"""Gated Delta Rule-2 (arXiv:2605.22791) mixing layers.

Three interchangeable implementations of the same recurrent operator:

* ``GatedDelta2``       -- token-by-token reference loop (``serial.py``)
* ``GatedDelta2Scan``   -- chunked parallel scan with manual backward
                           (``chunked.py`` + ``scan.py``) -- recommended

All classes share the same parameters and buffers, so state dicts are
interchangeable between them.
"""

from .serial import GatedDelta2
from .chunked import GatedDelta2Scan

__version__ = "0.1.0"

__all__ = ["GatedDelta2", "GatedDelta2Scan"]
