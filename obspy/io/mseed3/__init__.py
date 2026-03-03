# -*- coding: utf-8 -*-
"""
MiniSEED 3 (mseed3) read support via the external 'pymseed' library.

This module exposes the plugin entry points:
- _is_mseed3(file): format detection
- _read_mseed3(file, ...): read into ObsPy Stream

Note: Writing MiniSEED 3 is not implemented yet.
"""
from .core import _is_mseed3, _read_mseed3  # noqa: F401
