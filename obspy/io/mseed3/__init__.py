# -*- coding: utf-8 -*-
"""
MiniSEED v2 and v3 (mseed3) support for via the external 'pymseed' package.

This module exposes the plugin entry points:
- _is_mseed3(file): format detection
- _read_mseed3(file, ...): read into ObsPy Stream

"""

from .core import _is_mseed3, _read_mseed3  # noqa: F401
