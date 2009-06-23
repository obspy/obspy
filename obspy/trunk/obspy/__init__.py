# -*- coding: utf-8 -*-

"""Python for Seismological Observatories."""

# See http://peak.telecommunity.com/DevCenter/setuptools#namespace-packages
try:
    __import__('pkg_resources').declare_namespace(__name__)
except ImportError:
    from pkgutil import extend_path
    __path__ = extend_path(__path__, __name__)

from obspy.core import util
from obspy.core import Trace, Stream, read, supportedFormats
from obspy.core import numpy
from obspy.core import parser

__all__ = ['util', 'numpy', 'parser']
