# -*- coding: utf-8 -*-

"""Python for observatories."""

from obspy.core.core import *
from obspy.core import util
from obspy.core import parser

# See http://peak.telecommunity.com/DevCenter/setuptools#namespace-packages
try:
    __import__('pkg_resources').declare_namespace(__name__)
except ImportError:
    from pkgutil import extend_path
    __path__ = extend_path(__path__, __name__)
