# -*- coding: utf-8 -*-

"""Python for Seismological Observatories."""

# See http://peak.telecommunity.com/DevCenter/setuptools#namespace-packages
try:
    __import__('pkg_resources').declare_namespace(__name__)
except ImportError:
    from pkgutil import extend_path
    __path__ = extend_path(__path__, __name__)


from obspy.core import Trace
from obspy.core import Stream
#
import core
import filter
import gse2
import imaging
import mseed
import numpy
import parser
import picker
import sac
import util
import wav
import xseed
