# -*- coding: utf-8 -*-
"""
obspy.core.util - Various utilities for ObsPy
=============================================

.. note:: Please import all utilities within your custom applications from this
    module rather than from any sub module, e.g.

    >>> from obspy.core.util import OrderedDict  # good

    instead of

    >>> from obspy.core.util.obspy_types import OrderedDict  # bad

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""

# import order matters - NamedTemporaryFile must be one of the first!
from obspy.core.util.attribdict import AttribDict
from obspy.core.util.base import NamedTemporaryFile, \
    ALL_MODULES, DEFAULT_MODULES, NATIVE_BYTEORDER, \
    createEmptyDataChunk, getExampleFile, getMatplotlibVersion, \
    NETWORK_MODULES, _readFromPlugin, getScriptDirName
from obspy.core.util.testing import add_doctests, add_unittests
from obspy.core.util.decorator import deprecated, deprecated_keywords, \
    skip, skipIf, uncompressFile
from obspy.core.util.geodetics import FlinnEngdahl
from obspy.core.util.geodetics import calcVincentyInverse, gps2DistAzimuth, \
    kilometer2degrees, locations2degrees
from obspy.core.util.misc import BAND_CODE, complexifyString, guessDelta, \
    scoreatpercentile, toIntOrZero, loadtxt, CatchOutput
from obspy.core.util.obspy_types import OrderedDict, Enum, \
    ComplexWithUncertainties, FloatWithUncertainties
from obspy.core.util.xmlwrapper import XMLParser, tostring, register_namespace
from obspy.core.util.version import get_git_version as _getVersionString
