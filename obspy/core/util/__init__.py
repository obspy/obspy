# -*- coding: utf-8 -*-
"""
obspy.core.util - Various utilities for ObsPy
=============================================

.. note:: Please import all utilities within your custom applications from this
    module rather than from any sub module, e.g.

    >>> from obspy.core.util import AttribDict  # good

    instead of

    >>> from obspy.core.util.attribdict import AttribDict  # bad

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import sys

# import order matters - NamedTemporaryFile must be one of the first!
from obspy.core.util.attribdict import AttribDict
from obspy.core.util.base import (ALL_MODULES, DEFAULT_MODULES,
                                  NATIVE_BYTEORDER, NETWORK_MODULES,
                                  NamedTemporaryFile, _read_from_plugin,
                                  create_empty_data_chunk, get_example_file,
                                  get_matplotlib_version, get_script_dir_name)
from obspy.core.util.deprecation_helpers import \
    DynamicAttributeImportRerouteModule
from obspy.core.util.misc import (BAND_CODE, CatchOutput, complexify_string,
                                  guess_delta, loadtxt, score_at_percentile,
                                  to_int_or_zero)
from obspy.core.util.obspy_types import (ComplexWithUncertainties, Enum,
                                         FloatWithUncertainties)
from obspy.core.util.testing import add_doctests, add_unittests
from obspy.core.util.version import get_git_version as _get_version_string


# Remove once 0.11 has been released!
sys.modules[__name__] = DynamicAttributeImportRerouteModule(
    name=__name__, doc=__doc__, locs=locals(),
    original_module=sys.modules[__name__],
    import_map={"geodetics": "obspy.geodetics"},
    function_map={
        "FlinnEngdahl": "obspy.geodetics.FlinnEngdahl",
        "calcVincentyInverse": "obspy.geodetics.calc_vincenty_inverse",
        "degrees2kilometers": "obspy.geodetics.degrees2kilometers",
        "gps2DistAzimuth": "obspy.geodetics.gps2dist_azimuth",
        "kilometer2degrees": "obspy.geodetics.kilometer2degrees",
        "locations2degrees": "obspy.geodetics.locations2degrees",
        "getMatplotlibVersion": "obspy.core.util.get_matplotlib_version",
        'complexifyString': 'obspy.core.util.complexify_string',
        'createEmptyDataChunk': 'obspy.core.util.create_empty_data_chunk',
        'getExampleFile': 'obspy.core.util.get_example_file',
        'getScriptDirName': 'obspy.core.util.get_script_dir_name',
        'guessDelta': 'obspy.core.util.guess_delta',
        'toIntOrZero': 'obspy.core.util.misc.to_int_or_zero',
        'scoreatpercentile': 'obspy.core.util.score_at_percentile',
        'uncompressFile': 'obspy.core.util.decorator.uncompress_file'})
