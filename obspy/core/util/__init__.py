# -*- coding: utf-8 -*-
"""
Various utilities for ObsPy

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
# import order matters - NamedTemporaryFile must be one of the first!
from obspy.core.util.attribdict import AttribDict
from obspy.core.util.base import (ALL_MODULES, DEFAULT_MODULES,
                                  NATIVE_BYTEORDER, NETWORK_MODULES,
                                  NamedTemporaryFile, _read_from_plugin,
                                  create_empty_data_chunk, get_example_file,
                                  get_script_dir_name, MATPLOTLIB_VERSION,
                                  SCIPY_VERSION, NUMPY_VERSION,
                                  CARTOPY_VERSION, CatchAndAssertWarnings)
from obspy.core.util.misc import (BAND_CODE, CatchOutput, complexify_string,
                                  guess_delta, score_at_percentile,
                                  to_int_or_zero, SuppressOutput)
from obspy.core.util.obspy_types import (ComplexWithUncertainties, Enum,
                                         FloatWithUncertainties)
from obspy.core.util.version import get_git_version as _get_version_string
