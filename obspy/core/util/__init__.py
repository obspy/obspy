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

# import order matters - NamedTemporaryFile must be one of the first!
from obspy.core.util.attribdict import AttribDict
from obspy.core.util.base import (ALL_MODULES, DEFAULT_MODULES,
                                  NATIVE_BYTEORDER, NETWORK_MODULES,
                                  NamedTemporaryFile, _read_from_plugin,
                                  create_empty_data_chunk, get_example_file,
                                  get_matplotlib_version, get_script_dir_name)
from obspy.core.util.misc import (BAND_CODE, complexify_string, guess_delta,
                                  loadtxt, score_at_percentile, to_int_or_zero)
from obspy.core.util.obspy_types import (ComplexWithUncertainties, Enum,
                                         FloatWithUncertainties)
from obspy.core.util.testing import add_doctests, add_unittests
from obspy.core.util.version import get_git_version as _get_version_string
from obspy.core.util.capture import PyCatchOutput, CCatchOutput, SuppressOutput
