# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import warnings

from obspy.core.util.deprecation_helpers import ObsPyDeprecationWarning
from obspy.signal.array_analysis.array_analysis import (
    SeismicArray, get_geometry)  # noqa
from obspy.signal.array_analysis.array_rotation_strain \
    import array_rotation_strain


def array_processing(*arg, **kwargs):
    msg = ("obspy.array_analysis.%s() is deprecated. Please use "
           "the new class based obspy.array_analysis.SeismicArray interface."
           % "array_processing")
    warnings.warn(msg, category=ObsPyDeprecationWarning)


def array_transff_freqslowness(*arg, **kwargs):
    msg = ("obspy.array_analysis.%s() is deprecated. Please use "
           "the new class based obspy.array_analysis.SeismicArray interface."
           % "array_transff_freqslowness")
    warnings.warn(msg, category=ObsPyDeprecationWarning)


def array_transff_wavenumber(*arg, **kwargs):
    msg = ("obspy.array_analysis.%s() is deprecated. Please use "
           "the new class based obspy.array_analysis.SeismicArray interface."
           % "array_transff_wavenumber")
    warnings.warn(msg, category=ObsPyDeprecationWarning)


def get_spoint(*arg, **kwargs):
    msg = ("obspy.array_analysis.%s() is deprecated. Please use "
           "the new class based obspy.array_analysis.SeismicArray interface."
           % "get_spoint")
    warnings.warn(msg, category=ObsPyDeprecationWarning)
