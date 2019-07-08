# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
#  Purpose: Convenience imports for obspy
#   Author: Robert Barsch
#           Moritz Beyreuther
#           Lion Krischer
#           Tobias Megies
#
# Copyright (C) 2008-2014 Robert Barsch, Moritz Beyreuther, Lion Krischer,
#                         Tobias Megies
# -----------------------------------------------------------------------------
"""
ObsPy: A Python Toolbox for seismology/seismological observatories
==================================================================

ObsPy is an open-source project dedicated to provide a Python framework for
processing seismological data. It provides parsers for common file formats
and seismological signal processing routines which allow the manipulation of
seismological time series.

The goal of the ObsPy project is to facilitate rapid application development
for seismology.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA
from future.utils import PY2, native_str

import warnings
import requests

# don't change order
from obspy.core.utcdatetime import UTCDateTime  # NOQA
from obspy.core.util import _get_version_string
__version__ = _get_version_string(abbrev=10)
from obspy.core.trace import Trace  # NOQA
from obspy.core.stream import Stream, read
from obspy.core.event import read_events, Catalog
from obspy.core.inventory import read_inventory, Inventory  # NOQA
from obspy.core.util.obspy_types import (  # NOQA
    ObsPyException, ObsPyReadingError)


__all__ = ["UTCDateTime", "Trace", "__version__", "Stream", "read",
           "read_events", "Catalog", "read_inventory", "ObsPyException",
           "ObsPyReadingError"]
__all__ = [native_str(i) for i in __all__]


# insert supported read/write format plugin lists dynamically in docstrings
from obspy.core.util.base import _add_format_plugin_table


_add_format_plugin_table(read, "waveform", "read", numspaces=4)
_add_format_plugin_table(read_events, "event", "read", numspaces=4)
_add_format_plugin_table(read_inventory, "inventory", "read", numspaces=4)
_add_format_plugin_table(Stream.write, "waveform", "write", numspaces=8)
_add_format_plugin_table(Catalog.write, "event", "write", numspaces=8)
_add_format_plugin_table(Inventory.write, "inventory", "write", numspaces=8)


if requests.__version__ in ('2.12.0', '2.12.1', '2.12.2'):
    msg = ("ObsPy has some known issues with 'requests' version {} (see "
           "github issue #1599). Please consider updating module 'requests' "
           "to a newer version.").format(requests.__version__)
    warnings.warn(msg)


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
