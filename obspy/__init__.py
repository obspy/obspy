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
import sys

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


# insert supported read/write format plugin lists dynamically in docstrings
from obspy.core.util.base import _add_format_plugin_table


_add_format_plugin_table(read, "waveform", "read", numspaces=4)
_add_format_plugin_table(read_events, "event", "read", numspaces=4)
_add_format_plugin_table(read_inventory, "inventory", "read", numspaces=4)
_add_format_plugin_table(Stream.write, "waveform", "write", numspaces=8)
_add_format_plugin_table(Catalog.write, "event", "write", numspaces=8)
_add_format_plugin_table(Inventory.write, "inventory", "write", numspaces=8)

if int(sys.version[0]) < 3:
    raise ImportError("""You are running ObsPy >= 1.3 on Python 2

ObsPy version 1.3 and above is not compatible with Python 2, and you still
ended up with this version installed. This should not have happened.
Make sure you have pip >= 9.0 and setuptools >= 24.2:

 $ pip install pip setuptools --upgrade

Your choices:

- Upgrade to Python 3.

- Install an older version of ObsPy:

 $ pip install 'obspy<1.3'
""")


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
