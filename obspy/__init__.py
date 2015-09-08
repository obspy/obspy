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
    (http://www.gnu.org/copyleft/lesser.html)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA
from future.utils import PY2, native_str

import imp
import importlib
import warnings
import sys

# don't change order
from obspy.core.utcdatetime import UTCDateTime  # NOQA
from obspy.core.util import _get_version_string
from obspy.core.util.deprecation_helpers import (
    ObsPyDeprecationWarning, DynamicAttributeImportRerouteModule)
__version__ = _get_version_string(abbrev=10)
from obspy.core.trace import Trace  # NOQA
from obspy.core.stream import Stream, read
from obspy.core.event import read_events, Catalog
from obspy.core.inventory import read_inventory, Inventory  # NOQA


__all__ = ["UTCDateTime", "Trace", "__version__", "Stream", "read",
           "read_events", "Catalog", "read_inventory"]
__all__ = [native_str(i) for i in __all__]


# Maps all imports to their new imports.
_import_map = {
    # I/O modules
    "obspy.ah": "obspy.io.ah",
    "obspy.cnv": "obspy.io.cnv",
    "obspy.css": "obspy.io.css",
    "obspy.datamark": "obspy.io.datamark",
    "obspy.gse2": "obspy.io.gse2",
    "obspy.kinemetrics": "obspy.io.kinemetrics",
    "obspy.mseed": "obspy.io.mseed",
    "obspy.ndk": "obspy.io.ndk",
    "obspy.nlloc": "obspy.io.nlloc",
    "obspy.pdas": "obspy.io.pdas",
    "obspy.pde": "obspy.io.pde",
    "obspy.sac": "obspy.io.sac",
    "obspy.seg2": "obspy.io.seg2",
    "obspy.segy": "obspy.io.segy",
    "obspy.seisan": "obspy.io.seisan",
    "obspy.sh": "obspy.io.sh",
    "obspy.wav": "obspy.io.wav",
    "obspy.xseed": "obspy.io.xseed",
    "obspy.y": "obspy.io.y",
    "obspy.zmap": "obspy.io.zmap",
    # Clients
    "obspy.arclink": "obspy.clients.arclink",
    "obspy.earthworm": "obspy.clients.earthworm",
    "obspy.fdsn": "obspy.clients.fdsn",
    "obspy.iris": "obspy.clients.iris",
    "obspy.neic": "obspy.clients.neic",
    "obspy.seedlink": "obspy.clients.seedlink",
    "obspy.seishub": "obspy.clients.seishub",
    # geodetics
    "obspy.core.util.geodetics": "obspy.geodetics",
    # obspy.station
    "obspy.station": "obspy.core.inventory",
    # Misc modules originally in core.
    "obspy.core.ascii": "obspy.io.ascii",
    "obspy.core.quakeml": "obspy.io.quakeml",
    "obspy.core.stationxml": "obspy.io.stationxml",
    "obspy.core.json": "obspy.io.json"
}

_function_map = {
    "readEvents": "obspy.read_events"
}


class ObsPyRestructureMetaPathFinderAndLoader(object):
    """
    Meta path finder and module loader helping users in transitioning to the
    new module structure.

    Make sure to remove this once 0.11 has been released!
    """
    def find_module(self, fullname, path=None):
        # Compatibility with namespace paths.
        if hasattr(path, "_path"):
            path = path._path

        if not path or not path[0].startswith(__path__[0]):
            return None

        for key in _import_map.keys():
            if fullname.startswith(key):
                break
        else:
            return None
        # Use this instance also as the loader.
        return self

    def load_module(self, name):
        # Use cached modules.
        if name in sys.modules:
            return sys.modules[name]
        # Otherwise check if the name is part of the import map.
        elif name in _import_map:
            new_name = _import_map[name]
        else:
            new_name = name
            for old, new in _import_map.items():
                if not new_name.startswith(old):
                    continue
                new_name = new_name.replace(old, new)
                break
            else:
                return None

        # Don't load again if already loaded.
        if new_name in sys.modules:
            module = sys.modules[new_name]
        else:
            module = importlib.import_module(new_name)

        # Warn here as at this point the module has already been imported.
        warnings.warn("Module '%s' is deprecated and will stop working "
                      "with the next ObsPy version. Please import module "
                      "'%s' instead." % (name, new_name),
                      ObsPyDeprecationWarning)
        sys.modules[new_name] = module
        sys.modules[name] = module
        return module


# Install meta path handler.
sys.meta_path.append(ObsPyRestructureMetaPathFinderAndLoader())


# Remove once 0.11 has been released.
sys.modules[__name__] = DynamicAttributeImportRerouteModule(
    name=__name__, doc=__doc__, locs=locals(),
    import_map={key.split(".")[1]: value for key, value in
                _import_map.items() if len(key.split(".")) == 2},
    function_map=_function_map)


# insert supported read/write format plugin lists dynamically in docstrings
from obspy.core.util.base import make_format_plugin_table
read.__doc__ = \
    read.__doc__ % make_format_plugin_table("waveform", "read", numspaces=4)
read_events.__doc__ = \
    read_events.__doc__ % make_format_plugin_table("event", "read",
                                                   numspaces=4)


if PY2:
    Stream.write.im_func.func_doc = \
        Stream.write.__doc__ % make_format_plugin_table("waveform", "write",
                                                        numspaces=8)
    Catalog.write.im_func.func_doc = \
        Catalog.write.__doc__ % make_format_plugin_table("event", "write",
                                                         numspaces=8)
else:
    Stream.write.__doc__ = \
        Stream.write.__doc__ % make_format_plugin_table("waveform", "write",
                                                        numspaces=8)
    Catalog.write.__doc__ = \
        Catalog.write.__doc__ % make_format_plugin_table("event", "write",
                                                         numspaces=8)


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
