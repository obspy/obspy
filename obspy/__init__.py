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
import warnings
import sys

# don't change order
from obspy.core.utcdatetime import UTCDateTime  # NOQA
from obspy.core.util import _getVersionString
__version__ = _getVersionString(abbrev=10)
from obspy.core.trace import Trace  # NOQA
from obspy.core.stream import Stream, read
from obspy.core.event import readEvents, Catalog
from obspy.core.inventory import read_inventory  # NOQA

# insert supported read/write format plugin lists dynamically in docstrings
from obspy.core.util.base import make_format_plugin_table
read.__doc__ = \
    read.__doc__ % make_format_plugin_table("waveform", "read", numspaces=4)
readEvents.__doc__ = \
    readEvents.__doc__ % make_format_plugin_table("event", "read", numspaces=4)


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

__all__ = ["UTCDateTime", "Trace", "__version__", "Stream", "read",
           "readEvents", "Catalog", "read_inventory"]
__all__ = [native_str(i) for i in __all__]


class ObsPyDeprecationWarning(UserWarning):
    """
    Make a custom deprecation warning as deprecation warnings or hidden by
    default since Python 2.7 and 3.2 and we really want users to notice these.
    """
    pass


class ObsPyRestructureMetaPathFinderAndLoader(object):
    """
    Meta path finder and module loader helping users in transitioning to the
    new module structure.

    Make sure to remove this once 0.11 has been released!
    """
    # Maps all imports to their new imports.
    import_map = {
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
        "obspy.arclink": "obspy.io.arclink",
        "obspy.earthworm": "obspy.io.earthworm",
        "obspy.fdsn": "obspy.io.fdsn",
        "obspy.iris": "obspy.io.iris",
        "obspy.neic": "obspy.io.neic",
        "obspy.neries": "obspy.io.neries",
        "obspy.seedlink": "obspy.io.seedlink",
        "obspy.seishub": "obspy.io.seishub",
        # geodetics
        "obspy.core.util.geodetics": "obspy.geodetics"
    }

    def find_module(self, fullname, path=None):
        if not path or not path[0].startswith(__path__[0]):
            return None

        for key in self.import_map.keys():
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
        elif name in self.import_map:
            new_name = self.import_map[name]

            # Don't load again if already loaded.
            if new_name in sys.modules:
                module = sys.modules[new_name]
            else:
                module = self._find_and_load_module(new_name)

            # Warn here as at this point the module has already been imported.
            warnings.warn("Module '%s' is deprecated and will stop working "
                          "with the next ObsPy version. Please import module "
                          "'%s'instead." % (name, new_name),
                          ObsPyDeprecationWarning)
            sys.modules[new_name] = module
        # This probably does not happen with a proper import. Not sure if we
        # should keep this condition as it might obsfuscate non-working
        # imports.
        else:
            module = self._find_and_load_module(name)

        sys.modules[name] = module
        return module

    def _find_and_load_module(self, name, path=None):
        """
        Finds and loads it. But if there's a . in the name, handles it
        properly.

        From the python-future module as it already did the painful steps to
        make it work on Python 2 and Python 3.
        """
        bits = name.split('.')
        while len(bits) > 1:
            # Treat the first bit as a package
            packagename = bits.pop(0)
            package = self._find_and_load_module(packagename, path)
            try:
                path = package.__path__
            except AttributeError:
                if name in sys.modules:
                    return sys.modules[name]
        name = bits[0]
        module_info = imp.find_module(name, path)
        return imp.load_module(name, *module_info)


# Install meta path handler.
sys.meta_path.append(ObsPyRestructureMetaPathFinderAndLoader())
