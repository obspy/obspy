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
    (http://www.gnu.org/copyleft/lesser.html)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import importlib
from types import ModuleType
import sys
import warnings

# import order matters - NamedTemporaryFile must be one of the first!
from obspy.core.util.attribdict import AttribDict
from obspy.core.util.base import (ALL_MODULES, DEFAULT_MODULES,
                                  NATIVE_BYTEORDER, NETWORK_MODULES,
                                  NamedTemporaryFile, _readFromPlugin,
                                  createEmptyDataChunk, getExampleFile,
                                  getMatplotlibVersion, getScriptDirName)
from obspy.core.util.decorator import (deprecated, deprecated_keywords, skip,
                                       skipIf, uncompressFile)
from obspy.core.util.misc import (BAND_CODE, CatchOutput, complexifyString,
                                  guessDelta, loadtxt, scoreatpercentile,
                                  toIntOrZero)
from obspy.core.util.obspy_types import (ComplexWithUncertainties, Enum,
                                         FloatWithUncertainties)
from obspy.core.util.testing import add_doctests, add_unittests
from obspy.core.util.version import get_git_version as _getVersionString


class ObsPyDeprecationWarning(UserWarning):
    """
    Make a custom deprecation warning as deprecation warnings or hidden by
    default since Python 2.7 and 3.2 and we really want users to notice these.
    """
    pass


class DynamicAttributeImportRerouteModule(ModuleType):
    """
    Class assisting in dynamically rerouting attribute access like imports.

    This essentially makes

    >>> import obspy  # doctest: +SKIP
    >>> obspy.station.Inventory  # doctest: +SKIP

    work. Remove this once 0.11 has been released!
    """
    def __init__(self, name, doc, locs, import_map):
        super(DynamicAttributeImportRerouteModule, self).__init__(name=name)
        self.import_map = import_map
        # Keep the metadata of the module.
        self.__dict__.update(locs)

    def __getattr__(self, name):
        try:
            real_module_name = self.import_map[name]
        except:
            raise AttributeError
        warnings.warn("Module '%s' is deprecated and will stop working with "
                      "the next ObsPy version. Please import module "
                      "'%s'instead." % (self.__name__ + "." + name,
                                        self.import_map[name]),
                      ObsPyDeprecationWarning)
        return importlib.import_module(real_module_name)


# Remove once 0.11 has been released!
sys.modules[__name__] = DynamicAttributeImportRerouteModule(
    name=__name__, doc=__doc__, locs=locals(),
    import_map={"geodetics": "obspy.geodetics"})
