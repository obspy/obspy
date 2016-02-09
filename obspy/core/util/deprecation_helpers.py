# -*- coding: utf-8 -*-
"""
Library name handling for ObsPy.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import importlib
import warnings
from types import ModuleType


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
    def __init__(self, name, doc, locs, import_map, function_map=None):
        super(DynamicAttributeImportRerouteModule, self).__init__(name=name)
        self.import_map = import_map
        self.function_map = function_map
        # Keep the metadata of the module.
        self.__dict__.update(locs)

    def __getattr__(self, name):
        # Functions, and not modules.
        if self.function_map and name in self.function_map:
            new_name = self.function_map[name].split(".")
            module = importlib.import_module(".".join(new_name[:-1]))
            warnings.warn("Function '%s' is deprecated and will stop working "
                          "with the next ObsPy version. Please use '%s' "
                          "instead." % (self.__name__ + "." + name,
                                        self.function_map[name]),
                          ObsPyDeprecationWarning)
            return getattr(module, new_name[-1])

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
