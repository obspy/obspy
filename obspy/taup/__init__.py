# -*- coding: utf-8 -*-
"""
obspy.taup - Travel time calculation tool
=========================================

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import inspect
import os

# Convenience imports.
from .tau import TauPyModel  # NOQA
from .taup import getTravelTimes, travelTimePlot  # NOQA


# Most generic way to get the data directory.
__DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(inspect.getfile(
    inspect.currentframe()))), "data")


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
