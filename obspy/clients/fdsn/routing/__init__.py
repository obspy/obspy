#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
obspy.clients.fdsn.routing - Routing services for FDSN web services
===================================================================

:copyright:
    The ObsPy Development Team (devs@obspy.org)
    Celso G Reyes, 2017
    IRIS-DMC
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA


# Extremely ugly way to avoid a race condition the first time strptime is
# imported which is not thread safe...
#
# See https://bugs.python.org/issue7980
import time
time.strptime("2000/11/30", "%Y/%m/%d")


if __name__ == '__main__':  # pragma: no cover
    import doctest
    doctest.testmod(exclude_empty=True)
