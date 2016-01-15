# -*- coding: utf-8 -*-
"""
obspy.clients.syngine - Client for the IRIS Syngine Service
===========================================================

Contains requests abilities that naturally integrate with the rest of ObsPy
for the IRIS syngine service (http://ds.iris.edu/ds/products/syngine/).

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)


Please see the documentation for each method for further information and
examples.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA
from future.utils import native_str

from .client import Client  # NOQA

__all__ = [native_str("Client")]


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
