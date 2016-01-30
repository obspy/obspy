"""
obspy.io.pde - NEIC PDE Bulletin read support for ObsPy
=======================================================

This module provides read support for NEIC Preliminary Determination of
Epicenters (PDE) Bulletin.

Currently, only the mchedr (machine readable Earthquake Data Report)
format is supported.

.. seealso:: http://earthquake.usgs.gov/data/pde.php

:copyright:
    The ObsPy Development Team (devs@obspy.org), Claudio Satriano
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)

"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
