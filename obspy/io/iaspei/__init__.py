"""
obspy.io.iaspei - Read support for IASPEI formats
=================================================

This module provides read support for the IASPEI Seismic Format (ISF,
http://www.isc.ac.uk/standards/isf/).  Currently only supports reading IMS1.0
bulletin files.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
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
