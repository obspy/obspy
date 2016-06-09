# -*- coding: utf-8 -*-
"""
obspy.io.gcf - Guralp Compressed Format read support for ObsPy
==============================================================
This module provides read support for GCF waveform data and header info.
Most methods are based on info from Guralp site
http://geophysics.eas.gatech.edu/GTEQ/Scream4.4/GCF_Specification.htm

:copyright:
    The ObsPy Development Team (devs@obspy.org) & Ran Novitsky Nof
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)

Reading
-------
Similar to reading any other waveform data format using :mod:`obspy.core`
The format will be determined automatically.

>>> from obspy import read
>>> st = read("/path/to/20160603_1955n.gcf")

You can also specify the following keyword arguments that change the
behavior of reading the file:
* ``headonly=True``: Read only the header part, not the actual data
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
