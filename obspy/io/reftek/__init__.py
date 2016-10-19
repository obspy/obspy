# -*- coding: utf-8 -*-
"""
obspy.io.reftek - REFTEK130 read support
========================================

This module provides read support for the RefTek 130 data format.

Currently the low level read routines are designed to operate on waveform files
written by RefTek 130 digitizers which are composed of event header/trailer and
data packages. These packages do not store information on network or location
code during acquisition and channels are simply enumerated (although
information on the first two channel code characters, the band and instrument
code, are present). Additional information on network, location and component
codes have to be supplied when reading files with
:func:`~obspy.core.stream.read` (or have to be filled in manually after
reading). See the low-level routine
:func:`obspy.io.reftek.core._read_reftek130` for additional arguments that can
be supplied to :func:`~obspy.core.stream.read`.

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
