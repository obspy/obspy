"""
obspy.io.dmc - INGV DMX file format reader for ObsPy
====================================================

Functions to read waveform data from the standard INGV DMX format.

:author:
    Thomas Lecocq
    Andrea Cannatta
:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)


Reading the waveforms
---------------------
Reading DMX is handled by using ObsPy's standard
:func:`~obspy.core.stream.read` function. The format can be detected
automatically, however setting the ``format`` parameter as "DMX" lead to a
speed up.
One optional keyword argument is available: ``station``. It is automatically
passed to the :func:`obspy.io.dmx.core._read_dmx`. Its format 

>>> from obspy import read
>>> # these two are equivalent, but the second case should be faster:
>>> st = read("/path/to/131114_090600.dmx")
>>> print(st)  #doctest: +NORMALIZE_WHITESPACE
2 Trace(s) in Stream:
ETNA.EMFO..Z | 2013-11-14T09:06:00.000000Z - 2013-11-14T09:06:59.990000Z | 100.0 Hz, 6000 samples
ETNA.EMPL..Z | 2013-11-14T09:06:00.000000Z - 2013-11-14T09:06:59.990000Z | 100.0 Hz, 6000 samples

>>> st = read("/path/to/131114_090600.dmx", format='DMX')
>>> print(st)  #doctest: +NORMALIZE_WHITESPACE
2 Trace(s) in Stream:
ETNA.EMFO..Z | 2013-11-14T09:06:00.000000Z - 2013-11-14T09:06:59.990000Z | 100.0 Hz, 6000 samples
ETNA.EMPL..Z | 2013-11-14T09:06:00.000000Z - 2013-11-14T09:06:59.990000Z | 100.0 Hz, 6000 samples

If the file is very large and only one station code needs to be fetched,
using the ``station`` parameter may speed the reading process:

>>> st = read("/path/to/131114_090600.dmx", station="EMFO")
>>> print(st)  # doctest: +ELLIPSIS
1 Trace(s) in Stream:
ETNA.EMFO..Z | 2013-11-14T09:06:00.000000Z - 2013-11-14T09:06:59.990000Z | 100.0 Hz, 6000 samples

"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
