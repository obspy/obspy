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
>>> st = read("/path/to/181223_120000.DMX")
>>> print(st)  #doctest: +NORMALIZE_WHITESPACE
186 Trace(s) in Stream:
IT.STR1..E | 2018-12-23T12:00:00.000000Z - 2018-12-23T12:00:59.980000Z | 50.0 Hz, 3000 samples
...
(184 other traces)
...
IT.SSY..Z | 2018-12-23T12:00:00.000000Z - 2018-12-23T12:00:59.990000Z | 100.0 Hz, 6000 samples
[Use "print(Stream.__str__(extended=True))" to print all Traces]
>>> st = read("/path/to/181223_120000.DMX", format='DMX')
>>> print(st)  #doctest: +NORMALIZE_WHITESPACE
186 Trace(s) in Stream:
IT.STR1..E | 2018-12-23T12:00:00.000000Z - 2018-12-23T12:00:59.980000Z | 50.0 Hz, 3000 samples
...
(184 other traces)
...
IT.SSY..Z | 2018-12-23T12:00:00.000000Z - 2018-12-23T12:00:59.990000Z | 100.0 Hz, 6000 samples
[Use "print(Stream.__str__(extended=True))" to print all Traces]

If the file is very large and only one station code needs to be fetched,
using the ``station`` parameter may speed the reading process:

>>> st = read("/path/to/181223_120000.DMX", station="STR1")
>>> print(st)  # doctest: +ELLIPSIS
3 Trace(s) in Stream:
IT.STR1..E | 2018-12-23T12:00:00.000000Z - 2018-12-23T12:00:59.980000Z | 50.0 Hz, 3000 samples
IT.STR1..N | 2018-12-23T12:00:00.000000Z - 2018-12-23T12:00:59.980000Z | 50.0 Hz, 3000 samples
IT.STR1..Z | 2018-12-23T12:00:00.000000Z - 2018-12-23T12:00:59.980000Z | 50.0 Hz, 3000 samples
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
