"""
obspy.io.dmc - INGV DMX file format reader for ObsPy
===========================================================

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
automatically, however setting the format parameter as "DMX" lead to a
speed up.
One optional keyword argument is available: ``station``. It is automatically
passed to the :func:`obspy.io.dmx.core._read_dmx`. Its format 

>>> import obspy
>>> from obspy.core.util import get_example_file
>>> filename = get_example_file('181223 120000.DMX')
>>> # these two are equivalent, but the second case should be faster:
>>> st = obspy.read(filename)
>>> st = obspy.read(filename, format='DMX')

If the file is very large and only one station code needs to be fetched,
using the ``station`` parameter may speed the reading process:

>>> st = obspy.read(filename, station="STR1")


"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
