"""
obspy.io.sh - Q and ASC read and write, EVT read support (Seismic Handler)
==========================================================================

The obspy.io.sh package contains methods in order to read and write seismogram
files in the Q and ASC format used by the Seismic Handler software package
(https://www.seismic-handler.org). EVT event files can also be read.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)

Reading
-------
Importing Q or ASC files is done similar to reading any other waveform data
format within ObsPy by using the :func:`~obspy.core.stream.read()` method of
the :mod:`obspy.core` module. Examples seismograms files may be found at
https://examples.obspy.org.

>>> from obspy import read
>>> st = read("/path/to/QFILE-TEST-ASC.ASC")
>>> st #doctest: +ELLIPSIS
<obspy.core.stream.Stream object at 0x...>
>>> print(st)  # doctest: +ELLIPSIS
3 Trace(s) in Stream:
.TEST..BHN | 2009-10-01T12:46:01.000000Z - ... | 20.0 Hz, 801 samples
.TEST..BHE | 2009-10-01T12:46:01.000000Z - ... | 20.0 Hz, 801 samples
.WET..HHZ  | 2010-01-01T01:01:05.999000Z - ... | 100.0 Hz, 4001 samples

The file format will be determined automatically. Each trace (multiple channels
are mapped to multiple traces) will have a stats attribute containing the usual
information.

>>> print(st[0].stats) #doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
             network:
             station: TEST
            location:
             channel: BHN
           starttime: 2009-10-01T12:46:01.000000Z
             endtime: 2009-10-01T12:46:41.000000Z
       sampling_rate: 20.0
               delta: 0.05
                npts: 801
               calib: 1.5
             _format: SH_ASC
                  sh: AttribDict({...})

>>> print(st[0].stats.sh['COMMENT'])
TEST TRACE IN QFILE #1

The actual data is stored as numpy.ndarray in the data attribute of each trace.

>>> st[0].data #doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
array([ 0.00000000e+00,   0.00000000e+00,   0.00000000e+00, ...
       -4.03387604e+01,  -3.99515305e+01,  -3.95423012e+01], dtype=float32)

Writing
-------
Writing is also done in the usual way:

>>> st.write('file.q', format = 'Q') #doctest: +SKIP

or

>>> st.write('file.asc', format = 'SH_ASC') #doctest: +SKIP
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
