# -*- coding: utf-8 -*-
"""
obspy.io.mseed3 - miniseed3 read and write support for ObsPy
===================================================
This module provides read and write support for miniseed3-files as
defined by the FDSN (http://docs.fdsn.org/projects/miniseed3/en/latest/).

:copyright:
    The ObsPy Development Team (devs@obspy.org) & H. P. Crotwell
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)

The mseed3 implementation in ObsPy uses simplemseed,
https://github.com/crotwell/simplemseed

Reading
-------
Similar to reading any other waveform data format using
:func:`~obspy.core.stream.read()`:

>>> from obspy import read
>>> st = read('/path/to/test.ms3', debug_headers=True)
>>> st #doctest: +ELLIPSIS
<obspy.core.stream.Stream object at 0x...>
>>> print(st) #doctest: +ELLIPSIS
1 Trace(s) in Stream:
.STA..Q | 1978-07-18T08:00:10.000000Z - ... | 1.0 Hz, 100 samples

The format will be determined automatically. Each trace will have a stats
attribute containing the essential meta data (station
name, channel, location, start time, end time, sampling rate, number of
points). Additionally, when reading a mseed3-file it will have one additional
attribute, 'eh', which contains the miniseed3 extra headers.

>>> print(st[0].stats)  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
         network:
         station: STA
        location:
         channel: Q
       starttime: 1978-07-18T08:00:10.000000Z
         endtime: 1978-07-18T08:01:49.000000Z
   sampling_rate: 1.0
           delta: 1.0
            npts: 100
           calib: 1.0
         _format: SAC
             sac: AttribDict({...})
>>> print(st[0].stats.sac.dist)
-12345.0

The data is stored in the data attribute.

>>> st[0].data #doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
array([ -8.74227766e-08,  -3.09016973e-01,..., 3.09007347e-01], dtype=float32)

Writing
-------
Writing is also straight forward. All changes on the data as well as in
stats and stats['eh'] are written with the following command to a file:

>>> st.write('tmp.ms3', format='MSEED3') #doctest: +SKIP

"""


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
