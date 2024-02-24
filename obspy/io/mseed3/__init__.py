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
>>> st = read('/path/to/casee_two.ms3', debug_headers=True)
>>> st #doctest: +ELLIPSIS
<obspy.core.stream.Stream object at 0x...>
>>> print(st) #doctest: +ELLIPSIS
1 Trace(s) in Stream:
CO.CASEE.00.HHZ | 2023-06-17T04:53:50.008392Z - 2023-06-17T04:53:55.498392Z | 100.0 Hz, 550 samples

The format will be determined automatically. Each trace will have a stats
attribute containing the essential meta data (station
name, channel, location, start time, end time, sampling rate, number of
points). Additionally, when reading a mseed3-file it will have one additional
attribute, 'eh', which contains the miniseed3 extra headers.

>>> print(st[0].stats)  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
           network: CO
           station: CASEE
          location: 00
           channel: HHZ
         starttime: 2023-06-17T04:53:50.008392Z
           endtime: 2023-06-17T04:53:55.498392Z
     sampling_rate: 100.0
             delta: 0.01
              npts: 550
             calib: 1.0
           _format: MSEED3
                eh: AttribDict({'FDSN': AttribDict({'Time': AttribDict({'Quality': 0})})})
publicationVersion: 4

>>> st[0].data #doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
array([129, 123, 101, 103, 115,..., 136, 140, 143, 137], dtype=int32)

Writing
-------
Writing is also straight forward. All changes on the data as well as in
stats and stats['eh'] are written with the following command to a file:

>>> st.write('tmp.ms3', format='MSEED3') #doctest: +SKIP

"""


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
