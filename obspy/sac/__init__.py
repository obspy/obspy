# -*- coding: utf-8 -*-
"""
obspy.sac - SAC read and write support for ObsPy
================================================
This module provides read and write support for ASCII and binary SAC-files as
defined by IRIS (http://www.iris.edu/manuals/sac/manual.html).

:copyright:
    The ObsPy Development Team (devs@obspy.org) & C. J. Annon
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)

Reading
-------
Similiar to reading any other waveform data format using
:func:`~obspy.core.read`:

>>> from obspy import read
>>> st = read('/path/to/test.sac')
>>> st #doctest: +ELLIPSIS
<obspy.core.stream.Stream object at 0x...>
>>> print(st) #doctest: +ELLIPSIS
1 Trace(s) in Stream:
.STA..Q | 1978-07-18T08:00:10.000000Z - ... | 1.0 Hz, 100 samples

The format will be determined automatically. As SAC-files can contain only one
data trace (as opposed to Mini-SEED or GSE2), the length of 'st' will be one.
'st[0]' will have a stats attribute containing the issential meta data (station
name, channel, location, start time, end time, sampling rate, number of
points). Additionally, when reading a SAC-file it will have one additional
attribute, 'sac', which contains all SAC-specific attributes (SAC header
values).

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
             sac: AttribDict({'dist': -12345.0, 'isynth': -12345, ...'})

The data is stored in the data attribut.

>>> st[0].data #doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
array([ -8.74227766e-08,  -3.09016973e-01,..., 3.09007347e-01], dtype=float32)

Writing
-------
Writing is also straight forward. All changes on the data as well as in
stats and stats['sac'] are written with the following command to a file:

>>> st.write('tmp.sac', format='SAC') #doctest: +SKIP
"""

from sacio import SacIO, SacError, SacIOError, attach_paz, attach_resp


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
