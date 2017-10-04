"""
obspy.io.wav - WAV (audio) read and write support for ObsPy
===========================================================

The obspy.io.wav package contains methods in order to read and write seismogram
files in the WAV(audio) format. The data are squeezed to audible frequencies.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)

Reading
-------
Importing WAV files is done similar to reading any other waveform data
format within ObsPy by using the :func:`~obspy.core.stream.read()` method of
the :mod:`obspy.core` module. Examples seismograms files may be found at
https://examples.obspy.org.

>>> from obspy import read
>>> st = read("/path/to/3cssan.near.8.1.RNON.wav")
>>> print(st) #doctest: +NORMALIZE_WHITESPACE
1 Trace(s) in Stream:
... | 1970-01-01T00:00:00.000000Z - 1970-01-01T00:00:00.371143Z
| 7000.0 Hz, 2599 samples

The format will be determined automatically. As WAV files can contain only one
data trace (as opposed to Mini-SEED or GSE2), the length of 'st' will be one.
'st[0]' will have a stats attribute containing the essential meta information
of the WAV file.

>>> print(st[0].stats) #doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
         network:
         station:
        location:
         channel:
       starttime: 1970-01-01T00:00:00.000000Z
         endtime: 1970-01-01T00:00:00.371143Z
   sampling_rate: 7000.0
           delta: 0.000142857142857...
            npts: 2599
           calib: 1.0
         _format: WAV

The data is stored in the data attribute.

>>> st[0].data
array([ 64,  78,  99, ..., 106, 103, 102], dtype=uint8)

Writing
-------
is also straight forward.

>>> st.write('myfile.wave', format='WAV', framerate=7000) #doctest: +SKIP

The framerate specifies the framerate to which the seismogram should be
squeezed. Using the original sampling_rate results in an WAV file with
frequencies which cannot be heard by a human, therefore it makes sense to
set the framerate to a high value.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
