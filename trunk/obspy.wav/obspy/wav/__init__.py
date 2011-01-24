"""
obspy.wav - WAVE (audio) read and write support
===============================================
Python method in order to read and write seismograms to WAV audio files. The
data are squeezed to audible frequencies.

:copyright: The ObsPy Development Team (devs@obspy.org)
:license: GNU Lesser General Public License, Version 3 (LGPLv3)

Reading
-------
Similar to reading any other waveform data format using obspy.core:

(Lines 2&3 are just to get the absolute path of our test data)

>>> from obspy.core import read
>>> from obspy.core import path
>>> filename = path("3cssan.near.8.1.RNON.wav")
>>> st = read(filename)
>>> print st
1 Trace(s) in Stream:
... | 1970-01-01T00:00:00.000000Z - 1970-01-01T00:00:00.371143Z | 7000.0 Hz, 2599 samples

The format will be determined automatically. As WAVE-files can contain only one
data trace (as opposed to Mini-SEED or GSE2), the length of 'st' will be one.
'st[0]' will have a stats attribute containing the essential meta information
of the WAVE file.

>>> print st[0].stats
         network: 
         station: 
        location: 
         channel: 
       starttime: 1970-01-01T00:00:00.000000Z
         endtime: 1970-01-01T00:00:00.371143Z
   sampling_rate: 7000.0
           delta: 0.000142857142857
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
squeezed. Using the original sampling_rate results in an WAVE file with
frequencies which cannot be heard by a human, therefore it makes sense to
set the framerate to a high value.
"""

from obspy.core.util import _getVersionString


__version__ = _getVersionString("obspy.wav")
