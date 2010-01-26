"""
obspy.wav Read & Write Seismograms to WAVE audio Format
=======================================================
Python method in order to read and write seismograms to WAV audio
files. The data are squeezed to audible frequencies.

:copyright: The ObsPy Development Team (devs@obspy.org)
:license: GNU Lesser General Public License, Version 3 (LGPLv3)

Reading using obspy.core
------------------------
Similiar to reading any other waveform data format using obspy.core:

>>> from obspy.core import read
>>> st = read("tests/data/3cssan.near.8.1.RNON.wav")
>>> print st
1 Trace(s) in Stream:
... | 1970-01-01T00:00:00.000000Z - 1970-01-01T00:00:00.371143Z | 7000.0 Hz, 2599 samples

The format will be determined automatically. As WAVE-files can contain only
one data trace (as opposed to Mini-SEED or GSE2), the length of 'st' will
be one. 'st[0]' will have a stats attribute containing the issential meta
information of the WAVE file.

>>> print st[0].stats
Stats({'network': '', 'delta': 0.00014285714285714287, 'station': '',
'location': '', 'starttime': UTCDateTime(1970, 1, 1, 0, 0), 'npts': 2599,
'calib': 1.0, 'sampling_rate': 7000, 'endtime': UTCDateTime(1970, 1, 1, 0,
0, 0, 371143), 'channel': ''})

The data is stored in the data attribut.

>>> st[0].data
>>> array([ 64,  78,  99, ..., 106, 103, 102], dtype=uint8)

Writing using obspy.core
------------------------
is also straight forward.

>>> st.write('myfile.wave', format='WAV', framerate=7000)

The framerate specifies the framerate to which the seismogram should be
squeezed. Using the originial sampling_rate results in an WAVE file with
frequencies which cannot be heard by a human, therefore it makes sence to
set the framerate to a high value.
"""
