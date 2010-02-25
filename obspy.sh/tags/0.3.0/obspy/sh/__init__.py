"""
obspy.sh - SH read and write support
====================================
This modules provides facilities to:

* Import and export seismogram files in the Q format.
* Import and export seismogram files in the ASC format.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)

Reading
-------
Importing Q or ASC files is done similiar to reading any other waveform data
format within ObsPy by using the :func:`~obspy.core.stream.read()` method of
the :mod:`obspy.core` module. Test files for the following examples may be
found at http://examples.obspy.org.

>>> from obspy.core import read
>>> st = read("QFILE-TEST-ASC.ASC")
>>> st
<obspy.core.stream.Stream object at 0x...>
>>> print st
3 Trace(s) in Stream:
.TEST..BHN | 2009-10-01T12:46:01.000000Z - 2009-10-01T12:46:41.000000Z | 20.0 Hz, 801 samples
.TEST..BHE | 2009-10-01T12:46:01.000000Z - 2009-10-01T12:46:41.000000Z | 20.0 Hz, 801 samples
.WET..HHZ | 2010-01-01T01:01:05.999000Z - 2010-01-01T01:01:45.999000Z | 100.0 Hz, 4001 samples

The file format will be determined automatically. Each trace (multiple channels
are mapped to multiple traces) will have a stats attribute containing the usual
information.

>>> st[0].stats
Stats({
    'network': '', 
    'station': 'TEST', 
    'npts': 801, 
    'sh': AttribDict({
        'COMMENT': 'TEST TRACE IN QFILE #1'
    }), 
    'location': '', 
    'starttime': UTCDateTime(2009, 10, 1, 12, 46, 1), 
    'delta': 0.050000000000000003, 
    'calib': 1.5, 
    'sampling_rate': 20.0, 
    'endtime': UTCDateTime(2009, 10, 1, 12, 46, 41), 
    'channel': 'BHN',
    '_format': 'SH_ASC'
})

The actual data is stored as numpy.ndarray in the data attribute of each trace.

>>> st[0].data
array([ 0.00000000e+00,   0.00000000e+00,   0.00000000e+00, ...
       -4.03387604e+01,  -3.99515305e+01,  -3.95423012e+01], dtype=float32)

Writing
-------
Writing is also done in the usual way:

>>> st.write('file.q', format = 'Q')

or 

>>> st.write('file.asc', format = 'SH_ASC')

"""

from obspy.core.util import _getVersionString


__version__ = _getVersionString("obspy.sh")
