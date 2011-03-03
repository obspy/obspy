# -*- coding: utf-8 -*-
"""
obspy.segy - SEG Y and SU read and write support for ObsPy
==========================================================

The obspy.segy package contains methods in order to read and write seismogram
files in the SEG Y (rev. 1) and SU (Seismic Unix) format.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)

Reading
-------
Importing SEG Y or SU files is done similar to reading any other waveform data
format within ObsPy by using the :func:`~obspy.core.stream.read()` method of
the :mod:`obspy.core` module. Examples seismograms files may be found at
http://examples.obspy.org.

>>> from obspy.core import read
>>> st = read("/path/to/00001034.sgy_first_trace")
>>> st #doctest: +ELLIPSIS
<obspy.core.stream.Stream object at 0x...>
>>> print(st)
1 Trace(s) in Stream:
Seq. No. in line:    1 | 2009-06-22T14:47:37.000000Z - 2009-06-22T14:47:41.000000Z | 500.0 Hz, 2001 samples

The file format will be determined automatically. Each trace (multiple channels
are mapped to multiple traces) will have a stats attribute containing the usual
information.

>>> print(st[0].stats) #doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
             network: 
             station: 
            location: 
             channel: 
           starttime: 2009-06-22T14:47:37.000000Z
             endtime: 2009-06-22T14:47:41.000000Z
       sampling_rate: 500.0
               delta: 0.002
                npts: 2001
               calib: 1.0
                segy: AttribDict({'trace_header': ...})
             _format: SEGY

The actual data is stored as numpy.ndarray in the data attribute of each trace.

>>> st[0].data #doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
array([ -2.84501867e-11,  -5.32782846e-11,  -1.13144355e-10, ...,
        -4.55348870e-10,  -8.47760084e-10,  -7.45420170e-10], dtype=float32)


The above way of reading a file will become very slow and memory intensive for
a large number of traces due to the huge number of objects created.

Use the internal reading method which is much faster and yields a structure
similar to the standard ObsPy Stream/Trace structure:

>>> from obspy.segy import readSEGY #doctest: +SKIP
>>> segy = readSEGY("/path/to/00001034.sgy_first_trace") #doctest: +SKIP
>>> segy #doctest: +SKIP
<obspy.segy.segy.SEGYFile object at 0x...>
>>> print(segy) #doctest: +SKIP
1 traces in the SEG Y structure.

The traces are a list stored in segy.traces. The trace header values are stored
in the traces attributes. They are just the raw SEGY attributes.

By default these values will not be unpacked and thus will not show up in
ipython's tab completion. See the header.py file for a list of all available
trace header attributes. They will be unpacked on the fly.

Writing
-------
Writing is also done in the usual way:

>>> st.write('file.segy', format = 'SEGY') #doctest: +SKIP

or 

>>> st.write('file.su', format = 'SU') #doctest: +SKIP
"""

from obspy.core.util import _getVersionString
from segy import readSEGY
from segy import readSU


__version__ = _getVersionString("obspy.segy")


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
