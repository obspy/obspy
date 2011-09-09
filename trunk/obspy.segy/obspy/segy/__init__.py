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

SEG Y files contain a large amount of additional meta data which are not
unpacked by default. However you may access those values by just calling the
header key directly or you may just use the ``unpack_trace_headers`` keyword on
the ``read()`` method to unpack all related meta data.

>>> st1 = read("/path/to/00001034.sgy_first_trace")
>>> len(st1[0].stats.segy.trace_header)
6
>>> st1[0].stats.segy.trace_header.data_use # unpacking a value on the fly
1
>>> len(st1[0].stats.segy.trace_header)
7
>>> st2 = read("/path/to/00001034.sgy_first_trace", unpack_trace_headers=True)
>>> len(st2[0].stats.segy.trace_header)
92

Reading SEG Y files with ``unpack_trace_headers=True`` will become very slow
and memory intensive for a large number of traces due to the huge number of
objects created.

A slightly faster way to read data is the internal reading method:

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

>>> st.write('file.segy', format='SEGY') #doctest: +SKIP

or

>>> st.write('file.su', format='SU') #doctest: +SKIP


SEGY files are sensitive to their headers and wrong headers might break them.

If some or all headers are missing, obspy.segy will attempt to autogenerate
them and fill them with senseful values. Most header values will be 0
nonetheless.

The following script demonstrated how to write SEGY without reusing some
headers and optionally setting custom header values::

    from obspy.core import read, Trace, AttribDict, Stream, UTCDateTime
    from obspy.segy.segy import SEGYTraceHeader, SEGYBinaryFileHeader
    from obspy.segy.core import readSEGY
    import numpy as np
    import sys
    
    stream = Stream()
    
    for _i in xrange(3):
        # Create some random data.
        data = np.random.ranf(1000)
        data = np.require(data, dtype='float32')
        trace = Trace(data=data)
    
        # Attributes in trace.stats will overwrite everything in
        # trace.stats.segy.trace_header
        trace.stats.delta = 0.01
        # SEGY does not support microsecond precission! Any microseconds will be
        # discarded.
        trace.stats.starttime = UTCDateTime(2011,11,11,11,11,11)
    
        # If you want to set some additional attributes in the trace header, add
        # one and only set the attributes you want to be set. Otherwise the header
        # will be created for you with default values.
        if not hasattr(trace.stats, 'segy.trace_header'):
            trace.stats.segy = {}
        trace.stats.segy.trace_header = SEGYTraceHeader()
        trace.stats.segy.trace_header.trace_sequence_number_within_line = _i + 1
        trace.stats.segy.trace_header.receiver_group_elevation = 444
    
        # Add trace to stream
        stream.append(trace)
    
    # A SEGY file has file wide headers. This can be attached to the stream object.
    # If these are not set, they will be autocreated with default values.
    stream.stats = AttribDict()
    stream.stats.textual_file_header = 'Textual Header!'
    stream.stats.binary_file_header = SEGYBinaryFileHeader()
    stream.stats.binary_file_header.trace_sorting_code = 5
    
    print "Stream object before writing..."
    print stream
    
    stream.write("TEST.sgy", format="SEGY", data_encoding=1, byteorder=sys.byteorder)
    print "Stream object after writing. Will have some segy attributes..."
    print stream
    
    print "Reading using obspy.segy..."
    st1 = readSEGY("TEST.sgy")
    print st1
    
    print "Reading using obspy.core..."
    st2 = read("TEST.sgy")
    print st2
    
    print "Just to show that the values are written..."
    print [tr.stats.segy.trace_header.receiver_group_elevation for tr in stream]
    print [tr.stats.segy.trace_header.receiver_group_elevation for tr in st2]
    print stream.stats.binary_file_header.trace_sorting_code
    print st1.stats.binary_file_header.trace_sorting_code
"""

from obspy.core.util import _getVersionString
from segy import readSEGY
from segy import readSU


__version__ = _getVersionString("obspy.segy")


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
