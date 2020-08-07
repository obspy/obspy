# -*- coding: utf-8 -*-
"""
obspy.io.segy - SEG Y and SU read and write support for ObsPy
=============================================================

The obspy.io.segy package contains methods in order to read and write files
in the
`SEG Y (rev. 1) <https://www.seg.org/documents/10161/77915/seg_y_rev1.pdf>`_
and SU (Seismic Unix) format.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)

.. note::
    The module can currently read files that are in accordance to the SEG Y
    rev. 1 specification but has been designed to be able to handle custom
    headers as well. This functionality is not yet exposed because the
    developers have no files to test it with. If you have access to some files
    with custom headers please consider sending them to ``devs@obspy.org``.

Reading
-------
The SEG Y and Seismic Unix (SU) file formats are quite different from the
file formats usually used in observatories (GSE2, MiniSEED, ...). The
:class:`~obspy.core.stream.Stream`/:class:`~obspy.core.trace.Trace` structures
of ObsPy are therefore not fully suited to handle them. Nonetheless they work
well enough if some potential problems are kept in mind.

SEG Y files can be read in four different ways that have different
advantages/disadvantages. Most of the following also applies to SU files with
some changes (keep in mind that SU files have no file wide headers).

1. Using the standard :func:`~obspy.core.stream.read` function.
2. Using the :mod:`obspy.io.segy` specific
   :func:`obspy.io.segy.core._read_segy` function.
3. Using the internal :func:`obspy.io.segy.segy._read_segy` function.
4. Some SEG-Y files are too large to be read into memory. The
   :func:`obspy.io.segy.segy.iread_segy` function reads a large file trace
   by trace circumventing this problem.

Reading using methods 1 and 2
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The first two methods will return a :class:`~obspy.core.stream.Stream` object
and they are identical except that the file wide SEGY headers are only
accessible if method 2 is used. These headers are stored in Stream.stats.

The obvious advantage of these methods is that the returned
:class:`~obspy.core.stream.Stream` object interfaces very well with other
functionality provided by ObsPy (file format conversion, filtering, ...).

Due to the fact that a single SEG Y file can contain several tens of thousands
of traces and each trace will be a :class:`~obspy.core.trace.Trace` instance
which in turn will contain other objects these methods are quite slow and
memory intensive.

To somewhat rectify this issue all SEG Y specific trace header attributes are
only unpacked on demand by default.

>>> from obspy.io.segy.core import _read_segy
>>> from obspy.core.util import get_example_file
>>> # or 'from obspy import read' if file wide headers are of no interest
>>> filename = get_example_file("00001034.sgy_first_trace")
>>> st = _read_segy(filename)
>>> st #doctest: +ELLIPSIS
<obspy.core.stream.Stream object at 0x...>
>>> print(st) #doctest: +ELLIPSIS
1 Trace(s) in Stream:
Seq. No. in line:    1 | 2009-06-22T14:47:37.000000Z - 2009-06-22T14:47:41...

SEG Y files contain a large amount of additional trace header fields which are
not unpacked by default. However these values can be accessed by calling the
header key directly or by using the ``unpack_trace_headers`` keyword with the
:func:`~obspy.core.stream.read`/ :func:`~obspy.io.segy.core._read_segy`
functions to unpack all header fields.

>>> st1 = _read_segy(filename)
>>> len(st1[0].stats.segy.trace_header)
8
>>> st1[0].stats.segy.trace_header.data_use # Unpacking a value on the fly.
1
>>> len(st1[0].stats.segy.trace_header) # This value will remain unpacked.
9
>>> st2 = _read_segy(filename, unpack_trace_headers=True)
>>> len(st2[0].stats.segy.trace_header)
92

Reading SEG Y files with ``unpack_trace_headers=True`` will be very slow and
memory intensive for a large number of traces due to the huge number of objects
created.

Reading using method 3
^^^^^^^^^^^^^^^^^^^^^^
The internal reading method is much faster and less of a memory hog but does
not return a :class:`~obspy.core.stream.Stream` object. Instead it returns a
:class:`~obspy.io.segy.segy.SEGYFile` object which is somewhat similar to the
:class:`~obspy.core.stream.Stream` object used in ObsPy but specific to
:mod:`~obspy.io.segy`.

>>> from obspy.io.segy.segy import _read_segy
>>> segy = _read_segy(filename)
>>> segy #doctest: +ELLIPSIS
<obspy.io.segy.segy.SEGYFile object at 0x...>
>>> print(segy)
1 traces in the SEG Y structure.

The traces are a list of :class:`~obspy.io.segy.segy.SEGYTrace` objects
stored in ``segy.traces``. The trace header values are stored in
``trace.header`` as a :class:`~obspy.io.segy.segy.SEGYTraceHeader` object.

By default these header values will not be unpacked and thus will not show up
in ipython's tab completion. See
:const:`obspy.io.segy.header.TRACE_HEADER_FORMAT`
`(source)
<https://github.com/obspy/obspy/blob/master/obspy/io/segy/header.py#L53>`_
for a list of all available trace header attributes. They will be unpacked
on the fly if they are accessed as class attributes.

By default trace data are read into memory, but this may be impractical for
very large datasets. To skip loading data into memory, read SEG Y files with
``headonly=True``.  The ``data`` class attribute will not show up in ipython's
tab completion, but data are read directly from the disk when it is accessed:

>>> from obspy.io.segy.segy import _read_segy
>>> segy = _read_segy(filename, headonly=True)
>>> print(len(segy.traces[0].data))
2001


Iteratively reading a file using method 4
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If a file is too big to be read into memory in one go (as all the other
methods do), read the file trace by trace using the
:func:`obspy.io.segy.segy.iread_segy` function:

>>> from obspy.io.segy.segy import iread_segy
>>> for trace in iread_segy(filename):
...    # Do something meaningful.
...    print(int(trace.data.sum() * 1E9))
-5


Writing
-------

Writing ObsPy :class:`~obspy.core.stream.Stream` objects
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Writing :class:`~obspy.core.stream.Stream` objects is done in the usual way.

>>> st.write('file.segy', format='SEGY') #doctest: +SKIP

or

>>> st.write('file.su', format='SU') #doctest: +SKIP

It is possible to control the data encoding, the byte order and the textual
header encoding of the final file either via the file wide stats object (see
sample code below) or directly via the write method. Possible values and their
meaning are documented here: :func:`~obspy.io.segy.core._write_segy`


Writing :class:`~obspy.io.segy.segy.SEGYFile` objects
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:class:`~obspy.io.segy.segy.SEGYFile` objects are written using its
:func:`~obspy.io.segy.segy.SEGYFile.write` method. Optional kwargs are able to
enforce the data encoding and the byte order.

>>> segy.write('file.segy') #doctest: +SKIP


Converting other file formats to SEG Y
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

SEGY files are sensitive to their headers and wrong headers might break them.

If some or all headers are missing, obspy.io.segy will attempt to autogenerate
them and fill them with somehow meaningful values. It is a wise idea to
manually check the headers because some other programs might use them and
misinterpret the data. Most header values will be 0 nonetheless.

One possibility to get valid headers for files to be converted is to read one
correct SEG Y file and use its headers.

The other possibility is to autogenerate the headers with the help of ObsPy and
a potential manual review of them which is demonstrated in the following
script::

    from obspy import read, Trace, Stream, UTCDateTime
    from obspy.core import AttribDict
    from obspy.io.segy.segy import SEGYTraceHeader, SEGYBinaryFileHeader
    from obspy.io.segy.core import _read_segy
    import numpy as np
    import sys

    stream = Stream()

    for _i in xrange(3):
        # Create some random data.
        data = np.random.ranf(1000)
        data = np.require(data, dtype=np.float32)
        trace = Trace(data=data)

        # Attributes in trace.stats will overwrite everything in
        # trace.stats.segy.trace_header
        trace.stats.delta = 0.01
        # SEGY does not support microsecond precision! Any microseconds will
        # be discarded.
        trace.stats.starttime = UTCDateTime(2011,11,11,11,11,11)

        # If you want to set some additional attributes in the trace header,
        # add one and only set the attributes you want to be set. Otherwise the
        # header will be created for you with default values.
        if not hasattr(trace.stats, 'segy.trace_header'):
            trace.stats.segy = {}
        trace.stats.segy.trace_header = SEGYTraceHeader()
        trace.stats.segy.trace_header.trace_sequence_number_within_line = \
_i + 1
        trace.stats.segy.trace_header.receiver_group_elevation = 444

        # Add trace to stream
        stream.append(trace)

    # A SEGY file has file wide headers. This can be attached to the stream
    # object.  If these are not set, they will be autocreated with default
    # values.
    stream.stats = AttribDict()
    stream.stats.textual_file_header = 'Textual Header!'
    stream.stats.binary_file_header = SEGYBinaryFileHeader()
    stream.stats.binary_file_header.trace_sorting_code = 5

    print "Stream object before writing..."
    print stream

    stream.write("TEST.sgy", format="SEGY", data_encoding=1,
                 byteorder=sys.byteorder)
    print "Stream object after writing. Will have some segy attributes..."
    print stream

    print "Reading using obspy.io.segy..."
    st1 = _read_segy("TEST.sgy")
    print st1

    print "Reading using obspy.core..."
    st2 = read("TEST.sgy")
    print st2

    print "Just to show that the values are written..."
    print [tr.stats.segy.trace_header.receiver_group_elevation
           for tr in stream]
    print [tr.stats.segy.trace_header.receiver_group_elevation for tr in st2]
    print stream.stats.binary_file_header.trace_sorting_code
    print st1.stats.binary_file_header.trace_sorting_code
"""
if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
