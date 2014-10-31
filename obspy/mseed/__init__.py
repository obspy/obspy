# -*- coding: utf-8 -*-
"""
obspy.mseed - MiniSEED read and write support
==============================================
This module provides read and write support for the `MiniSEED
<http://www.iris.edu/ds/nodes/dmc/data/formats/#miniseed>`_ (and the
data part of full SEED) waveform data format and some other convenient
methods to handle MiniSEED files. It utilizes
`libmseed <http://www.iris.edu/ds/nodes/dmc/software/downloads/libmseed/>`_,
a C library by Chad Trabant.

.. seealso::

    The format is  defined in the
    `SEED Manual <http://www.fdsn.org/seed_manual/SEEDManual_V2.4.pdf>`_.

:copyright:
    The ObsPy Development Team (devs@obspy.org) & Chad Trabant
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)

Reading
-------

Reading MiniSEED (and the data part of full SEED files) is handled by using
ObsPy's standard :func:`~obspy.core.stream.read` function. The format is
detected automatically.

>>> from obspy import read
>>> st = read("/path/to/test.mseed")
>>> st #doctest: +ELLIPSIS
<obspy.core.stream.Stream object at 0x...>
>>> print(st)  #doctest: +ELLIPSIS
1 Trace(s) in Stream:
NL.HGN.00.BHZ | 2003-05-29T02:13:22.043400Z - ... | 40.0 Hz, 11947 samples
>>> print(st[0].stats) # doctest: +ELLIPSIS
         network: NL
         station: HGN
        location: 00
         channel: BHZ
       starttime: 2003-05-29T02:13:22.043400Z
         endtime: 2003-05-29T02:18:20.693400Z
   sampling_rate: 40.0
           delta: 0.025
            npts: 11947
           calib: 1.0
         _format: MSEED
           mseed: ...

MiniSEED specific metadata is stored in ``stats.mseed``.

>>> for k, v in sorted(st[0].stats.mseed.items()):
...     print("'%s': %s" % (k, str(v))) # doctest: +NORMALIZE_WHITESPACE
    'byteorder':  >
    'dataquality': R
    'encoding': STEIM2
    'filesize': 8192
    'number_of_records':  2
    'record_length': 4096

The actual data is stored as a :class:`~numpy.ndarray` in the ``data``
attribute of each trace.

>>> print(st[0].data)
[2787 2776 2774 ..., 2850 2853 2853]


Several key word arguments are available which can be used for example to
only read certain records from a file or force the header byteorder:
``starttime``, ``endtime``, ``headonly``, ``sourcename``, ``reclen``,
``details``, and ``header_byteorder``. They are passed to the
:meth:`~obspy.mseed.core.readMSEED` method so refer to it for details to
each parameter.

Writing
-------
Write data back to disc or a file like object using the
:meth:`~obspy.core.stream.Stream.write` method of a
:class:`~obspy.core.stream.Stream` or
:class:`~obspy.core.trace.Trace` object.

>>> st.write('Mini-SEED-filename.mseed', format='MSEED') #doctest: +SKIP

You can also specify several keyword arguments that change the resulting
Mini-SEED file: ``reclen``, ``encoding``, ``byteorder``, ``flush``,  and
``verbose``.
They are are passed to the :meth:`~obspy.mseed.core.writeMSEED` method so
refer to it for details to each parameter.

Refer to the :meth:`~obspy.mseed.core.writeMSEED` method for details to
each parameter.

So in order to write a STEIM1 encoded Mini-SEED file with a record length of
512 byte do the following:

>>> st.write('out.mseed', format='MSEED', reclen=512,  # doctest: +SKIP
...          encoding='STEIM1')

Many of these can also be set at each Trace`s ``stats.mseed`` attribute which
allows for per Trace granularity. Values passed to the
:func:`~obspy.core.stream.read` function have priority.


Encoding Support
----------------

MiniSEED is a format with a large variety of different blockettes,
byte order issues, and encodings. The capabilities of ``obspy.mseed``
largely coincide with the capabilities of ``libmseed``, which is the de-facto
standard MiniSEED library and used internally by ObsPy.

In regards to the different encodings for the data part of MiniSEED files
this means the following:

* Read support for: ACSII, 16 and 32 bit integers, 32 and 64 bit floats,
  STEIM 1 + 2, all GEOSCOPE encodings, the CDSN encoding, the SRO encoding,
  and the DWWSSN encoding
* Write support for: ACSII, 16 and 32 bit integers, 32 and 64 bit floats,
  and STEIM 1 + 2
* Unsupported: 24 bit integers, US National Network compression,
  Graefenberg 16 bit gain ranged encoding, IPG - Strasbourg 16 bit gain ranged
  encoding, STEIM 3, the HGLP encoding, and the RSTN 16 bit gain ranged
  encoding

Utilities
---------

This module also contains a couple of utility functions which are useful for
some purposes. Refer to the documentation of each for details.

+------------------------------------------------------+--------------------------------------------------------------------------+
| :func:`~obspy.mseed.util.getStartAndEndTime`         | Fast way of getting the temporal bounds of a well-behaved MiniSEED file. |
+------------------------------------------------------+--------------------------------------------------------------------------+
| :func:`~obspy.mseed.util.getTimingAndDataQuality`    |  Returns information about the data and timing quality flags in a file.  |
+------------------------------------------------------+--------------------------------------------------------------------------+
| :func:`~obspy.mseed.util.shiftTimeOfFile`            |      Shifts the time of a file preserving all blockettes and flags.      |
+------------------------------------------------------+--------------------------------------------------------------------------+
| :func:`~obspy.mseed.util.getRecordInformation`       |   Returns record information about given files and file-like object.     |
+------------------------------------------------------+--------------------------------------------------------------------------+
| :func:`~obspy.mseed.util.set_flags_in_fixed_headers` |   Updates a given miniSEED file with some fixed header flags.            |
+------------------------------------------------------+--------------------------------------------------------------------------+
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
