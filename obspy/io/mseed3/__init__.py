# -*- coding: utf-8 -*-
"""
obspy.io.mseed3 - miniSEED v2/v3 read and write support for ObsPy
==================================================================
This module provides read and write support for miniSEED, covering both
the miniSEED v2 and v3 formats. It is a thin ObsPy layer
around the external `pymseed <https://pypi.org/project/pymseed/>`_ package,
which itself is based on the `libmseed
<https://github.com/EarthScope/libmseed>`_ C library.

.. seealso::

    The miniSEED version 3 format is defined in the
    `FDSN miniSEED3 specification
    <https://docs.fdsn.org/projects/miniseed3/>`_.
    The (still widely used) version 2 format is defined in the
    `SEED Manual <https://doi.org/10.7914/en3h-2318>`_.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)

Reading
-------

Reading miniSEED (version 2 or 3) is handled by using ObsPy's standard
:func:`~obspy.core.stream.read` function. Because :mod:`obspy.io.mseed`
(libmseed version 2) also claims miniSEED v2 files, and is tried first
during format autodetection, pass ``format="MSEED3"`` explicitly to make
sure this module is used, e.g. to read a version 2 file with this reader
instead.

>>> from obspy import read
>>> st = read("/path/to/testdata-3channel-signal.mseed3")
>>> st  # doctest: +ELLIPSIS
<obspy.core.stream.Stream object at 0x...>
>>> print(st)  # doctest: +ELLIPSIS
3 Trace(s) in Stream:
IU.COLA.00.LH1 | 2010-02-27T06:50:00.069539Z - ... | 1.0 Hz, 4200 samples
IU.COLA.00.LH2 | 2010-02-27T06:50:00.069539Z - ... | 1.0 Hz, 4200 samples
IU.COLA.00.LHZ | 2010-02-27T06:50:00.069539Z - ... | 1.0 Hz, 4200 samples
>>> print(st[0].stats)  # doctest: +ELLIPSIS
         network: IU
         station: COLA
        location: 00
         channel: LH1
       starttime: 2010-02-27T06:50:00.069539Z
         endtime: 2010-02-27T07:59:59.069539Z
   sampling_rate: 1.0
           delta: 1.0
            npts: 4200
           calib: 1.0
         _format: MSEED3
          mseed3: ...

miniSEED specific metadata is stored in ``stats.mseed3``.  By default,
``details=False``, so ``stats.mseed3`` contains only a source_id field.
See example below for how to get more detailed metadata.

>>> for k, v in sorted(st[0].stats.mseed3.items()):  # doctest: +SKIP
...     print("'%s': %s" % (k, str(v)))
    'source_id': FDSN:IU_COLA_00_L_H_1

``source_id`` is the trace's `FDSN Source Identifier
<https://docs.fdsn.org/projects/source-identifiers/>`_, the FDSN
standard for identifying data sources used by miniSEED v3 internally
that is a superset of SEED network/station/location/channel (NSLC) codes.
It is converted to the usual ``network``/``station``/``location``/``channel``
fields so most code does not need to deal with it directly.

The actual data is stored as a :class:`~numpy.ndarray` in the ``data``
attribute of each trace.

>>> print(st[0].data)  # doctest: +SKIP
[-502676 -504105 -507491 ...]

Several keyword arguments are available to control what is read, e.g. to
only read a certain time window or a subset of channels: ``starttime``,
``endtime``, ``headonly``, ``sourceid``, ``sourcename``, ``details``,
``skip_not_data``, ``validate_crc``, ``split_version``, and ``verbose``.
They are passed to the :func:`~obspy.io.mseed3.core._read_mseed3` method,
so refer to it for details on each parameter. For example, restrict
reading to a single channel with ``sourceid`` (which accepts FDSN Source ID
wildcard patterns) or ``sourcename`` (which accepts SEED ID wildcard
patterns):

>>> st = read(
...     "/path/to/testdata-3channel-signal.mseed3",
...     sourceid="FDSN:IU_COLA_00_L_H_Z",
... )
>>> print(st)  # doctest: +ELLIPSIS
1 Trace(s) in Stream:
IU.COLA.00.LHZ | 2010-02-27T06:50:00.069539Z - ... | 1.0 Hz, 4200 samples

Passing ``details=True`` additionally populates ``stats.mseed3`` with
run-length-deduplicated lists of the timing quality, publication version,
and encoding of the records that make up each trace, in the order that
they were encountered:

>>> st = read("/path/to/testdata-3channel-signal.mseed3", details=True)
>>> for k, v in sorted(st[0].stats.mseed3.items()):  # doctest: +SKIP
...     print("'%s': %s" % (k, str(v)))
    'encodings': ['STEIM-2 integer compression']
    'number_of_records': 36
    'publication_versions': [4]
    'source_id': FDSN:IU_COLA_00_L_H_1
    'timing_qualities': [100]

Writing
-------
Write data back to disc or a file-like object using the
:meth:`~obspy.core.stream.Stream.write` method of a
:class:`~obspy.core.stream.Stream` or
:class:`~obspy.core.trace.Trace` object.

>>> st.write("mseed3-filename.mseed3", format="MSEED3")  # doctest: +SKIP

You can also specify keyword arguments that change the resulting file:
``encoding``, ``max_record_length``, ``format_version``, and ``overwrite``.
They are passed to the :func:`~obspy.io.mseed3.core._write_mseed3` method,
so refer to it for details on each parameter.

So in order to write a STEIM1-encoded, version 2 miniSEED file with a
record length of 512 bytes do the following:

>>> st.write(
...     "out.mseed2",
...     format="MSEED3",
...     format_version=2,  # doctest: +SKIP
...     max_record_length=512,
...     encoding="STEIM1",
... )

Encoding Support
----------------

Reading follows whatever encodings the underlying ``pymseed``/``libmseed``
support, which covers essentially all encodings ever used in miniSEED,
including STEIM 1 & 2, integer, and floating point data.

Writing is more restricted. Supported encodings are: ``ASCII``/``TEXT``
(``0``), ``INT16`` (``1``), ``INT32`` (``3``), ``FLOAT32`` (``4``)*,
``FLOAT64`` (``5``)*, ``STEIM1`` (``10``), and ``STEIM2`` (``11``)*. If no
``encoding`` is given it is derived from the ``dtype`` of the data and the
appropriate default encoding (marked with an asterisk above) is chosen.
"""

from .core import _is_mseed3, _read_mseed3, _write_mseed3  # noqa: F401


if __name__ == "__main__":
    import doctest

    doctest.testmod(exclude_empty=True)
