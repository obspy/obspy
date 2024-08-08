# -*- coding: utf-8 -*-
"""
obspy.io.gcf - Guralp Compressed Format, GCF, read and write support for ObsPy
==============================================================================
This module provides read and write support for `GCF
<https://www.guralp.com/apps/ok?doc=GCF_Intro>`_ waveform data and header info
acording to GCF Reference `SWA-RFC-GCFR Issue F, December 2021
<https://www.guralp.com/apps/ok?doc=GCF_format>`_

:copyright:
    The ObsPy Development Team (devs@obspy.org), Ran Novitsky Nof, 
    Peter Schmidt and Reynir Bödvarsson
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)

Reading
-------

Similar to reading any other waveform data format using :mod:`obspy.core`
The format will be determined automatically.

>>> from obspy import read
>>> st = read("/path/to/20160603_1955n.gcf")

GCF specific metadata is stored in ``stats.gcf``

>>> for k, v in sorted(st[0].stats.gcf.items()):
...     print("'%s': %s" % (k, str(v)))  # doctest: +NORMALIZE_WHITESPACE
'FIC': -49378
'RIC': -49312
'blk': 0
'digi': 0
'gain': 1
'stat': 0
'stream_id': 6018N4
'sys_type': 1
'system_id': 6281
't_leap': False
'ttl': 6

Several keyword arguments are available to set parameters required by the
returned :class:`~obspy.core.stream.Stream` object as well as to control
merging of data blocks on reading: ``network``, ``station``, ``locationcode``,
``bandcode``, ``instrumentcode``, ``channel_prefix``, ``blockmerge``,
``headonly``, ``cleanoverlap``, ``errorret``. They are passed to the
:meth:`~obspy.io.gcf.core._read_gcf` method so refer to this for details of
each parameter.

Function may raise ``TypeError`` on bad arguments (e.g. out of range) and
``IOError`` on failure to read/decode file or errors in data blocks in file
(bad blocks may be returned by use of keyword argument ``errorret`` though,
preventing IOError's raised by these)

Writing
-------

Write data back to disc or a file like object using the
:meth:`~obspy.core.stream.Stream.write` method of a
:class:`~obspy.core.stream.Stream` or
:class:`~obspy.core.trace.Trace` object.

>>> st.write('GCF-filename.gcf', format='GCF')  # doctest: +SKIP


Several key word arguments are available to set required GCF specific header
information and allowed misalignment of supported starttime of first data
sample: ``stream_id``, ``system_id``, ``is_leap``, ``gain``, ``ttl``, ``digi``,
``sys_type``, ``misalign``

GCF specific header information can also be provided in ``stats.gcf`` on each
:class:`~obspy.core.trace.Trace` object. If a specific piece of header
information is available both in ``stats.gcf`` and set by its correspondng
keyworde argument, the keyword argument will have precedenc. For default values
if neither set by ``stats.gcf`` nor by use of appropriate keyword arguments
refer to :meth:`~obspy.io.gcf.core._read_gcf`

.. note:: 
    The GCF format is only guaranteed to support 32-bit signed integer values.
    While data with values out of range may be properly stored in the GCF
    format (if first and last data sample can be represented as a 32-bit signed
    integer as well as all first difference values of the data vector) the
    current implementation only permits input data to be representable as a
    32-bit signed integer. If input waveforms cannot be represented as 32-bit
    signed integers they will be clipped at -2,147,483,648 and 2,147,483,647

The GCF format only supports a restricted set of sampling rates. To check if a
sampling rate is supported, function :func:`~obspy.io.gcf.core.compatible_sps`
is provided.

The GCF format only supports fractional start of first data sample for sampling
rates > 250 Hz. For lower sampling rates data must first be sampled at whole
second. For greater sampling rates fractional start time is permitted but
restricted and smapling rate specific. For info on permitted fractional start
time for a given sampling rate, function
:func:`~obspy.io.gcf.core.get_time_denominator` is provided.


Utilities
---------

This module also contains a couple of utility functions which are useful for
some purposes. Refer to the documentation of each for details.

+-------------------------------------------------+---------------------------+
| :func:`~obspy.io.gcf.core.compatible_sps`       | Checks if a sampling rate |
|                                                 | is compatible with the    |
|                                                 | implemented GCF format    |
+-------------------------------------------------+---------------------------+
| :func:`~obspy.io.gcf.core.get_time_denominator` | Returns the time          |
|                                                 | fractional offset         |
|                                                 | offset denominator, d,    |
|                                                 | associated with an input  |
|                                                 | sampling rate.            |
+-------------------------------------------------+---------------------------+
"""
if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
