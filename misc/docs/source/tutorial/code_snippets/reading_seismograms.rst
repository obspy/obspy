.. _reading-seismogramms:

===================
Reading Seismograms
===================

Seismograms of various formats (e.g. SAC, MiniSEED, GSE2, SEISAN, Q, etc.) can
be imported into a :class:`~obspy.core.stream.Stream` object using the
:func:`~obspy.core.stream.read` function.

:class:`Streams <obspy.core.stream.Stream>` are list-like objects which
contain multiple :class:`~obspy.core.trace.Trace` objects, i.e.
gap-less continuous time series and related header/meta information.

Each :class:`~obspy.core.trace.Trace` object has a attribute called ``data``
pointing to a NumPy_ :class:`~numpy.ndarray` of
the actual time series and the attribute ``stats`` which contains all meta
information in a dictionary-like :class:`~obspy.core.trace.Stats` object. Both
attributes ``starttime`` and ``endtime`` of the
:class:`~obspy.core.trace.Stats` object are
:class:`~obspy.core.utcdatetime.UTCDateTime` objects.

The following example demonstrates how a single GSE2_-formatted seismogram file
is read into a ObsPy :class:`~obspy.core.stream.Stream` object. There exists
only one :class:`~obspy.core.trace.Trace` in the given seismogram:

.. doctest::

   >>> from obspy import read
   >>> st = read('http://examples.obspy.org/RJOB_061005_072159.ehz.new')
   >>> print(st)
   1 Trace(s) in Stream:
   .RJOB..Z | 2005-10-06T07:21:59.849998Z - 2005-10-06T07:24:59.844998Z | 200.0 Hz, 36000 samples
   >>> len(st)
   1
   >>> tr = st[0]  # assign first and only trace to new variable
   >>> print(tr)
   .RJOB..Z | 2005-10-06T07:21:59.849998Z - 2005-10-06T07:24:59.844998Z | 200.0 Hz, 36000 samples

-------------------
Accessing Meta Data
-------------------

Seismogram meta data, data describing the actual waveform data, are accessed
via the ``stats`` keyword on each :class:`~obspy.core.trace.Trace`:

.. doctest::

    >>> print(tr.stats)  # doctest: +NORMALIZE_WHITESPACE
             network:
             station: RJOB
            location:
             channel: Z
           starttime: 2005-10-06T07:21:59.849998Z
             endtime: 2005-10-06T07:24:59.844998Z
       sampling_rate: 200.0
               delta: 0.005
                npts: 36000
               calib: 0.0948999971151
             _format: GSE2
                gse2: AttribDict({'instype': '      ', 'datatype': 'CM6', 'hang': -1.0, 'auxid': 'RJOB', 'vang': -1.0, 'calper': 1.0})
    >>> tr.stats.station
    'RJOB'
    >>> tr.stats.gse2.datatype
    'CM6'

-----------------------
Accessing Waveform Data
-----------------------

The actual waveform data may be retrieved via the ``data`` keyword on each
:class:`~obspy.core.trace.Trace`:

.. doctest::

    >>> tr.data
    array([-38,  12,  -4, ..., -14,  -3,  -9])
    >>> tr.data[0:3]
    array([-38,  12,  -4])
    >>> len(tr)
    36000

------------
Data Preview
------------

:class:`~obspy.core.stream.Stream` objects offer a
:meth:`~obspy.core.stream.Stream.plot` method for fast
preview of the waveform (requires the :mod:`obspy.imaging` module):

    >>> st.plot()

.. plot:: tutorial/code_snippets/reading_seismograms.py

.. _NumPy: http://www.numpy.org/
.. _GSE2: https://github.com/obspy/obspy/blob/master/obspy/io/gse2/docs/provisional_GSE2.1.pdf?raw=true
