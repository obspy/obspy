===================
Reading Seismograms
===================

Seismograms of the formats SAC, MiniSEED, GSE2, SEISAN, Q, etc. can be imported
into a :py:class:`~obspy.core.stream.Stream` object using the 
:py:func:`~obspy.core.stream.read` function.

:py:class:`Streams <obspy.core.stream.Stream>` are list-like objects which
contain multiple :py:class:`~obspy.core.trace.Trace` objects, i.e.
gap-less continuous time series and related header/meta information.

Each :py:class:`~obspy.core.trace.Trace` object has a attribute called ``data``
pointing to a NumPy_ :py:class:`~numpy.ndarray` of
the actual time series and the attribute ``stats`` which contains all meta
information in a dictionary-like Stats object. Both attributes ``starttime``
and ``endtime`` of the :py:class:`~obspy.core.trace.Stats` object are
:py:class:`~obspy.core.utcdatetime.UTCDateTime` objects.

The following example demonstrates how a single GSE2_-formatted seismogram file
is read into a ObsPy :py:class:`~obspy.core.stream.Stream` object. There exists
only one :py:class:`~obspy.core.trace.Trace` in the given seismogram:

.. doctest::

   >>> from obspy.core import read
   >>> st = read('http://examples.obspy.org/RJOB_061005_072159.ehz.new')
   >>> print st
   1 Trace(s) in Stream:
   .RJOB..Z | 2005-10-06T07:21:59.849998Z - 2005-10-06T07:24:59.844998Z | 200.0 Hz, 36000 samples

Seismogram meta data are accessed via the ``stats`` keyword on each
:py:class:`~obspy.core.trace.Trace`:

.. doctest::

    >>> print st[0].stats
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
    >>> st[0].stats.station
    'RJOB'
    >>> st[0].stats.gse2.datatype
    'CM6'

The actual waveform data may be retrieved via the ``data`` keyword on each
:py:class:`~obspy.core.trace.Trace`:

.. doctest::

    >>> st[0].data
    array([-38,  12,  -4, ..., -14,  -3,  -9])
    >>> st[0].data[0:3]
    array([-38,  12,  -4])
    >>> len(st[0])
    36000

:py:class:`~obspy.core.stream.Stream` objects offer a plotting method for fast
preview of the waveform (requires the obspy.imaging module):

.. doctest::

    >>> st.plot(color='k')

.. plot:: source/tutorial/reading_seismograms.py

.. _NumPy: http://numpy.scipy.org/
.. _GSE2: http://obspy.org/export/2593/obspy/trunk/obspy.gse2/docs/other/provisional_GSE2.1.pdf
