===========================
Export Seismograms to ASCII
===========================

----------------
Built-in Formats
----------------

You may directly export waveform data to any ASCII format available by ObsPy
using the :meth:`~obspy.core.stream.Stream.write` method on the generated
:class:`~obspy.core.stream.Stream` object.

    >>> from obspy.core import read
    >>> stream = read('https://examples.obspy.org/RJOB20090824.ehz')
    >>> stream.write('outfile.ascii', format='SLIST')

The following ASCII formats are currently supported:

* ``SLIST``, a ASCII time series format represented with a header line
  followed by a sample lists (see also
  :func:`SLIST format description<obspy.io.ascii.core._write_slist>`)::

    TIMESERIES BW_RJOB__EHZ_D, 6001 samples, 200 sps, 2009-08-24T00:20:03.000000, SLIST, INTEGER, 
    288 300 292 285 265 287
    279 250 278 278 268 258    
    ...

* ``TSPAIR``, a ASCII format where data is written in time-sample pairs
  (see also
  :func:`TSPAIR format description<obspy.io.ascii.core._write_tspair>`)::

    TIMESERIES BW_RJOB__EHZ_D, 6001 samples, 200 sps, 2009-08-24T00:20:03.000000, TSPAIR, INTEGER, 
    2009-08-24T00:20:03.000000  288
    2009-08-24T00:20:03.005000  300
    2009-08-24T00:20:03.010000  292
    2009-08-24T00:20:03.015000  285
    2009-08-24T00:20:03.020000  265
    2009-08-24T00:20:03.025000  287
    ...

* ``SH_ASC``, ASCII format supported by `Seismic Handler`_::

    DELTA: 5.000000e-03
    LENGTH: 6001
    START: 24-AUG-2009_00:20:03.000
    COMP: Z
    CHAN1: E
    CHAN2: H
    STATION: RJOB
    CALIB: 1.000000e+00
    2.880000e+02 3.000000e+02 2.920000e+02 2.850000e+02 
    2.650000e+02 2.870000e+02 2.790000e+02 2.500000e+02 
    ...

-------------
Custom Format
-------------

In the following, a small Python script is shown which converts each
:class:`~obspy.core.trace.Trace` of a seismogram file to an ASCII file with
a custom header. Waveform data will be multiplied by a given ``calibration``
factor and written using NumPy_'s :func:`~numpy.savetxt` function. 

.. include:: export_seismograms_to_ascii.py
   :literal:


.. _`Seismic Handler`: https://www.seismic-handler.org
.. _NumPy: http://www.numpy.org/
