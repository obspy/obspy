.. _waveform_data:

======================
Handling Waveform Data
======================


.. doctest::

   >>> from obspy.core import read
   >>> st = read()
   >>> print st
   3 Trace(s) in Stream:
   BW.RJOB..EHZ | 2009-08-24T00:20:03.000000Z - 2009-08-24T00:20:32.990000Z | 100.0 Hz, 3000 samples
   BW.RJOB..EHN | 2009-08-24T00:20:03.000000Z - 2009-08-24T00:20:32.990000Z | 100.0 Hz, 3000 samples
   BW.RJOB..EHE | 2009-08-24T00:20:03.000000Z - 2009-08-24T00:20:32.990000Z | 100.0 Hz, 3000 samples


* Automatic file format detection.
* Always results in a Stream object.
* Raw data available as a numpy.ndarray.

.. doctest::

   >>> st[0].data
   array([ 0.        ,  0.00694644,  0.07597424, ...,  1.93449584,
           0.98196204,  0.44196924])
