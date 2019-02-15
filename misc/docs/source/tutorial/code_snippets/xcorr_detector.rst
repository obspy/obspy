==========================
Cross Correlation Detector
==========================

This code snippets shows how to use the functions
:func:`~obspy.signal.cross_correlation.correlate_stream_template` and
:func:`~obspy.signal.cross_correlation.similarity_detector`.

In the first example we will determine the origin time of the 2017
Korean nuclear test, by using a template of another test in 2013. We will
use only the Z component of the station IC.MDJ.

.. plot:: tutorial/code_snippets/xcorr_detector_1.py
   :include-source:

The detection corresponds to the starttime of the template. If we know the
origin time from the first explosion, we can directly get the origin time
of the 2017 explosion:

.. plot:: tutorial/code_snippets/xcorr_detector_2.py
   :include-source: