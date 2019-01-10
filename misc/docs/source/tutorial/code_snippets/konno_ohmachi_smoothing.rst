=======================
Konno-Ohmachi Smoothing
=======================

This example shows how to apply smoothing to amplitude spectra following the
approach by developed by [Konno1998]_. This method applies a smoothing window
whose width is constant in log space rather than linear space. This can be
useful, for example, to reduce noise before fitting a spectral model.

The following example will apply Konno-Ohmachi smoothing and a Savitzky-Golay
filter (included in scipy) and compare the results on a log-log plot. Note how
the Savitzky-Golay filter appears over-smoothed at lower frequencies and
under-smoothed at higher frequencies but the Konno-Ohmachi filtered data
appears reasonable throughout.

.. plot:: tutorial/code_snippets/konno_ohmachi_smoothing_1.py
   :include-source:

For large spectra it can be very time-consuming and memory intensive to
calculate the smoothing coefficients for all frequencies. In these cases
it is usually best specify a subset of frequencies of interest using the
``center_frequencies`` parameter. Multiple spectra can also be smoothed
simultaneously which is more efficient than smoothing each individually.

.. plot:: tutorial/code_snippets//konno_ohmachi_smoothing_2.py
   :include-source:
