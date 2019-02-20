=================================
Cross-Correlation Pick Correction
=================================

This example shows how to align the waveforms of phase onsets of two
earthquakes in order to correct the original pick times that can never be set
perfectly consistent in routine analysis. A parabola is fit to the concave part
of the cross correlation function around its maximum, following the approach by
[Deichmann1992]_.

To adjust the parameters (i.e. the used time window around the pick and the
filter settings) and to validate and check the results the options `plot` and
`filename` can be used to open plot windows or save the figure to a file.

See the documentation of
:func:`~obspy.signal.cross_correlation.xcorr_pick_correction` for more details.

The example will print the time correction for pick 2 and the respective
correlation coefficient and open a plot window for correlations on both the
original and preprocessed data::

    No preprocessing:
      Time correction for pick 2: -0.014459
      Correlation coefficient: 0.92
    Bandpass prefiltering:
      Time correction for pick 2: -0.013025
      Correlation coefficient: 0.98

.. plot:: tutorial/code_snippets/xcorr_pick_correction.py
   :include-source:
