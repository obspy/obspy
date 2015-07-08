=========================
Beamforming - FK Analysis
=========================

The following code shows how to do an FK Analysis with ObsPy. The data are from
the blasting of the AGFA skyscraper in Munich. To use array analysis, we first
manually set up an inventory for this array (usually provided by a web service
or by station XMLs).

* The slowness grid is set to corner values of -3.0 to 3.0 s/km with a step
  fraction of ``sls = 0.03``.
* The window length is 1.0 s, using a step fraction of 0.05 s.
* The data is bandpass filtered, using corners at 1.0 and 8.0 Hz.

The output is a :class:`~obspy.signal.array_analysis.BeamformerResult` object,
stored as ``results`` and provides plot options. We first plot relative power,
absolute power, backazimuth and slowness over time; the colorbar corresponds
to relative power. The second plot is a polar plot, which sums the relative
power in gridded bins, each defined by backazimuth and slowness of the
analyzed signal part.

.. plot:: tutorial/code_snippets/beamforming_fk_analysis_1.py
   :include-source:
