============================
Continuous Wavelet Transform
============================

Using ObsPy
-----------

The following is a short example for a continuous wavelet transform using
ObsPy's internal routine based on [Kristekova2006]_.

.. plot:: tutorial/code_snippets/continuous_wavelet_transform_obspy.py
   :include-source:

Using MLPY
----------

Small script doing the continuous wavelet transform using the
`mlpy <https://mlpy.fbk.eu/>`_ package (version 3.5.0) for infrasound data recorded at Yasur
in 2008. Further details on wavelets can be found at `Wikipedia
<https://en.wikipedia.org/wiki/Morlet_wavelet>`_ - in the article the
omega0 factor is denoted as sigma. *(really sloppy and possibly incorrect: the
omega0 factor tells you how often the wavelet fits into the time window, dj
defines the spacing in the scale domain)*

.. plot:: tutorial/code_snippets/continuous_wavelet_transform_mlpy.py
   :include-source:
