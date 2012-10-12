============================
Continuous Wavelet Transform
============================

Small script doing the continuous wavelet transform using the
`mlpy <https://mlpy.fbk.eu/>`_ package for infrasound data recorded at Yasur
in 2008. Further details on wavelets can be found at
`Wikipedia <http://en.wikipedia.org/wiki/Morlet_wavelet>`_ - in the article the
omega0 factor is denoted as sigma. *(really sloppy and possibly incorrect: the
omega0 factor tells you how often the wavelet fits into the time window, dj
defines the spacing in the scale domain)* 

.. include:: continuous_wavelet_transform.py
   :literal:

.. plot:: source/tutorial/code_snippets/continuous_wavelet_transform.py
