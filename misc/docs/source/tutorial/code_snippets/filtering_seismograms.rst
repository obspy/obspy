=====================
Filtering Seismograms
=====================

The following script shows how to filter a seismogram. The example uses a
zero-phase-shift low-pass filter with a corner frequency of 1 Hz using 2
corners. This is done in two runs forward and backward, so we end up with 4
corners de facto.

The available filters are:

* ``bandpass``
* ``bandstop``
* ``lowpass``
* ``highpass`` 

.. plot:: tutorial/code_snippets/filtering_seismograms.py
   :include-source:
