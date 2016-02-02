====================
Seismogram Envelopes
====================

The following script shows how to filter a seismogram and plot it together with
its envelope.

This example uses a zero-phase-shift bandpass to filter the data
with corner frequencies 1 and 3 Hz, using 2 corners (two runs due to zero-phase
option, thus 4 corners overall). Then we calculate the envelope and plot it
together with the Trace. Data can be found
`here <https://examples.obspy.org/RJOB_061005_072159.ehz.new>`_.

.. plot:: tutorial/code_snippets/seismogram_envelopes.py
   :include-source:
