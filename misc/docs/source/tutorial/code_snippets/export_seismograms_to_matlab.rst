============================
Export Seismograms to MATLAB
============================

The following example shows how to read in a waveform file with Python and save
each :class:`~obspy.core.trace.Trace` in the resulting
:class:`~obspy.core.stream.Stream` object to one MATLAB_ .MAT file.
The data can the be loaded from within MATLAB with the ``load`` function.

.. include:: export_seismograms_to_matlab.py
   :literal:

.. _MATLAB: https://www.mathworks.com/products/matlab/
