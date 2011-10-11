===================================
Poles and Zeros, Frequency Response
===================================

The following lines show how to calculate and visualize the frequency response
of a LE-3D/1s seismometer with sampling interval 0.005s and 16384 points of
fft. Two things have to be taken into account for the phase (actually for the
imaginary part of the response):

* the fft that is used is defined as exp(-i*phi), but this minus sign is
  missing for the visualization, so we have to add it again
* we want the phase to go from 0 to 2*pi, instead of the output from atan2
  that goes from -pi to pi 

.. include:: frequency_response.py
   :literal:

.. plot:: source/tutorial/frequency_response.py
