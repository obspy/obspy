=======================
Konno-Ohmachi Smoothing
=======================

Obspy includes the
:func:`~obspy.signal.konnoohmachismoothing.konno_ohmachi_smoothing`
function which follows the approach described by [Konno1998]_. Unlike most
smoothing algorithms, this method applies a smoothing window with constant
width in logarithmic space. Such an approach can be useful, for example, to
reduce noise before fitting a spectral model.

The following example will apply Konno-Ohmachi smoothing and a Savitzky-Golay
filter (included in scipy) and compare the results on a log-log plot. Note how
the Savitzky-Golay filter appears over-smoothed at lower frequencies and
under-smoothed at higher frequencies but the Konno-Ohmachi smoothed data
appears reasonable throughout.

.. plot:: tutorial/code_snippets/konno_ohmachi_smoothing_1.py
   :include-source:

-------------------
Efficient Smoothing
-------------------

For large spectra it can be very time-consuming and memory intensive to
calculate the smoothing coefficients for all frequencies. In these cases
it may be best to specify a subset of frequencies using the
``center_frequencies`` parameter. The output will then only contain values
corresponding to the frequencies of interest.

Multiple spectra can also be smoothed simultaneously by passing a numpy array
to the smoothing function, which is more efficient than smoothing each
spectrum individually.

The following example will generate a few large(ish) random time-series,
calculate the amplitude spectra, then apply Konno-Ohmachi smoothing to all
spectra simultaneously for selected frequencies.

.. plot:: tutorial/code_snippets//konno_ohmachi_smoothing_2.py
   :include-source:
