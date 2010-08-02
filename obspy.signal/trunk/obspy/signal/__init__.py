# -*- coding: utf-8 -*-
"""
obspy.signal - Signal Processing Routines for Seismology
========================================================
Capabilities include filtering, triggering, rotation, instrument
correction and coordinate transformations.

:copyright: The ObsPy Development Team (devs@obspy.org)
:license: GNU Lesser General Public License, Version 3 (LGPLv3)

Filter
------
Available filters are :func:`~obspy.signal.filter.bandpass()`,
:func:`~obspy.signal.filter.lowpass()`,
:func:`~obspy.signal.filter.highpass()`,
:func:`~obspy.signal.filter.bandstop()`. Zero-phase filtering can be done by
specifying ``zerophase=True``.
The following example shows how to highpass a seismogram at 1.0Hz.
In the example only the first trace is processed to see the changes in
comparison with the other traces in the plot.

:Note: The filter takes the data explicitly as argument (i.e. a
       numpy.ndarray) and therefore the sampling_rate needs to be also
       specified. It returns the filtered data.
       For :class:`~obspy.core.stream.Stream` and
       :class:`~obspy.core.trace.Trace` objects simply use their respective
       filtering methods :meth:`~obspy.core.stream.Stream.filter()` and
       :meth:`~obspy.core.trace.Trace.filter`.

>>> from obspy.core import read
>>> import obspy.signal
>>> st = read()
>>> tr = st[0]
>>> tr.data = obspy.signal.highpass(tr.data, 1.0,
...         df=tr.stats.sampling_rate, corners=1, zerophase=True)
>>> st.plot() #doctest: +SKIP

Working with stream's implemented method works similar:

>>> from obspy.core import read
>>> st = read()
>>> tr = st[0]
>>> tr.filter('highpass', {'freq': 1.0, 'corners': 1, 'zerophase': True})
>>> st.plot() #doctest: +SKIP

.. plot::

    from obspy.core import read
    import obspy.signal
    st = read()
    tr = st[0]
    tr.data = obspy.signal.highpass(tr.data, 1.0,
            df=tr.stats.sampling_rate, corners=1, zerophase=True)
    st.plot()

Instrument Correction
---------------------
The response of the instrument can be removed by the invsim module. The
following example shows how to remove the the instrument response of a
STS2 and simulate an instrument with 2Hz corner frequency.
In the example only the first trace is processed to see the changes in
comparison with the other traces in the plot.

>>> from obspy.core import read
>>> from obspy.signal import seisSim, cornFreq2Paz
>>> inst2hz = cornFreq2Paz(2.0)
>>> st = read()
>>> tr = st[0]
>>> tr.data = tr.data - tr.data.mean()
>>> sts2 = {'gain': 60077000.0,
...         'poles': [(-0.037004000000000002+0.037016j),
...                   (-0.037004000000000002-0.037016j),
...                   (-251.33000000000001+0j),
...                   (-131.03999999999999-467.29000000000002j),
...                   (-131.03999999999999+467.29000000000002j)],
...         'sensitivity': 2516778400.0,
...         'zeros': [0j, 0j]}
>>> df = tr.stats.sampling_rate
>>> tr.data = seisSim(tr.data, df, paz_remove=sts2, paz_simulate=inst2hz, water_level=60.0)
>>> st.plot() #doctest: +SKIP

.. plot::

    from obspy.core import read
    from obspy.signal import seisSim, cornFreq2Paz
    inst2hz = cornFreq2Paz(2.0)
    st = read()
    tr = st[0]
    tr.data = tr.data - tr.data.mean()
    sts2 = {'gain': 60077000.0,
            'poles': [(-0.037004000000000002+0.037016j),
                      (-0.037004000000000002-0.037016j),
                      (-251.33000000000001+0j),
                      (-131.03999999999999-467.29000000000002j),
                      (-131.03999999999999+467.29000000000002j)],
            'sensitivity': 2516778400.0,
            'zeros': [0j, 0j]}
    df = tr.stats.sampling_rate
    tr.data = seisSim(tr.data, df, paz_remove=sts2, paz_simulate=inst2hz, water_level=60.0)
    st.plot()

**There are many more functions available (rotation, pazToFreqResp, triggers,
cpxtrace analysis, ...), please also check the tutorial.**
"""

from obspy.core.util import _getVersionString
from filter import bandpass, bandstop, lowpass, highpass, remezFIR, lowpassFIR
from filter import envelope
# import the depreciated ones
from filter import bandpassZPHSH, bandstopZPHSH, lowpassZPHSH, highpassZPHSH
from rotate import rotate_NE_RT, gps2DistAzimuth
from trigger import recStalta, recStaltaPy, carlStaTrig, classicStaLta, \
        delayedStaLta, zdetect, triggerOnset, pkBaer, arPick
from seismometer import PAZ_WOOD_ANDERSON
from invsim import cosTaper, detrend, cornFreq2Paz
from invsim import pazToFreqResp, seisSim, specInv, estimateMagnitude
from cpxtrace import normEnvelope, centroid, instFreq, instBwith
from util import xcorr, xcorr_3C, utlGeoKm, utlLonLat


__version__ = _getVersionString("obspy.signal")

if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
