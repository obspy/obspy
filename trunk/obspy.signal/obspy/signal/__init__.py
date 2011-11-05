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

.. warning::

    Before filtering you should make sure that data is demeaned/detrended, e.g.
    using :func:`~obspy.signal.invsim.detrend`. Otherwise there can be massive
    artifacts from filtering.

.. note::
    
    The filter takes the data explicitly as argument (i.e. an
    :class:`~numpy.ndarray`) and therefore the `sampling_rate` needs to be also
    specified. It returns the filtered data.  For
    :class:`~obspy.core.stream.Stream` and :class:`~obspy.core.trace.Trace`
    objects simply use their respective filtering methods
    :meth:`~obspy.core.stream.Stream.filter` and
    :meth:`~obspy.core.trace.Trace.filter`.

>>> from obspy.core import read
>>> import obspy.signal
>>> st = read()
>>> tr = st[0]
>>> tr.data = obspy.signal.highpass(tr.data, 1.0,
...         df=tr.stats.sampling_rate, corners=1, zerophase=True)
>>> st.plot() #doctest: +SKIP

Working with the convenience methods implemented on
:class:`~obspy.core.stream.Stream`/:class:`~obspy.core.trace.Trace`
works similar:

>>> tr.filter('highpass', freq=1.0, corners=1, zerophase=True)

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
>>> sts2 = {'gain': 60077000.0,
...         'poles': [(-0.037004000000000002+0.037016j),
...                   (-0.037004000000000002-0.037016j),
...                   (-251.33000000000001+0j),
...                   (-131.03999999999999-467.29000000000002j),
...                   (-131.03999999999999+467.29000000000002j)],
...         'sensitivity': 2516778400.0,
...         'zeros': [0j, 0j]}
>>> df = tr.stats.sampling_rate
>>> tr.data = seisSim(tr.data, df, paz_remove=sts2, paz_simulate=inst2hz,
...                   water_level=60.0, remove_sensitivity=False,
...                   simulate_sensitivity=False)
>>> st.plot() #doctest: +SKIP

Again, there are convenience methods implemented on
:class:`~obspy.core.stream.Stream`/:class:`~obspy.core.trace.Trace`:

>>> tr.simulate(paz_remove=sts2, paz_simulate=inst2hz, water_level=60.0,
...             remove_sensitivity=False, simulate_sensitivity=False)

.. plot::

    from obspy.core import read
    from obspy.signal import seisSim, cornFreq2Paz
    inst2hz = cornFreq2Paz(2.0)
    st = read()
    tr = st[0]
    sts2 = {'gain': 60077000.0,
            'poles': [(-0.037004000000000002+0.037016j),
                      (-0.037004000000000002-0.037016j),
                      (-251.33000000000001+0j),
                      (-131.03999999999999-467.29000000000002j),
                      (-131.03999999999999+467.29000000000002j)],
            'sensitivity': 2516778400.0,
            'zeros': [0j, 0j]}
    df = tr.stats.sampling_rate
    tr.data = seisSim(tr.data, df, paz_remove=sts2, paz_simulate=inst2hz,
                      water_level=60.0)
    st.plot()

Trigger
-------

The :mod:`~obspy.signal.trigger` module provides various triggering algorithms,
including different Sta/Lta routines, Z-Detector, AR picker and the P-picker by
M. Bear. The implementation is based on these articles:

.. note:: Withers, M., Aster, R., Young, C., Beiriger, J., Harris, M.,
    Moore, S., & Trujillo, J., 1998. A comparison of
    selected trigger algorithms for automated global seismic phase and event
    detection, Bulletin of the Seismological
    Society of America, 88, 95-106.

.. note:: Baer, M. & Kradolfer, U., 1987. An automatic phase picker for
    local and teleseismic events, Bulletin of the
    Seismological Society of America, 77, 1437-1445.


The following example demonstrates a recursive Sta/Lta triggering:

>>> from obspy.core import read
>>> from obspy.signal import recStalta
>>> from obspy.imaging.waveform import plot_trigger
>>>
>>> st = read()
>>> tr = st.select(component="Z")[0]
>>> tr.filter("bandpass", freqmin=1, freqmax=20)
>>> sta = 0.5
>>> lta = 4
>>> cft = recStalta(tr.data, int(sta * tr.stats.sampling_rate),
...                 int(lta * tr.stats.sampling_rate))
>>> thrOn = 4
>>> thrOff = 0.7
>>> plot_trigger(tr, cft, thrOn, thrOff) #doctest: +SKIP

.. plot::

    from obspy.core import read
    from obspy.signal import recStalta
    from obspy.imaging.waveform import plot_trigger
    st = read()
    tr = st.select(component="Z")[0]
    tr.filter("bandpass", freqmin=1, freqmax=20)
    sta = 0.5
    lta = 4
    cft = recStalta(tr.data, int(sta * tr.stats.sampling_rate),
                    int(lta * tr.stats.sampling_rate))
    thrOn = 4
    thrOff = 0.7
    plot_trigger(tr, cft, thrOn, thrOff)

There is also a convenience method implemented on
:class:`~obspy.core.stream.Stream`/:class:`~obspy.core.trace.Trace`.
It works on and overwrites the traces waveform data and is intended for batch
processing rather than for interactive determination of triggering parameters.
But it also means that the trace's built-in methods can be used.

>>> tr.trigger("recstalta", sta=0.5, lta=4)
>>> tr.plot() #doctest: +SKIP

.. plot::

    from obspy.core import read
    from obspy.signal import recStalta
    from obspy.imaging.waveform import plot_trigger
    st = read()
    tr = st.select(component="Z")[0]
    tr.filter("bandpass", freqmin=1, freqmax=20)
    tr.trigger("recstalta", sta=0.5, lta=4)
    tr.plot()

For more examples check out the `triggering page`_ in the `Tutorial`_. For
automated use and network coincidence there are some example scripts in the
`svn repository`_.

.. _`triggering page`: http://www.obspy.org/wiki/TriggerTutorial
.. _`Tutorial`: http://www.obspy.org/wiki/ObspyTutorial
.. _`svn repository`: http://www.obspy.org/browser/branches/sandbox/stalta

**There are many more functions available (rotation, pazToFreqResp,
cpxtrace analysis, ...), please also check the tutorial.**
"""

from obspy.core.util import _getVersionString
from filter import bandpass, bandstop, lowpass, highpass, remezFIR, lowpassFIR
from filter import envelope, integerDecimation
from rotate import rotate_NE_RT, rotate_ZNE_LQT, rotate_LQT_ZNE, \
        gps2DistAzimuth
from trigger import recStalta, recStaltaPy, carlStaTrig, classicStaLta, \
        delayedStaLta, zdetect, triggerOnset, pkBaer, arPick
from seismometer import PAZ_WOOD_ANDERSON
from invsim import cosTaper, detrend, cornFreq2Paz
from invsim import pazToFreqResp, seisSim, specInv, estimateMagnitude
from cpxtrace import normEnvelope, centroid, instFreq, instBwith
from util import utlGeoKm, utlLonLat
from cross_correlation import xcorr, xcorr_3C
from freqattributes import cfrequency, bwith, domperiod, logcep
from hoctavbands import sonogram
from polarization import eigval
from psd import psd, PPSD
from konnoohmachismoothing import konnoOhmachiSmoothing


__version__ = _getVersionString("obspy.signal")


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
