# -*- coding: utf-8 -*-
"""
obspy.signal - Signal Processing Routines for ObsPy
===================================================
Capabilities include filtering, triggering, rotation, instrument
correction and coordinate transformations.

:copyright: The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)

Filter
------
The :mod:`~obspy.signal.filter` module provides various filters, including
different bandpass, lowpass, highpass, bandstop and FIR filter.

.. warning::

    Before filtering you should make sure that data is demeaned/detrended, e.g.
    using :meth:`~obspy.core.stream.Stream.detrend`. Otherwise there can be
    massive artifacts from filtering.

The following example shows how to highpass a seismogram at 1.0Hz.
In the example only the first trace is processed to see the changes in
comparison with the other traces in the plot.

.. note::

    The filter takes the data explicitly as argument (i.e. an
    :class:`numpy.ndarray`) and therefore the ``sampling_rate`` needs to be
    also specified. It returns the filtered data.  For
    :class:`~obspy.core.stream.Stream` and :class:`~obspy.core.trace.Trace`
    objects simply use their respective filtering methods
    :meth:`Stream.filter() <obspy.core.stream.Stream.filter>` and
    :meth:`Trace.filter() <obspy.core.trace.Trace.filter>`.

>>> from obspy import read
>>> import obspy.signal
>>> st = read()
>>> tr = st[0]
>>> tr.data = obspy.signal.highpass(tr.data, 1.0, corners=1, zerophase=True,
...                                 df=tr.stats.sampling_rate)
>>> st.plot()  # doctest: +SKIP

Working with the convenience methods implemented on
:class:`~obspy.core.stream.Stream`/:class:`~obspy.core.trace.Trace`
works similar:

>>> tr.filter('highpass', freq=1.0, corners=1, zerophase=True)
... # doctest: +ELLIPSIS
<...Trace object at 0x...>

.. plot::

    from obspy import read
    import obspy.signal
    st = read()
    tr = st[0]
    tr.data = obspy.signal.highpass(tr.data, 1.0,
            df=tr.stats.sampling_rate, corners=1, zerophase=True)
    st.plot()

Instrument Correction
---------------------
The response of the instrument can be removed by the
:mod:`~obspy.signal.invsim` module. The following example shows how to remove
the the instrument response of a STS2 and simulate an instrument with 2Hz
corner frequency.

>>> from obspy import read
>>> st = read()
>>> st.plot() #doctest: +SKIP

.. plot::

    from obspy import read
    st = read()
    st.plot()

Now we apply the instrument correction and simulation:

>>> from obspy.signal import seisSim, cornFreq2Paz
>>> inst2hz = cornFreq2Paz(2.0)
>>> sts2 = {'gain': 60077000.0,
...         'poles': [(-0.037004+0.037016j),
...                   (-0.037004-0.037016j),
...                   (-251.33+0j),
...                   (-131.04-467.29j),
...                   (-131.04+467.29j)],
...         'sensitivity': 2516778400.0,
...         'zeros': [0j, 0j]}
>>> for tr in st:
...     df = tr.stats.sampling_rate
...     tr.data = seisSim(tr.data, df, paz_remove=sts2, paz_simulate=inst2hz,
...                       water_level=60.0)
>>> st.plot()  # doctest: +SKIP

Again, there are convenience methods implemented on
:class:`~obspy.core.stream.Stream`/:class:`~obspy.core.trace.Trace`:

>>> tr.simulate(paz_remove=sts2, paz_simulate=inst2hz, water_level=60.0)
... # doctest: +ELLIPSIS
<...Trace object at 0x...>

.. plot::

    from obspy import read
    from obspy.signal import seisSim, cornFreq2Paz
    inst2hz = cornFreq2Paz(2.0)
    st = read()
    tr = st[0]
    sts2 = {'gain': 60077000.0,
            'poles': [(-0.037004+0.037016j),
                      (-0.037004-0.037016j),
                      (-251.33+0j),
                      (-131.04-467.29j),
                      (-131.04+467.29j)],
            'sensitivity': 2516778400.0,
            'zeros': [0j, 0j]}
    for tr in st:
        df = tr.stats.sampling_rate
        tr.data = seisSim(tr.data, df, paz_remove=sts2, paz_simulate=inst2hz,
                          water_level=60.0)
    st.plot()

Trigger
-------

The :mod:`~obspy.signal.trigger` module provides various triggering algorithms,
including different STA/LTA routines, Z-Detector, AR picker and the P-picker by
M. Bear. The implementation is based on [Withers1998]_ and [Baer1987]_.

The following example demonstrates a recursive STA/LTA triggering:

>>> from obspy import read
>>> from obspy.signal.trigger import recSTALTA, plotTrigger
>>> st = read()
>>> tr = st.select(component="Z")[0]
>>> tr.filter("bandpass", freqmin=1, freqmax=20)  # doctest: +ELLIPSIS
<...Trace object at 0x...>
>>> sta = 0.5
>>> lta = 4
>>> cft = recSTALTA(tr.data, int(sta * tr.stats.sampling_rate),
...                 int(lta * tr.stats.sampling_rate))
>>> thrOn = 4
>>> thrOff = 0.7
>>> plotTrigger(tr, cft, thrOn, thrOff) #doctest: +SKIP

.. plot::

    from obspy import read
    from obspy.signal.trigger import recSTALTA, plotTrigger
    st = read()
    tr = st.select(component="Z")[0]
    tr.filter("bandpass", freqmin=1, freqmax=20)
    sta = 0.5
    lta = 4
    cft = recSTALTA(tr.data, int(sta * tr.stats.sampling_rate),
                    int(lta * tr.stats.sampling_rate))
    thr_on = 4
    thr_off = 0.7
    plotTrigger(tr, cft, thr_on, thr_off)

There is also a convenience method implemented on
:class:`~obspy.core.stream.Stream`/:class:`~obspy.core.trace.Trace`.
It works on and overwrites the traces waveform data and is intended for batch
processing rather than for interactive determination of triggering parameters.
But it also means that the trace's built-in methods can be used.

>>> tr.trigger("recstalta", sta=0.5, lta=4)  # doctest: +ELLIPSIS
<...Trace object at 0x...>
>>> tr.plot()  # doctest: +SKIP

.. plot::

    from obspy import read
    st = read()
    tr = st.select(component="Z")[0]
    tr.filter("bandpass", freqmin=1, freqmax=20)
    tr.trigger("recstalta", sta=0.5, lta=4)
    tr.plot()

For more examples check out the `trigger`_ in the `Tutorial`_. For
network coincidence refer to :func:`obspy.signal.trigger.coincidenceTrigger`
and the same page in the `Tutorial`_. For automated use see the following
`stalta`_ example scripts.

.. _`trigger`: http://tutorial.obspy.org/code_snippets/trigger_tutorial.html
.. _`Tutorial`: http://tutorial.obspy.org
.. _`stalta`: https://github.com/obspy/branches/tree/master/sandbox/stalta
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

from obspy.signal.cpxtrace import centroid, instBwith, instFreq, normEnvelope
from obspy.signal.cross_correlation import xcorr, xcorr_3C, xcorrPickCorrection
from obspy.signal.filter import (bandpass, bandstop, envelope, highpass,
                                 integerDecimation, lowpass, lowpassFIR,
                                 remezFIR)
from obspy.signal.freqattributes import bwith, cfrequency, domperiod, logcep
from obspy.signal.hoctavbands import sonogram
from obspy.signal.invsim import (cornFreq2Paz, cosTaper, estimateMagnitude,
                                 pazToFreqResp, seisSim, specInv)
from obspy.signal.konnoohmachismoothing import konnoOhmachiSmoothing
from obspy.signal.polarization import eigval
from obspy.signal.rotate import (rotate_LQT_ZNE, rotate_NE_RT, rotate_RT_NE,
                                 rotate_ZNE_LQT)
from obspy.signal.spectral_estimation import PPSD, psd
from obspy.signal.trigger import (arPick, carlSTATrig, classicSTALTA,
                                  classicSTALTAPy, coincidenceTrigger,
                                  delayedSTALTA, pkBaer, recSTALTA,
                                  recSTALTAPy, triggerOnset, zDetect)
from obspy.signal.util import utlGeoKm, utlLonLat

if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
