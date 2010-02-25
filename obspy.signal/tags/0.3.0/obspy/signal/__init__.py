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
Available filters are bandpass, lowpass, highpass, bandstop. Zerophase
correspondence are also included (bandpassZPHSH, lowpassZPHSH, highpassZPHSH,
bandstopZPHSH). The following example shows how to lowpass a seismogram
at 1.5Hz.

:Note: The filter takes the data explicitly as argument (i.e. a
       numpy.ndarray) and therefore the sampling_rate needs to be also specified.
       It returns the filtered data.

>>> from obspy.core import read
>>> import obspy.signal
>>> st = read("RJOB_061005_072159.ehz.new")
>>> data = obspy.signal.lowpassZPHSH(st[0].data, 1.5, 
                                     df=st[0].stats.sampling_rate, corners=2)

Instrument Correction
---------------------
The response of the instrument can be removed by the invsim module. The
following example shows how to remove the the instrument response of a
le3d and simulate an instrument with 2Hz corner frequency.

>>> from obspy.core import read
>>> from obspy.signal import seisSim, cornFreq2Paz
>>> inst2hz = cornFreq2Paz(2.0)
>>> st = read("RJOB_061005_072159.ehz.new")
>>> data = st[0].data - st[0].data.mean()
>>> le3d = {'poles' :  [-4.21000 +4.66000j, -4.21000 -4.66000j,
                        -2.105000+0.00000j],
            'zeros' : [0.0 +0.0j, 0.0 +0.0j, 0.0 +0.0j],
            'gain' : 0.4}
>>> npts, df = (st[0].stats.npts, st[0].stats.sampling_rate)
>>> data2 = seisSim(data, df, le3d,
                    inst_sim=inst2hz, water_level=60.0)

**There are many more functions available (rotation, pazToFreqResp, triggers,
cpxtrace analysis, ...), please also check the tutorial.**
"""

from obspy.core.util import _getVersionString
from filter import *
from rotate import *
from trigger import *
from seismometer import *
from invsim import cosTaper, detrend, cornFreq2Paz
from invsim import pazToFreqResp, seisSim, specInv, estimateMagnitude
from cpxtrace import *


__version__ = _getVersionString("obspy.signal")
