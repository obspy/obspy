# -*- coding: utf-8 -*-
"""
obspy.io.cybershake - CyberShake read support for ObsPy
=======================================================
This module provides read support for the `CyberShake
<https://strike.scec.org/scecpedia/Accessing_CyberShake_Seismograms#Seismogram_Format>`_
waveform data format.

:copyright:
    The ObsPy Development Team (devs@obspy.org) & John Rekoske
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)

Reading
-------

Reading CyberShake waveform data is handled by using
ObsPy's standard :func:`~obspy.core.stream.read` function. The format is
detected automatically.

>>> from obspy import read
>>> st = read("/path/to/test.grm")
>>> st #doctest: +ELLIPSIS
<obspy.core.stream.Stream object at 0x...>
>>> print(st)  #doctest: +ELLIPSIS
2 Trace(s) in Stream:
CS.USC.00.MXE | 1970-01-01T00:00:00.000000Z - ... | 20.0 Hz, 8000 samples
CS.USC.00.MXN | 1970-01-01T00:00:00.000000Z - ... | 20.0 Hz, 8000 samples
>>> print(st[0].stats) # doctest: +ELLIPSIS
         network: CS
         station: USC
        location: 00
         channel: MXE
       starttime: 1970-01-01T00:00:00.000000Z
         endtime: 1970-01-01T00:06:39.950006Z
   sampling_rate: 19.99999970197678
           delta: 0.05000000074505806
            npts: 8000
           calib: 1.0
         _format: CYBERSHAKE
      cybershake: AttribDict({'source_id': 12, ...})

CyberShake specific metadata is stored in ``stats.cybershake``.

>>> for k, v in sorted(st[0].stats.cybershake.items()):
...     print("'%s': %s" % (k, str(v))) # doctest: +NORMALIZE_WHITESPACE
    'det_max_freq': 1.0
    'rup_var_id': 144
    'rupture_id': 0
    'source_id': 12
    'stoch_max_freq': -1.0
"""

if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
