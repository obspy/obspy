# -*- coding: utf-8 -*-
"""
obspy.io.kinemetrics - Evt format support for ObsPy
===================================================

Evt read support for ObsPy.

This module provides read support for the Evt Kinemetrics data format.
It is based on the Kinemetrics description of the format and the provided
C code (Kw2asc.c (see "KW2ASC.SRC" File in /doc section)).

:copyright:
    The ObsPy Development Team (devs@obspy.org), Henri Martin, Thomas Lecocq,
    Kinemetrics(c)
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)

Reading
-------
Similar to reading any other waveform data format using obspy.core:

    >>> from obspy import read
    >>> st = read("/path/to/BI008_MEMA-04823.evt", apply_calib=True)
    >>> st  # doctest: +ELLIPSIS
    <obspy.core.stream.Stream object at 0x...>
    >>> print (st)  # doctest: +NORMALIZE_WHITESPACE
    3 Trace(s) in Stream:
    .MEMA..0 | 2013-08-15T09:20:28.000000Z - 2013-08-15T09:20:50.996000Z |
    250.0 Hz, 5750 samples
    .MEMA..1 | 2013-08-15T09:20:28.000000Z - 2013-08-15T09:20:50.996000Z |
    250.0 Hz, 5750 samples
    .MEMA..2 | 2013-08-15T09:20:28.000000Z - 2013-08-15T09:20:50.996000Z |
    250.0 Hz, 5750 samples

Each trace will have a ``stats`` attribute containing the usual information and
a ``kinemetrics_evt`` dictionary with specific attributes.

.. note::
    All the Header's attributes are not read (can be implemented if necessary
    for someone.)

.. code-block:: python

    >>> stats_evt = st[0].stats.pop('kinemetrics_evt')
    >>> print(st[0].stats)  # doctest: +NORMALIZE_WHITESPACE
             network:
             station: MEMA
            location:
             channel: 0
           starttime: 2013-08-15T09:20:28.000000Z
             endtime: 2013-08-15T09:20:50.996000Z
       sampling_rate: 250.0
               delta: 0.004
                npts: 5750
               calib: 1.1694431304931641e-06
             _format: KINEMETRICS_EVT
    >>> for k, v in sorted(stats_evt.items()):
    ...     print(k, v)
    ...     # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
    a2dbits 24
    batteryvoltage -134
    chan_azimuth 0
    chan_calcoil 0.05000000074...
    chan_damping 0.70700001716...
    chan_east 0
    chan_fullscale 2.5
    chan_gain 1
    chan_id
    chan_natfreq 196.0
    chan_north 0
    chan_range 3
    chan_sensitivity 2.5
    chan_sensorgain 1
    chan_up 0
    comment MEMBACH
    duration 230
    elevation 298
    gpslastlock 2013-08-15T09:19:20.000000Z
    gpsstatus Present ON
    installedchan 4
    instrument New Etna
    latitude 50.609794616...
    longitude 6.0092501640...
    maxchannels 12
    nchannels 3
    nscans 6
    samplebytes 3
    serialnumber 4823
    starttime 2013-08-15T09:20:28.000000Z
    stnid MEMA
    temperature 76
    triggertime 2013-08-15T09:20:34.600000Z

The actual data is stored as :class:`numpy.ndarray` in the ``data`` attribute
of each trace.

    >>> type(st[0].data)  # doctest: +ELLIPSIS
    <... 'numpy.ndarray'>
    >>> print(st[0].data)
    [-0.02446475 -0.02453492 -0.02446709 ..., -0.02452556 -0.02450685
     -0.02442499]

Writing
-------
Not implemented


"""
if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
