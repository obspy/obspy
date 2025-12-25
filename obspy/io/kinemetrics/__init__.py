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
    >>> st = read("/path/to/BI008_MEMA-04823.evt", apply_calib=False)
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
    acqdelay 0
    appblkversion 303
    apw 0
    aqoffset 0
    batteryvoltage -134
    bootblkversion 112
    buffersize 64
    chan_altitude 0
    chan_azimuth 0
    chan_calcoil 0.0500...
    chan_channel 0
    chan_damping 0.7070...
    chan_east 0
    chan_fullscale 2.5
    chan_gain 1
    chan_id
    chan_iirtrigfilter iira IIR bandpass, 1.2 to 20Hz @ 200sps
    chan_ltaseconds 20
    chan_natfreq 196.0
    chan_north 0
    chan_range 1g
    chan_sensitivity 2.5
    chan_sensorgain 1
    chan_sensorserialnumber 0
    chan_sensorserialnumberext 0
    chan_sensortype 32
    chan_staltaprecent 10
    chan_staltaratio 1.5
    chan_stasecondstten 0.1
    chan_triggertype threshold, default
    chan_up 0
    clocksource Internal GPS
    comment MEMBACH PARAMETERS FAC+EEP/v3.02
    crc 0
    daccount 2124
    datafmt 1
    dspblkversion 102
    duration 230
    elevation 298
    errors 0
    eventnumber 0
    filterflag 0
    filtertype 0
    flags 0
    gpsaltitude 294
    gpslastdrift1 1
    gpslastdrift2 0
    gpslastlock1 2013-08-15T09:19:20.000Z
    gpslastlock2 2013-08-15T09:16:50.000Z
    gpslastturnontime1 2013-07-19T14:22:31.000Z
    gpslastturnontime2 1980-01-01T00:00:00.000Z
    gpslastupdatetime1 2013-08-03T14:37:25.000Z
    gpslastupdatetime2 1980-01-01T00:00:00.000Z
    gpslatitude 5060
    gpslockfailcount 0
    gpslongitude 600
    gpsmaxturnontime 30
    gpssoh 0
    gpsstatus Present ON
    gpsturnoninterval 0
    gpsupdatertccount 1
    headerbytes 2040
    headerversion 130
    id KMI
    installedchan 4
    instrument New Etna
    latitude 50.609...
    localoffset 0
    longitude 6.009...
    maxchannels 12
    maxpeak -19494
    maxpeakoffset 1722
    mean -20944
    minpeak -22142
    minpeakoffset 1714
    minruntime 0
    modem_autoansweroffcmd ATS0=0
    modem_autoansweroncmd ATS0=1
    modem_calloutmsg MEMBACH CALLS SEISMOLOGY
    modem_cellontime 0
    modem_cellshare 0
    modem_cellstarttime1 -1
    modem_cellstarttime2 -1
    modem_cellstarttime3 -1
    modem_cellstarttime4 -1
    modem_cellstarttime5 -1
    modem_cellwarmuptime 0
    modem_dialingprefix ATDT
    modem_dialingsuffix
    modem_flags 63
    modem_hangupcmd ATH0
    modem_initcmd AT&FE0&C1S0=1S25=10&W
    modem_maxdialattempts 15
    modem_pausebetweencalls 200
    modem_phonenumber1 023757...
    modem_phonenumber2 027903...
    modem_phonenumber3
    modem_phonenumber4
    modem_waitforconnection 45
    nchannels 3
    nscans 5750
    postevent 15
    preevent 6
    primarystorage 0
    restartsource 0
    samplebytes 3
    samplerate 0
    secondarystorage 0
    serialnumber 4823
    sps 250
    starttime 2013-08-15T09:20:28.000Z
    starttimemsec 0
    stnid MEMA
    stream_flags functional test, FT
    sysblkversion 0
    temperature 76
    timeout 0
    triggerbitmap 6
    triggertime 2013-08-15T09:20:34.600Z
    triggertimemsec 600
    txblksize 0
    txchanmap 15
    voter_number 0
    voter_type channel
    voter_weight 1
    votestodetrigger 1
    votestotrigger 1


The actual data is stored as :class:`numpy.ndarray` in the ``data`` attribute
of each trace.

    >>> type(st[0].data)  # doctest: +ELLIPSIS
    <... 'numpy.ndarray'>
    >>> print(st[0].data)  # doctest: +ELLIPSIS
    [-20920. -20980. -20922. ... -20972. -20956. -20886.]

Writing
-------
Not implemented


"""
if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
