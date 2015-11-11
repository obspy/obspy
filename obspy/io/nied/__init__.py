"""
obspy.io.nied.fnetmt - F-net's moment tensors TEXT format support for ObsPy
===================================================================

This module provides read support for the moment tensor files (TEXT format)
provided for the F-net broadband seismograph network operated by the National
Research Institute for Earth Science and Disaster Prevention in Japan
(NIED; http://www.fnet.bosai.go.jp/).

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)

Example
-------

It works by utilizing ObsPy's :func:`~obspy.core.event.read_events` function.

>>> import obspy
>>> cat = obspy.read_events("/path/to/FNETMTCATALOG")
>>> print(cat)
1 Event(s) in Catalog:
2011-03-11T05:46:18.120000Z | +38.103, +142.861 | 8.7 Mw

The event will contain a couple of origins and magnitudes.

>>> print(cat[0])  # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
Event:  2011-03-11T05:46:18.120000Z | +38.103, +142.861 | 8.7 Mw
<BLANKLINE>
              resource_id: ResourceIdentifier(id="...")
               event_type: 'earthquake'
        ---------
         focal_mechanisms: 1 Elements
                  origins: 2 Elements
               magnitudes: 2 Elements

obspy.io.nied.knet - K-NET and KiK-net ASCII format support for ObsPy
================================================================

This module provides read support for the ASCII format for waveforms from the
K-NET and KiK-net strong-motion seismograph networks operated by the National
Research Institute for Earth Science and Disaster Prevention in Japan
(NIED; http://www.kyoshin.bosai.go.jp/).

KiK-net stations consist of one borehole and one surface sensor which are
distinguished by their channel names:

    ============ ============= ======================
    Channel name  Sensor type   Sensor orientation
    ============ ============= ======================
       NS1        Borehole          N
       EW1        Borehole          E
       UD1        Borehole          Z
       NS2        Surface           N
       EW2        Surface           E
       UD2        Surface           Z
    ============ ============= ======================

K-NET stations only have one surface sensor with the following channel naming
conventions:

    ============ ======================
    Channel name  Sensor orientation
    ============ ======================
       NS            N
       EW            E
       UD            Z
    ============ ======================

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)

Example
-------

Reading K-NET/KiK-net files is handled by using ObsPy's
standard :func:`~obspy.core.stream.read` function. The format is detected
automatically.

>>> from obspy import read
>>> st = read('/path/to/test.knet')
>>> print(st) # doctest: +ELLIPSIS
1 Trace(s) in Stream:
BO.AKT013..EW | 1996-08-10T... - 1996-08-10T... | 100.0 Hz, 5900 samples

Note that K-NET/KiK-net station names are 6 characters long. This will cause
problems if you want to write MiniSEED as it only allows 5 character station
names. In this case you can opt to write the last 2 characters of the station
name into the location field. This is possible because the location is encoded
in the channel name (see table above).

>>> st = read('/path/to/test.knet',convert_stnm=True)
>>> print(st) # doctest: +ELLIPSIS
1 Trace(s) in Stream:
BO.AKT0.13.EW | 1996-08-10T18:12... - 1996-08-10T... | 100.0 Hz, 5900 samples

Additional header entries from the K-NET/KiK-net file are written to a
dictionary called 'knet':

>>> print(st[0].stats) # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
         network: BO
         station: AKT0
        location: 13
         channel: EW
       starttime: 1996-08-10T18:12:24.000000Z
         endtime: 1996-08-10T18:13:22.990000Z
   sampling_rate: 100.0
           delta: 0.01
            npts: 5900
           calib: 2.3841857...e-06
         _format: KNET
            knet: AttribDict(...)

>>> print(st[0].stats.knet.stlo)
140.3213

>>> print(st[0].stats.knet.comment)
A dummy comment

The meaning of the entries in the 'knet' dictionary is as follows:
    ==================== ==================================================
       Name                Description
    ==================== ==================================================
       evot               Event origin time (UTC)
       evla               Event latitude
       evlo               Event longitude
       evdp               Event depth
       mag                Event magnitude
       stla               Station latitude
       stlo               Station longitude
       stel               Station elevation
       accmax             Maximum acceleration (after baseline removal)
       duration           Recording duration time [s]
       comment            Comment
       last correction    Time of last correction (Japanese standard time)
    ==================== ==================================================

"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
