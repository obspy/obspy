# -*- coding: utf-8 -*-
"""
obspy.io.reftek - REFTEK130 read support for ObsPy
==================================================

This module provides read support for the RefTek 130 data format.

Currently the low level read routines are designed to operate on waveform files
written by RefTek 130 digitizers which are composed of event header/trailer and
data packages. These packages do not store information on network or location
code during acquisition. Furthermore, it is unclear how consistently the level
of detail on the recorded channel codes is set in the headers (real world test
data at hand recorded with a Reftek 130 do contain the information on the first
two channel code characters, the band and instrument code but lack information
on the component codes, i.e. ZNE, which with high likelihood were set in the
acquisition parameters). Therefore, additional information on network and
location codes should be supplied and ideally component codes should be
supplied as well when reading files with :func:`~obspy.core.stream.read` (or
should be filled in manually after reading). See the low-level routine
:func:`obspy.io.reftek.core._read_reftek130` for additional arguments that can
be supplied to :func:`~obspy.core.stream.read`.
Currently, only event header/trailer (EH/ET) and data packets (DT) are
implemented and any other packets will be ignored (a warning is shown if any
other packets are encountered during reading). So far, only data encodings
"C0", "C2", "16" and "32" are implemented (due to the lack of test data in
other encodings).

Reading
-------

Reading Reftek130 data is handled by using ObsPy's standard
:func:`~obspy.core.stream.read` function. The format is detected
automatically and optionally can be explicitly set if known
beforehand to skip format detection.

>>> from obspy import read
>>> st = read("/path/to/225051000_00008656")  # doctest: +SKIP
>>> st  # doctest: +SKIP
<obspy.core.stream.Stream object at 0x...>
>>> print(st)  # doctest: +SKIP
8 Trace(s) in Stream:
.KW1..EH0 | 2015-10-09T22:50:51.000000Z - ... | 200.0 Hz, 3165 samples
.KW1..EH0 | 2015-10-09T22:51:06.215000Z - ... | 200.0 Hz, 892 samples
.KW1..EH0 | 2015-10-09T22:51:11.675000Z - ... | 200.0 Hz, 2743 samples
.KW1..EH1 | 2015-10-09T22:50:51.000000Z - ... | 200.0 Hz, 3107 samples
.KW1..EH1 | 2015-10-09T22:51:05.925000Z - ... | 200.0 Hz, 768 samples
.KW1..EH1 | 2015-10-09T22:51:10.765000Z - ... | 200.0 Hz, 2925 samples
.KW1..EH2 | 2015-10-09T22:50:51.000000Z - ... | 200.0 Hz, 3405 samples
.KW1..EH2 | 2015-10-09T22:51:08.415000Z - ... | 200.0 Hz, 3395 samples

Network, location and component codes can be specified during reading:

>>> st = read("/path/to/225051000_00008656", network="BW", location="",
...           component_codes="ZNE")
>>> st  # doctest: +ELLIPSIS
<obspy.core.stream.Stream object at 0x...>
>>> print(st)  # doctest: +ELLIPSIS
8 Trace(s) in Stream:
BW.KW1..EHE | 2015-10-09T22:50:51.000000Z - ... | 200.0 Hz, 3405 samples
BW.KW1..EHE | 2015-10-09T22:51:08.415000Z - ... | 200.0 Hz, 3395 samples
BW.KW1..EHN | 2015-10-09T22:50:51.000000Z - ... | 200.0 Hz, 3107 samples
BW.KW1..EHN | 2015-10-09T22:51:05.925000Z - ... | 200.0 Hz, 768 samples
BW.KW1..EHN | 2015-10-09T22:51:10.765000Z - ... | 200.0 Hz, 2925 samples
BW.KW1..EHZ | 2015-10-09T22:50:51.000000Z - ... | 200.0 Hz, 3165 samples
BW.KW1..EHZ | 2015-10-09T22:51:06.215000Z - ... | 200.0 Hz, 892 samples
BW.KW1..EHZ | 2015-10-09T22:51:11.675000Z - ... | 200.0 Hz, 2743 samples

Reftek 130 specific metadata (from event header packet) is stored
in ``stats.reftek130``.

>>> print(st[0].stats)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
         network: BW
         station: KW1
        location:
         channel: EHE
       starttime: 2015-10-09T22:50:51.000000Z
         endtime: 2015-10-09T22:51:08.020000Z
   sampling_rate: 200.0
           delta: 0.005
            npts: 3405
           calib: 1.0
         _format: REFTEK130
       reftek130: ...

Details on the individual packets can be retrieved with the low level
:class:`~obspy.io.reftek.core.Reftek130` object:

>>> from obspy.core.util.base import get_example_file
>>> from obspy.io.reftek.core import Reftek130
>>> rt = Reftek130.from_file(get_example_file("225051000_00008656"))
>>> print(rt)  # doctest: +ELLIPSIS
Reftek130 (29 packets, file: ...225051000_00008656)
Packet Sequence  Byte Count  Data Fmt  Sampling Rate      Time
  | Packet Type   |  Event #  | Station | Channel #         |
  |   |  Unit ID  |    | Data Stream #  |   |  # of samples |
  |   |   |  Exper.#   |   |  |  |      |   |    |          |
0000 EH AE4C  0  416  427  0 C0 KW1    200         2015-10-09T22:50:51.000000Z
0001 DT AE4C  0 1024  427  0 C0             0  549 2015-10-09T22:50:51.000000Z
0002 DT AE4C  0 1024  427  0 C0             1  447 2015-10-09T22:50:51.000000Z
0003 DT AE4C  0 1024  427  0 C0             2  805 2015-10-09T22:50:51.000000Z
0004 DT AE4C  0 1024  427  0 C0             0  876 2015-10-09T22:50:53.745000Z
0005 DT AE4C  0 1024  427  0 C0             1  482 2015-10-09T22:50:53.235000Z
0006 DT AE4C  0 1024  427  0 C0             1  618 2015-10-09T22:50:55.645000Z
0007 DT AE4C  0 1024  427  0 C0             2  872 2015-10-09T22:50:55.025000Z
0008 DT AE4C  0 1024  427  0 C0             0  892 2015-10-09T22:50:58.125000Z
0009 DT AE4C  0 1024  427  0 C0             1  770 2015-10-09T22:50:58.735000Z
0010 DT AE4C  0 1024  427  0 C0             2  884 2015-10-09T22:50:59.385000Z
0011 DT AE4C  0 1024  427  0 C0             0  848 2015-10-09T22:51:02.585000Z
0012 DT AE4C  0 1024  427  0 C0             1  790 2015-10-09T22:51:02.585000Z
0013 DT AE4C  0 1024  427  0 C0             2  844 2015-10-09T22:51:03.805000Z
0014 DT AE4C  0 1024  427  0 C0             0  892 2015-10-09T22:51:06.215000Z
0015 DT AE4C  0 1024  427  0 C0             1  768 2015-10-09T22:51:05.925000Z
0016 DT AE4C  0 1024  427  0 C0             2  884 2015-10-09T22:51:08.415000Z
0017 DT AE4C  0 1024  427  0 C0             1  778 2015-10-09T22:51:10.765000Z
0018 DT AE4C  0 1024  427  0 C0             0  892 2015-10-09T22:51:11.675000Z
0019 DT AE4C  0 1024  427  0 C0             2  892 2015-10-09T22:51:12.835000Z
0020 DT AE4C  0 1024  427  0 C0             1  736 2015-10-09T22:51:14.655000Z
0021 DT AE4C  0 1024  427  0 C0             0  892 2015-10-09T22:51:16.135000Z
0022 DT AE4C  0 1024  427  0 C0             2  860 2015-10-09T22:51:17.295000Z
0023 DT AE4C  0 1024  427  0 C0             1  738 2015-10-09T22:51:18.335000Z
0024 DT AE4C  0 1024  427  0 C0             0  892 2015-10-09T22:51:20.595000Z
0025 DT AE4C  0 1024  427  0 C0             1  673 2015-10-09T22:51:22.025000Z
0026 DT AE4C  0 1024  427  0 C0             2  759 2015-10-09T22:51:21.595000Z
0027 DT AE4C  0 1024  427  0 C0             0   67 2015-10-09T22:51:25.055000Z
0028 ET AE4C  0  416  427  0 C0 KW1    200         2015-10-09T22:50:51.000000Z
(detailed packet information with: 'print(Reftek130.__str__(compact=False))')
>>> print(rt.__str__(compact=False)) # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
Reftek130 (29 packets, file: ...225051000_00008656)
EH Packet
    packet_sequence: 0
    experiment_number: 0
    unit_id: AE4C
    byte_count: 416
    time: 2015-10-09T22:50:51.000000Z
    event_number: 427
    data_stream_number: 0
    data_format: C0
    flags: 0
    --------------------
    _reserved_2:
    _reserved_3:
    ...
    detrigger_time: None
    digital_filter_list:
    first_sample_time: 2015-10-09T22:50:51.000000Z
    last_sample_time: None
    position:
    ...
    sampling_rate: 200
    station_channel_number: (None, None, None, None, None, None, None, ...)
    station_comment: STATION COMMENT
    station_name: KW1
    station_name_extension:
    stream_name: EH
    time_quality: ?
    time_source: 1
    total_installed_channels: 3
    trigger_time: 2015-10-09T22:50:51.000000Z
    trigger_time_message: Trigger Time = 2015282225051000
<BLANKLINE>
    trigger_type: CON
DT Packet
    packet_sequence: 1
    experiment_number: 0
    unit_id: AE4C
    byte_count: 1024
    time: 2015-10-09T22:50:51.000000Z
    event_number: 427
    data_stream_number: 0
    channel_number: 0
    number_of_samples: 549
    data_format: C0
    flags: 0
DT Packet
    packet_sequence: 2
    experiment_number: 0
    unit_id: AE4C
    byte_count: 1024
    time: 2015-10-09T22:50:51.000000Z
    event_number: 427
    data_stream_number: 0
    channel_number: 1
    number_of_samples: 447
    data_format: C0
    flags: 0
...


:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
