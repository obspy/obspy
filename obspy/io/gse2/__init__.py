# -*- coding: utf-8 -*-
"""
obspy.io.gse2 - GSE2/GSE1 and GSE2 bulletin support for ObsPy
=============================================================
This module provides read and write support for GSE2 CM6 compressed as well as
GSE1 CM6/INT waveform data and header info. Most methods are based on the C
library `GSE_UTI <ftp://www.orfeus-eu.org/pub/software/conversion/GSE_UTI/>`_
of Stefan Stange, which is interfaced via Python :mod:`ctypes`.

This module also provides read support for GSE2.0 bulletin format.

.. seealso:: ftp://www.orfeus-eu.org/pub/software/conversion/GSE_UTI/\
gse2001.pdf.

:copyright:
    The ObsPy Development Team (devs@obspy.org) & Stefan Stange
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)

Reading waveform data
---------------------
Similar to reading any other waveform data format using :mod:`obspy.core`:

>>> from obspy import read
>>> st = read("/path/to/loc_RJOB20050831023349.z")

You can also specify the following keyword arguments that change the
behavior of reading the file:

* ``headonly=True``: Read only the header part, not the actual data
* ``verify_chksum=False``: Do not verify the checksum of the GSE2 file. This is
  very useful if the program, which wrote the checksum, calculated it in a
  wrong way.

>>> st #doctest: +ELLIPSIS
<obspy.core.stream.Stream object at 0x...>
>>> print(st) #doctest: +NORMALIZE_WHITESPACE
1 Trace(s) in Stream:
.RJOB..Z | 2005-08-31T02:33:49.850000Z - 2005-08-31T02:34:49.845000Z
| 200.0 Hz, 12000 samples

The format will be determined automatically. Each trace (multiple 'WID2'
entries are mapped to multiple traces) will have a ``stats`` attribute
containing the usual information. When reading a GSE2 file it will have one
additional attribute named ``gse2``. This attribute contains all GSE2 specific
attributes:

>>> gse2 = st[0].stats.pop('gse2')
>>> print(st[0].stats) #doctest: +NORMALIZE_WHITESPACE
         network:
         station: RJOB
        location:
         channel: Z
       starttime: 2005-08-31T02:33:49.850000Z
         endtime: 2005-08-31T02:34:49.845000Z
   sampling_rate: 200.0
           delta: 0.005
            npts: 12000
           calib: 0.0949
         _format: GSE2

>>> for k, v in sorted(gse2.items()):
...     print(k, v) #doctest: +NORMALIZE_WHITESPACE
auxid RJOB
calper 1.0
coordsys
datatype CM6
edepth -0.999
elev -0.999
hang -1.0
instype
lat -99.0
lon -999.0
vang -1.0

The actual data is stored as :class:`~numpy.ndarray` in the ``data`` attribute
of each trace.

>>> print(st[0].data)
[ 12 -10  16 ...,   8   0 -40]

Writing waveform data
---------------------
You may export the data to the file system using the
:meth:`~obspy.core.stream.Stream.write` method of an existing
:class:`~obspy.core.stream.Stream` object :

>>> st.write('GSE2-filename.gse', format='GSE2') #doctest: +SKIP

Reading bulletin
----------------
Only GSE2.0 bulletins are currently supported. IMS1.0 (or GSE2.1 before
renaming) bulletins are similar to GSE2.0 but have significant differences.

Read support works via the ObsPy plugin structure:

>>> from obspy import read_events
>>> catalog = read_events('/path/to/bulletin/gse_2.0_standard.txt')
>>> print(catalog)
1 Event(s) in Catalog:
1995-01-16T07:26:52.400000Z | +39.450,  +20.440 | 3.6  mb | manual

For details on further parameters see
:meth:`~obspy.io.gse2.bulletin._read_gse2`

Data example::

    BEGIN GSE2.0
    MSG_TYPE DATA
    MSG_ID example GSE_IDC
    DATA_TYPE BULLETIN GSE2.0
    Reviewed Event Bulletin (REB) of the GSE_IDC for January 16, 1995
    EVENT 280435
       Date       Time       Latitude Longitude    Depth    Ndef Nsta Gap    Mag1  N    Mag2  N    Mag3  N  Author          ID
           rms   OT_Error      Smajor Sminor Az        Err   mdist  Mdist     Err        Err        Err     Quality

    1995/01/16 07:26:52.4     39.4500   20.4400     66.8            8 322  mb 3.6  3  ML 4.0  1             GSE_IDC     282672
          0.53   +- 12.69      93.6   83.7   27    +- 83.8   10.56  78.21                                   m i ke

    GREECE-ALBANIA BORDER REGION
    Sta    Dist   EvAz     Phase       Date      Time     TRes  Azim  AzRes  Slow  SRes Def  SNR        Amp   Per   Mag1   Mag2 Arr ID
    GERES  10.56 150.3     P       1995/01/16 07:29:20.7  -0.2 163.7   13.4  13.8   0.1 T     6.8       0.6   0.3 ML 4.0         3586432
    GERES  10.56 150.3     S       1995/01/16 07:31:17.5  -0.6 153.4    3.1  23.4  -1.0 T     4.9       2.9   0.6                3586513
    NORES  22.02 161.4     P       1995/01/16 07:31:41.2   0.3 155.0   -6.4  11.5   0.7 T     9.9       3.5   0.3                3586453
    FINES  22.29 191.6     P       1995/01/16 07:31:44.1   0.2 182.0   -9.6   8.6  -2.2 T     7.3       4.5   0.8 mb 3.7         3586555
    ARCES  30.27 187.8     P       1995/01/16 07:32:57.8   1.2 191.3    3.5  10.7   1.9 T     7.7       1.2   0.6 mb 3.7         3586456
    MBC    61.77  34.6     P       1995/01/16 07:37:03.8   0.5   5.9  -28.6   4.4  -2.3 T     4.6       0.3   0.4 mb 3.3         3586481
    FCC    68.12  49.4     P       1995/01/16 07:37:45.3   0.4                          T                                        3604094
    YKA    72.17  35.1     P       1995/01/16 07:38:09.5  -0.1                          T                                        3604095
    WHY    78.21  19.3     P       1995/01/16 07:38:44.0  -0.5                          T                                        3604093


    EVENT 280436
       Date       Time       Latitude Longitude    Depth    Ndef Nsta Gap    Mag1  N    Mag2  N    Mag3  N  Author          ID
           rms   OT_Error      Smajor Sminor Az        Err   mdist  Mdist     Err        Err        Err     Quality

    1995/01/16 07:27:07.3     50.7700 -129.7600     36.7       7    7 252  mb 4.0  2                        GSE_IDC     281990
          0.79   +-  9.63     129.3   23.5   37    +- 60.1   10.32  25.90                                   m i ke

    VANCOUVER ISLAND REGION
    Sta    Dist   EvAz     Phase       Date      Time     TRes  Azim  AzRes  Slow  SRes Def  SNR        Amp   Per   Mag1   Mag2 Arr ID
    WHY    10.32 161.5     Pn      1995/01/16 07:29:33.7   0.8 161.4   -0.2  13.7  -0.0 T     5.9      52.9   0.4                3586419
    WALA   10.37 285.5     Pn      1995/01/16 07:29:34.0   0.4 262.7  -22.8   9.5  -4.2 T     7.6      11.2   0.5                3586401
    YKA    14.35 222.0     Pn      1995/01/16 07:30:26.6  -1.3 223.3    1.2  11.3  -2.3 T     6.9       1.4   1.0                3586540
    INK    17.69 172.1     Pn      1995/01/16 07:31:10.7   0.0                          T                                        3604812
    ULM    21.45 284.5     P       1995/01/16 07:31:51.1  -1.0 287.5    3.0  10.8   0.1 T    15.0      15.7   0.8 mb 4.3         3586452
    FCC    21.85 264.2     P       1995/01/16 07:31:56.6   0.3                          T                                        3604813
    MBC    25.90 195.2     P       1995/01/16 07:32:34.5  -0.8 217.6   22.4   4.6  -4.5 T     4.4       1.7   1.0 mb 3.6         3586477


    STOP
"""
if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
