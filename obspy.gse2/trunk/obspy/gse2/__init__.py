# -*- coding: utf-8 -*-
"""
obspy.gse2 - Read & write seismograms, Format GSE2
==================================================
This module provides read and write support for GSE2 CM6 compressed
waveform data and header info. Most methods are based on the C library
GSE_UTI of Stefan Stange, which is interfaced via Python ctypes.
See: http://www.orfeus-eu.org/Software/softwarelib.html#gse.

Reading
-------
Similiar to reading any other waveform data format using obspy.core:

>>> from obspy.core import read
>>> st = read("loc_RJOB20050831023349.z")

You can also specify the following keyword arguments that change the
behavior of reading the file:

* headonly=True: Read only the header part, not the data part
* verify_chksum=False: Do not verify the checksum of the GSE2 file. This is
  very useful if the program, which wrote the checksum, calculated it in a
  wrong way. 

>>> st
<obspy.core.stream.Stream object at 0xb7df752c>
>>> print st
1 Trace(s) in Stream:
.RJOB ..  Z | 2005-08-31T02:33:49.849998Z - 2005-08-31T02:34:49.844998Z | 200.0 Hz, 12000 samples

The format will be determined automatically. Each trace (multiple 'WID2'
entries are mapped to multiple traces) will have a stats attribute
containing the usual information. When reading a GSE2 file it will have one
additional attribute: 'gse2'. This attribute contains all GSE2 specific
attributes:

>>> print st[0].stats
Stats({
    'network': '', 
    'gse2': Stats({
        'instype': '      ', 
        'datatype': 'CM6', 
        'hang': -1.0, 
        'auxid': 'RJOB', 
        'vang': -1.0, 
        'calper': 1.0, 
        'calib': 0.094899997115135193
    }), 
    'station': 'RJOB ', 
    'location': '', 
    'starttime': UTCDateTime(2005, 8, 31, 2, 33, 49, 849998), 
    'sampling_rate': 200.0, 
    'npts': 12000, 
    'endtime': UTCDateTime(2005, 8, 31, 2, 34, 49, 844998), 
    'channel': '  Z'
})

Writing
-------
Writing is also done in the usual way:

>>> st.write('GSE2-filename.gse', format = 'GSE2')

License
-------
GNU General Public License (GPL)

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA
02110-1301, USA.
"""

