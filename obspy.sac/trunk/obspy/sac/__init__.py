# -*- coding: utf-8 -*-
"""
obspy.sac - SAC read and write support
======================================
This module provides read and write support for ascii and binary SAC-files as
defined by IRIS (http://www.iris.edu/manuals/sac/manual.html). It depends on
numpy and obspy.core. 

:copyright: The ObsPy Development Team (devs@obspy.org) & C. J. Annon
:license: GNU Lesser General Public License, Version 3 (LGPLv3)

Reading using obspy.core
------------------------
Similiar to reading any other waveform data format using obspy.core:

>>> from obspy.core import read
>>> st = read('test.sac')
>>> st
<obspy.core.stream.Stream object at 0x101700150>
>>> print st
1 Trace(s) in Stream:
.STA     ..Q        | 1978-07-18T08:00:00.000000Z - 1978-07-18T08:01:40.000000Z | 1.0 Hz, 100 samples

The format will be determined automatically. As SAC-files can contain only one
data trace (as opposed to Mini-SEED or GSE2), the length of 'st' will be one.
'st[0]' will have a stats attribute containing the issential meta data (station
name, channel, location, start time, end time, sampling rate, number of
points). Additionally, when reading a SAC-file it will have one additional
attribute, 'sac', which contains all SAC-specific attributes (SAC header
values).

>>> print st[0].stats
Stats({'network': '', 'sac': Stats({'dist': -12345.0, 'isynth': -12345, 'depmin': -1.0, 'iftype': 1, 'kuser0': '-12345  ', 'kuser1': '-12345  ', 'k
user2': '-12345  ', 'gcarc': -12345.0, 'iinst': -12345, 'kevnm': 'FUNCGEN: SINE   ', 'iqual': -12345, 'cmpinc': -12345.0, 'imagsrc': -12345, 'norid': -12345
, 'lpspol': 0, 'leven': 1, 't8': -12345.0, 't9': -12345.0, 't6': -12345.0, 't7': -12345.0, 't4': -12345.0, 't5': -12345.0, 't2': -12345.0, 't3': -12345.0, '
t0': -12345.0, 't1': -12345.0, 'istreg': -12345, 'baz': -12345.0, 'evla': -12345.0, 'nzhour': 8, 'idep': -12345, 'stdp': -12345.0, 'evlo': -12345.0, 'scale'
: -12345.0, 'nwfid': -12345, 'ievreg': -12345, 'nzsec': 0, 'ievtype': -12345, 'stel': -12345.0, 'depmax': 1.0, 'lovrok': 1, 'nzmsec': 0, 'kinst': '-12345  '
, 'o': -12345.0, 'cmpaz': -12345.0, 'knetwk': '-12345  ', 'khole': '-12345  ', 'lcalda': 1, 'user3': -12345.0, 'kt8': '-12345  ', 'kt9': '-12345  ', 'nvhdr'
: 6, 'kt4': '-12345  ', 'kt5': '-12345  ', 'kt6': '-12345  ', 'kt7': '-12345  ', 'kt0': '-12345  ', 'kt1': '-12345  ', 'kt2': '-12345  ', 'kt3': '-12345  ',
 'imagtyp': -12345, 'nzjday': 199, 'user9': -12345.0, 'odelta': -12345.0, 'b': 10.0, 'stla': -12345.0, 'f': -12345.0, 'stlo': -12345.0, 'nzmin': 0, 'evdp':
-12345.0, 'user6': -12345.0, 'user7': -12345.0, 'user4': -12345.0, 'user5': -12345.0, 'user2': -12345.0, 'nzyear': 1978, 'user0': -12345.0, 'user1': -12345.
0, 'user8': -12345.0, 'iztype': -12345, 'az': -12345.0, 'nevid': -12345, 'depmen': 8.7539462e-08, 'mag': -12345.0, 'kdatrd': '-12345  ', 'a': -12345.0, 'ka'
: '-12345  ', 'e': 109.0, 'kf': '-12345  ', 'ko': '-12345  '}), 'station': 'STA     ', 'location': '', 'starttime': UTCDateTime(1978, 7, 18, 8, 0), 'npts':
100, 'sampling_rate': 1.0, 'endtime': UTCDateTime(1978, 7, 18, 8, 1, 40), 'channel': 'Q       '})

The data is stored in the data attribut.

>>> st[0].data
array([-8.74227766e-08,  -3.09016973e-01, ..., 5.87777138e-01,3.09007347e-01], dtype=int32)

Writing using obspy.core
------------------------
Writing is also straight forward. All changes on the data as well as in
stats and stats['sac'] are written with the following command to a file:

>>> st.write('tmp.sac', format='SAC')




Additonal methods of obspy.sac
------------------------------
More SAC-specific functionality is available if you import the ReadSac
class from the obspy.sac module. All of the following methods can only be
accessed as an instance of the ReadSac class.
"""

from obspy.core.util import _getVersionString
from sacio import *

__version__ = _getVersionString("obspy.sac")
