# -*- coding: utf-8 -*-
"""
obspy.sac - SAC read and write support for ObsPy
================================================
This module provides read and write support for ASCII and binary SAC-files as
defined by IRIS (http://www.iris.edu/manuals/sac/manual.html). It depends on
numpy and obspy.core.

:copyright:
    The ObsPy Development Team (devs@obspy.org) & C. J. Annon
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)

Reading
-------
Similiar to reading any other waveform data format using obspy.core:

>>> from obspy.core import read
>>> st = read('/path/to/test.sac')
>>> st #doctest: +ELLIPSIS
<obspy.core.stream.Stream object at 0x...>
>>> print(st) #doctest: +ELLIPSIS
1 Trace(s) in Stream:
.STA..Q | 1978-07-18T08:00:10.000000Z - ... | 1.0 Hz, 100 samples

The format will be determined automatically. As SAC-files can contain only one
data trace (as opposed to Mini-SEED or GSE2), the length of 'st' will be one.
'st[0]' will have a stats attribute containing the issential meta data (station
name, channel, location, start time, end time, sampling rate, number of
points). Additionally, when reading a SAC-file it will have one additional
attribute, 'sac', which contains all SAC-specific attributes (SAC header
values).

>>> print(st[0].stats)  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
         network:
         station: STA
        location:
         channel: Q
       starttime: 1978-07-18T08:00:10.000000Z
         endtime: 1978-07-18T08:01:49.000000Z
   sampling_rate: 1.0
           delta: 1.0
            npts: 100
           calib: 1.0
             sac: AttribDict({'dist': -12345.0, 'isynth': -12345, 'depmin':
             -1.0, 'iftype': 1, 'kuser0': '-12345  ', 'kuser1': '-12345  ',
             'kuser2': '-12345  ', 'gcarc': -12345.0, 'iinst': -12345,
             'unused10': -12345.0, 'kevnm': 'FUNCGEN: SINE   ', 'iqual':
             -12345, 'cmpinc': -12345.0, 'imagsrc': -12345, 'norid': -12345,
             'unused11': -12345.0, 'ymaximum': -12345.0, 'lpspol': 0,
             'unused12': -12345.0, 'leven': 1, 't8': -12345.0, 't9': -12345.0,
             't6': -12345.0, 't7': -12345.0, 't4': -12345.0, 't5': -12345.0,
             't2': -12345.0, 't3': -12345.0, 't0': -12345.0, 't1': -12345.0,
             'istreg': -12345, 'baz': -12345.0, 'yminimum': -12345.0, 'evla':
             -12345.0, 'xminimum': -12345.0, 'idep': -12345, 'stdp': -12345.0,
             'evlo': -12345.0, 'nwfid': -12345, 'ievreg': -12345, 'ievtype':
             -12345, 'stel': -12345.0, 'depmax': 1.0, 'lovrok': 1, 'imagtyp':
             -12345, 'kinst': '-12345  ', 'xmaximum': -12345.0, 'o': -12345.0,
             'cmpaz': -12345.0, 'lcalda': 1, 'kt8': '-12345  ', 'kt9': '-12345
             ', 'nvhdr': 6, 'kt4': '-12345  ', 'kt5': '-12345  ', 'kt6':
             '-12345  ', 'kt7': '-12345  ', 'kt0': '-12345  ', 'kt1': '-12345
             ', 'kt2': '-12345  ', 'kt3': '-12345  ', 'unused7': -12345.0,
             'unused6': -12345.0, 'unused9': -12345.0, 'unused8': -12345.0,
             'user9': -12345.0, 'odelta': -12345.0, 'b': 10.0, 'stla':
             -12345.0, 'f': -12345.0, 'stlo': -12345.0, 'evdp': -12345.0,
             'user6': -12345.0, 'user7': -12345.0, 'user4': -12345.0, 'user5':
             -12345.0, 'user2': -12345.0, 'user3': -12345.0, 'user0': -12345.0,
             'user1': -12345.0, 'user8': -12345.0, 'iztype': -12345, 'az':
             -12345.0, 'nevid': -12345, 'depmen': 8.3446501e-0...8, 'mag':
             -12345.0, 'kdatrd': '-12345  ', 'a': -12345.0, 'ka': '-12345  ',
             'e': 109.0, 'kf': '-12345  ', 'ko': '-12345  '})
         _format: SAC

The data is stored in the data attribut.

>>> st[0].data #doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
array([ -8.74227766e-08,  -3.09016973e-01,..., 3.09007347e-01], dtype=float32)

Writing
-------
Writing is also straight forward. All changes on the data as well as in
stats and stats['sac'] are written with the following command to a file:

>>> st.write('tmp.sac', format='SAC') #doctest: +SKIP
"""

from obspy.core.util import _getVersionString
from sacio import SacIO, SacError, SacIOError, attach_paz, attach_resp

__version__ = _getVersionString("obspy.sac")


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
