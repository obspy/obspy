# -*- coding: utf-8 -*-
"""
obspy.mseed - MiniSEED read and write support
=============================================
This module provides read and write support for Mini-SEED waveform data and
some other convinient methods to handle Mini-SEED files. Most methods are based
on libmseed, a C library framework by Chad Trabant and interfaced via python
ctypes.

:copyright: The ObsPy Development Team (devs@obspy.org) & Chad Trabant
:license: GNU General Public License (GPLv2)

Reading via obspy.core
----------------------
Similiar to reading any other waveform data format using obspy.core:

>>> from obspy.core import read
>>> st = read('COP.BHE.DK.2009.050') #doctest: +SKIP
>>> st #doctest: +SKIP
<obspy.core.stream.Stream object at 0x101700150>
>>> print st #doctest: +SKIP
1 Trace(s) in Stream:
DK.COP..BHE | 2009-02-19T00:00:00.035100Z - 2009-02-19T23:59:59.985100Z | 20.0 Hz, 1728000 samples

The format will be determined automatically.

Each trace will have a stats attribute containing the usual information. When
reading a Mini-SEED file it will have one additional attribute: 'mseed'. This
attribute contains all Mini-SEED specific attributes which actually is just the
dataquality.

>>> print st[0].stats #doctest: +SKIP
Stats({
    'network': 'DK',
    'mseed': Stats({
        'dataquality': 'D'
    }),
    'station': 'COP',
    'location': '',
    'starttime': UTCDateTime(2009, 2, 19, 0, 0, 0, 35100), 
    'npts': 1728000, 
    'sampling_rate': 20.0,
    'endtime': UTCDateTime(2009, 2, 19, 23, 59, 59, 985100), 
    'channel': 'BHE'
})

The actual data is stored as numpy.ndarray in the data attribute of each trace.

>>> st[0].data #doctest: +SKIP
array([1085, 1167, 1131, ...,  -19,  -46,  -55], dtype=int32)

Writing via obspy.core
----------------------
WARNING: No matter what is written in the Stream[i].stats.mseed['dataquality']
field, the dataquality in the resulting Mini-SEED file will always be 'D'.

Writing is also done in the usual way.

>>> st.write('Mini-SEED-filename.mseed', format='MSEED') #doctest: +SKIP

You can also specify several keyword arguments that change the resulting
Mini-SEED file:

* reclen : Record length in bytes of the resulting Mini-SEED file. The record
  length needs to be expressible as 2 to the power of X where X is in between
  and including 8 and 20. If no reclen is given it will default to 4096 bytes.
* encoding: Encoding of the Mini-SEED file. You can either give the a string or
  the corresponding number. If no encoding is given it will default to STEIM2.
  Available encodings:
  |   o INT16 or 1
  |   o INT32 or 3
  |   o STEIM1 or 10
  |   o STEIM2 or 11 
* byteorder: Byte order of the Mini-SEED file. 0 will result in a little-endian
  file and 1 in a big-endian file. Defaults to big-endian. Do not change this
  if you don't know what you are doing because most other programs can only
  read big-endian Mini-SEED files.
* flush: If it is not zero all of the data will be packed into records,
  otherwise records will only be packed while there are enough data samples to
  completely fill a record. The default value is -1 and thus every data value
  will be packed by default.
* verbose: Controls verbosity of the underlaying libmseed. A value higher than
  0 will give diagnostic output. Defaults to 0. 

So in order to write a STEIM1 encoded Mini-SEED file with a record_length of
512 byte do the following:

>>> st.write('out.mseed', format='MSEED', reclen=512, encoding='STEIM1') \
        #doctest: +SKIP
"""

from obspy.core.util import _getVersionString
from obspy.mseed.libmseed import LibMSEED


__version__ = _getVersionString("obspy.mseed")
