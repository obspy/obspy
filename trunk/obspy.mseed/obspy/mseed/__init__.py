# -*- coding: utf-8 -*-
"""
obspy.mseed - MiniSEED read and write support
=============================================
This module provides read and write support for Mini-SEED waveform data and
some other convenient methods to handle Mini-SEED files. Most methods are based
on libmseed, a C library framework by Chad Trabant and interfaced via python
ctypes.

:copyright: The ObsPy Development Team (devs@obspy.org) & Chad Trabant
:license: GNU General Public License (GPLv2)

Reading
-------
Similar to reading any other waveform data format using obspy.core:

>>> from obspy.core import read
>>> st = read("/path/to/test.mseed")
>>> st #doctest: +ELLIPSIS
<obspy.core.stream.Stream object at 0x...>
>>> print(st)
1 Trace(s) in Stream:
NL.HGN.00.BHZ | 2003-05-29T02:13:22.043400Z - 2003-05-29T02:18:20.693400Z | 40.0 Hz, 11947 samples

The format will be determined automatically.

Each trace will have a stats attribute containing the usual information. When
reading a Mini-SEED file it will have one additional attribute: 'mseed'. This
attribute contains all Mini-SEED specific attributes which actually is just the
dataquality.

>>> print(st[0].stats) #doctest: +NORMALIZE_WHITESPACE
         network: NL
         station: HGN
        location: 00
         channel: BHZ
       starttime: 2003-05-29T02:13:22.043400Z
         endtime: 2003-05-29T02:18:20.693400Z
   sampling_rate: 40.0
           delta: 0.025
            npts: 11947
           calib: 1.0
         _format: MSEED
           mseed: AttribDict({'dataquality': 'R', 'record_length': 4096, 'encoding': 'STEIM2', 'byteorder': '>'})

The actual data is stored as numpy.ndarray in the data attribute of each trace.

>>> print(st[0].data)
[2787 2776 2774 ..., 2850 2853 2853]

Writing
-------
Writing is also done in the usual way.

>>> st.write('Mini-SEED-filename.mseed', format='MSEED') #doctest: +SKIP

You can also specify several keyword arguments that change the resulting
Mini-SEED file. Please refer to :

* reclen : Record length in bytes of the resulting Mini-SEED file. The record
  length needs to be expressible as 2 to the power of X where X is in between
  and including 8 and 20. If no reclen is given it will default to 4096 bytes.
* encoding: Encoding of the Mini-SEED file. You can either give the a string or
  the corresponding number. Available encodings are ASCII (0)*, INT16 (1),
  INT32 (3), FLOAT32 (4)*, FLOAT64 (5)*, STEIM1 (10) and STEIM2 (11)*. Default
  data types a marked with an asterisk. Currently INT24 (2) is not supported
  due to lacking NumPy support.
* byteorder: Byte order of the Mini-SEED file. 0 will result in a little-endian
  file and 1 in a big-endian file. Defaults to big-endian. Do not change this
  if you don't know what you are doing because most other programs can only
  read big-endian Mini-SEED files.
* flush: If it is not zero all of the data will be packed into records,
  otherwise records will only be packed while there are enough data samples to
  completely fill a record. The default value is -1 and thus every data value
  will be packed by default.
* verbose: Controls verbosity of the underlying libmseed. A value higher than
  0 will give diagnostic output. Defaults to 0. 

So in order to write a STEIM1 encoded Mini-SEED file with a record_length of
512 byte do the following:

>>> st.write('out.mseed', format='MSEED', reclen=512, encoding='STEIM1') \
        #doctest: +SKIP
"""

from obspy.core.util import _getVersionString
from obspy.mseed.libmseed import LibMSEED


__version__ = _getVersionString("obspy.mseed")


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
