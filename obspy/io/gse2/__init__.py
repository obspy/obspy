# -*- coding: utf-8 -*-
"""
obspy.io.gse2 - GSE2/GSE1 read and write support for ObsPy
==========================================================
This module provides read and write support for GSE2 CM6 compressed as well as
GSE1 CM6/INT waveform data and header info. Most methods are based on the C
library `GSE_UTI <ftp://www.orfeus-eu.org/pub/software/conversion/GSE_UTI/>`_
of Stefan Stange, which is interfaced via Python :mod:`ctypes`.

.. seealso:: ftp://www.orfeus-eu.org/pub/software/conversion/GSE_UTI/\
gse2001.pdf.

:copyright:
    The ObsPy Development Team (devs@obspy.org) & Stefan Stange
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)

Reading
-------
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

Writing
-------
You may export the data to the file system using the
:meth:`~obspy.core.stream.Stream.write` method of an existing
:class:`~obspy.core.stream.Stream` object :

>>> st.write('GSE2-filename.gse', format='GSE2') #doctest: +SKIP
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
