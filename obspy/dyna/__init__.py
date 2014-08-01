# -*- coding: utf-8 -*-
"""
obspy_dyna - DYNA and ITACA read and write support for ObsPy
=======================================================================

The obspy_dyna package contains methods in order to read and write seismogram
files in the DYNA and ITACA format as defined by INGV Milano

:copyright:
    The ITACA Development Team (itaca@mi.ingv.it)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)

Reading
-------
Importing DYNA or ITACA files is done similar to reading any other waveform
data format within ObsPy by using the :func:`~obspy.core.stream.read()` method
of the :mod:`obspy.core` module.

>>> from obspy import read
>>> st = read("/path/to/IT.ARL..HGE.D.20140120.071240.X.ACC.ASC")
>>> st #doctest: +ELLIPSIS
<obspy.core.stream.Stream object at 0x...>
>>> print(st)  # doctest: +ELLIPSIS
1 Trace(s) in Stream:
IT.ARL..HGE | 2014-01-20T07:12:30.000000Z - 2014-01-20T07:13:14.980000Z | 200.0 Hz, 8997 samples

The file format will be determined automatically.
The trace will have a stats attribute containing the usual information.

>>> print(st[0].stats) #doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
         network: IT
         station: ARL
        location:
         channel: HGE
       starttime: 2014-01-20T07:12:30.000000Z
         endtime: 2014-01-20T07:13:14.980000Z
   sampling_rate: 200.0
           delta: 0.005
            npts: 8997
           calib: 1.0
         _format: DYNA
            dyna: AttribDict(...)

The actual data is stored as numpy.ndarray in the data attribute of each trace.

>>> st[0].data #doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
array([-0.006763,  0.001193,  0.007042, ..., -0.037417, -0.030865,
       -0.021271], dtype=float32)

Writing
-------
Writing is also done in the usual way:

>>> st.write('filename.ASC', format = 'DYNA') #doctest: +SKIP

"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
