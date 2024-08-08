# -*- coding: utf-8 -*-
"""
obspy.io.alsep - Apollo seismic data support for ObsPy
======================================================

This module provides read support for the seismic data obtained by
Apollo Lunar Surface Experiments Package (ALSEP).

.. seealso::

    The format detail is shown in the `UTIG Technical Report No. 118
    <http://www-udc.ig.utexas.edu/external/yosio/PSE/catsrepts/TechRept118.pdf>`_.

    The ALSEP seismic data is downloadable from `DARTS website
    <http://darts.isas.jaxa.jp/planet/seismology/apollo/>`_.

:author:
    Yukio Yamamoto (yukio@planeta.sci.isas.jaxa.jp) Nov. 15, 2018
:copyright:
    The ObsPy Development Team (devs@obspy.org) & Yukio Yamamoto
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)

Reading
-------

There are three types of ALSEP tape: PSE, WTN, and WTH. They are handled by
using ObsPy's standard:func:`~obspy.core.stream.read` function. The format is
detected automatically.

>>> from obspy import read
>>> url = "http://darts.isas.jaxa.jp/pub/apollo/pse/p15s/pse.a15.1.2"
>>> st = read(url)  # doctest: +SKIP
>>> st  # doctest: +SKIP
<obspy.core.stream.Stream object at 0x...>
>>> st_spz = st.select(id='XA.S15..SPZ')  # doctest: +SKIP
>>> print(st_spz)  # doctest: +SKIP
13 Trace(s) in Stream:
XA.S15..SPZ | 1971-08-01T18:52:00.515000Z - ... | 53.0 Hz, 903296 samples
...
XA.S15..SPZ | 1971-08-02T13:46:53.409000Z - ... | 53.0 Hz, 970304 samples
"""


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
