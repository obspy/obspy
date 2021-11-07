# -*- coding: utf-8 -*-
"""
obspy.io.hyposat - HYPOSAT file formats support for ObsPy
=========================================================

This module provides read/write support for some file formats used by
`HYPOSAT <ftp://ftp.norsar.no/pub/outgoing/johannes/hyposat/>`_.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)


Usage Example
-------------

>>> from obspy import read_events
>>> cat = read_events('/path/to/nlloc.qml')
>>> cat.write('my_hyposat.in', format='HYPOSAT_PHASES')  # doctest: +SKIP

"""
if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
