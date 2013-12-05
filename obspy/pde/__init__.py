"""
obspy.pde - NEIC PDE Bulletin read support for ObsPy
====================================================

This module provides read support for NEIC Preliminary Determination of
Epicenters (PDE) Bulletin.

Currently, only the mchedr (machine readable Earthquake Data Report)
format is supported.

.. seealso:: http://earthquake.usgs.gov/research/data/pde.php

:copyright:
    The ObsPy Development Team (devs@obspy.org), Claudio Satriano
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)

"""


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
