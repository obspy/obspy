"""
obspy.io.pde - NEIC PDE Bulletin read support for ObsPy
=======================================================

This module provides read support for NEIC Preliminary Determination of
Epicenters (PDE) Bulletin.

Currently, only the mchedr (machine readable Earthquake Data Report)
format is supported.

.. seealso:: https://earthquake.usgs.gov/data/comcat/catalog/us/

:copyright:
    The ObsPy Development Team (devs@obspy.org), Claudio Satriano
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)

"""
if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
