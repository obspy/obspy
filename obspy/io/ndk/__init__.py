#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
obspy.io.ndk - NDK file support for ObsPy
=========================================

This module provides read support for the NDK files distributed by the
Global Centroid-Moment-Tensor (CMT) catalog.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)

The  `NDK
<https://www.ldeo.columbia.edu/~gcmt/projects/CMT/catalog/
allorder.ndk_explained>`_ format is used by the
`Global Centroid-Moment-Tensor (CMT) Project <http://www.globalcmt.org/>`_
and also in couple of other places to describe centroid moment tensor
solutions for earthquakes. Write support is not provided due to limited
potential use. Furthermore NDK files require some very specific information
about earthquakes which is not available all the timer


Acquiring NDK Files
-------------------

.. note::

    You always can either download the file and then pass the file location to
    ObsPy or you can directly pass the URL to ObsPy which will download it for
    you as demonstrated below.

The GCMT project offers two types of solutions: *Quick* (calculated within
hours of an event) and *Standard* (calculated later and with greater
accuracy). Quick solutions are available `individually
<https://www.ldeo.columbia.edu/~gcmt/projects/CMT/catalog/NEW_QUICK>`_ or
as a `single file
<https://www.ldeo.columbia.edu/~gcmt/projects/CMT/catalog/NEW_QUICK/qcmt.ndk>`_
collecting the most recent solutions. Standard solution are available via the
`CMT Catalog Files <http://www.globalcmt.org/CMTfiles.html>`_ website.


Basic Usage
-----------

.. note::

    During reading all values will be converted from ``dyne-cm`` to ``N m``.
    Furthermore the depth values will be converted to ``m`` and the source time
    function duration will be doubled to adhere to the QuakeML standard.  The
    region name in the NDK file will be replaced with its full Flinn-Engdahl
    region name using ObsPy's built-in region parser.

The :func:`~obspy.core.event.read_events` method is used to read NDK files to
:class:`~obspy.core.event.Catalog` objects.

>>> import obspy
>>> cat = obspy.read_events("/path/to/C200604092050A.ndk")
>>> print(cat)
1 Event(s) in Catalog:
2006-04-09T20:50:51.300000Z | -20.460,  -70.730 | 5.73 Mwc

One of the main purposes of this module is the conversion to QuakeML which
can easily be achieved with ObsPy.

>>> cat.write("C200604092050A.xml", format="quakeml")  # doctest: +SKIP

Instead of passing a filename it is also possible to specify a URL (in this
case to the solutions from February 2011).

>>> cat = obspy.read_events("https://www.ldeo.columbia.edu/~gcmt/projects/CMT/"
...                         "catalog/NEW_MONTHLY/2011/"
...                         "feb11.ndk")  # doctest: +SKIP
>>> print(cat)  # doctest: +SKIP
135 Event(s) in Catalog:
2011-02-01T03:32:08.600000Z | +22.520, +144.790 | 5.02 Mwc
2011-02-01T08:16:31.700000Z | +24.190, +121.730 | 5.33 Mwc
...
2011-02-28T20:45:45.700000Z | -20.520,  -69.440 | 5.27 Mwc
2011-02-28T23:42:21.500000Z | -29.650, -112.280 | 5.54 Mwc
To see all events call 'print(CatalogObject.__str__(print_all=True))'
>>> cat.plot()  # doctest: +SKIP

.. plot::

    import obspy
    cat = obspy.read_events("https://www.ldeo.columbia.edu/~gcmt/projects/CMT/"
                            "catalog/NEW_MONTHLY/2011/feb11.ndk")
    cat.plot()

It can also deal with large amounts of data, though it might take a while.

>>> cat = obspy.read_events("/path/to/jan76_dec10.ndk")  # doctest: +SKIP
>>> print(cat)  # doctest: +SKIP
33872 Event(s) in Catalog:
1976-01-01T01:29:53.400000Z | -29.250, -176.960 | 7.25  Mwc
1976-01-05T02:31:44.700000Z | -13.420,  -75.140 | 5.65  Mwc
...
2010-12-31T04:11:08.400000Z | -18.980, +167.590 | 5.07  Mwc
2010-12-31T16:30:59.700000Z |  +0.810,  -26.060 | 5.45  Mwc
To see all events call 'print(CatalogObject.__str__(print_all=True))'
>>> cat.plot()  # doctest: +SKIP

.. figure:: /_images/expensive_plots/GCMT_Catalog_1976-2010.png

"""
if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
