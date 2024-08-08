=============
Cartopy Plots
=============

Cartopy Plot with Custom Projection Setup
=========================================

Simple Cartopy plots of e.g. :class:`~obspy.core.inventory.inventory.Inventory`
or :class:`~obspy.core.event.catalog.Catalog` objects can be performed with
builtin methods, see e.g.
:meth:`Inventory.plot() <obspy.core.inventory.inventory.Inventory.plot>` or
:meth:`Catalog.plot() <obspy.core.event.catalog.Catalog.plot>`.

For full control over the projection and map extent, a custom map can be
set up (e.g. following the examples in the
`cartopy documentation <https://scitools.org.uk/cartopy/docs/latest/gallery/index.html>`_),
and then be reused for plots of
e.g. :class:`~obspy.core.inventory.inventory.Inventory` or
:class:`~obspy.core.event.catalog.Catalog` objects:

.. plot:: tutorial/code_snippets/cartopy_plot_custom.py
   :include-source:

Cartopy Plot of a Local Area with Beachballs
============================================

The following example shows how to plot beachballs into a cartopy plot together
with some stations. The example requires the cartopy_ package (pypi_)
to be installed. The SRTM file used can be downloaded here_. The first lines of
our SRTM data file (from CGIAR_) look like this::

    ncols         400
    nrows         200
    xllcorner     12°40'E
    yllcorner     47°40'N
    xurcorner     13°00'E
    yurcorner     47°50'N
    cellsize      0.00083333333333333
    NODATA_value  -9999
    682 681 685 690 691 689 678 670 675 680 681 679 675 671 674 680 679 679 675 671 668 664 659 660 656 655 662 666 660 659 659 658 ....

.. plot:: tutorial/code_snippets/cartopy_plot_with_beachballs.py
   :include-source:

**Some notes:**

* The Python package GDAL_ allows you to directly read a GeoTiff into NumPy_
  :class:`~numpy.ndarray`

      >>> geo = gdal.Open("file.geotiff")  # doctest: +SKIP
      >>> x = geo.ReadAsArray()  # doctest: +SKIP

* GeoTiff elevation data is available e.g. from ASTER_
* Shading/Illumination can be added.

.. _cartopy: https://scitools.org.uk/cartopy/docs/latest/
.. _pypi: https://pypi.org/project/Cartopy/
.. _here: https://examples.obspy.org/srtm_1240-1300E_4740-4750N.asc.gz
.. _CGIAR: https://srtm.csi.cgiar.org/
.. _NumPy: https://www.numpy.org/
.. _GDAL: https://trac.osgeo.org/gdal/wiki/GdalOgrInPython
.. _ASTER: https://gdem.ersdac.jspacesystems.or.jp/search.jsp


Cartopy Plot of the Globe with Beachballs
=========================================

.. plot:: tutorial/code_snippets/cartopy_plot_with_beachballs2.py
   :include-source:

Cartopy Plot of the Globe with Beachball using read_events
==========================================================

.. plot:: tutorial/code_snippets/cartopy_with_beachball_read_events.py
   :include-source:
