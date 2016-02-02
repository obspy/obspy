============================
Basemap Plot with Beachballs
============================

Basemap Plot of a Local Area
============================

The following example shows how to plot beachballs into a basemap plot together
with some stations. The example requires the basemap_ package (download_ site)
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

.. plot:: tutorial/code_snippets/basemap_plot_with_beachballs.py
   :include-source:

**Some notes:**

* The Python package GDAL_ allows you to directly read a GeoTiff into NumPy_
  :class:`~numpy.ndarray`

      >>> geo = gdal.Open("file.geotiff")  # doctest: +SKIP
      >>> x = geo.ReadAsArray()  # doctest: +SKIP

* GeoTiff elevation data is available e.g. from ASTER_
* Shading/Illumination can be added. See the basemap example plotmap_shaded.py_
  for more info.

.. _basemap: http://matplotlib.org/basemap/
.. _download: http://sourceforge.net/projects/matplotlib/files/matplotlib-toolkits/
.. _here: https://examples.obspy.org/srtm_1240-1300E_4740-4750N.asc.gz
.. _CGIAR: http://srtm.csi.cgiar.org/
.. _NumPy: http://www.numpy.org/
.. _GDAL: https://trac.osgeo.org/gdal/wiki/GdalOgrInPython
.. _ASTER: http://gdem.ersdac.jspacesystems.or.jp/search.jsp
.. _plotmap_shaded.py: https://github.com/matplotlib/basemap/blob/master/examples/plotmap_shaded.py?raw=true


Basemap Plot of the Globe
=========================

.. plot:: tutorial/code_snippets/basemap_plot_with_beachballs2.py
   :include-source:
