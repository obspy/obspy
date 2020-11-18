# -*- coding: utf-8 -*-
"""
obspy.io.zmap - ZMAP read and write support for ObsPy
=====================================================

This module provides read and write support for the ZMAP format.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)


Usage Example
-------------

The ZMAP reader and writer hooks into the standard ObsPy event handling
mechanisms including format autodetection.

>>> from obspy.core.event import read_events
>>> cat = read_events('/path/to/zmap_events.txt')
>>> print(cat)
2 Event(s) in Catalog:
2012-04-04T14:21:42.300000Z | +41.818,  +79.689 | 4.4  None
2012-04-04T14:21:42.300000Z | +41.822,  +79.684 | 5.1  None
>>> cat.write('example.txt', format='ZMAP')  # doctest: +SKIP


Reading ZMAP
------------

Reading ZMAP is lenient, i.e. ``obspy.io.zmap`` will try to import a file
even if it doesn't strictly conform to 10 or 13 column ZMAP. Namely the
following deviations from standard ZMAP format are acceptable:

* Less or more than 10 or 13 columns. Extra columns are ignored. Missing
  values are set to ``None``.
* Integer years without a fractional part. If the fractional part is
  present, the date/time is computed from the year column. All other
  date/time fields are ignored.
  If the year column is an integer number, date and time are computed from all
  date/time related fields.

Note that ZMAP format autodetection in :func:`~obspy.core.event.read_events`
only works with strictly 10 or 13 column files. To read non-standard ZMAP
files, the ``format='ZMAP'`` keyword argument must be provided.

When reading ZMAP, the following mappings are used for uncertainties

* *Column 11 (Horizontal error):* stored in
  ``origin.origin_uncertainty.horizontal_uncertainty`` with
  ``preferred_description`` set accordingly.
* *Column 12 (Depth error):* stored in ``origin.depth_errors.uncertainty``
* *Column 13 (Magnitude error):* stored in
  ``magnitude.mag_errors.uncertainty``


Writing ZMAP
------------

When writing to ZMAP, the preferred origin and magnitude are used to fill the
origin and magnitude columns. Any missing values are exported as ``'NaN'``.

.. rubric:: Extended ZMAP

Writing extended ZMAP, is supported by using the keyword argument
``with_uncertainties``.

If :class:`~obspy.core.event.OriginUncertainty` specifies a
*horizontal uncertainty* the value for column 11 is extracted from there.
*Uncertainty ellipse* and *confidence ellipsoid* are not currently supported.
If no horizontal uncertainty is given, :class:`~obspy.core.event.Origin`'s
``latitude_errors`` and ``longitude_errors`` are used instead. Depth and
magnitude errors are always read from the respective ``_errors`` attribute in
:class:`~obspy.core.event.Origin`.


The ZMAP Format
---------------

ZMAP is a simple 10 column CSV file (technically TSV) format for basic catalog
data. It originates from ZMAP, a Matlab® based earthquake statistics package
(see [Wiemer2001]_). Since ZMAP files are purely numerical they are easily
imported into Matlab® using the ``dlmread`` function.

=================   ==============================================
Column #            Value
=================   ==============================================
 1                  Longitude [deg]
 2                  Latitude [deg]
 3                  Decimal year (e.g., 2005.5 for July 2nd, 2005)
 4                  Month
 5                  Day
 6                  Magnitude
 7                  Depth [km]
 8                  Hour
 9                  Minute
10                  Second
=================   ==============================================

CSEP (http://www.cseptesting.org) defines an extension to the format to
include uncertainties. The extended CSEP format has the following extra
columns:

=================   ==============================================
Column #            Value
=================   ==============================================
11                  Horizontal error
12                  Depth error
13                  Magnitude error
=================   ==============================================
"""
if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
