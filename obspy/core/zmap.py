# -*- coding: utf-8 -*-
"""
ZMAP write support.

ZMAP is a simple 10 column csv file format for basic catalog data
[Wiemer2001]_. Since ZMAP files are purely numerical they are easily
imported into Matlab® using the ``dlmread`` function.

=================   ==============================================
Column #            Value
=================   ==============================================
 1                  Longitude [deg]
 2                  Latitude [deg]
 3                  Decimal year (e.g., 2005.5 for July 1st, 2005)
 4                  Month
 5                  Day
 6                  Magnitude
 7                  Depth [km]
 8                  Hour
 9                  Minute
10                  Second
=================   ==============================================

When writing to ZMAP, the preferred origin and magnitude are used to fill the
origin and magnitude columns. Any missing values will be exported as 'NaN'

.. rubric:: Extended ZMAP

Extended ZMAP format as used in CSEP (http://www.cseptesting.org) is supported
by using the keyword argument ``withUncertainties``. The resulting non
standard columns will be added at the end as follows:

=================   ==============================================
Column #            Value
=================   ==============================================
11                  Horizontal error
12                  Depth error
13                  Magnitude error
=================   ==============================================

If :class:`~obspy.core.event.OriginUncertainty` specifies a
*horizontal uncertainty* the value for column 11 is extracted from there.
*Uncertainty ellipse* and *confidence ellipsoid* are not currently supported.
If no horizontal uncertainty is given, :class:`~obspy.core.event.Origin`'s
``latitude_errors`` and ``longitude_errors`` are used instead. Depth and
magnitude errors are always taken from the respective ``_errors`` attribute in
:class:`~obspy.core.event.Origin`.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import math
from obspy.core import UTCDateTime


class Pickler(object):

    def __init__(self, with_uncertainties=False):
        self.with_uncertainties = with_uncertainties
        # This is ZMAP column order, don't change it
        zmap_columns = ['lon', 'lat', 'year', 'month', 'day', 'mag', 'depth',
                        'hour', 'minute', 'second']
        if with_uncertainties:
            zmap_columns += ['h_err', 'z_err', 'mag_err']
        self.zmap_columns = zmap_columns

    def dump(self, catalog, filename):
        """
        Writes ObsPy Catalog into given file.

        :type catalog: :class:`~obspy.core.event.Catalog`
        :param catalog: ObsPy Catalog object.
        :type fh: file
        :param fh: File name.
        """
        # Open filehandler or use an existing file like object.
        if not hasattr(filename, "write"):
            file_opened = True
            fh = open(filename, "wb")
        else:
            file_opened = False
            fh = filename
        cat_string = self._serialize(catalog)
        fh.write(cat_string.encode('utf-8'))
        if file_opened:
            fh.close()

    def dumps(self, catalog):
        """
        Returns ZMAP string of given ObsPy Catalog object.

        :type catalog: :class:`~obspy.core.event.Catalog`
        :param catalog: ObsPy Catalog object.
        :rtype: str
        :returns: ZMAP formatted string.
        """
        return self._serialize(catalog)

    @staticmethod
    def _hz_error(origin):
        """
        Compute horizontal error of origin.

        If the origin has an associated origin uncertainty object, we will try
        to extract the horizontal uncertainty from there. Otherwise we compute
        it from the individual lat/lon uncertainties stored in origin.
        """
        ou = origin.origin_uncertainty
        # TODO: implement the other two possible descriptions
        if ou and ou.preferred_description == 'horizontal uncertainty':
            return ou.horizontal_uncertainty
        else:
            km_per_deg = math.pi * 6371.0087714 / 180.0
            lat_err = origin.latitude_errors.uncertainty
            lon_err = origin.longitude_errors.uncertainty
            if lat_err is None or lon_err is None:
                return None

            h_err = math.sqrt(math.pow(lat_err * km_per_deg, 2) +
                              math.pow(lon_err * math.cos(origin.latitude *
                                       math.pi/180.0)
                                       * km_per_deg, 2))
            return h_err

    @staticmethod
    def _depth_error(origin):
        """
        Return the absolute depth error in km
        """
        # TODO: add option to extract depth error from origin uncertainty
        # when ellipsoid is used
        if origin.depth_errors.uncertainty is None:
            return None
        return origin.depth_errors.uncertainty / 1000.0

    @staticmethod
    def _num2str(num, precision=6):
        """
        Convert num into a matlab (and thus zmap) compatible string
        """
        if num is None:
            return 'NaN'
        else:
            return '{n:.{p}f}'.format(p=precision, n=num)

    @staticmethod
    def _decimal_year(time):
        """
        Return (floating point) decimal year representation of UTCDateTime
        input value
        """
        start_of_year = UTCDateTime(time.year, 1, 1).timestamp
        end_of_year = UTCDateTime(time.year + 1, 1, 1).timestamp
        timestamp = time.timestamp
        year_fraction = ((timestamp - start_of_year) /
                         (end_of_year - start_of_year))
        return time.year + year_fraction

    def _serialize(self, catalog):
        zmap = ''
        for ev in catalog:
            strings = dict.fromkeys(self.zmap_columns, 'NaN')
            # origin
            origin = ev.preferred_origin()
            if origin:
                dec_year = self._decimal_year(origin.time)
                strings.update({
                    'depth':   self._num2str(origin.depth/1000.0),  # m to km
                    'z_err':   self._num2str(self._depth_error(origin)),
                    'lat':     self._num2str(origin.latitude),
                    'lon':     self._num2str(origin.longitude),
                    'h_err':   self._num2str(self._hz_error(origin)),
                    'year':    self._num2str(dec_year, 12),
                    'month':   self._num2str(origin.time.month, 0),
                    'day':     self._num2str(origin.time.day, 0),
                    'hour':    self._num2str(origin.time.hour, 0),
                    'minute':  self._num2str(origin.time.minute, 0),
                    'second':  self._num2str(origin.time.second, 0)
                })
            # magnitude
            magnitude = ev.preferred_magnitude()
            if magnitude:
                strings.update({
                    'mag':     self._num2str(magnitude.mag),
                    'mag_err': self._num2str(magnitude.mag_errors.uncertainty)
                })
            # create tab separated row
            zmap += '\t'.join([strings[c] for c in self.zmap_columns]) + '\n'
        return zmap


def writeZmap(catalog, filename, with_uncertainties=False,
              **kwargs):  # @UnusedVariable
    """
    Writes a ZMAP file.

    Extended ZMAP format as used in CSEP (http://www.cseptesting.org) is
    supported by using keyword argument with_uncertainties=True (default:
    False)

    .. warning::
        This function should NOT be called directly, it registers via the
        the :meth:`~obspy.core.event.Catalog.write` method of an
        ObsPy :class:`~obspy.core.event.Catalog` object, call this instead.

    :type catalog: :class:`~obspy.core.stream.Catalog`
    :param catalog: The ObsPy Catalog object to write.
    :type filename: str or file
    :param filename: Filename to write or open file-like object.
    :type with_uncertainties: boolean
    :param with_uncertainties: appends non-standard columns for horizontal,
        magnitude and depth uncertainty (see :mod:`~obspy.core.zmap`).
    """
    Pickler(with_uncertainties).dump(catalog, filename)


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
