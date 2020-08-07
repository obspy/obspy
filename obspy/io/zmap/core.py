# -*- coding: utf-8 -*-
"""
ZMAP read/write support.

ZMAP is a simple 10 column CSV file (technically TSV) format for basic catalog
data (see [Wiemer2001]_).

:copyright:
    The ObsPy Development Team (devs@obspy.org), Lukas Heiniger
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""
import math

from obspy.core import UTCDateTime
from obspy.core.event import (Catalog, Event, Magnitude, Origin,
                              OriginUncertainty)
from obspy.core.util.decorator import map_example_filename


_STD_ZMAP_COLUMNS = ('lon', 'lat', 'year', 'month', 'day', 'mag', 'depth',
                     'hour', 'minute', 'second')
_EXT_ZMAP_COLUMNS = ('h_err', 'z_err', 'm_err')


class Pickler(object):

    def __init__(self, with_uncertainties=False):
        # This is ZMAP column order, don't change it
        zmap_columns = _STD_ZMAP_COLUMNS
        if with_uncertainties:
            zmap_columns += _EXT_ZMAP_COLUMNS
        self.zmap_columns = zmap_columns

    def dump(self, catalog, filename):
        """
        Writes ObsPy Catalog into given file.

        :type catalog: :class:`~obspy.core.event.Catalog`
        :param catalog: ObsPy Catalog object.
        :type filename: str or file
        :param filename: Target file name or open file-like object.
        """
        # Open filehandler or use an existing file like object.
        if not hasattr(filename, "write"):
            file_opened = True
            fh = open(filename, "wb")
        else:
            file_opened = False
            fh = filename
        try:
            cat_string = self._serialize(catalog)
            fh.write(cat_string.encode('utf-8'))
        finally:
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
            d_lat = lat_err
            d_lon = lon_err * math.cos(origin.latitude * math.pi / 180.0)
            h_err = km_per_deg * math.hypot(d_lat, d_lon)
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
        rows = []
        for ev in catalog:
            strings = dict.fromkeys(self.zmap_columns, 'NaN')
            # origin
            origin = ev.preferred_origin()
            if origin is None and ev.origins:
                origin = ev.origins[0]
            if origin:
                dec_year = self._decimal_year(origin.time)
                dec_second = origin.time.second + \
                    origin.time.microsecond / 1e6
                strings.update({
                    'depth': self._num2str(origin.depth / 1000.0),  # m to km
                    'z_err': self._num2str(self._depth_error(origin)),
                    'lat': self._num2str(origin.latitude),
                    'lon': self._num2str(origin.longitude),
                    'h_err': self._num2str(self._hz_error(origin)),
                    'year': self._num2str(dec_year, 12),
                    'month': self._num2str(origin.time.month, 0),
                    'day': self._num2str(origin.time.day, 0),
                    'hour': self._num2str(origin.time.hour, 0),
                    'minute': self._num2str(origin.time.minute, 0),
                    'second': str(dec_second)
                })
            # magnitude
            magnitude = ev.preferred_magnitude()
            if magnitude is None and ev.magnitudes:
                magnitude = ev.magnitudes[0]
            if magnitude:
                strings.update({
                    'mag': self._num2str(magnitude.mag),
                    'm_err': self._num2str(magnitude.mag_errors.uncertainty)
                })
            # create tab separated row
            rows.append('\t'.join([strings[c] for c in self.zmap_columns]))
        zmap = '\n'.join(rows) + '\n'
        return zmap


class Unpickler(object):

    def load(self, filename):
        """
        Returns an ObsPy Catalog object from a ZMAP file.

        :type filename: str or file
        :param filename: Source file name or open file-like object.
        :rtype: :class:`~obspy.core.event.Catalog`
        :returns: ObsPy catalog
        """
        # Open filehandler or use an existing file like object.
        if not hasattr(filename, "read"):
            file_opened = True
            fh = open(filename, 'rb')
        else:
            file_opened = False
            fh = filename
        try:
            zmap_str = fh.read()
            if hasattr(zmap_str, 'decode'):
                zmap_str = zmap_str.decode('utf-8')
            catalog = self._deserialize(zmap_str)
            return catalog
        finally:
            if file_opened:
                fh.close()

    def loads(self, zmap_str):
        """
        Returns an ObsPy Catalog object from a ZMAP string.

        :type zmap_str: str
        :param zmap_str: ObsPy Catalog object.
        :rtype: :class:`~obspy.core.event.Catalog`
        :returns: ObsPy catalog
        """
        return self._deserialize(zmap_str)

    def _deserialize(self, zmap_str):
        catalog = Catalog()
        for row in zmap_str.split('\n'):
            if len(row) == 0:
                continue
            origin = Origin()
            event = Event(origins=[origin])
            event.preferred_origin_id = origin.resource_id.id
            # Begin value extraction
            columns = row.split('\t', 13)[:13]  # ignore extra columns
            values = dict(zip(_STD_ZMAP_COLUMNS + _EXT_ZMAP_COLUMNS, columns))
            # Extract origin
            origin.longitude = self._str2num(values.get('lon'))
            origin.latitude = self._str2num(values.get('lat'))
            depth = self._str2num(values.get('depth'))
            if depth is not None:
                origin.depth = depth * 1000.0
            z_err = self._str2num(values.get('z_err'))
            if z_err is not None:
                origin.depth_errors.uncertainty = z_err * 1000.0
            h_err = self._str2num(values.get('h_err'))
            if h_err is not None:
                ou = OriginUncertainty()
                ou.horizontal_uncertainty = h_err
                ou.preferred_description = 'horizontal uncertainty'
                origin.origin_uncertainty = ou
            year = self._str2num(values.get('year'))
            if year is not None:
                t_fields = ['year', 'month', 'day', 'hour', 'minute', 'second']
                comps = [self._str2num(values.get(f)) for f in t_fields]
                if year % 1 != 0:
                    origin.time = self._decyear2utc(year)
                elif any(v > 0 for v in comps[1:]):
                    # no seconds involved
                    if len(comps) < 6:
                        utc_args = [int(v) for v in comps if v is not None]
                    # we also have to handle seconds
                    else:
                        utc_args = [int(v) if v is not None else 0
                                    for v in comps[:-1]]
                        # just leave float seconds as is
                        utc_args.append(comps[-1])
                    origin.time = UTCDateTime(*utc_args)
            mag = self._str2num(values.get('mag'))
            # Extract magnitude
            if mag is not None:
                magnitude = Magnitude(mag=mag)
                m_err = self._str2num(values.get('m_err'))
                magnitude.mag_errors.uncertainty = m_err
                event.magnitudes.append(magnitude)
                event.preferred_magnitude_id = magnitude.resource_id.id
            event.scope_resource_ids()
            catalog.append(event)
        return catalog

    @staticmethod
    def _str2num(num_str):
        try:
            if num_str is not None and num_str.lower() != 'nan':
                num = float(num_str)
            else:
                return None
        except ValueError:
            return None
        return num

    @staticmethod
    def _decyear2utc(decimal_year):
        """
        Return UTCDateTime from decimal year
        """
        start_of_year = UTCDateTime(int(decimal_year), 1, 1)
        end_of_year = UTCDateTime(int(decimal_year) + 1, 1, 1)
        t = start_of_year + (decimal_year % 1) * (end_of_year - start_of_year)
        return t


def _write_zmap(catalog, filename, with_uncertainties=False,
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

    :type catalog: :class:`~obspy.core.event.Catalog`
    :param catalog: The ObsPy Catalog object to write.
    :type filename: str or file
    :param filename: Filename to write or open file-like object.
    :type with_uncertainties: bool
    :param with_uncertainties: appends non-standard columns for horizontal,
        magnitude and depth uncertainty (see :mod:`~obspy.io.zmap.core`).
    """
    Pickler(with_uncertainties).dump(catalog, filename)


def _read_zmap(filename, **kwargs):
    """
    Reads a ZMAP file and returns an ObsPy Catalog object.

    .. warning::
        This function should NOT be called directly, it registers via the
        ObsPy :func:`~obspy.core.event.read_events` function, call this
        instead.

    :type filename: str or file
    :param filename: ZMAP string, name of ZMAP file to be read or open
        file-like object.
    :rtype: :class:`~obspy.core.event.Catalog`
    :return: An ObsPy Catalog object.
    """
    if not hasattr(filename, 'read'):
        try:
            with open(filename):
                pass
        except Exception:
            # we assume it's a string now
            return Unpickler().loads(filename)
    return Unpickler().load(filename)


@map_example_filename("filename")
def _is_zmap(filename):
    """
    Checks whether a file is ZMAP format.

    Unlike :func:`~obspy.io.zmap.core._read_zmap` *_is_zmap* is strict, i.e. it
    will not detect a ZMAP file unless it consists of exactly 10 or 13
    numerical columns.

    :type filename: str or file
    :param filename: ZMAP string, name of the file to be checked or open
        file-like object.
    :rtype: bool
    :return: ``True`` if ZMAP file.

    .. rubric:: Example

    >>> _is_zmap('/path/to/zmap_events.txt')
    True
    """
    if all(hasattr(filename, attr) for attr in ['tell', 'seek', 'read']):
        # we got an open file
        pos = filename.tell()
        filename.seek(0)
        first_line = filename.readline()
        filename.seek(pos)
        if hasattr(first_line, 'decode'):
            first_line = first_line.decode()
    else:
        try:
            with open(filename, 'rb') as f:
                first_line = f.readline().decode()
        except Exception:
            try:
                first_line = filename.decode()
            except Exception:
                first_line = str(filename)
            line_ending = first_line.find("\n")
            if line_ending == -1:
                return False
            first_line = first_line[:line_ending]

    # we expect 10 (standard) or 13 columns (extended)
    columns = first_line.split('\t')
    if len(columns) not in [10, 13]:
        return False

    # only numerical values are allowed (including NaN)
    try:
        [float(col) for col in columns]
        return True
    except ValueError:
        return False


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
