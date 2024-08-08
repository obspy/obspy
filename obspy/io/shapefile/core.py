# -*- coding: utf-8 -*-
import datetime
import re
import warnings

from obspy import Catalog, UTCDateTime
from obspy.core.event import Origin, Magnitude
from obspy.core.inventory import Inventory
from obspy.core.util.misc import to_int_or_zero

try:
    import shapefile
except ImportError as e:
    HAS_PYSHP = False
    PYSHP_VERSION = None
    PYSHP_VERSION_AT_LEAST_1_2_11 = False
    IMPORTERROR_MSG = str(e) + (
        ". ObsPy's write support for shapefiles requires the 'pyshp' module "
        "to be installed in addition to the general ObsPy dependencies.")
else:
    HAS_PYSHP = True
    try:
        PYSHP_VERSION = list(map(to_int_or_zero,
                                 shapefile.__version__.split('.')))
    except AttributeError:
        PYSHP_VERSION = None
        PYSHP_VERSION_AT_LEAST_1_2_11 = False
    else:
        PYSHP_VERSION_AT_LEAST_1_2_11 = PYSHP_VERSION >= [1, 2, 11]
PYSHP_VERSION_WARNING = (
    'pyshp versions < 1.2.11 are buggy, e.g. in writing numerical values to '
    'the dbf table, so e.g. timestamp float values might lack proper '
    'precision. You should update to a newer pyshp version.')


def _write_shapefile(obj, filename, extra_fields=None, **kwargs):
    """
    Write :class:`~obspy.core.inventory.inventory.Inventory` or
    :class:`~obspy.core.event.Catalog` object to a ESRI shapefile.

    :type obj: :class:`~obspy.core.event.Catalog` or
        :class:`~obspy.core.inventory.inventory.Inventory`
    :param obj: ObsPy object for shapefile output
    :type filename: str
    :param filename: Filename to write to. According to ESRI shapefile
        definition, multiple files with the following suffixes will be written:
        ".shp", ".shx", ".dbj", ".prj". If filename does not end with ".shp",
        it will be appended. Other files will be created with respective
        suffixes accordingly.
    :type extra_fields: list
    :param extra_fields: List of extra fields to write to
        the shapefile table. Each item in the list has to be specified as a
        tuple of: field name (i.e. name of database column, ``str``), field
        type (single character as used by ``pyshp``: ``'C'`` for string
        fields, ``'N'`` for integer/float fields - use precision ``None`` for
        integer fields, ``'L'`` for boolean fields), field width (``int``),
        field precision (``int``) and field values (``list`` of individual
        values, must have same length as given catalog object or as the sum of
        all station objects across all networks of a given inventory).
    """
    if not HAS_PYSHP:
        raise ImportError(IMPORTERROR_MSG)
    if not PYSHP_VERSION_AT_LEAST_1_2_11:
        warnings.warn(PYSHP_VERSION_WARNING)
    if not filename.endswith(".shp"):
        filename += ".shp"

    if PYSHP_VERSION >= [2., 0, 0]:
        writer = shapefile.Writer(target=filename, shapeType=shapefile.POINT)
    else:
        writer = shapefile.Writer(shapeType=shapefile.POINT)
    writer.autoBalance = 1

    # create the layer
    if isinstance(obj, Catalog):
        _add_catalog_layer(writer, obj, extra_fields=extra_fields)
    elif isinstance(obj, Inventory):
        _add_inventory_layer(writer, obj, extra_fields=extra_fields)
    else:
        msg = ("Object for shapefile output must be "
               "a Catalog or Inventory.")
        raise TypeError(msg)

    if PYSHP_VERSION >= [2.0, 0, 0]:
        writer.close()
    else:
        writer.save(filename)

    _save_projection_file(filename.rsplit('.', 1)[0] + '.prj')


def _add_catalog_layer(writer, catalog, extra_fields=None):
    """
    :type writer: :class:`shapefile.Writer`.
    :param writer: pyshp Writer object
    :type catalog: :class:`~obspy.core.event.Catalog`
    :param catalog: Event data to add as a new layer.
    :type extra_fields: list
    :param extra_fields: List of extra fields to write to the shapefile table.
        For details see :func:`_write_shapefile()`.
    """
    # [name, type, width, precision]
    # field name is 10 chars max
    # ESRI shapefile attributes are stored in dbf files, which can not
    # store datetimes, only dates, see:
    # http://www.gdal.org/drv_shapefile.html
    # use POSIX timestamp for exact origin time, set time of first pick
    # for events with no origin
    field_definitions = [
        ["EventID", 'C', 100, None],
        ["OriginID", 'C', 100, None],
        ["MagID", 'C', 100, None],
        ["Date", 'D', None, None],
        ["OriginTime", 'N', 20, 6],
        ["FirstPick", 'N', 20, 6],
        ["Longitude", 'N', 16, 10],
        ["Latitude", 'N', 16, 10],
        ["Depth", 'N', 8, 3],
        ["MinHorUncM", 'N', 12, 3],
        ["MaxHorUncM", 'N', 12, 3],
        ["MaxHorAzi", 'N', 7, 3],
        ["OriUncDesc", 'C', 40, None],
        ["Magnitude", 'N', 8, 3],
    ]

    _create_layer(writer, field_definitions, extra_fields)

    if extra_fields:
        for name, type_, width, precision, values in extra_fields:
            if len(values) != len(catalog):
                msg = ("list of values for each item in 'extra_fields' must "
                       "have same length as Catalog object")
                raise ValueError(msg)

    for i, event in enumerate(catalog):
        # try to use preferred origin/magnitude, fall back to first or use
        # empty one with `None` values in it
        origin = (event.preferred_origin() or
                  event.origins and event.origins[0] or
                  Origin(force_resource_id=False))
        magnitude = (event.preferred_magnitude() or
                     event.magnitudes and event.magnitudes[0] or
                     Magnitude(force_resource_id=False))
        t_origin = origin.time
        pick_times = [pick.time for pick in event.picks
                      if pick.time is not None]
        t_pick = pick_times and min(pick_times) or None
        date = t_origin or t_pick

        feature = {}

        # setting fields with `None` results in values of `0.000`
        # need to really omit setting values if they are `None`
        if event.resource_id is not None:
            feature["EventID"] = str(event.resource_id)
        if origin.resource_id is not None:
            feature["OriginID"] = str(origin.resource_id)
        if t_origin is not None:
            # Use timestamp for exact timing
            feature["OriginTime"] = t_origin.timestamp
        if t_pick is not None:
            # Use timestamp for exact timing
            feature["FirstPick"] = t_pick.timestamp
        if date is not None:
            # ESRI shapefile attributes are stored in dbf files, which can
            # not store datetimes, only dates. We still need to use the
            # GDAL API with precision up to seconds (aiming at other output
            # drivers of GDAL; `100` stands for GMT)
            feature["Date"] = date.datetime
        if origin.latitude is not None:
            feature["Latitude"] = origin.latitude
        if origin.longitude is not None:
            feature["Longitude"] = origin.longitude
        if origin.depth is not None:
            feature["Depth"] = origin.depth / 1e3
        if magnitude.mag is not None:
            feature["Magnitude"] = magnitude.mag
        if magnitude.resource_id is not None:
            feature["MagID"] = str(magnitude.resource_id)
        if origin.origin_uncertainty is not None:
            ou = origin.origin_uncertainty
            ou_description = ou.preferred_description
            if ou_description == 'uncertainty ellipse':
                feature["MinHorUncM"] = ou.min_horizontal_uncertainty
                feature["MaxHorUncM"] = ou.max_horizontal_uncertainty
                feature["MaxHorAzi"] = \
                    ou.azimuth_max_horizontal_uncertainty
                feature["OriUncDesc"] = ou_description
            elif ou_description == 'horizontal uncertainty':
                feature["MinHorUncM"] = ou.horizontal_uncertainty
                feature["MaxHorUncM"] = ou.horizontal_uncertainty
                feature["MaxHorAzi"] = 0.0
                feature["OriUncDesc"] = ou_description
            else:
                msg = ('Encountered an event with origin uncertainty '
                       'description of type "{}". This is not yet '
                       'implemented for output as shapefile. No origin '
                       'uncertainty will be added to shapefile for such '
                       'events.').format(ou_description)
                warnings.warn(msg)

        if origin.latitude is not None and origin.longitude is not None:
            writer.point(origin.longitude, origin.latitude)
            if extra_fields:
                for name, _, _, _, values in extra_fields:
                    feature[name] = values[i]
            _add_record(writer, feature)


def _add_inventory_layer(writer, inventory, extra_fields=None):
    """
    :type writer: :class:`shapefile.Writer`.
    :param writer: pyshp Writer object
    :type inventory: :class:`~obspy.core.inventory.inventory.Inventory`
    :param inventory: Inventory data to add as a new layer.
    :type extra_fields: list
    :param extra_fields: List of extra fields to write to the shapefile table.
        For details see :func:`_write_shapefile()`.
    """
    # [name, type, width, precision]
    # field name is 10 chars max
    # ESRI shapefile attributes are stored in dbf files, which can not
    # store datetimes, only dates, see:
    # http://www.gdal.org/drv_shapefile.html
    # use POSIX timestamp for exact origin time, set time of first pick
    # for events with no origin
    field_definitions = [
        ["Network", 'C', 20, None],
        ["Station", 'C', 20, None],
        ["Longitude", 'N', 16, 10],
        ["Latitude", 'N', 16, 10],
        ["Elevation", 'N', 9, 3],
        ["StartDate", 'D', None, None],
        ["EndDate", 'D', None, None],
        ["Channels", 'C', 254, None],
    ]

    _create_layer(writer, field_definitions, extra_fields)

    station_count = sum(len(net) for net in inventory)
    if extra_fields:
        for name, type_, width, precision, values in extra_fields:
            if len(values) != station_count:
                msg = ("list of values for each item in 'extra_fields' must "
                       "have same length as the count of all Stations "
                       "combined across all Networks.")
                raise ValueError(msg)

    i = 0
    for net in inventory:
        for sta in net:
            channel_list = ",".join(["%s.%s" % (cha.location_code, cha.code)
                                     for cha in sta])

            feature = {}

            # setting fields with `None` results in values of `0.000`
            # need to really omit setting values if they are `None`
            if net.code is not None:
                feature["Network"] = net.code
            if sta.code is not None:
                feature["Station"] = sta.code
            if sta.latitude is not None:
                feature["Latitude"] = sta.latitude
            if sta.longitude is not None:
                feature["Longitude"] = sta.longitude
            if sta.elevation is not None:
                feature["Elevation"] = sta.elevation
            if sta.start_date is not None:
                # ESRI shapefile attributes are stored in dbf files, which
                # can not store datetimes, only dates. We still need to use
                # the GDAL API with precision up to seconds (aiming at
                # other output drivers of GDAL; `100` stands for GMT)
                feature["StartDate"] = sta.start_date.datetime
            if sta.end_date is not None:
                # ESRI shapefile attributes are stored in dbf files, which
                # can not store datetimes, only dates. We still need to use
                # the GDAL API with precision up to seconds (aiming at
                # other output drivers of GDAL; `100` stands for GMT)
                feature["EndDate"] = sta.end_date.datetime
            if channel_list:
                feature["Channels"] = channel_list

            if extra_fields:
                for name, _, _, _, values in extra_fields:
                    feature[name] = values[i]

            if sta.latitude is not None and sta.longitude is not None:
                writer.point(sta.longitude, sta.latitude)
                _add_record(writer, feature)

            i += 1


wgs84_wkt = \
    """
    GEOGCS["WGS 84",
        DATUM["WGS_1984",
            SPHEROID["WGS 84",6378137,298.257223563,
                AUTHORITY["EPSG","7030"]],
            AUTHORITY["EPSG","6326"]],
        PRIMEM["Greenwich",0,
            AUTHORITY["EPSG","8901"]],
        UNIT["degree",0.0174532925199433,
            AUTHORITY["EPSG","9122"]],
        AUTHORITY["EPSG","4326"]]
    """
wgs84_wkt = re.sub(r'\s+', '', wgs84_wkt)


def _save_projection_file(filename):
    with open(filename, 'wt') as fh:
        fh.write(wgs84_wkt)


def _add_field(writer, name, type_, width, precision):
    # default field width is not set correctly for dates and booleans in
    # shapefile <=1.2.10, see
    # GeospatialPython/pyshp@ba61854aa7161fd7d4cff12b0fd08b6ec7581bb7 and
    # GeospatialPython/pyshp#71 so work around this
    if type_ == 'D':
        width = 8
        precision = 0
    elif type_ == 'L':
        width = 1
        precision = 0
    kwargs = dict(fieldType=type_, size=width, decimal=precision)
    # remove None's because shapefile.Writer.field() doesn't use None as
    # placeholder but the default values directly
    for key in list(kwargs.keys()):
        if kwargs[key] is None:
            kwargs.pop(key)
    writer.field(name, **kwargs)


def _create_layer(writer, field_definitions, extra_fields=None):
    # Add the fields we're interested in
    for name, type_, width, precision in field_definitions:
        _add_field(writer, name, type_, width, precision)
    field_names = [name for name, _, _, _ in field_definitions]
    # add custom fields
    if extra_fields is not None:
        for name, type_, width, precision, _ in extra_fields:
            if name in field_names:
                msg = "Conflict with existing field named '{}'.".format(name)
                raise ValueError(msg)
            _add_field(writer, name, type_, width, precision)


def _add_record(writer, feature):
    values = []
    for key, type_, width, precision in writer.fields:
        value = feature.get(key)
        # various hacks for old pyshp < 1.2.11
        if not PYSHP_VERSION_AT_LEAST_1_2_11:
            if type_ == 'C':
                # mimic pyshp 1.2.12 behavior of putting 'None' in string
                # fields for value of `None`
                if value is None:
                    value = 'None'
            # older pyshp is not correctly writing dates as used nowadays
            # '%Y%m%d' (8 chars), work around this
            elif type_ == 'D':
                if isinstance(value, (UTCDateTime, datetime.date)):
                    value = value.strftime('%Y%m%d')
            # work around issues with older pyshp, backport 1.2.12 behavior
            elif type_ == 'L':
                # logical: 1 byte - initialized to 0x20 (space)
                # otherwise T or F
                if value in [True, 1]:
                    value = "T"
                elif value in [False, 0]:
                    value = "F"
                else:
                    value = ' '
            # work around issues with older pyshp, backport 1.2.12 behavior
            elif type_ in ('N', 'F'):
                # numeric or float: number stored as a string, right justified,
                # and padded with blanks to the width of the field.
                if value in (None, ''):
                    value = ' ' * width  # QGIS NULL
                elif not precision:
                    # caps the size if exceeds the field size
                    value = format(value, "d")[:width].rjust(width)
                else:
                    # caps the size if exceeds the field size
                    value = format(value, ".%sf" % precision)[:width].rjust(
                        width)
            # work around older pyshp not converting `None`s properly (e.g. for
            # float fields)
            elif value is None:
                value = ''
        values.append(value)
    writer.record(*values)


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
