import re

from obspy import UTCDateTime
from obspy.core.event import (
    Catalog, Event, Origin, Comment, EventDescription, OriginUncertainty,
    QuantityError, OriginQuality)

test_file = '19670130012028.isf'


def evaluation_mode_and_status(my_string):
    """
    Return QuakeML standard evaluation mode and status based on the single
    field ISF "analysis type" field.
    """
    my_string = my_string.lower()
    # TODO check this matching
    if my_string == 'a':
        mode = 'automatic'
        status = None
    elif my_string == 'm':
        mode = 'manual'
        status = 'reviewed'
    elif my_string == 'g':
        mode = 'manual'
        status = 'preliminary'
    elif not my_string.strip():
        return None, None
    else:
        raise ValueError()
    return mode, status


def type_or_none(my_string, type_, multiplier=None):
    my_string = my_string.strip() or None
    my_string = my_string and type_(my_string)
    if my_string is not None and multiplier is not None:
        my_string = my_string * multiplier
    return my_string and type_(my_string)


def float_or_none(my_string, **kwargs):
    return type_or_none(my_string, float)


def int_or_none(my_string, **kwargs):
    return type_or_none(my_string, int)


def fixed_flag(my_char):
    if len(my_char) != 1:
        raise ValueError()
    return my_char.lower() == 'f'


class ISFReader(object):
    encoding = 'UTF-8'
    resource_id_prefix = 'smi:local'

    def __init__(self, fh):
        self.fh = fh
        self.cat = Catalog()

    def deserialize(self):
        try:
            self._deserialize()
        except StopIteration:
            pass
        return self.cat

    def _deserialize(self):
        line = self._get_next_line()
        if not line.startswith('DATA_TYPE BULLETIN IMS1.0:short'):
            raise Exception()
        line = self._get_next_line()
        catalog_description = line.strip()
        line = self._get_next_line()
        if not line.startswith('Event'):
            raise Exception()
        event = self._read_event(line)
        self.cat.append(event)

    def _get_next_line(self):
        line = self.fh.readline().decode(self.encoding).rstrip()
        print(line)
        if line.startswith('STOP'):
            raise StopIteration
        return line

    def _read_event(self, line):
        # TODO check if the various blocks always come ordered same aas in our
        # test data or if oreder of blocks is random.. then we would have to
        # acoount for random order..
        event_id = '/'.join([self.resource_id_prefix, 'event',
                             line[6:14].strip()])
        region = line[15:80].strip()
        event = Event(
            resource_id=event_id,
            event_descriptions=[EventDescription(text=region,
                                                 type='region name')])

        # read origins block
        while True:
            line = self._get_next_line()
            if line.split()[:3] != ['Date', 'Time', 'Err']:
                continue
            event.origins.extend(self._read_origins())
            break

        # read publications block
        while True:
            line = self._get_next_line()
            if line.split()[:3] != ['Year', 'Volume', 'Page1']:
                continue
            event.comments.extend(self._read_bibliography())
            break

        # read magnitudes block
        while True:
            line = self._get_next_line()
            if line.split()[:3] != ['Magnitude', 'Err', 'Nsta']:
                continue
            event.magnitudes.extend(self._read_magnitudes())
            break

        return event

    def _read_origins(self):
        origins = []
        while True:
            line = self._get_next_line()
            if re.match('[0-9]{4}/[0-9]{2}/[0-9]{2}', line):
                origins.append(self._parse_origin(line))
                continue
            if line.strip().startswith('('):
                origins[-1].comments.append(
                    self._parse_origin_comment(line))
                continue
            return origins

    def _read_magnitudes(self):
        magnitudes = []
        while True:
            line = self._get_next_line()
            if re.match('[0-9]{4}/[0-9]{2}/[0-9]{2}', line):
                magnitudes.append(self._parse_magnitude(line))
                continue
            if line.strip().startswith('('):
                magnitudes[-1].comments.append(
                    self._parse_magnitude_comment(line))
                continue
            return magnitudes

    def _read_bibliography(self):
        comments = []
        while True:
            line = self._get_next_line()
            if re.match('[0-9]{4}', line):
                comments.append(self._parse_bibliography_item(line))
                continue
            if line.strip().startswith('('):
                # TODO parse bibliography comment blocks
                continue
            return comments

    @staticmethod
    def _parse_bibliography_item(line):
        comment = Comment(text=line)
        return comment

    def _parse_origin(self, line):
        # 1-10    i4,a1,i2,a1,i2    epicenter date (yyyy/mm/dd)
        # 12-22   i2,a1,i2,a1,f5.2  epicenter time (hh:mm:ss.ss)
        time = UTCDateTime.strptime(line[:17], '%Y/%m/%d %H:%M:')
        time += float(line[17:22])
        # 23      a1    fixed flag (f = fixed origin time solution, blank if
        #                           not a fixed origin time)
        time_fixed = fixed_flag(line[22])
        # 25-29   f5.2  origin time error (seconds; blank if fixed origin time)
        time_error = float_or_none(line[24:29])
        time_error = time_error and QuantityError(uncertainty=time_error)
        # 31-35   f5.2  root mean square of time residuals (seconds)
        rms = float_or_none(line[30:35])
        # 37-44   f8.4  latitude (negative for South)
        latitude = float_or_none(line[36:44])
        # 46-54   f9.4  longitude (negative for West)
        longitude = float_or_none(line[45:54])
        # 55      a1    fixed flag (f = fixed epicenter solution, blank if not
        #                           a fixed epicenter solution)
        epicenter_fixed = fixed_flag(line[54])
        # 56-60   f5.1  semi-major axis of 90% ellipse or its estimate
        #               (km, blank if fixed epicenter)
        _uncertainty_major_m = float_or_none(line[55:60], multiplier=1e3)
        # 62-66   f5.1  semi-minor axis of 90% ellipse or its estimate
        #               (km, blank if fixed epicenter)
        _uncertainty_minor_m = float_or_none(line[61:66], multiplier=1e3)
        # 68-70   i3    strike (0 <= x <= 360) of error ellipse clock-wise from
        #                       North (degrees)
        _uncertainty_major_azimuth = float_or_none(line[67:70])
        # 72-76   f5.1  depth (km)
        depth = float_or_none(line[71:76])
        # 77      a1    fixed flag (f = fixed depth station, d = depth phases,
        #                           blank if not a fixed depth)
        epicenter_fixed = fixed_flag(line[76])
        # 79-82   f4.1  depth error 90% (km; blank if fixed depth)
        depth_error = float_or_none(line[78:82], multiplier=1e3)
        # 84-87   i4    number of defining phases
        used_phase_count = int_or_none(line[83:87])
        # 89-92   i4    number of defining stations
        used_station_count = int_or_none(line[88:92])
        # 94-96   i3    gap in azimuth coverage (degrees)
        azimuthal_gap = float_or_none(line[93:96])
        # 98-103  f6.2  distance to closest station (degrees)
        minimum_distance = float_or_none(line[97:103])
        # 105-110 f6.2  distance to furthest station (degrees)
        maximum_distance = float_or_none(line[104:110])
        # 112     a1    analysis type: (a = automatic, m = manual, g = guess)
        evaluation_mode, evaluation_status = \
            evaluation_mode_and_status(line[112])
        # 114     a1    location method: (i = inversion, p = pattern
        #                                 recognition, g = ground truth, o =
        #                                 other)
        location_method = line[114]
        # 116-117 a2    event type:
        #                 uk = unknown
        #                 de = damaging earthquake ( Not standard IMS )
        #                 fe = felt earthquake ( Not standard IMS )
        #                 ke = known earthquake
        #                 se = suspected earthquake
        #                 kr = known rockburst
        #                 sr = suspected rockburst
        #                 ki = known induced event
        #                 si = suspected induced event
        #                 km = known mine expl.
        #                 sm = suspected mine expl.
        #                 kh = known chemical expl. ( Not standard IMS )
        #                 sh = suspected chemical expl. ( Not standard IMS )
        #                 kx = known experimental expl.
        #                 sx = suspected experimental expl.
        #                 kn = known nuclear expl.
        #                 sn = suspected nuclear explosion
        #                 ls = landslide
        # XXX map to QuakeML enum
        event_type = None
        # 119-127 a9    author of the origin
        author = line[118:127].strip()
        # 129-136 a8    origin identification
        origin_id = '/'.join([self.resource_id_prefix, 'origin',
                              line[128:136].strip()])

        # do some combinations
        depth_error = depth_error and dict(uncertainty=depth_error,
                                           confidence_level=90)
        if all(v is not None for v in (_uncertainty_major_m,
                                       _uncertainty_minor_m,
                                       _uncertainty_major_azimuth)):
            origin_uncertainty = OriginUncertainty(
                min_horizontal_uncertainty=_uncertainty_minor_m,
                max_horizontal_uncertainty=_uncertainty_major_m,
                azimuth_max_horizontal_uncertainty=_uncertainty_major_azimuth,
                preferred_description='uncertainty ellipse',
                confidence_level=90)
        else:
            origin_uncertainty = None
        origin_quality = OriginQuality(
            standard_error=rms, used_phase_count=used_phase_count,
            used_station_count=used_station_count, azimuthal_gap=azimuthal_gap,
            minimum_distance=minimum_distance,
            maximum_distance=maximum_distance)
        origin = Origin(
            time=time, resource_id=origin_id, longitude=longitude,
            latitude=latitude, depth=depth, depth_errors=depth_error,
            origin_uncertainty=origin_uncertainty, time_fixed=time_fixed,
            epicenter_fixed=epicenter_fixed, origin_quality=origin_quality)
        return origin

    @staticmethod
    def _parse_origin_comment(line):
        comment = Comment()
        return comment


def _read_isf(fh):
    cat = ISFReader(fh).deserialize()
    return cat


with open(test_file, 'rb') as fh:
    cat = _read_isf(fh)
