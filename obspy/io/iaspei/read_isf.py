# -*- coding: utf-8 -*-
import re
import warnings

from obspy import UTCDateTime
from obspy.core.util.obspy_types import ObsPyReadingError
from obspy.core.event import (
    Catalog, Event, Origin, Comment, EventDescription, OriginUncertainty,
    QuantityError, OriginQuality, CreationInfo, Magnitude, ResourceIdentifier)

test_file = '19670130012028.isf'


EVENT_TYPE_CERTAINTY = {
    "uk": (None, None),
    "de": ("earthquake", "known"),
    "fe": ("earthquake", "known"),
    "ke": ("earthquake", "known"),
    "se": ("earthquake", "suspected"),
    "kr": ("rock burst", "known"),
    "sr": ("rock burst", "suspected"),
    "ki": ("induced or triggered event", "known"),
    "si": ("induced or triggered event", "suspected"),
    "km": ("mining explosion", "known"),
    "sm": ("mining explosion", "suspected"),
    "kh": ("chemical explosion", "known"),
    "sh": ("chemical explosion", "suspected"),
    "kx": ("experimental explosion", "known"),
    "sx": ("experimental explosion", "suspected"),
    "kn": ("nuclear explosion", "known"),
    "sn": ("nuclear explosion", "suspected"),
    "ls": ("landslide", "known"),
    "": (None, None),
    }
LOCATION_METHODS = {'i': 'inversion', 'p': 'pattern recognition',
                    'g': 'ground truth', 'o': 'other', '': None}


class ISFEndOfFile(StopIteration):
    pass


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
        except ISFEndOfFile:
            pass
        return self.cat

    def _deserialize(self):
        line = self._get_next_line()
        if not line.startswith('DATA_TYPE BULLETIN IMS1.0:short'):
            raise ObsPyReadingError()
        line = self._get_next_line()
        catalog_description = line.strip()
        self.cat.description = catalog_description
        line = self._get_next_line()
        if not line.startswith('Event'):
            raise ObsPyReadingError()
        self._read_event(line)

    def _construct_id(self, *parts):
        return '/'.join([str(self.cat.resource_id)] + list(parts))

    def _get_next_line(self):
        line = self.fh.readline().decode(self.encoding).rstrip()
        if line.startswith('STOP'):
            raise ISFEndOfFile
        return line

    def _read_event(self, line):
        # TODO check if the various blocks always come ordered same aas in our
        # test data or if oreder of blocks is random.. then we would have to
        # acoount for random order..
        event_id = self._construct_id('event', line[6:14].strip())
        region = line[15:80].strip()
        event = Event(
            resource_id=event_id,
            event_descriptions=[EventDescription(text=region,
                                                 type='region name')])
        self.cat.append(event)

        while True:
            # get next line stops the loop eventually, raising a controlled
            # exception
            line = self._get_next_line()
            # ignore blank lines
            if not line.strip():
                continue
            self._process_block(line, event)

    def _process_block(self, line, event):
        header_start = [x.lower() for x in line.split()[:4]]
        # read origins block
        if header_start == ['date', 'time', 'err', 'rms']:
            origins, event_type, event_type_certainty = self._read_origins()
            event.origins.extend(origins)
            event.event_type = event_type
            event.event_type_certainty = event_type_certainty
        # read publications block
        elif header_start == ['year', 'volume', 'page1', 'page2']:
            event.comments.extend(self._read_bibliography())
        # read magnitudes block
        elif header_start == ['magnitude', 'err', 'nsta', 'author']:
            event.magnitudes.extend(self._read_magnitudes())
        # unexpected block header line
        else:
            msg = ('Unexpected line while reading file (line will be '
                   'ignored):\n' + line)
            warnings.warn(msg)

    def _read_origins(self):
        origins = []
        event_types_certainties = []
        while True:
            line = self._get_next_line()
            if re.match('[0-9]{4}/[0-9]{2}/[0-9]{2}', line):
                origin, event_type, event_type_certainty = \
                    self._parse_origin(line)
                origins.append(origin)
                event_types_certainties.append(
                    (event_type, event_type_certainty))
                continue
            if line.strip().startswith('('):
                origins[-1].comments.append(
                    self._parse_generic_comment(line))
                continue
            break
        # check event types/certainties for consistency
        event_types = set(type_ for type_, _ in event_types_certainties)
        event_types.discard(None)
        if len(event_types) == 1:
            event_type = event_types.pop()
            certainties = set(
                cert for type_, cert in event_types_certainties
                if type_ == event_type)
            if "known" in certainties:
                event_type_certainty = "known"
            elif "suspected" in certainties:
                event_type_certainty = "suspected"
            else:
                event_type_certainty = None
        return origins, event_type, event_type_certainty

    def _read_magnitudes(self):
        magnitudes = []
        while True:
            line = self._get_next_line()
            # regex assumes that at least an integer or float for magnitude
            # value is present
            if re.match('[a-z ]{5}[<> ][\d ]\d[\. ][\d ]', line):
                magnitudes.append(self._parse_magnitude(line))
                continue
            if line.strip().startswith('('):
                magnitudes[-1].comments.append(
                    self._parse_generic_comment(line))
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
        location_method = LOCATION_METHODS[line[114].strip().lower()]
        # 116-117 a2    event type:
        # XXX event type and event type certainty is specified per origin,
        # XXX not sure how to bset handle this, for now only use it if
        # XXX information on the individual origins do not clash.. not sure yet
        # XXX how to identify the preferred origin..
        event_type, event_type_certainty = \
            EVENT_TYPE_CERTAINTY[line[115:117].strip().lower()]
        # 119-127 a9    author of the origin
        author = line[118:127].strip()
        # 129-136 a8    origin identification
        origin_id = self._construct_id('origin', line[128:136].strip())

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
        comments = []
        if location_method:
            comments.append(
                Comment(text='location method: ' + location_method))
        if author:
            creation_info = CreationInfo(author=author)
        else:
            creation_info = None
        # assemble whole event
        origin = Origin(
            time=time, resource_id=origin_id, longitude=longitude,
            latitude=latitude, depth=depth, depth_errors=depth_error,
            origin_uncertainty=origin_uncertainty, time_fixed=time_fixed,
            epicenter_fixed=epicenter_fixed, origin_quality=origin_quality,
            comments=comments, creation_info=creation_info)
        return origin, event_type, event_type_certainty

    def _parse_magnitude(self, line):
        #    1-5  a5   magnitude type (mb, Ms, ML, mbmle, msmle)
        magnitude_type = line[0:5].strip()
        #      6  a1   min max indicator (<, >, or blank)
        # TODO figure out the meaning of this min max indicator
        min_max_indicator = line[5:6].strip()
        #   7-10  f4.1 magnitude value
        mag = float_or_none(line[6:10])
        #  12-14  f3.1 standard magnitude error
        mag_errors = float_or_none(line[11:14])
        #  16-19  i4   number of stations used to calculate magni-tude
        station_count = int_or_none(line[15:19])
        #  21-29  a9   author of the origin
        author = line[20:29].strip()
        #  31-38  a8   origin identification
        origin_id = line[30:38].strip()

        # process items
        if author:
            creation_info = CreationInfo(author=author)
        else:
            creation_info = None
        mag_errors = mag_errors and QuantityError(uncertainty=mag_errors)
        if origin_id:
            origin_id = self._construct_id('origin', origin_id)
        else:
            origin_id = None
        if not magnitude_type:
            magnitude_type = None
        # magnitudes have no id field, so construct a unique one at least
        resource_id = ResourceIdentifier(
            prefix=self._construct_id('magnitude'))

        if min_max_indicator:
            msg = 'Magnitude min/max indicator field not yet implemented'
            warnings.warn(msg)

        # combine and return
        return Magnitude(
            magnitude_type=magnitude_type, mag=mag,
            station_count=station_count, creation_info=creation_info,
            mag_errors=mag_errors, origin_id=origin_id,
            resource_id=resource_id)

    def _get_pick_time(self, my_string):
        """
        Look up absolute time of pick including date, based on the time-of-day
        only representation in the phase line
        """
        # TODO maybe we should defer phases block parsing.. but that will make
        # the whole reading more complex
        if not self.cat.events:
            msg = ('Can not parse phases block before parsing origins block, '
                   'because phase lines do not contain date information, only '
                   'time-of-day')
            raise NotImplementedError(msg)
        origin_times = [origin.time for origin in self.cat.events[-1].origins]
        if not origin_times:
            msg = ('Can not parse phases block unless origins with origin '
                   'time information are present, because phase lines do not '
                   'contain date information, only time-of-day')
            raise NotImplementedError(msg)
        # XXX this whole routine is on shaky ground..
        # since picks only have a time-of-day and there's not even an
        # association to one of the origins, in principle this would need some
        # real tough logic to make it failsafe. actually this would mean using
        # taup with the given epicentral distance of the pick and check what
        # date is appropriate.
        # for now just do a very simple logic and raise exceptions when things
        # look fishy. this is ugly but it's not worth spending more time on
        # this, unless somebody starts bumping into one of the explicitly
        # raised exceptions below.
        origin_time_min = min(origin_times)
        origin_time_max = max(origin_times)
        hour = int(my_string[0:2])
        minute = int(my_string[3:5])
        seconds = float(my_string[6:])

        all_guesses = []
        for origin in self.cat.events[-1].origins:
            first_guess = UTCDateTime(
                origin.time.year, origin.time.month, origin.time.day, hour,
                minute, seconds)
            all_guesses.append((first_guess, origin.time))
            all_guesses.append((first_guess - 86400, origin.time))
            all_guesses.append((first_guess + 86400, origin.time))

        pick_date = sorted(all_guesses, key=lambda x: abs(x[0] - x[1]))[0][0]

        # make sure event origin times are reasonably close together
        if origin_time_max - origin_time_min > 5 * 3600:
            msg = ('Origin times in event differ by more than 5 hours, this '
                   'is currently not implemented as determining the date of '
                   'the pick might be tricky. Sorry.')
            raise NotImplementedError(msg)
        # now try the date of the latest origin and raise if things seem fishy
        t = UTCDateTime(pick_date.year, pick_date.month, pick_date.day, hour,
                        minute, seconds)
        for origin_time in origin_times:
            if t - origin_time > 6 * 3600:
                msg = ('This pick would have a time more than 6 hours after '
                       'or before one of the origins in the event. This seems '
                       'fishy. Please report an issue on our github.')
                raise NotImplementedError(msg)
        return t

    @staticmethod
    def _parse_generic_comment(line):
        comment = Comment(text=line)
        return comment


def _read_isf(fh):
    cat = ISFReader(fh).deserialize()
    return cat


with open(test_file, 'rb') as fh:
    cat = _read_isf(fh)
cat.write('/tmp/isc.xml', format="QuakeML")
