# -*- coding: utf-8 -*-
"""
IASPEI Seismic Format (ISF) support for ObsPy

Currently only supports reading IMS1.0 bulletin files.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""
import warnings

from obspy import UTCDateTime
from obspy.core.event import (
    Catalog, Event, Origin, Comment, EventDescription, OriginUncertainty,
    QuantityError, OriginQuality, CreationInfo, Magnitude, ResourceIdentifier,
    Pick, StationMagnitude, WaveformStreamID, Amplitude, Arrival)
from obspy.core.event.header import AmplitudeUnit
from obspy.core.util.obspy_types import ObsPyReadingError
from .util import (
    float_or_none, int_or_none, fixed_flag, evaluation_mode_and_status,
    _block_header)


PICK_EVALUATION_MODE = {'m': 'manual', 'a': 'automatic', '_': None, '': None}
POLARITY = {'c': 'positive', 'd': 'negative', '_': None, '': None}
ONSET = {'i': 'impulsive', 'e': 'emergent', 'q': 'questionable', '_': None,
         '': None}
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


def _decode_if_possible(value, encoding="UTF-8"):
    try:
        return value.decode(encoding)
    except Exception:
        return value


class ISFReader(object):
    encoding = 'UTF-8'
    resource_id_prefix = 'smi:local'

    def __init__(self, fh, **kwargs):
        self.lines = [_decode_if_possible(line, self.encoding).rstrip()
                      for line in fh.readlines()
                      if line.strip()]
        self.cat = Catalog()
        self._no_uuid_hashes = kwargs.get('_no_uuid_hashes', False)

    def deserialize(self):
        if not self.lines:
            raise ObsPyReadingError()
        header_flag = False
        while not self._next_line_type():
            # Skip the lines preceding the header
            # (if the bulletin is enveloped as an AutoDRM response message)
            # (BEGIN IMS1.0, MSG_TYPE DATA, ...)
            line = self._get_next_line()
            if line.upper().startswith('DATA_TYPE BULLETIN IMS1.0:SHORT'):
                header_flag = True
                break
        if not header_flag:
            raise ObsPyReadingError()
        try:
            self._deserialize()
        except ISFEndOfFile:
            pass
        return self.cat

    def _deserialize(self):
        line = self._get_next_line()
        catalog_description = line.strip()
        self.cat.description = catalog_description
        if not self.lines[0].lower().startswith('event'):
            raise ObsPyReadingError()
        # get next line stops the loop eventually, raising a controlled
        # exception
        while True:
            next_line_type = self._next_line_type()
            if next_line_type == 'event':
                # scope last event when a new one starts
                # avoid trying to do it when starting to read first event,
                # which results in index error
                if len(self.cat):
                    self.cat[-1].scope_resource_ids()
                self._read_event_header()
            elif next_line_type:
                self._process_block()
            else:
                raise ObsPyReadingError
        # scope last event when no new events are encountered anymore
        # in principle we should get rid of the above scoping and should just
        # scope every event at the end in a loop? not sure if this has negative
        # side effects so doing it like this for now, just fixing the
        # IndexError for first event above
        if len(self.cat):
            self.cat[-1].scope_resource_ids()

    def _construct_id(self, parts, add_hash=False):
        id_ = '/'.join([str(self.cat.resource_id)] + list(parts))
        if add_hash and not self._no_uuid_hashes:
            id_ = str(ResourceIdentifier(prefix=id_))
        return id_

    def _get_next_line(self):
        if not self.lines:
            raise ISFEndOfFile
        line = self.lines.pop(0)
        if line.startswith('STOP'):
            raise ISFEndOfFile
        return line

    def _read_event_header(self):
        line = self._get_next_line()
        event_id = self._construct_id(['event', line[6:14].strip()])
        region = line[15:80].strip()
        event = Event(
            resource_id=event_id,
            event_descriptions=[EventDescription(text=region,
                                                 type='region name')])
        self.cat.append(event)

    def _next_line_type(self):
        if not self.lines:
            raise ISFEndOfFile
        return _block_header(self.lines[0])

    def _process_block(self):
        if not self.cat:
            raise ObsPyReadingError
        line = self._get_next_line()
        block_type = _block_header(line)
        # read origins block
        if block_type == 'origins':
            self._read_origins()
            self._specify_preferred_origin()
        # read publications block
        elif block_type == 'bibliography':
            self._read_bibliography()
        # read magnitudes block
        elif block_type == 'magnitudes':
            self._read_magnitudes()
        # read phases block
        elif block_type == 'phases':
            self._read_phases()
        # unexpected block header line
        else:
            msg = ('Unexpected line while reading file (line will be '
                   'ignored):\n' + line)
            warnings.warn(msg)

    def _read_phases(self):
        event = self.cat[-1]
        origin = event.preferred_origin()
        line = self.lines[0].strip()
        # OrigID Phase and Phase Information Comment
        if line.startswith('(#OrigID'):
            origin_id = ResourceIdentifier(
                self._construct_id(['origin', line[8:].rstrip(") ").strip()]))
            origin = origin_id.get_referred_object()
        if origin:
            origin_id = origin.resource_id
        else:
            origin_id = None

        while not self._next_line_type():
            line = self._get_next_line()
            if line.strip().startswith('('):
                comment = self._parse_generic_comment(line)
                if len(event.picks):
                    event.picks[-1].comments.append(comment)
                continue
            pick, amplitude, station_magnitude, arrival = self._parse_phase(
                                                             line, origin_id)
            if (pick, amplitude, station_magnitude) == (None, None, None):
                continue
            event.picks.append(pick)
            if amplitude:
                event.amplitudes.append(amplitude)
            if station_magnitude:
                event.station_magnitudes.append(station_magnitude)
            if origin:
                # The list of Arrival items is stored in the specified origin
                if arrival:
                    origin.arrivals.append(arrival)
            continue

    def _read_origins(self):
        event = self.cat[-1]
        origins = []
        event_types_certainties = []
        # just in case origin block is at end of file, make sure the event type
        # routine below gets executed, even if next line is EOF at some point
        try:
            while not self._next_line_type():
                line = self._get_next_line()
                if line.strip().startswith('('):
                    origins[-1].comments.append(
                        self._parse_generic_comment(line))
                    continue
                origin, event_type, event_type_certainty = \
                    self._parse_origin(line)
                origins.append(origin)
                event_types_certainties.append(
                    (event_type, event_type_certainty))
                continue
        finally:
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
            else:
                event_type = None
                event_type_certainty = None
            event.origins.extend(origins)
            event.event_type = event_type
            event.event_type_certainty = event_type_certainty

    def _specify_preferred_origin(self):
        '''
        Find the preferred origin in two simple cases:
          - only one origin for an event
          - tagged #PRIME
        Affects: event.preferred_origin_id
        '''
        event = self.cat[-1]
        event.preferred_origin_id = None
        if len(event.origins) == 1:
            event.preferred_origin_id = event.origins[0].resource_id.id
            return
        for origin in event.origins:
            for comment in origin.comments:
                if "#PRIME" in comment.text.upper():
                    event.preferred_origin_id = origin.resource_id.id

    def _read_magnitudes(self):
        event = self.cat[-1]
        while not self._next_line_type():
            line = self._get_next_line()
            if line.strip().startswith('('):
                event.magnitudes[-1].comments.append(
                    self._parse_generic_comment(line))
                continue
            event.magnitudes.append(self._parse_magnitude(line))
            continue

    def _read_bibliography(self):
        event = self.cat[-1]
        while not self._next_line_type():
            line = self._get_next_line()
            if line.strip().startswith('('):
                # TODO parse bibliography comment blocks
                continue
            event.comments.append(self._parse_bibliography_item(line))
            continue

    def _make_comment(self, text):
        id_ = self._construct_id(['comment'], add_hash=True)
        comment = Comment(text=text, resource_id=id_)
        return comment

    def _parse_bibliography_item(self, line):
        return self._make_comment(line)

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
        #               ([m], blank if fixed epicenter)
        _uncertainty_major_m = float_or_none(line[55:60], multiplier=1e3)
        # 62-66   f5.1  semi-minor axis of 90% ellipse or its estimate
        #               ([m], blank if fixed epicenter)
        _uncertainty_minor_m = float_or_none(line[61:66], multiplier=1e3)
        # 68-70   i3    strike (0 <= x <= 360) of error ellipse clock-wise from
        #                       North (degrees)
        _uncertainty_major_azimuth = float_or_none(line[67:70])
        # 72-76   f5.1  depth [m]
        depth = float_or_none(line[71:76], multiplier=1e3)
        # 77      a1    fixed flag (f = fixed depth station, d = depth phases,
        #                           blank if not a fixed depth)
        depth_fixed = fixed_flag(line[76])
        # 79-82   f4.1  depth error 90% ([m]; blank if fixed depth)
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
            evaluation_mode_and_status(line[111])
        # 114     a1    location method: (i = inversion, p = pattern
        #                                 recognition, g = ground truth, o =
        #                                 other)
        location_method = LOCATION_METHODS[line[113].strip().lower()]
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
        origin_id = self._construct_id(['origin', line[128:136].strip()])

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
            # event init always sets an empty QuantityError, even when
            # specifying None, which is strange
            for key in ['confidence_ellipsoid']:
                setattr(origin_uncertainty, key, None)
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
                self._make_comment('location method: ' + location_method))
        if author:
            creation_info = CreationInfo(author=author)
        else:
            creation_info = None
        # assemble whole event
        origin = Origin(
            time=time, resource_id=origin_id, longitude=longitude,
            latitude=latitude, depth=depth, depth_errors=depth_error,
            origin_uncertainty=origin_uncertainty, time_fixed=time_fixed,
            epicenter_fixed=epicenter_fixed or depth_fixed,
            origin_quality=origin_quality,
            comments=comments, creation_info=creation_info)
        # event init always sets an empty QuantityError, even when specifying
        # None, which is strange
        for key in ('time_errors', 'longitude_errors', 'latitude_errors',
                    'depth_errors'):
            setattr(origin, key, None)
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
            origin_id = self._construct_id(['origin', origin_id])
        else:
            origin_id = None
        if not magnitude_type:
            magnitude_type = None
        # magnitudes have no id field, so construct a unique one at least
        resource_id = self._construct_id(['magnitude'], add_hash=True)

        if min_max_indicator:
            msg = 'Magnitude min/max indicator field not yet implemented'
            warnings.warn(msg)

        # combine and return
        mag = Magnitude(
            magnitude_type=magnitude_type, mag=mag,
            station_count=station_count, creation_info=creation_info,
            mag_errors=mag_errors, origin_id=origin_id,
            resource_id=resource_id)
        # event init always sets an empty QuantityError, even when specifying
        # None, which is strange
        for key in ['mag_errors']:
            setattr(mag, key, None)
        return mag

    def _get_pick_time(self, my_string):
        """
        Look up absolute time of pick including date, based on the time-of-day
        only representation in the phase line

        Returns absolute pick time or None if it can not be determined safely.
        """
        if not my_string.strip():
            return None
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
            warnings.warn(msg)
            return None
        # now try the date of the latest origin and raise if things seem fishy
        t = UTCDateTime(pick_date.year, pick_date.month, pick_date.day, hour,
                        minute, seconds)
        for origin_time in origin_times:
            if t - origin_time > 6 * 3600:
                msg = ('This pick would have a time more than 6 hours after '
                       'or before one of the origins in the event. This seems '
                       'fishy. Please report an issue on our github.')
                warnings.warn(msg)
                return None
        return t

    def _parse_phase(self, line, origin_id):

        # Data that we cannot interpret and assign
        # to "Arrival", "Amplitude", "Pick" objects
        # are still written as "Pick.comments" text.
        # In the case that no reference to the origin is known,
        # the origin-specific data was only stored as a comment.
        # It's not very helpful. I recommend deleting this feature.
        # It is currently commented out (# DD if origin_id is None: ...)
        comments = []

        # 1-5     a5      station code
        station_code = line[0:5].strip()
        # 7-12    f6.2    station-to-event distance (degrees)
        # Arrival.distance (float)
        distance = float_or_none(line[6:12])
        # DD if origin_id is None:
        # DD     comments.append(
        # DD     'station-to-event distance (degrees): "{}"'.format(
        # DD                                                    line[6:12]))
        # 14-18   f5.1    event-to-station azimuth (degrees)
        # Arrival.azimuth (float)
        event_azimuth = float_or_none(line[13:18])
        # DD if origin_id is None:
        # DD     comments.append(
        # DD     'event-to-station azimuth (degrees): "{}"'.format(
        # DD                                                   line[13:18]))
        # 20-27   a8      phase code
        # Pick.phase_hint (str)
        # Arrival.phase (str)
        phase_hint = line[19:27].strip()
        # 29-40   i2,a1,i2,a1,f6.3        arrival time (hh:mm:ss.sss)
        # Pick.time (UTCDateTime, ATTRIBUTE_HAS_ERRORS)
        time = self._get_pick_time(line[28:40])
        # 42-46   f5.1    time residual (seconds)
        # Arrival.time_residual (float)
        time_residual = float_or_none(line[41:46])
        # DD if origin_id is None:
        # DD     comments.append(
        # DD     'time residual (seconds): "{}"'.format(line[41:46]))
        # 48-52   f5.1    observed azimuth (degrees)
        # Pick.backazimuth (float, ATTRIBUTE_HAS_ERRORS)
        backazimuth = float_or_none(line[47:52])
        # DD if origin_id is None:
        # DD     comments.append(
        # DD     'observed azimuth (degrees): "{}"'.format(line[47:52]))
        # 54-58   f5.1    azimuth residual (degrees)
        # Arrival.backazimuth_residual (float)
        backazimuth_residual = float_or_none(line[53:58])
        # DD if origin_id is None:
        # DD     comments.append(
        # DD     'azimuth residual (degrees): "{}"'.format(line[53:58]))
        # 60-65   f5.1    observed slowness (seconds/degree)
        # Pick.horizontal_slowness (float, ATTRIBUTE_HAS_ERRORS)
        horizontal_slowness = float_or_none(line[59:65])
        # DD comments.append(
        # DD   'observed slowness (seconds/degree): "{}"'.format(line[59:65]))
        # 67-72   f5.1    slowness residual (seconds/degree)
        # Arrival.horizontal_slowness_residual (float)
        horizontal_slowness_residual = float_or_none(line[66:71])
        # DD if origin_id is None:
        # DD     comments.append(
        # DD     'slowness residual (seconds/degree): "{}"'.format(
        # DD                                                   line[66:71]))
        # 74      a1      time defining flag (T or _)
        # Arrival.time_weight (float)
        time_defining_flag = line[73]
        time_weight = 0
        if time_defining_flag == 'T':
            time_weight = 1
        # DD comments.append(
        # DD   'time defining flag (T or _): "{}"'.format(line[73]))
        # 75      a1      azimuth defining flag (A or _)
        # Arrival.backazimuth_weight (float)
        azimuth_defining_flag = line[74]
        backazimuth_weight = 0
        if azimuth_defining_flag == 'A':
            backazimuth_weight = 1
        # DD comments.append(
        # DD   'azimuth defining flag (A or _): "{}"'.format(line[74]))
        # 76      a1      slowness defining flag (S or _)
        # Arrival.horizontal_slowness_weight (float)
        slowness_defining_flag = line[75]
        horizontal_slowness_weight = 0
        if slowness_defining_flag == 'S':
            horizontal_slowness_weight = 1
        # DD comments.append(
        # DD   'slowness defining flag (S or _): "{}"'.format(line[75]))
        comments.append(
            'TAS flag: "{}"'.format(line[73:76]))
        # 78-82   f5.1    signal-to-noise ratio
        # Amplitude.snr (float)
        snr = float_or_none(line[77:82])
        # comments.append('signal-to-noise ratio: "{}"'.format(line[77:82]))
        # 84-92   f9.1    amplitude (nanometers)
        # Amplitude.generic_amplitude (float, ATTRIBUTE_HAS_ERRORS)
        amp = float_or_none(line[83:92])
        # 94-98   f5.2    period (seconds)
        # Amplitude.period (float, ATTRIBUTE_HAS_ERRORS)
        period = float_or_none(line[93:98])
        # 100     a1      type of pick (a = automatic, m = manual)
        # Pick.evaluation_mode (EvaluationMode)
        evaluation_mode = line[99]
        # 101     a1      direction of short period motion
        #                 (c = compression, d = dilatation, _= null)
        # Pick.polarity (PickPolarity)
        polarity = POLARITY[line[100].strip().lower()]
        # 102     a1      onset quality (i = impulsive, e = emergent,
        #                                q = questionable, _ = null)
        # Pick.onset (PickOnset)
        onset = ONSET[line[101].strip().lower()]
        # 104-108 a5      magnitude type (mb, Ms, ML, mbmle, msmle)
        # StationMagnitude.station_magnitude_type (str)
        magnitude_type = line[103:108].strip()
        # 109     a1      min max indicator (<, >, or blank)
        min_max_indicator = line[108]
        # 110-113 f4.1    magnitude value
        # StationMagnitude.mag (float, ATTRIBUTE_HAS_ERRORS)
        mag = float_or_none(line[109:113])
        # 115-122 a8      arrival identification
        phase_id = line[114:122].strip()
        #
        if (time is None) and (amp is None) and (mag is None):
            msg = ('Could not determine absolute time of pick. This phase '
                   'line will be ignored:\n{}').format(line)
            warnings.warn(msg)
            return None, None, None, None
        # process items
        waveform_id = WaveformStreamID(station_code=station_code)
        evaluation_mode = PICK_EVALUATION_MODE[evaluation_mode.strip().lower()]
        comments = [self._make_comment(', '.join(comments))]
        # assemble
        # pick
        if phase_id:
            pick_id = self._construct_id(['pick', phase_id])
        else:
            pick_id = self._construct_id(['pick'], add_hash=True)
        pick = Pick(resource_id=pick_id,
                    phase_hint=phase_hint, time=time, waveform_id=waveform_id,
                    evaluation_mode=evaluation_mode, comments=comments,
                    polarity=polarity, onset=onset, backazimuth=backazimuth,
                    horizontal_slowness=horizontal_slowness
                    )
        # event init always sets an empty QuantityError, even when specifying
        # None, which is strange
        for key in ('time_errors', 'horizontal_slowness_errors',
                    'backazimuth_errors'):
            setattr(pick, key, None)
        # amplitude
        if amp:
            if phase_id:
                amplitude_id = self._construct_id(['amplitude', phase_id])
            else:
                amplitude_id = self._construct_id(['amplitude'], add_hash=True)
            # Convert from nanometers to meters
            amp /= 1e9
            amplitude = Amplitude(resource_id=amplitude_id, pick_id=pick_id,
                                  unit=AmplitudeUnit.M, generic_amplitude=amp,
                                  period=period, snr=snr
                                  )
        else:
            amplitude = None
            amplitude_id = None
        # station_magnitude
        if mag:
            comment = ('min max indicator (<, >, or blank): ' +
                       min_max_indicator)
            if phase_id:
                station_magnitude_id = self._construct_id(
                        ['station_magnitude', phase_id])
            else:
                station_magnitude_id = self._construct_id(
                        ['station_magnitude'], add_hash=True)
            station_magnitude = StationMagnitude(
                    resource_id=station_magnitude_id,
                    amplitude_id=amplitude_id, waveform_id=waveform_id,
                    origin_id=origin_id, mag=mag,
                    magnitude_type=magnitude_type,
                    comments=[self._make_comment(comment)]
                    )
            if origin_id is None:
                setattr(station_magnitude, 'origin_id', None)
            for key in ['mag_errors']:
                setattr(station_magnitude, key, None)
        else:
            station_magnitude = None
        # arrival
        if distance or time_residual:
            if phase_id:
                arrival_id = self._construct_id(['arrival', phase_id])
            else:
                arrival_id = self._construct_id(['arrival'], add_hash=True)
            arrival = Arrival(
                    resource_id=arrival_id, pick_id=pick_id, phase=phase_hint,
                    azimuth=event_azimuth, distance=distance,
                    time_residual=time_residual,
                    backazimuth_residual=backazimuth_residual,
                    horizontal_slowness_residual=horizontal_slowness_residual,
                    time_weight=time_weight,
                    backazimuth_weight=backazimuth_weight,
                    horizontal_slowness_weight=horizontal_slowness_weight
                    )
        else:
            arrival = None

        return pick, amplitude, station_magnitude, arrival

    def _parse_generic_comment(self, line):
        return self._make_comment(line)


def _buffer_proxy(filename_or_buf, function, reset_fp=True,
                  file_mode="rb", *args, **kwargs):
    """
    Calls a function with an open file or file-like object as the first
    argument. If the file originally was a filename, the file will be
    opened, otherwise it will just be passed to the underlying function.

    :param filename_or_buf: File to pass.
    :type filename_or_buf: str, open file, or file-like object
    :param function: The function to call.
    :param reset_fp: If True, the file pointer will be set to the initial
        position after the function has been called.
    :type reset_fp: bool
    :param file_mode: Mode to open file in if necessary.
    """
    try:
        position = filename_or_buf.tell()
        is_buffer = True
    except AttributeError:
        is_buffer = False

    if is_buffer is True:
        ret_val = function(filename_or_buf, *args, **kwargs)
        if reset_fp:
            filename_or_buf.seek(position, 0)
        return ret_val
    else:
        with open(filename_or_buf, file_mode) as fh:
            return function(fh, *args, **kwargs)


def _read_ims10_bulletin(filename_or_buf, **kwargs):
    """
    Reads an ISF IMS1.0 bulletin file to a :class:`~obspy.core.event.Catalog`
    object.

    :param filename_or_buf: File or file-like object
    """
    try:
        return _buffer_proxy(filename_or_buf, __read_ims10_bulletin,
                             reset_fp=False, **kwargs)
    # Happens for example when passing the data as a string which would be
    # interpreted as a filename.
    except OSError:
        return False


def __read_ims10_bulletin(fh, **kwargs):  # NOQA
    """
    Reads an ISF IMS1.0 bulletin file to a :class:`~obspy.core.event.Catalog`
    object.

    :param fh: File or file-like object
    """
    return ISFReader(fh, **kwargs).deserialize()


def _is_ims10_bulletin(filename_or_buf, **kwargs):
    """
    Checks whether a file is ISF IMS1.0 bulletin format.

    :type filename_or_buf: str or file
    :param filename_or_buf: name of the file to be checked or open file-like
        object.
    :rtype: bool
    :return: ``True`` if ISF IMS1.0 bulletin file.
    """
    try:
        return _buffer_proxy(filename_or_buf, __is_ims10_bulletin,
                             reset_fp=True, **kwargs)
    # Happens for example when passing the data as a string which would be
    # interpreted as a filename.
    except OSError:
        return False


def __is_ims10_bulletin(fh, **kwargs):  # NOQA
    """
    Checks whether a file is ISF IMS1.0 bulletin format.

    :type fh: open file or file-like object
    :param filename: name of the file to be checked or open file-like object.
    :rtype: bool
    :return: ``True`` if ISF IMS1.0 bulletin file.
    """
    iline = 0
    try:
        for line in fh:
            iline += 1
            # the header can be preceded by a max of 40 lines of text
            if iline > 40:
                break
            sline = _decode_if_possible(line).rstrip()
            if sline.upper().startswith("DATA_TYPE BULLETIN IMS1.0:SHORT"):
                return True
    except Exception:
        return False
    return False
