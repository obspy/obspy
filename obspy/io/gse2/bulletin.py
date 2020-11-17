# -*- coding: utf-8 -*-
"""
GSE2.0 bulletin read support.

:author:
    EOST (École et Observatoire des Sciences de la Terre)
:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""
import re
import warnings

from obspy.core.event import (Amplitude, Arrival, Catalog, Comment,
                              CreationInfo, Event, EventDescription,
                              Magnitude, Origin, OriginQuality,
                              OriginUncertainty, Pick, ResourceIdentifier,
                              StationMagnitude,
                              StationMagnitudeContribution,
                              WaveformStreamID)
from obspy.core.event.header import (
    EvaluationMode, EventDescriptionType, EventType, EventTypeCertainty,
    OriginDepthType, OriginUncertaintyDescription, PickOnset, PickPolarity)
from obspy.core.utcdatetime import UTCDateTime


# Convert GSE2 depth flag to ObsPy depth type
DEPTH_TYPES = {
    'f': OriginDepthType('operator assigned'),
    'd': OriginDepthType('constrained by depth phases'),
}

# Convert GSE2 analysis type to ObsPy evaluation modes
EVALUATION_MODES = {
    'm': EvaluationMode.MANUAL,
    'a': EvaluationMode.AUTOMATIC,
    'g': EvaluationMode.MANUAL,
}


# Convert GSE2 to ObsPy location methods
LOCATION_METHODS = {
    'i': 'inversion',
    'p': 'pattern recognition',
    'g': 'ground truth',
    'o': 'other',
}


# Convert GSE2 to ObsPy event types
EVENT_TYPES = {
    'ke': (EventTypeCertainty.KNOWN,
           EventType('earthquake')),
    'se': (EventTypeCertainty.SUSPECTED,
           EventType('earthquake')),
    'kr': (EventTypeCertainty.KNOWN,
           EventType('rock burst')),
    'sr': (EventTypeCertainty.SUSPECTED,
           EventType('rock burst')),
    'ki': (EventTypeCertainty.KNOWN,
           EventType('induced or triggered event')),
    'si': (EventTypeCertainty.SUSPECTED,
           EventType('induced or triggered event')),
    'km': (EventTypeCertainty.KNOWN,
           EventType('mining explosion')),
    'sm': (EventTypeCertainty.SUSPECTED,
           EventType('mining explosion')),
    'kx': (EventTypeCertainty.KNOWN,
           EventType('experimental explosion')),
    'sx': (EventTypeCertainty.SUSPECTED,
           EventType('experimental explosion')),
    'kn': (EventTypeCertainty.KNOWN,
           EventType('nuclear explosion')),
    'sn': (EventTypeCertainty.SUSPECTED,
           EventType('nuclear explosion')),
    'ls': (EventTypeCertainty.KNOWN,
           EventType('landslide')),
    'uk': (None,
           EventType('other')),
}


# Convert GSE2 to ObsPy polarity
PICK_POLARITIES = {
    'c': PickPolarity.POSITIVE,
    'd': PickPolarity.NEGATIVE,
}


# Convert GSE2 to ObsPy pick onset
PICK_ONSETS = {
    'E': PickOnset.EMERGENT,
    'I': PickOnset.IMPULSIVE,
    'Q': PickOnset.QUESTIONABLE,
}


class GSE2BulletinSyntaxError(Exception):
    """Raised when the file is not a valid GSE2 file"""


def _is_gse2(filename):
    """
    Checks whether a file is GSE2.0 format.

    :type filename: str
    :param filename: Name of the GSE2.0 file to be checked.
    :rtype: bool
    :return: ``True`` if GSE2.0 file.
    """
    try:
        with open(filename, 'rb') as fh:
            temp = fh.read(12)
    except Exception:
        return False
    if temp != b'BEGIN GSE2.0':
        return False
    return True


class LinesIterator(object):
    """
    Iterator to iterate file lines and count lines. Usefull for warning
    messages.
    """
    def __init__(self, lines):
        self.lines = iter(lines)
        self.line_nb = 0

    def __iter__(self):
        return self

    def __next__(self):
        self.line_nb += 1
        return next(self.lines)


class Unpickler(object):
    """
    De-serialize a GSE2.0 string into an ObsPy Catalog object.
    """

    def __init__(self, inventory, default_network_code, default_location_code,
                 default_channel_code, res_id_prefix, fields,
                 event_point_separator, agency):
        self.default_network_code = default_network_code
        self.default_location_code = default_location_code
        self.default_channel_code = default_channel_code
        self.inventory = inventory
        self.res_id_prefix = res_id_prefix
        self.event_point_separator = event_point_separator
        self.agency = agency
        self.author = ''

        self.fields = {
            'line_1': {
                'time': slice(0, 21),
                'time_fixf': slice(22, 23),
                'lat': slice(25, 33),
                'lon': slice(34, 43),
                'epicenter_fixf': slice(44, 45),
                'depth': slice(47, 52),
                'depth_fixf': slice(53, 54),
                'n_def': slice(56, 60),
                'n_sta': slice(61, 65),
                'gap': slice(66, 69),
                'mag_type_1': slice(71, 73),
                'mag_1': slice(73, 77),
                'mag_n_sta_1': slice(78, 80),
                'mag_type_2': slice(82, 84),
                'mag_2': slice(84, 88),
                'mag_n_sta_2': slice(89, 91),
                'mag_type_3': slice(93, 95),
                'mag_3': slice(95, 99),
                'mag_n_sta_3': slice(100, 102),
                'author': slice(104, 112),
                'id': slice(114, 122),
            },
            'line_2': {
                'rms': slice(5, 10),
                'ot_error': slice(15, 21),
                's_major': slice(25, 31),
                's_minor': slice(32, 38),
                'az': slice(40, 43),
                'depth_err': slice(49, 54),
                'min_dist': slice(56, 62),
                'max_dist': slice(63, 69),
                'mag_err_1': slice(74, 77),
                'mag_err_2': slice(85, 88),
                'mag_err_3': slice(96, 99),
                'antype': slice(104, 105),
                'loctype': slice(106, 107),
                'evtype': slice(108, 110),
            },
            'arrival': {
                'sta': slice(0, 5),
                'dist': slice(6, 12),
                'ev_az': slice(13, 18),
                'picktype': slice(19, 20),
                'direction': slice(20, 21),
                'detchar': slice(21, 22),
                'phase': slice(23, 30),
                'time': slice(31, 52),
                't_res': slice(53, 58),
                'azim': slice(59, 64),
                'az_res': slice(65, 71),
                'slow': slice(72, 77),
                's_res': slice(78, 83),
                't_def': slice(84, 85),
                'a_def': slice(85, 86),
                's_def': slice(86, 87),
                'snr': slice(88, 93),
                'amp': slice(94, 103),
                'per': slice(104, 109),
                'mag_type_1': slice(110, 112),
                'mag_1': slice(112, 116),
                'mag_type_2': slice(117, 119),
                'mag_2': slice(119, 123),
                'id': slice(124, 132),
            },
        }

        if fields:
            if 'line_1' in fields:
                self.fields['line_1'].update(fields['line_1'])
            if 'line_2' in fields:
                self.fields['line_2'].update(fields['line_2'])
            if 'arrival' in fields:
                self.fields['arrival'].update(fields['arrival'])

    def load(self, filename):
        """
        Read GSE2.0 file into ObsPy catalog object.

        :type filename: str
        :param filename: File name to read.
        :rtype: :class:`~obspy.core.event.Catalog`
        :return: ObsPy Catalog object.
        """
        with open(filename, 'r') as f:
            self.lines = LinesIterator(f.readlines())
        return self._deserialize()

    def _add_line_nb(self, message):
        """
        Add line number at the end of a str message.

        :type message: str
        :param message: Message for warnings or exceptions.
        :rtype: str
        :return: Message with line number.
        """
        return "%s, line %s" % (message, self.lines.line_nb)

    def _warn(self, message):
        """
        Display a warning message with the line number.

        :type message: str
        :param message: Message to be displayed
        """
        warnings.warn(self._add_line_nb(message))

    def _skip_empty_lines(self):
        line = next(self.lines)
        while not line or line.isspace():
            line = next(self.lines)
        return line

    def _get_res_id(self, ident, parent=None, parent_res_id=None):
        """
        Create a :class:`~obspy.core.event.resourceid.ResourceIdentifier`
        object.

        :type ident: str
        :param ident: Id of
            the :class:`~obspy.core.event.resourceid.ResourceIdentifier`.
        :type parent: :class:`~obspy.core.event.origin.Origin`,
            :class:`~obspy.core.event.event.Event` or any other object
            with a resource_id attribute.
        :param parent: The resource_id attribute of the parent will be
            used as a prefix for the new
            :class:`~obspy.core.event.resourceid.ResourceIdentifier`.
        :type parent_res_id:
            :class:`~obspy.core.event.resourceid.ResourceIdentifier` of the
            parent.
        :param parent_res_id:
            :class:`~obspy.core.event.resourceid.ResourceIdentifier`
        :rtype: :class:`~obspy.core.event.resourceid.ResourceIdentifier`
        :return: ResourceIdentifier object.
        """
        prefix = self.res_id_prefix
        # Put the parent id as prefix
        # Example: smi:local/origin/351412/arrival/6389611
        #          |        prefix        |     ident    |
        if parent:
            prefix = parent.resource_id.id
        elif parent_res_id:
            prefix = parent_res_id.id

        public_id = "%s/%s" % (prefix, ident)
        return ResourceIdentifier(public_id)

    def _comment(self, text):
        comment = Comment()
        comment.text = text
        comment.resource_id = ResourceIdentifier(prefix=self.res_id_prefix)
        return comment

    def _get_creation_info(self):
        creation_info = CreationInfo(creation_time=UTCDateTime())
        if self.agency:
            creation_info.agency_id = self.agency
        if self.author:
            creation_info.author = self.author
        return creation_info

    def _check_header(self, first_line):
        """
        Just check some stuff in header.

        :type first_line: str
        :param first_line: First line of header.
        """
        line_1_pattern = r'BEGIN\sGSE2.0'
        line_2_pattern = r'MSG_TYPE\s(REQUEST|DATA|SUBSCRIPTION|PROBLEM)'
        line_3_pattern = r'MSG_ID\s\w{1,20}\s?\w{1,8}'

        if not re.match(line_1_pattern, first_line):
            raise GSE2BulletinSyntaxError('Wrong GSE2.0 header')
        if not re.match(line_2_pattern, next(self.lines)):
            raise GSE2BulletinSyntaxError('Wrong message type in header')
        if not re.match(line_3_pattern, next(self.lines)):
            raise GSE2BulletinSyntaxError('Wrong message ID in header')

    def _get_channel(self, station, time):
        """
        Use inventory to retrieve channel and location code.

        :type station: str
        :param station: Station code
        :type time: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param time: Only use channel active at given point in time.
        :rtype: str, :class:`~obspy.core.inventory.channel.Channel`
        :return: Network code, channel retrieved.
        """
        if self.inventory is None:
            return None

        sub_inventory = self.inventory.select(station=station, time=time)

        if len(sub_inventory) == 0:
            self._warn("Can't find station %s in inventory" % station)
            return None

        if len(sub_inventory) > 1:
            self._warn('Several stations %s have been found with different '
                       'network code in inventory' %
                       station)
            return None

        network = sub_inventory[0]

        sta = [cha for sta in network for cha in sta]

        # Select only vertical channels
        channels = [cha for cha in sta if cha.code[-1] == 'Z']

        if len(channels) == 0:
            self._warn("Can't find channel for station %s" % station)
            return network.code, None

        if len(channels) > 1:
            # Choose first location code
            location_codes = [channel.location_code for channel in channels]
            location_code = sorted(location_codes)[0]
            channels = [channel for channel in channels
                        if channel.location_code == location_code]

            codes = [channel.code for channel in channels]
            code = None
            if len(codes) == 1:
                code = codes[0]
            else:
                # Choose channel code by priority
                # HHZ > EHZ > ELZ > BHZ > LHZ > SHZ
                priority = ['HHZ', 'EHZ', 'ELZ', 'BHZ', 'LHZ', 'SHZ']
                code = next((code for code in priority if code in codes), None)

            if code is None:
                if len(network) > 1:
                    self._warn('Several stations %s or location code have '
                               'been found in inventory' % station)
                else:
                    self._warn('Several channels have been found for '
                               'station %s' % station)

                return network.code, None

            channel = next((channel for channel in channels
                           if channel.code == code))

            self._warn('Several stations, location codes or channels have '
                       'been found, choose %s.%s.%s.%s' %
                       (network.code, station, channel.location_code,
                        channel.code))
            return network.code, channel

        return network.code, channels[0]

    def _parse_event(self, first_line):
        """
        Parse an event.

        :type first_line: str
        :param first_line: First line of an event block, which contains
            the event id.
        :rtype: :class:`~obspy.core.event.event.Event`
        :return: The parsed event or None.
        """
        event_id = first_line[5:].strip()
        # Skip event without id
        if not event_id:
            self._warn('Missing event id')
            return None

        event = Event()

        origin, origin_res_id = self._parse_origin(event)
        # Skip event without origin
        if not origin:
            return None

        line = self._skip_empty_lines()

        self._parse_region_name(line, event)
        self._parse_arrivals(event, origin, origin_res_id)

        # Origin ResourceIdentifier should be set at the end, when
        # Arrivals are already set.
        origin.resource_id = origin_res_id
        event.origins.append(origin)

        event.preferred_origin_id = origin.resource_id.id

        # Must be done after the origin parsing
        event.creation_info = self._get_creation_info()

        public_id = "event/%s" % event_id
        event.resource_id = self._get_res_id(public_id)

        event.scope_resource_ids()

        return event

    def _parse_origin(self, event):
        """
        Parse an origin.

        :type event: :class:`~obspy.core.event.event.Event`
        :param event: Event of the origin.
        :rtype: :class:`~obspy.core.event.origin.Origin`,
            :class:`~obspy.core.event.resourceid.ResourceIdentifier`
        :returns: Parsed origin or None, resource identifier of the
            origin.
        """
        # Store magnitudes in a list to keep their positions.
        # Needed for the magnitude errors.
        magnitudes = []

        # Skip 2 lines of header
        next(self.lines)
        next(self.lines)

        line = self._skip_empty_lines()

        origin, origin_res_id = \
            self._parse_first_line_origin(line, event, magnitudes)

        # If some crucial information can't be parsed, return None
        if not origin:
            return None, None

        line = self._skip_empty_lines()

        # File from ldg can have author on several lines
        while re.match(r'\s{105,}\w+\s*', line):
            fields = self.fields['line_1']
            origin.creation_info.author += line[fields['author']].strip()
            line = self._skip_empty_lines()

        self._parse_second_line_origin(line, event, origin, magnitudes)

        # Magnitude resource_id can only be set after that magnitude
        # errors are set.
        for i in range(3):
            magnitude = magnitudes[i]
            if magnitude is not None:
                public_id = "magnitude/%s" % i
                mag_res_id = \
                    self._get_res_id(public_id, parent_res_id=origin_res_id)
                magnitude.resource_id = mag_res_id

                if event.preferred_magnitude_id is None:
                    event.preferred_magnitude_id = magnitude.resource_id.id

        return origin, origin_res_id

    def _parse_first_line_origin(self, line, event, magnitudes):
        """
        Parse the first line of origin data.

        :type line: str
        :param line: Line to parse.
        :type event: :class:`~obspy.core.event.event.Event`
        :param event: Event of the origin.
        :type magnitudes: list of
            :class:`~obspy.core.event.magnitude.Magnitude`
        :param magnitudes: Store magnitudes in a list to keep
            their positions.
        :rtype: :class:`~obspy.core.event.origin.Origin`,
            :class:`~obspy.core.event.resourceid.ResourceIdentifier`
        :returns: Parsed origin or None, resource identifier of the
            origin.
        """
        magnitude_types = []
        magnitude_values = []
        magnitude_station_counts = []

        fields = self.fields['line_1']

        time_origin = line[fields['time']].strip()
        time_fixed_flag = line[fields['time_fixf']].strip()
        latitude = line[fields['lat']].strip()
        longitude = line[fields['lon']].strip()
        epicenter_fixed_flag = line[fields['epicenter_fixf']].strip()
        depth = line[fields['depth']].strip()
        depth_fixed_flag = line[fields['depth_fixf']].strip()
        phase_count = line[fields['n_def']].strip()
        station_count = line[fields['n_sta']].strip()
        azimuthal_gap = line[fields['gap']].strip()
        magnitude_types.append(line[fields['mag_type_1']].strip())
        magnitude_values.append(line[fields['mag_1']].strip())
        magnitude_station_counts.append(line[fields['mag_n_sta_1']].strip())
        magnitude_types.append(line[fields['mag_type_2']].strip())
        magnitude_values.append(line[fields['mag_2']].strip())
        magnitude_station_counts.append(line[fields['mag_n_sta_2']].strip())
        magnitude_types.append(line[fields['mag_type_3']].strip())
        magnitude_values.append(line[fields['mag_3']].strip())
        magnitude_station_counts.append(line[fields['mag_n_sta_3']].strip())
        author = line[fields['author']].strip()
        origin_id = line[fields['id']].strip()

        origin = Origin()
        origin.quality = OriginQuality()

        try:
            origin.time = UTCDateTime(time_origin.replace('/', '-'))
            origin.latitude = float(latitude)
            origin.longitude = float(longitude)
        except (TypeError, ValueError):
            self._warn('Missing origin data, skipping event')
            return None, None

        origin.time_fixed = time_fixed_flag.lower() == 'f'
        origin.epicenter_fixed = epicenter_fixed_flag.lower() == 'f'

        try:
            # Convert value from km to m
            origin.depth = float(depth) * 1000
        except ValueError:
            pass
        try:
            origin.depth_type = DEPTH_TYPES[depth_fixed_flag]
        except KeyError:
            origin.depth_type = OriginDepthType('from location')
        try:
            origin.quality.used_phase_count = int(phase_count)
            origin.quality.associated_phase_count = int(phase_count)
        except ValueError:
            pass
        try:
            origin.quality.used_station_count = int(station_count)
            origin.quality.associated_station_count = int(station_count)
        except ValueError:
            pass
        try:
            origin.quality.azimuthal_gap = float(azimuthal_gap)
        except ValueError:
            pass

        self.author = author
        origin.creation_info = self._get_creation_info()

        public_id = "origin/%s" % origin_id
        origin_res_id = self._get_res_id(public_id)

        for i in range(3):
            try:
                magnitude = Magnitude()
                magnitude.creation_info = self._get_creation_info()
                magnitude.magnitude_type = magnitude_types[i]
                magnitude.mag = float(magnitude_values[i])
                magnitude.station_count = int(magnitude_station_counts[i])
                magnitude.origin_id = origin_res_id
                magnitudes.append(magnitude)
                event.magnitudes.append(magnitude)
            except ValueError:
                # Magnitude can be empty but we need to keep the
                # position between mag1, mag2 or mag3.
                magnitudes.append(None)

        return origin, origin_res_id

    def _find_magnitude_by_type(self, event, origin_res_id, magnitude_type):
        for mag in event.magnitudes:
            if mag.origin_id == origin_res_id \
                    and mag.magnitude_type == magnitude_type:
                return mag

    def _parse_second_line_origin(self, line, event, origin, magnitudes):
        magnitude_errors = []

        fields = self.fields['line_2']

        standard_error = line[fields['rms']].strip()
        time_uncertainty = line[fields['ot_error']].strip()
        max_horizontal_uncertainty = line[fields['s_major']].strip()
        min_horizontal_uncertainty = line[fields['s_minor']].strip()
        azimuth_max_horizontal_uncertainty = line[fields['az']].strip()
        depth_uncertainty = line[fields['depth_err']].strip()
        min_distance = line[fields['min_dist']].strip()
        max_distance = line[fields['max_dist']].strip()
        magnitude_errors.append(line[fields['mag_err_1']].strip())
        magnitude_errors.append(line[fields['mag_err_2']].strip())
        magnitude_errors.append(line[fields['mag_err_3']].strip())
        analysis_type = line[fields['antype']].strip().lower()
        location_method = line[fields['loctype']].strip().lower()
        event_type = line[fields['evtype']].strip().lower()

        try:
            origin.quality.standard_error = float(standard_error)
        except ValueError:
            pass

        try:
            origin.time_errors.uncertainty = float(time_uncertainty)
        except ValueError:
            pass

        try:
            uncertainty = OriginUncertainty()
            # Convert values from km to m
            min_value = float(min_horizontal_uncertainty) * 1000
            max_value = float(max_horizontal_uncertainty) * 1000
            azimuth_value = float(azimuth_max_horizontal_uncertainty)
            description = OriginUncertaintyDescription('uncertainty ellipse')

            uncertainty.min_horizontal_uncertainty = min_value
            uncertainty.max_horizontal_uncertainty = max_value
            uncertainty.azimuth_max_horizontal_uncertainty = azimuth_value
            uncertainty.preferred_description = description
            origin.origin_uncertainty = uncertainty
        except ValueError:
            pass

        try:
            # Convert value from km to m
            origin.depth_errors.uncertainty = float(depth_uncertainty) * 1000
        except ValueError:
            pass

        try:
            origin.quality.minimum_distance = float(min_distance)
            origin.quality.maximum_distance = float(max_distance)
        except ValueError:
            self._warn('Missing minimum/maximum distance')

        for i in range(2):
            try:
                mag_errors = magnitudes[i].mag_errors
                mag_errors.uncertainty = float(magnitude_errors[i])
            except (AttributeError, ValueError):
                pass

        # No match for 'g' (guess)
        # We map 'g' to 'manual' and create a comment for origin
        try:
            origin.evaluation_mode = EVALUATION_MODES[analysis_type]

            if analysis_type == 'g':
                # comment: 'GSE2.0:antype=g'
                text = 'GSE2.0:antype=g'
                comment = self._comment(text)
                origin.comments.append(comment)
        except KeyError:
            self._warn('Wrong analysis type')

        if location_method not in LOCATION_METHODS.keys():
            location_method = 'o'
        method = LOCATION_METHODS[location_method]
        method_id = "method/%s" % method
        origin.method_id = self._get_res_id(method_id)

        if event_type not in EVENT_TYPES.keys():
            event_type = 'uk'
            self._warn('Wrong or unknown event type')
        event_data = EVENT_TYPES[event_type]
        event.event_type_certainty, event.event_type = event_data

        # comment: 'GSE2.0:evtype=<evtype>'
        if event_type:
            text = 'GSE2.0:evtype=%s' % event_type
            comment = self._comment(text)
            event.comments.append(comment)

    def _parse_region_name(self, line, event):
        event_description = EventDescription()
        event_description.text = line.strip()
        event_description.type = EventDescriptionType('region name')
        event.event_descriptions.append(event_description)

    def _parse_arrivals(self, event, origin, origin_res_id):
        # Skip header of arrivals
        next(self.lines)

        # Stop the loop after 2 empty lines (according to the standard).
        previous_line_empty = False

        for line in self.lines:
            line_empty = not line or line.isspace()

            if not self.event_point_separator:
                # Event are separated by two empty lines
                if line_empty and previous_line_empty:
                    break
            else:
                # Event are separated by '.'
                if line.startswith('.'):
                    break

            previous_line_empty = line_empty

            if line_empty:
                # Skip empty lines when the loop should be stopped by
                # point
                continue

            magnitude_types = []
            magnitude_values = []

            fields = self.fields['arrival']

            station = line[fields['sta']].strip()
            distance = line[fields['dist']].strip()
            event_azimuth = line[fields['ev_az']].strip()
            evaluation_mode = line[fields['picktype']].strip()
            direction = line[fields['direction']].strip()
            onset = line[fields['detchar']].strip()
            phase = line[fields['phase']].strip()
            time = line[fields['time']].strip().replace('/', '-')
            time_residual = line[fields['t_res']].strip()
            arrival_azimuth = line[fields['azim']].strip()
            azimuth_residual = line[fields['az_res']].strip()
            slowness = line[fields['slow']].strip()
            slowness_residual = line[fields['s_res']].strip()
            time_defining_flag = line[fields['t_def']].strip()
            azimuth_defining_flag = line[fields['a_def']].strip()
            slowness_defining_flag = line[fields['s_def']].strip()
            snr = line[fields['snr']].strip()
            amplitude_value = line[fields['amp']].strip()
            period = line[fields['per']].strip()
            magnitude_types.append(line[fields['mag_type_1']].strip())
            magnitude_values.append(line[fields['mag_1']].strip())
            magnitude_types.append(line[fields['mag_type_2']].strip())
            magnitude_values.append(line[fields['mag_2']].strip())
            line_id = line[fields['id']].strip()

            # Don't take pick and arrival with wrong time residual
            if '*' in time_residual:
                continue

            try:
                pick = Pick()
                pick.creation_info = self._get_creation_info()
                pick.waveform_id = WaveformStreamID()
                pick.waveform_id.station_code = station
                pick.time = UTCDateTime(time)

                network_code = self.default_network_code
                location_code = self.default_location_code
                channel_code = self.default_channel_code

                try:
                    network_code, channel = self._get_channel(station,
                                                              pick.time)
                    if channel:
                        channel_code = channel.code
                        location_code = channel.location_code
                except TypeError:
                    pass

                pick.waveform_id.network_code = network_code
                pick.waveform_id.channel_code = channel_code
                if location_code:
                    pick.waveform_id.location_code = location_code

                try:
                    ev_mode = EVALUATION_MODES[evaluation_mode]
                    pick.evaluation_mode = ev_mode
                except KeyError:
                    pass
                try:
                    pick.polarity = PICK_POLARITIES[direction]
                except KeyError:
                    pass
                try:
                    pick.onset = PICK_ONSETS[onset]
                except KeyError:
                    pass
                pick.phase_hint = phase
                try:
                    pick.backazimuth = float(arrival_azimuth)
                except ValueError:
                    pass
                try:
                    pick.horizontal_slowness = float(slowness)
                except ValueError:
                    pass

                public_id = "pick/%s" % line_id
                pick.resource_id = self._get_res_id(public_id)
                event.picks.append(pick)
            except (TypeError, ValueError, AttributeError):
                # Can't parse pick, skip arrival and amplitude parsing
                continue

            arrival = Arrival()
            arrival.creation_info = self._get_creation_info()

            try:
                arrival.pick_id = pick.resource_id.id
            except AttributeError:
                pass
            arrival.phase = phase
            try:
                arrival.azimuth = float(event_azimuth)
            except ValueError:
                pass
            try:
                arrival.distance = float(distance)
            except ValueError:
                pass
            try:
                arrival.time_residual = float(time_residual)
            except ValueError:
                pass
            try:
                arrival.backazimuth_residual = float(azimuth_residual)
            except ValueError:
                pass
            try:
                arrival.horizontal_slowness_residual = float(slowness_residual)
            except ValueError:
                pass

            if time_defining_flag == 'T':
                arrival.time_weight = 1

            if azimuth_defining_flag == 'A':
                arrival.backazimuth_weight = 1

            if slowness_defining_flag == 'S':
                arrival.horizontal_slowness_weight = 1

            public_id = "arrival/%s" % line_id
            arrival.resource_id = self._get_res_id(public_id,
                                                   parent_res_id=origin_res_id)
            origin.arrivals.append(arrival)

            try:
                amplitude = Amplitude()
                amplitude.creation_info = self._get_creation_info()
                amplitude.generic_amplitude = float(amplitude_value)
                try:
                    amplitude.pick_id = pick.resource_id
                    amplitude.waveform_id = pick.waveform_id
                except AttributeError:
                    pass
                try:
                    amplitude.period = float(period)
                except ValueError:
                    pass
                try:
                    amplitude.snr = float(snr)
                except ValueError:
                    pass

                for i in [0, 1]:
                    if magnitude_types[i] and not magnitude_types[i].isspace():
                        amplitude.magnitude_hint = magnitude_types[i]

                public_id = "amplitude/%s" % line_id
                amplitude.resource_id = self._get_res_id(public_id)
                event.amplitudes.append(amplitude)

                for i in [0, 1]:
                    sta_mag = StationMagnitude()
                    sta_mag.creation_info = self._get_creation_info()
                    sta_mag.origin_id = origin_res_id
                    sta_mag.amplitude_id = amplitude.resource_id
                    sta_mag.station_magnitude_type = magnitude_types[i]
                    sta_mag.mag = magnitude_values[i]
                    sta_mag.waveform_id = pick.waveform_id
                    public_id = "magnitude/station/%s/%s" % (line_id, i)
                    sta_mag.resource_id = self._get_res_id(public_id)
                    event.station_magnitudes.append(sta_mag)

                    # Associate station mag with network mag of same type
                    mag = self._find_magnitude_by_type(event, origin_res_id,
                                                       magnitude_types[i])
                    if mag:
                        contrib = StationMagnitudeContribution()
                        contrib.station_magnitude_id = sta_mag.resource_id
                        contrib.weight = 1.0
                        mag.station_magnitude_contributions.append(contrib)
            except ValueError:
                pass

    def _deserialize(self):
        catalog = Catalog()
        catalog.description = 'Created from GSE2 format'
        catalog.creation_info = self._get_creation_info()

        # Flag used to ignore line which aren't in a BEGIN-STOP block
        begin_block = False
        # Flag used to ignore line which aren't in a BULLETIN block
        bulletin_block = False

        try:
            for line in self.lines:
                if line.startswith('BEGIN'):
                    if begin_block:
                        # 2 BEGIN without STOP
                        message = self._add_line_nb('Missing STOP tag')
                        raise GSE2BulletinSyntaxError(message)
                    else:
                        # Enter a BEGIN block
                        begin_block = True

                    self._check_header(line)
                elif line.startswith('STOP'):
                    if begin_block:
                        # Exit a BEGIN-STOP block
                        begin_block = False
                    else:
                        # STOP without BEGIN
                        message = self._add_line_nb('Missing BEGIN tag')
                        raise GSE2BulletinSyntaxError(message)
                elif line.startswith('DATA_TYPE'):
                    bulletin_block = line[10:18] == 'BULLETIN'

                if not begin_block or not bulletin_block:
                    # Not in a BEGIN-STOP block, nor a DATA_TYPE BULLETIN
                    # block.
                    continue

                # If a "Reviewed Event Bulletin" or "Reviewed Bulletin"
                # line exists, put it in comment
                if 'Reviewed Event Bulletin' in line \
                        or 'Reviewed Bulletin' in line:
                    comment = self._comment(line.strip())
                    if comment.text:
                        catalog.comments.append(comment)
                # Detect start of an event
                elif line.startswith('EVENT'):
                    event = self._parse_event(line)
                    if event:
                        catalog.append(event)

        except StopIteration:
            message = self._add_line_nb('Unexpected EOF while parsing')
            raise GSE2BulletinSyntaxError(message)
        except Exception:
            self._warn('Unexpected error')
            raise

        if begin_block:
            # BEGIN-STOP block not closed
            text = 'Unexpected EOF while parsing, BEGIN-STOP block not closed'
            message = self._add_line_nb(text)
            raise GSE2BulletinSyntaxError(message)

        catalog.resource_id = self._get_res_id('event/evid')

        return catalog


def _read_gse2(filename, inventory=None, default_network_code='XX',
               default_location_code=None, default_channel_code=None,
               res_id_prefix='smi:local', fields=None,
               event_point_separator=False, agency=None):
    """
    Read a GSE2.0 bulletin file and returns an ObsPy Catalog object.

    .. warning::
        This function should NOT be called directly, it registers via the
        ObsPy :func:`~obspy.core.event.read_events` function, call this
        instead.

    :type filename: str
    :param filename: File or file-like object in text mode.
    :type inventory: :class:`~obspy.core.inventory.inventory.Inventory`
    :param inventory: Inventory used to retrieve network code, location code
        and channel code of stations.
    :type default_network_code: str
    :param default_network_code: Default network code used if stations
        are not found in the inventory.
    :type default_location_code: str
    :param default_location_code: Location code used if stations are
        not found in the inventory.
    :type default_channel_code: str
    :param default_channel_code: Default channel code used if stations
        are not found in the inventory.
    :type res_id_prefix: str
    :param res_id_prefix: Prefix used
        in :class:`~obspy.core.event.resourceid.ResourceIdentifier` attributes.
    :type fields: dict
    :param fields: dictionary of positions of input fields, used if input file
        is non-standard
    :type event_point_separator: bool
    :param event_point_separator: ``True`` if events are separated by
        point rather than 2 empty lines.
    :type agency: str
    :param agency: Agency that generated the file.
    :rtype: :class:`~obspy.core.event.Catalog`
    :return: An ObsPy Catalog object.

    .. rubric:: Example

    >>> from obspy import read_events
    >>> default_fields = {
    ...     'line_1': {
    ...         'time': slice(0, 21),
    ...         'time_fixf': slice(22, 23),
    ...         'lat': slice(25, 33),
    ...         'lon': slice(34, 43),
    ...         'epicenter_fixf': slice(44, 45),
    ...         'depth': slice(47, 52),
    ...         'depth_fixf': slice(53, 54),
    ...         'n_def': slice(56, 60),
    ...         'n_sta': slice(61, 65),
    ...         'gap': slice(66, 69),
    ...         'mag_type_1': slice(71, 73),
    ...         'mag_1': slice(73, 77),
    ...         'mag_n_sta_1': slice(78, 80),
    ...         'mag_type_2': slice(82, 84),
    ...         'mag_2': slice(84, 88),
    ...         'mag_n_sta_2': slice(89, 91),
    ...         'mag_type_3': slice(93, 95),
    ...         'mag_3': slice(95, 99),
    ...         'mag_n_sta_3': slice(100, 102),
    ...         'author': slice(104, 112),
    ...         'id': slice(114, 122),
    ...     },
    ...     'line_2': {
    ...         'rms': slice(5, 10),
    ...         'ot_error': slice(15, 21),
    ...         's_major': slice(25, 31),
    ...         's_minor': slice(32, 38),
    ...         'az': slice(40, 43),
    ...         'depth_err': slice(49, 54),
    ...         'min_dist': slice(56, 62),
    ...         'max_dist': slice(63, 69),
    ...         'mag_err_1': slice(74, 77),
    ...         'mag_err_2': slice(85, 88),
    ...         'mag_err_3': slice(96, 99),
    ...         'antype': slice(104, 105),
    ...         'loctype': slice(106, 107),
    ...         'evtype': slice(108, 110),
    ...     },
    ...     'arrival': {
    ...         'sta': slice(0, 5),
    ...         'dist': slice(6, 12),
    ...         'ev_az': slice(13, 18),
    ...         'picktype': slice(19, 20),
    ...         'direction': slice(20, 21),
    ...         'detchar': slice(21, 22),
    ...         'phase': slice(23, 30),
    ...         'time': slice(31, 52),
    ...         't_res': slice(53, 58),
    ...         'azim': slice(59, 64),
    ...         'az_res': slice(65, 71),
    ...         'slow': slice(72, 77),
    ...         's_res': slice(78, 83),
    ...         't_def': slice(84, 85),
    ...         'a_def': slice(85, 86),
    ...         's_def': slice(86, 87),
    ...         'snr': slice(88, 93),
    ...         'amp': slice(94, 103),
    ...         'per': slice(104, 109),
    ...         'mag_type_1': slice(110, 112),
    ...         'mag_1': slice(112, 116),
    ...         'mag_type_2': slice(117, 119),
    ...         'mag_2': slice(119, 123),
    ...         'id': slice(124, 132),
    ...     },
    ... }
    >>> # Only non-standard field indexes are required
    >>> fields = {
    ...     'line_1': {
    ...         'author': slice(105, 113),
    ...         'id': slice(114, 123),
    ...     },
    ...     'line_2': {
    ...         'az': slice(40, 46),
    ...         'antype': slice(105, 106),
    ...         'loctype': slice(107, 108),
    ...         'evtype': slice(109, 111),
    ...     },
    ...     'arrival': {
    ...         'amp': slice(94, 104),
    ...     },
    ... }
    >>> catalog = read_events('/path/to/bulletin/gse_2.0_non_standard.txt',
    ... default_network_code='FR', res_id_prefix='quakeml:abc',
    ... fields=fields, event_point_separator=True)
    >>> print(catalog)
    2 Event(s) in Catalog:
    1995-01-16T07:26:52.400000Z | +39.450,  +20.440 | 3.6  mb | manual
    1995-01-16T07:27:07.300000Z | +50.772, -129.760 | 1.2  Ml | manual
    """
    return Unpickler(inventory, default_network_code, default_location_code,
                     default_channel_code, res_id_prefix, fields,
                     event_point_separator, agency).load(filename)


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
