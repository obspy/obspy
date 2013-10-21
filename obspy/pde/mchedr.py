# -*- coding: utf-8 -*-
"""
NEIC PDE mchedr (machine-readable Earthquake Data Report) read support.

Only supports file format revision of February 24, 2004.

.. seealso:: http://earthquake.usgs.gov/research/data/pde.php

:copyright:
    The ObsPy Development Team (devs@obspy.org), Claudio Satriano
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""

from obspy.core.event import Catalog, Event, Origin, CreationInfo, Magnitude, \
    EventDescription, OriginUncertainty, OriginQuality, \
    ConfidenceEllipsoid, StationMagnitude, Comment, WaveformStreamID, Pick, \
    Arrival, FocalMechanism, MomentTensor, NodalPlanes, \
    PrincipalAxes, Axis, NodalPlane, Tensor, DataUsed, \
    ResourceIdentifier, Amplitude
from obspy.core.utcdatetime import UTCDateTime
from obspy.core.util import getExampleFile
from datetime import timedelta
import os
import StringIO
import csv
import numpy as np


# ResourceIdentifier prefix used throughout this code
res_id_prefix = 'smi:gov.usgs.earthquake'


def isMchedr(filename):
    """
    Checks whether a file format is mchedr
    (machine-readable Earthquake Data Report).

    :type filename: str
    :param filename: Name of the mchedr file to be checked.
    :rtype: bool
    :return: ``True`` if mchedr file.

    .. rubric:: Example

    >>> isMchedr('/path/to/mchedr.dat')  # doctest: +SKIP
    True
    """
    if not isinstance(filename, basestring):
        return False
    with open(filename, 'r') as fh:
        for line in fh.readlines():
            # skip blanck lines at beginnning, if any
            if line.strip() == '':
                continue
            # first record has to be 'HY':
            if line[0:2] == 'HY':
                return True
            else:
                return False


class Unpickler(object):
    """
    De-serializes a mchedr string into an ObsPy Catalog object.
    """
    def __init__(self):
        self.FE_regions = None

    def load(self, filename):
        """
        Reads mchedr file into ObsPy catalog object.

        :type file: str
        :param file: File name to read.
        :rtype: :class:`~obspy.core.event.Catalog`
        :returns: ObsPy Catalog object.
        """
        if not isinstance(filename, basestring):
            raise TypeError('File name must be a string.')
        self.filename = filename
        self.fh = open(filename, 'r')
        return self._deserialize()

    def loads(self, string):
        """
        Parses mchedr string into ObsPy catalog object.

        :type string: str
        :param string: QuakeML string to parse.
        :rtype: :class:`~obspy.core.event.Catalog`
        :returns: ObsPy Catalog object.
        """
        self.fh = StringIO.StringIO(string)
        self.filename = None
        return self._deserialize()

    def _int(self, string):
        try:
            return int(string)
        except ValueError:
            return None

    def _intUnused(self, string):
        val = self._int(string)
        if val < 0:
            val = None
        return val

    def _intZero(self, string):
        val = self._int(string)
        if val is None:
            val = 0
        return val

    def _float(self, string):
        try:
            return float(string)
        except ValueError:
            return None

    def _floatUnused(self, string):
        val = self._float(string)
        if val < 0:
            val = None
        return val

    def _floatWithFormat(self, string, format_string, scale=1):
        ndigits, ndec = map(int, format_string.split('.'))
        nint = ndigits-ndec
        val = self._float(string[0:nint] + '.' + string[nint:nint+ndec])
        if val is not None:
            val *= scale
        return val

    def _coordinateSign(self, type):
        if type == 'S' or type == 'W':
            return -1
        else:
            return 1

    def _tensorCodeSign(self, code):
        """
        Converts tensor from 'x,y,z' system to 'r,t,p'
        and translates 'f' code to 'p'
        """
        if code == 'xx':
            return 'tt', 1
        if code == 'yy':
            return 'pp', 1
        if code == 'zz':
            return 'rr', 1
        if code == 'xy':
            return 'tp', -1
        if code == 'xz':
            return 'rt', 1
        if code == 'yz':
            return 'rp', -1
        if code == 'ff':
            return 'pp', 1
        if code == 'rf':
            return 'rp', 1
        if code == 'tf':
            return 'tp', 1
        return code, 1

    def _tensorStore(self, tensor, code, value, error):
        code, sign = self._tensorCodeSign(code)
        if code == 'rr':
            tensor.m_rr = value * sign
            tensor.m_rr_errors['uncertainty'] = error
        if code == 'tt':
            tensor.m_tt = value * sign
            tensor.m_tt_errors['uncertainty'] = error
        if code == 'pp':
            tensor.m_pp = value * sign
            tensor.m_pp_errors['uncertainty'] = error
        if code == 'rt':
            tensor.m_rt = value * sign
            tensor.m_rt_errors['uncertainty'] = error
        if code == 'rp':
            tensor.m_rp = value * sign
            tensor.m_rp_errors['uncertainty'] = error
        if code == 'tp':
            tensor.m_tp = value * sign
            tensor.m_tp_errors['uncertainty'] = error

    def _decode_FE_region_number(self, number):
        """
        Converts Flinn-Engdahl region number to string.
        """
        #FIXME: this should be checked against #451 for redundancy
        if not isinstance(number, int):
            number = int(number)
        if self.FE_regions is None:
            FE_filename = os.path.join(os.path.dirname(__file__),
                                       'data',
                                       'Flinn-Engdahl.csv')
            with open(FE_filename, 'rb') as csvfile:
                FE_csv = csv.reader(csvfile, delimiter=';',
                                    quotechar='#', skipinitialspace=True)
                self.FE_regions =\
                    {int(row[0]): row[1] for row in FE_csv if len(row) > 1}
        return self.FE_regions[number]

    def _to_rad(self, degrees):
        radians = np.pi * degrees / 180
        return radians

    def _to_deg(self, radians):
        degrees = 180 * radians / np.pi
        return degrees

    def _spherical_to_cartesian(self, (lenght, azimuth, plunge)):
        plunge_rad = self._to_rad(plunge)
        azimuth_rad = self._to_rad(azimuth)
        x = lenght * np.sin(plunge_rad) * np.cos(azimuth_rad)
        y = lenght * np.sin(plunge_rad) * np.sin(azimuth_rad)
        z = lenght * np.cos(plunge_rad)
        return (x, y, z)

    def _angle_between(self, u1, u2):
        """
        Returns the angle in degrees between unit vectors 'u1' and 'u2':
        Source: http://stackoverflow.com/questions/2827393/
                       angles-between-two-n-dimensional-vectors-in-python
        """
        angle = np.arccos(np.dot(u1, u2))
        if np.isnan(angle):
            if (u1 == u2).all():
                angle = 0.0
            else:
                angle = np.pi
        return self._to_deg(angle)

    def _parseRecordHY(self, line):
        """
        Parses the 'hypocenter' record HY
        """
        date = line[2:10]
        time = line[11:20]
        #unused: location_quality = line[20]
        latitude = self._float(line[21:27])
        lat_type = line[27]
        longitude = self._float(line[29:36])
        lon_type = line[36]
        depth = self._float(line[38:43])
        #unused: depth_quality = line[43]
        standard_dev = self._float(line[44:48])
        station_number = self._int(line[48:51])
        #unused: version_flag = line[51]
        FE_region_number = line[52:55]
        FE_region_name = self._decode_FE_region_number(FE_region_number)
        source_code = line[55:60].strip()

        event = Event()
        event.resource_id =\
            ResourceIdentifier(
                resource_id='/'.join((res_id_prefix, date+time)))
        description = EventDescription(
            type='region name',
            text=FE_region_name)
        event.event_descriptions.append(description)
        description = EventDescription(
            type='Flinn-Engdahl region',
            text=FE_region_number)
        event.event_descriptions.append(description)
        origin = Origin()
        origin.resource_id = ResourceIdentifier(prefix=res_id_prefix)
        origin.creation_info = CreationInfo()
        if source_code:
            origin.creation_info.agency_id = source_code
        else:
            origin.creation_info.agency_id = 'USGS-NEIC'
        origin.earth_model_id =\
            ResourceIdentifier(
                resource_id='/'.join((res_id_prefix, 'emid=AK135')))
        origin.time = UTCDateTime(date+time)
        origin.latitude = latitude * self._coordinateSign(lat_type)
        origin.longitude = longitude * self._coordinateSign(lon_type)
        origin.depth = depth
        origin.depth_type = 'from location'
        origin.quality = OriginQuality()
        origin.quality.associated_station_count = station_number
        origin.quality.standard_error = standard_dev
        #associated_phase_count can be incremented in records 'P ' and 'S '
        origin.quality.associated_phase_count = 0
        #depth_phase_count can be incremented in record 'S '
        origin.quality.depth_phase_count = 0
        origin.type = 'hypocenter'
        origin.region = FE_region_name
        event.origins.append(origin)
        return event

    def _parseRecordE(self, line, event):
        """
        Parses the 'error and magnitude' record E
        """
        orig_time_stderr = self._float(line[2:7])
        latitude_stderr = self._float(line[8:14])
        longitude_stderr = self._float(line[15:21])
        depth_stderr = self._float(line[22:27])
        mb_mag = self._float(line[28:31])
        mb_nsta = self._int(line[32:35])
        Ms_mag = self._float(line[36:39])
        Ms_nsta = self._int(line[39:42])
        mag1 = self._float(line[42:45])
        mag1_type = line[45:47]
        mag1_source_code = line[47:51].strip()
        mag2 = self._float(line[51:54])
        mag2_type = line[54:56]
        mag2_source_code = line[56:60].strip()

        origin = event.origins[0]
        origin.time_errors['uncertainty'] = orig_time_stderr
        origin.latitude_errors['uncertainty'] = latitude_stderr
        origin.longitude_errors['uncertainty'] = longitude_stderr
        origin.depth_errors['uncertainty'] = depth_stderr
        if mb_mag is not None:
            mag = Magnitude()
            mag.resource_id = ResourceIdentifier(prefix=res_id_prefix)
            mag.creation_info = CreationInfo(agency_id='USGS-NEIC')
            mag.mag = mb_mag
            mag.magnitude_type = 'Mb'
            mag.station_count = mb_nsta
            mag.origin_id = origin.resource_id
            event.magnitudes.append(mag)
        if Ms_mag is not None:
            mag = Magnitude()
            mag.resource_id = ResourceIdentifier(prefix=res_id_prefix)
            mag.creation_info = CreationInfo(agency_id='USGS-NEIC')
            mag.mag = Ms_mag
            mag.magnitude_type = 'Ms'
            mag.station_count = Ms_nsta
            mag.origin_id = origin.resource_id
            event.magnitudes.append(mag)
        if mag1 is not None:
            mag = Magnitude()
            mag.resource_id = ResourceIdentifier(prefix=res_id_prefix)
            mag.creation_info = CreationInfo(agency_id=mag1_source_code)
            mag.mag = mag1
            mag.magnitude_type = mag1_type
            mag.origin_id = origin.resource_id
            event.magnitudes.append(mag)
        if mag2 is not None:
            mag = Magnitude()
            mag.resource_id = ResourceIdentifier(prefix=res_id_prefix)
            mag.creation_info = CreationInfo(agency_id=mag2_source_code)
            mag.mag = mag2
            mag.magnitude_type = mag2_type
            mag.origin_id = origin.resource_id
            event.magnitudes.append(mag)

    def _parseRecordL(self, line, event):
        """
        Parses the '90 percent error ellipse' record L
        """
        origin = event.origins[0]
        semi_major_axis_azimuth = self._float(line[2:8])
        if semi_major_axis_azimuth is None:
            return
        semi_major_axis_plunge = self._float(line[8:13])
        semi_major_axis_length = self._float(line[13:21])
        intermediate_axis_azimuth = self._float(line[21:27])
        intermediate_axis_plunge = self._float(line[27:32])
        # This is called "intermediate_axis_length",
        # but it is definitively a "semi_intermediate_axis_length",
        # since in most cases:
        #   (intermediate_axis_length / 2) < semi_minor_axis_lenght
        intermediate_axis_length = self._float(line[32:40])
        semi_minor_axis_azimuth = self._float(line[40:46])
        semi_minor_axis_plunge = self._float(line[46:51])
        semi_minor_axis_length = self._float(line[51:59])

        if (semi_minor_axis_azimuth ==
           semi_minor_axis_plunge ==
           semi_minor_axis_length == 0):
            semi_minor_axis_azimuth = intermediate_axis_azimuth
            semi_minor_axis_plunge = intermediate_axis_plunge
            semi_minor_axis_length = intermediate_axis_length
            origin.depth_type = 'operator assigned'

        #FIXME: The following code needs to be double-checked!
        semi_major_axis_unit_vect =\
            self._spherical_to_cartesian((
                1,
                semi_major_axis_azimuth,
                semi_major_axis_plunge
                ))
        semi_minor_axis_unit_vect =\
            self._spherical_to_cartesian((
                1,
                semi_minor_axis_azimuth,
                semi_minor_axis_plunge
                ))
        major_axis_rotation = self._angle_between(
            semi_major_axis_unit_vect,
            semi_minor_axis_unit_vect
            )

        origin.origin_uncertainty = OriginUncertainty()
        origin.origin_uncertainty.preferred_description =\
            'confidence ellipsoid'
        origin.origin_uncertainty.confidence_level = 90
        confidence_ellipsoid = ConfidenceEllipsoid()
        confidence_ellipsoid.semi_major_axis_length = semi_major_axis_length
        confidence_ellipsoid.semi_minor_axis_length = semi_minor_axis_length
        confidence_ellipsoid.semi_intermediate_axis_length =\
            intermediate_axis_length
        confidence_ellipsoid.major_axis_plunge = semi_major_axis_plunge
        confidence_ellipsoid.major_axis_azimuth = semi_major_axis_azimuth
        confidence_ellipsoid.major_axis_rotation = major_axis_rotation
        origin.origin_uncertainty.confidence_ellipsoid = confidence_ellipsoid

    def _parseRecordA(self, line, event):
        """
        Parses the 'additional parameters' record A
        """
        origin = event.origins[0]
        phase_number = self._int(line[2:6])
        station_number = self._int(line[7:10])
        gap = self._float(line[10:15])
        #unused: official_mag = line[16:19]
        #unused: official_mag_type = line[19:21]
        #unused: official_mag_source_code = line[21:26]
        #unused: deaths_field_descriptor = line[27]
        #unused: dead_people = line[28:35]
        #unused: injuries_field_descriptor = line[35]
        #unused: injured_people = line[36:43]
        #unused: damaged_buildings_descriptor = line[43]
        #unused: damaged_buildings = line[44:51]
        #unused: event_quality_flag = line[51]

        origin.quality.used_phase_count = phase_number
        origin.quality.used_station_count = station_number
        origin.quality.azimuthal_gap = gap

    def _parseRecordC(self, line, event):
        """
        Parses the 'general comment' record C
        """
        try:
            comment = event.comments[0]
            comment.text += line[2:60]
        except IndexError:
            comment = Comment()
            comment.resource_id = ResourceIdentifier(prefix=res_id_prefix)
            event.comments.append(comment)
            comment.text = line[2:60]

    def _parseRecordAH(self, line, event):
        """
        Parses the 'additional hypocenter' record AH
        """
        date = line[2:10]
        time = line[11:20]
        #unused: hypocenter_quality = line[20]
        latitude = self._float(line[21:27])
        lat_type = line[27]
        longitude = self._float(line[29:36])
        lon_type = line[36]
        #unused: preliminary_flag = line[37]
        depth = self._float(line[38:43])
        #unused: depth_quality = line[43]
        standard_dev = self._floatUnused(line[44:48])
        station_number = self._intUnused(line[48:51])
        phase_number = self._intUnused(line[51:55])
        source_code = line[56:60].strip()

        origin = Origin()
        origin.resource_id = ResourceIdentifier(prefix=res_id_prefix)
        origin.creation_info = CreationInfo(agency_id=source_code)
        origin.time = UTCDateTime(date+time)
        origin.latitude = latitude * self._coordinateSign(lat_type)
        origin.longitude = longitude * self._coordinateSign(lon_type)
        origin.depth = depth
        origin.depth_type = 'from location'
        origin.quality = OriginQuality()
        origin.quality.standard_error = standard_dev
        origin.quality.used_station_count = station_number
        origin.quality.used_phase_count = phase_number
        origin.type = 'hypocenter'
        event.origins.append(origin)

    def _parseRecordAE(self, line, event):
        """
        Parses the 'additional hypocenter error and magnitude record' AE
        """
        orig_time_stderr = self._floatUnused(line[2:7])
        latitude_stderr = self._floatUnused(line[8:14])
        longitude_stderr = self._floatUnused(line[15:21])
        depth_stderr = self._floatUnused(line[22:27])
        gap = self._floatUnused(line[28:33])
        mag1 = self._float(line[33:36])
        mag1_type = line[36:38]
        mag2 = self._float(line[43:46])
        mag2_type = line[46:48]

        #this record is to be associated to the latest origin
        origin = event.origins[-1]
        origin.time_errors['uncertainty'] = orig_time_stderr
        origin.latitude_errors['uncertainty'] = latitude_stderr
        origin.longitude_errors['uncertainty'] = longitude_stderr
        origin.depth_errors['uncertainty'] = depth_stderr
        origin.quality.azimuthal_gap = gap
        if mag1 > 0:
            mag = Magnitude()
            mag.resource_id = ResourceIdentifier(prefix=res_id_prefix)
            mag.creation_info = CreationInfo(
                agency_id=origin.creation_info.agency_id)
            mag.mag = mag1
            mag.magnitude_type = mag1_type
            mag.origin_id = origin.resource_id
            event.magnitudes.append(mag)
        if mag2 > 0:
            mag = Magnitude()
            mag.resource_id = ResourceIdentifier(prefix=res_id_prefix)
            mag.creation_info = CreationInfo(
                agency_id=origin.creation_info.agency_id)
            mag.mag = mag2
            mag.magnitude_type = mag2_type
            mag.origin_id = origin.resource_id
            event.magnitudes.append(mag)

    def _parseRecordDp(self, line, event):
        """
        Parses the 'source parameter data - primary' record Dp
        """
        source_contributor = line[2:6].strip()
        computation_type = line[6]
        exponent = self._intZero(line[7])
        scale = 10**exponent
        centroid_origin_time = line[8:14] + '.' + line[14]
        orig_time_stderr = line[15:17]
        if orig_time_stderr == 'FX':
            orig_time_stderr = 'Fixed'
        else:
            orig_time_stderr =\
                self._floatWithFormat(orig_time_stderr, '2.1', scale)
        centroid_latitude = self._floatWithFormat(line[17:21], '4.2')
        lat_type = line[21]
        if centroid_latitude is not None:
            centroid_latitude *= self._coordinateSign(lat_type)
        lat_stderr = line[22:25]
        if lat_stderr == 'FX':
            lat_stderr = 'Fixed'
        else:
            lat_stderr =\
                self._floatWithFormat(lat_stderr, '3.2', scale)
        centroid_longitude = self._floatWithFormat(line[25:30], '5.2')
        lon_type = line[30]
        if centroid_longitude is not None:
            centroid_longitude *= self._coordinateSign(lon_type)
        lon_stderr = line[31:34]
        if lon_stderr == 'FX':
            lon_stderr = 'Fixed'
        else:
            lon_stderr =\
                self._floatWithFormat(lon_stderr, '3.2', scale)
        centroid_depth = self._floatWithFormat(line[34:38], '4.1')
        depth_stderr = line[38:40]
        if depth_stderr == 'FX' or depth_stderr == 'BD':
            depth_stderr = 'Fixed'
        else:
            depth_stderr =\
                self._floatWithFormat(depth_stderr, '2.1', scale)
        station_number = self._intZero(line[40:43])
        component_number = self._intZero(line[43:46])
        station_number2 = self._intZero(line[46:48])
        component_number2 = self._intZero(line[48:51])
        #unused: half_duration = self._floatWithFormat(line[51:54], '3.1')
        moment = self._floatWithFormat(line[54:56], '2.1')
        moment_stderr = self._floatWithFormat(line[56:58], '2.1')
        moment_exponent = self._int(line[58:60])
        if (moment is not None) and (moment_exponent is not None):
            moment *= 10**moment_exponent
        if (moment_stderr is not None) and (moment_exponent is not None):
            moment_stderr *= 10**moment_exponent

        #Create a new origin only if centroid time is defined:
        origin = None
        if centroid_origin_time.strip() != '.':
            origin = Origin()
            origin.resource_id = ResourceIdentifier(prefix=res_id_prefix)
            origin.creation_info =\
                CreationInfo(agency_id=source_contributor)
            date = event.origins[0].time.strftime('%Y%m%d')
            origin.time = UTCDateTime(date+centroid_origin_time)
            #Check if centroid time is on the next day:
            if origin.time < event.origins[0].time:
                origin.time += timedelta(days=1)
            origin.time_errors['uncertainty'] = orig_time_stderr
            origin.latitude = centroid_latitude
            origin.longitude = centroid_longitude
            origin.depth = centroid_depth
            if lat_stderr == 'Fixed' and lon_stderr == 'Fixed':
                origin.epicenter_fixed = True
            else:
                origin.latitude_errors['uncertainty'] = lat_stderr
                origin.longitude_errors['uncertainty'] = lon_stderr
            if depth_stderr == 'Fixed':
                origin.depth_type = 'operator assigned'
            else:
                origin.depth_type = 'from location'
                origin.depth_errors['uncertainty'] = depth_stderr
            quality = OriginQuality()
            quality.used_station_count =\
                station_number + station_number2
            quality.used_phase_count =\
                component_number + component_number2
            origin.quality = quality
            origin.type = 'centroid'
            event.origins.append(origin)
        focal_mechanism = FocalMechanism()
        focal_mechanism.resource_id = ResourceIdentifier(prefix=res_id_prefix)
        focal_mechanism.creation_info =\
            CreationInfo(agency_id=source_contributor)
        moment_tensor = MomentTensor()
        if origin is not None:
            moment_tensor.derived_origin_id = origin.resource_id
        else:
            #this is required for QuakeML validation:
            moment_tensor.derived_origin_id =\
                ResourceIdentifier(
                    resource_id='/'.join((res_id_prefix, 'no-origin'))
                    )
        for mag in event.magnitudes:
            if mag.creation_info.agency_id == source_contributor:
                moment_tensor.moment_magnitude_id = mag.resource_id
        moment_tensor.resource_id = ResourceIdentifier(prefix=res_id_prefix)
        moment_tensor.scalar_moment = moment
        moment_tensor.scalar_moment_errors['uncertainty'] =\
            moment_stderr
        data_used = DataUsed()
        data_used.station_count = station_number + station_number2
        data_used.component_count = component_number + component_number2
        if computation_type == 'C':
            focal_mechanism.method_id =\
                ResourceIdentifier(
                    resource_id='/'.join((res_id_prefix, 'methodID=CMT')))
            #CMT algorithm uses long-period body waves,
            #very-long-period surface waves and
            #intermediate period surface waves (since 2004
            #for shallow and intermediate-depth earthquakes
            # --Ekstrom et al., 2012)
            data_used.wave_type = 'combined'
        if computation_type == 'M':
            focal_mechanism.method_id =\
                ResourceIdentifier(
                    resource_id='/'.join(
                        (res_id_prefix, 'methodID=moment_tensor'))
                    )
            #FIXME: not sure which kind of data is used by
            #"moment tensor" algorithm.
            data_used.wave_type = 'unknown'
        if computation_type == 'B':
            focal_mechanism.method_id =\
                ResourceIdentifier(
                    resource_id='/'.join(
                        (res_id_prefix, 'methodID=broadband_data'))
                    )
            #FIXME: is 'combined' correct here?
            data_used.wave_type = 'combined'
        if computation_type == 'F':
            focal_mechanism.method_id =\
                ResourceIdentifier(
                    resource_id='/'.join(
                        (res_id_prefix, 'methodID=P-wave_first_motion'))
                    )
            data_used.wave_type = 'P waves'
        if computation_type == 'S':
            focal_mechanism.method_id =\
                ResourceIdentifier(
                    resource_id='/'.join(
                        (res_id_prefix, 'methodID=scalar_moment'))
                    )
            #FIXME: not sure which kind of data is used
            #for scalar moment determination.
            data_used.wave_type = 'unknown'
        moment_tensor.data_used = data_used
        focal_mechanism.moment_tensor = moment_tensor
        event.focal_mechanisms.append(focal_mechanism)
        return focal_mechanism

    def _parseRecordDt(self, line, focal_mechanism):
        """
        Parses the 'source parameter data - tensor' record Dt
        """
        tensor = Tensor()
        exponent = self._intZero(line[3:5])
        scale = 10**exponent
        code = line[6:8]
        value = self._floatWithFormat(line[8:12], '4.2', scale)
        error = self._floatWithFormat(line[12:15], '3.2', scale)
        self._tensorStore(tensor, code, value, error)
        code = line[15:17]
        value = self._floatWithFormat(line[17:21], '4.2', scale)
        error = self._floatWithFormat(line[21:24], '3.2', scale)
        self._tensorStore(tensor, code, value, error)
        code = line[24:26]
        value = self._floatWithFormat(line[26:30], '4.2', scale)
        error = self._floatWithFormat(line[30:33], '3.2', scale)
        self._tensorStore(tensor, code, value, error)
        code = line[33:35]
        value = self._floatWithFormat(line[35:39], '4.2', scale)
        error = self._floatWithFormat(line[39:42], '3.2', scale)
        self._tensorStore(tensor, code, value, error)
        code = line[42:44]
        value = self._floatWithFormat(line[44:48], '4.2', scale)
        error = self._floatWithFormat(line[48:51], '3.2', scale)
        self._tensorStore(tensor, code, value, error)
        code = line[51:53]
        value = self._floatWithFormat(line[53:57], '4.2', scale)
        error = self._floatWithFormat(line[57:60], '3.2', scale)
        self._tensorStore(tensor, code, value, error)
        focal_mechanism.moment_tensor.tensor = tensor

    def _parseRecordDa(self, line, focal_mechanism):
        """
        Parses the 'source parameter data - principal axes and
        nodal planes' record Da
        """
        exponent = self._intZero(line[3:5])
        scale = 10**exponent
        t_axis_len = self._floatWithFormat(line[5:9], '4.2', scale)
        t_axis_stderr =\
            self._floatWithFormat(line[9:12], '3.2', scale)
        t_axis_plunge = self._int(line[12:14])
        t_axis_azimuth = self._int(line[14:17])
        n_axis_len = self._floatWithFormat(line[17:21], '4.2', scale)
        n_axis_stderr =\
            self._floatWithFormat(line[21:24], '3.2', scale)
        n_axis_plunge = self._int(line[24:26])
        n_axis_azimuth = self._int(line[26:29])
        p_axis_len = self._floatWithFormat(line[29:33], '4.2', scale)
        p_axis_stderr =\
            self._floatWithFormat(line[33:36], '3.2', scale)
        p_axis_plunge = self._int(line[36:38])
        p_axis_azimuth = self._int(line[38:41])
        np1_strike = self._int(line[42:45])
        np1_dip = self._int(line[45:47])
        np1_slip = self._int(line[47:51])
        np2_strike = self._int(line[51:54])
        np2_dip = self._int(line[54:56])
        np2_slip = self._int(line[56:60])

        t_axis = Axis()
        t_axis.length = t_axis_len
        t_axis.length_errors['uncertainity'] = t_axis_stderr
        t_axis.plunge = t_axis_plunge
        t_axis.azimuth = t_axis_azimuth
        n_axis = Axis()
        n_axis.length = n_axis_len
        n_axis.length_errors['uncertainity'] = n_axis_stderr
        n_axis.plunge = n_axis_plunge
        n_axis.azimuth = n_axis_azimuth
        p_axis = Axis()
        p_axis.length = p_axis_len
        p_axis.length_errors['uncertainity'] = p_axis_stderr
        p_axis.plunge = p_axis_plunge
        p_axis.azimuth = p_axis_azimuth
        principal_axes = PrincipalAxes()
        principal_axes.t_axis = t_axis
        principal_axes.n_axis = n_axis
        principal_axes.p_axis = p_axis
        focal_mechanism.principal_axes = principal_axes
        nodal_plane_1 = NodalPlane()
        nodal_plane_1.strike = np1_strike
        nodal_plane_1.dip = np1_dip
        nodal_plane_1.rake = np1_slip
        nodal_plane_2 = NodalPlane()
        nodal_plane_2.strike = np2_strike
        nodal_plane_2.dip = np2_dip
        nodal_plane_2.rake = np2_slip
        nodal_planes = NodalPlanes()
        nodal_planes.nodal_plane_1 = nodal_plane_1
        nodal_planes.nodal_plane_2 = nodal_plane_2
        focal_mechanism.nodal_planes = nodal_planes

    def _parseRecordDc(self, line, focal_mechanism):
        """
        Parses the 'source parameter data - comment' record Dc
        """
        try:
            comment = focal_mechanism.comments[0]
            comment.text += line[2:60]
        except IndexError:
            comment = Comment()
            comment.resource_id = ResourceIdentifier(prefix=res_id_prefix)
            focal_mechanism.comments.append(comment)
            comment.text = line[2:60]

    def _parseRecordP(self, line, event):
        """
        Parses the 'primary phase record' P

        The primary phase is the first phase of the reading,
        regardless its type.
        """
        station = line[2:7].strip()
        phase = line[7:15]
        arrival_time = line[15:24]
        residual = self._float(line[25:30])
        #unused: residual_flag = line[30]
        distance = self._float(line[32:38])  # degrees
        azimuth = self._float(line[39:44])
        backazimuth = azimuth % -360 + 180
        mb_period = self._float(line[44:48])
        mb_amplitude = self._float(line[48:55])  # nanometers
        mb_magnitude = self._float(line[56:59])
        #unused: mb_usage_flag = line[59]

        origin = event.origins[0]
        pick = Pick()
        pick.resource_id = ResourceIdentifier(prefix=res_id_prefix)
        date = origin.time.strftime('%Y%m%d')
        pick.time = UTCDateTime(date + arrival_time)
        #Check if pick is on the next day:
        if pick.time < origin.time:
            pick.time += timedelta(days=1)
        pick.waveform_id = WaveformStreamID()
        pick.waveform_id.station_code = station
        #network_code is required for QuakeML validation
        pick.waveform_id.network_code = '  '
        pick.waveform_id.resource_uri =\
            ResourceIdentifier(prefix=res_id_prefix)
        pick.backazimuth = backazimuth
        onset = phase[0]
        if onset == 'e':
            pick.onset = 'emergent'
            phase = phase[1:]
        if onset == 'i':
            pick.onset = 'impulsive'
            phase = phase[1:]
        if onset == 'q':
            pick.onset = 'questionable'
            phase = phase[1:]
        pick.phase_hint = phase.strip()
        event.picks.append(pick)
        if mb_amplitude is not None:
            amplitude = Amplitude()
            amplitude.resource_id =\
                ResourceIdentifier(prefix=res_id_prefix)
            amplitude.generic_amplitude = mb_amplitude * 1E-9
            amplitude.unit = 'm'
            amplitude.period = mb_period
            amplitude.type = 'AB'
            amplitude.magnitude_hint = 'Mb'
            amplitude.pick_id = pick.resource_id
            amplitude.waveform_id = pick.waveform_id
            event.amplitudes.append(amplitude)
            station_magnitude = StationMagnitude()
            station_magnitude.resource_id =\
                ResourceIdentifier(prefix=res_id_prefix)
            station_magnitude.origin_id = origin.resource_id
            station_magnitude.mag = mb_magnitude
            #station_magnitude.mag_errors['uncertainty'] = 0.0
            station_magnitude.station_magnitude_type = 'Mb'
            station_magnitude.amplitude_id = amplitude.resource_id
            station_magnitude.waveform_id = pick.waveform_id
            station_magnitude.method_id =\
                ResourceIdentifier(resource_id='/'.join(
                    (res_id_prefix,
                     'magnitude/generic/body_wave_magnitude')
                    )
                )
            event.station_magnitudes.append(station_magnitude)
        arrival = Arrival()
        arrival.resource_id = ResourceIdentifier(prefix=res_id_prefix)
        arrival.pick_id = pick.resource_id
        arrival.phase = pick.phase_hint
        arrival.azimuth = azimuth
        arrival.distance = distance
        arrival.time_residual = residual
        arrival.earth_model_id =\
            ResourceIdentifier(
                resource_id='/'.join((res_id_prefix, 'emid=AK135')))
        origin.arrivals.append(arrival)
        origin.quality.minimum_distance = min(
            d for d in (arrival.distance, origin.quality.minimum_distance)
            if d is not None)
        origin.quality.maximum_distance =\
            max(arrival.distance, origin.quality.minimum_distance)
        origin.quality.associated_phase_count += 1
        return pick, arrival

    def _parseRecordM(self, line, event, pick):
        """
        Parses the 'surface wave record' M
        """
        #unused: Z_comp = line[7]
        Z_period = self._float(line[9:13])
        # note: according to the format documentation,
        # column 20 should be blank. However, it seems that
        # Z_amplitude includes that column
        Z_amplitude = self._float(line[13:21])  # micrometers
        #TODO: N_comp and E_comp seems to be never there
        MSZ_mag = line[49:52]
        Ms_mag = self._float(line[53:56])
        #unused: Ms_usage_flag = line[56]

        amplitude = None
        if Z_amplitude is not None:
            amplitude = Amplitude()
            amplitude.resource_id = ResourceIdentifier(prefix=res_id_prefix)
            amplitude.generic_amplitude = Z_amplitude * 1E-6
            amplitude.unit = 'm'
            amplitude.period = Z_period
            amplitude.type = 'AS'
            amplitude.magnitude_hint = 'Ms'
            amplitude.pick_id = pick.resource_id
            event.amplitudes.append(amplitude)
        if MSZ_mag is not None:
            station_magnitude = StationMagnitude()
            station_magnitude.resource_id =\
                ResourceIdentifier(prefix=res_id_prefix)
            station_magnitude.origin_id = event.origins[0].resource_id
            station_magnitude.mag = Ms_mag
            station_magnitude.station_magnitude_type = 'Ms'
            if amplitude is not None:
                station_magnitude.amplitude_id = amplitude.resource_id
            event.station_magnitudes.append(station_magnitude)

    def _parseRecordS(self, line, event, p_pick, p_arrival):
        """
        Parses the 'secondary phases' record S

        Secondary phases are following phases of the reading,
        and can be P-type or S-type.
        """
        arrivals = []
        phase = line[7:15].strip()
        arrival_time = line[15:24]
        if phase:
            arrivals.append((phase, arrival_time))
        phase = line[25:33].strip()
        arrival_time = line[33:42]
        if phase:
            arrivals.append((phase, arrival_time))
        phase = line[43:51].strip()
        arrival_time = line[51:60]
        if phase:
            arrivals.append((phase, arrival_time))

        origin = event.origins[0]
        for phase, arrival_time in arrivals:
            if phase[0:2] == 'D=':
                #unused: depth = self._float(phase[2:7])
                depth_usage_flag = phase[7]
                #FIXME: I'm not sure that 'X' actually
                #means 'used'
                if depth_usage_flag == 'X':
                    #FIXME: is this enough to say that
                    #the event is constained by depth pahses?
                    origin.depth_type =\
                        'constrained by depth phases'
                    origin.quality.depth_phase_count += 1
            else:
                pick = Pick()
                pick.resource_id = ResourceIdentifier(prefix=res_id_prefix)
                date = origin.time.strftime('%Y%m%d')
                pick.time = UTCDateTime(date + arrival_time)
                #Check if pick is on the next day:
                if pick.time < origin.time:
                    pick.time += timedelta(days=1)
                pick.waveform_id = p_pick.waveform_id
                pick.backazimuth = p_pick.backazimuth
                onset = phase[0]
                if onset == 'e':
                    pick.onset = 'emergent'
                    phase = phase[1:]
                if onset == 'i':
                    pick.onset = 'impulsive'
                    phase = phase[1:]
                if onset == 'q':
                    pick.onset = 'questionable'
                    phase = phase[1:]
                pick.phase_hint = phase.strip()
                event.picks.append(pick)
                arrival = Arrival()
                arrival.resource_id = ResourceIdentifier(prefix=res_id_prefix)
                arrival.pick_id = pick.resource_id
                arrival.phase = pick.phase_hint
                arrival.azimuth = p_arrival.azimuth
                arrival.distance = p_arrival.distance
                origin.quality.associated_phase_count += 1
                origin.arrivals.append(arrival)

    def _deserialize(self):
        catalog = Catalog()
        catalog.resource_id =\
            ResourceIdentifier(
                resource_id='/'.join((res_id_prefix, self.filename)))
        catalog.description = 'Created from NEIC PDE mchedr format'
        catalog.comments = ''
        catalog.creation_info = CreationInfo(creation_time=UTCDateTime())
        for line in self.fh.readlines():
            record_id = line[0:2]
            if record_id == 'HY':
                event = self._parseRecordHY(line)
                catalog.append(event)
            if record_id == 'E ':
                self._parseRecordE(line, event)
            if record_id == 'L ':
                self._parseRecordL(line, event)
            if record_id == 'A ':
                self._parseRecordA(line, event)
            if record_id == 'C ':
                self._parseRecordC(line, event)
            if record_id == 'AH':
                self._parseRecordAH(line, event)
            if record_id == 'AE':
                self._parseRecordAE(line, event)
            if record_id == 'Dp':
                focal_mechanism = self._parseRecordDp(line, event)
            if record_id == 'Dt':
                self._parseRecordDt(line, focal_mechanism)
            if record_id == 'Da':
                self._parseRecordDa(line, focal_mechanism)
            if record_id == 'Dc':
                self._parseRecordDc(line, focal_mechanism)
            if record_id == 'P ':
                pick, arrival = self._parseRecordP(line, event)
            if record_id == 'M ':
                self._parseRecordM(line, event, pick)
            if record_id == 'S ':
                self._parseRecordS(line, event, pick, arrival)
        self.fh.close()
        return catalog


def readMchedr(filename):
    """
    Reads a NEIC PDE mchedr (machine-readable Earthquake Data Report) file
    and returns a ObsPy Catalog object.

    .. warning::
        This function should NOT be called directly, it registers via the
        ObsPy :func:`~obspy.core.event.readEvents` function, call this instead.

    :type filename: str
    :param filename: mchedr file to be read.
    :rtype: :class:`~obspy.core.event.Catalog`
    :return: An ObsPy Catalog object.

    .. rubric:: Example

    >>> from obspy.core.event import readEvents
    >>> cat = readEvents('/path/to/mchedr.dat')
    >>> print cat
    1 Event(s) in Catalog:
    2012-01-01T05:27:55.980000Z | +31.456, +138.072 | 6.2 Mb
    """
    # if filename starts with /path/to/ try to search in examples
    if isinstance(filename, basestring) and \
       filename.startswith('/path/to/'):
        try:
            filename = getExampleFile(filename[9:])
        except:
            # otherwise just try to read the given /path/to folder
            pass
    return Unpickler().load(filename)


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
