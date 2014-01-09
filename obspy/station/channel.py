#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Provides the Channel class.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2013
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""
from obspy.station import BaseNode
from obspy.station.util import Longitude, Latitude


class Channel(BaseNode):
    """
    From the StationXML definition:
        Equivalent to SEED blockette 52 and parent element for the related the
        response blockettes.
    """
    def __init__(self, code, location_code, latitude, longitude,
                 elevation, depth, azimuth=None, dip=None, types=None,
                 external_references=None, sample_rate=None,
                 sample_rate_ratio_number_samples=None,
                 sample_rate_ratio_number_seconds=None, storage_format=None,
                 clock_drift_in_seconds_per_sample=None,
                 calibration_units=None, calibration_units_description=None,
                 sensor=None, pre_amplifier=None, data_logger=None,
                 equipment=None, response=None, description=None,
                 comments=None, start_date=None, end_date=None,
                 restricted_status=None, alternate_code=None,
                 historical_code=None):
        """
        :type code: String
        :param code: The SEED channel code for this channel
        :type location_code: String
        :param location_code: The SEED location code for this channel
        :type latitude: :class:`~obspy.station.util.Latitude`
        :param latitude: Latitude coordinate of this channel's sensor.
        :type longitude: :class:`~obspy.station.util.Longitude`
        :param longitude: Longitude coordinate of this channel's sensor.
        :type elevation: float
        :param elevation: Elevation of the sensor.
        :type depth: float
        :param depth: The local depth or overburden of the instrument's
            location. For downhole instruments, the depth of the instrument
            under the surface ground level. For underground vaults, the
            distance from the instrument to the local ground level above.
        :type azimuth: float, optional
        :param azimuth: Azimuth of the sensor in degrees from north, clockwise.
        :type dip: float, optional
        :param dip: Dip of the instrument in degrees, down from horizontal.
        :type types: List of strings, optional
        :param types: The type of data this channel collects. Corresponds to
            channel flags in SEED blockette 52. The SEED volume producer could
            use the first letter of an Output value as the SEED channel flag.
            Possible values: TRIGGERED, CONTINUOUS, HEALTH, GEOPHYSICAL,
                WEATHER, FLAG, SYNTHESIZED, INPUT, EXPERIMENTAL, MAINTENANCE,
                BEAM
        :type external_references: List of
            :class:`~obspy.station.util.ExternalRefernce`, optional
        :param external_references: URI of any type of external report, such as
            data quality reports.
        :type sample_rate: float, optional
        :param sample_rate: This is a group of elements that represent sample
            rate. If this group is included, then SampleRate, which is the
            sample rate in samples per second, is required. SampleRateRatio,
            which is expressed as a ratio of number of samples in a number of
            seconds, is optional. If both are included, SampleRate should be
            considered more definitive.
        :type sample_rate_ratio_number_samples: int, optional
        :param sample_rate_ratio_number_samples: The sample rate expressed as
            number of samples in a number of seconds. This is the number of
            samples.
        :type channel.sample_rate_ratio_number_seconds: int, optional
        :param channel.sample_rate_ratio_number_seconds: The sample rate
            expressed as number of samples in a number of seconds. This is the
            number of seconds.
        :type storage_format: string, optional
        :param storage_format: The storage format of the recorded data (e.g.
            SEED)
        :type clock_drift_in_seconds_per_sample: float, optional
        :param clock_drift_in_seconds_per_sample: A tolerance value, measured
            in seconds per sample, used as a threshold for time error detection
            in data from the channel.
        :type calibration_units: String
        :param calibration_units: Name of units , e.g. "M/S", "M", ...
        :type calibration_units_description: String
        :param calibration_units_description: Description of units, e.g.
            "Velocity in meters per second", ...
        :type sensor: :class:~`obspy.station.util.Equipment`
        :param sensor: The sensor
        :type pre_amplifier: :class:~`obspy.station.util.Equipment`
        :param pre_amplifier: The pre-amplifier
        :type data_logger: :class:~`obspy.station.util.Equipment`
        :param data_logger: The data-logger
        :type equipment: :class:~`obspy.station.util.Equipment`
        :param equipment: Other station equipment
        :type response: :class:~`obspy.station.response.Response`, optional
        :param response: The response of the channel
        :type description: String, optional
        :param description: A description of the resource
        :type comments: List of :class:`~obspy.station.util.Comment`, optional
        :param comments: An arbitrary number of comments to the resource
        :type start_date: :class:`~obspy.core.utcdatetime.UTCDateTime`,
            optional
        :param start_date: The start date of the resource
        :type end_date: :class:`~obspy.core.utcdatetime.UTCDateTime`, optional
        :param end_date: The end date of the resource
        :type restricted_status: String, optional
        :param restricted_status: The restriction status
        :type alternate_code: String, optional
        :param alternate_code: A code used for display or association,
            alternate to the SEED-compliant code.
        :type historical_code: String, optional
        :param historical_code: A previously used code if different from the
            current code.
        """
        self.location_code = location_code
        self.latitude = latitude
        self.longitude = longitude
        self.elevation = elevation
        self.depth = depth
        self.azimuth = azimuth
        self.dip = dip
        self.types = types or []
        self.external_references = external_references or []
        self.sample_rate = sample_rate
        self.sample_rate_ratio_number_samples = \
            sample_rate_ratio_number_samples
        self.sample_rate_ratio_number_seconds = \
            sample_rate_ratio_number_seconds
        self.storage_format = storage_format
        self.clock_drift_in_seconds_per_sample = \
            clock_drift_in_seconds_per_sample
        self.calibration_units = calibration_units
        self.sensor = sensor
        self.pre_amplifier = pre_amplifier
        self.data_logger = data_logger
        self.equipment = equipment
        self.response = response
        super(Channel, self).__init__(
            code=code, description=description, comments=comments,
            start_date=start_date, end_date=end_date,
            restricted_status=restricted_status, alternate_code=alternate_code,
            historical_code=historical_code)

    def __str__(self):
        ret = (
            "Channel '{id}', Location '{location}' {description}\n"
            "\tTimerange: {start_date} - {end_date}\n"
            "\tLatitude: {latitude:.2f}, Longitude: {longitude:.2f}, "
            "Elevation: {elevation:.1f} m, Local Depth: {depth:.1f} m\n"
            "{azimuth}"
            "{dip}"
            "{channel_types}"
            "\tSampling Rate: {sampling_rate:.2f} Hz\n"
            "\tSensor: {sensor}\n"
            "{response}")\
            .format(
                id=self.code, location=self.location_code,
                description="(%s)" % self.description
                if self.description else "",
                start_date=str(self.start_date),
                end_date=str(self.end_date) if self.end_date else "--",
                latitude=self.latitude, longitude=self.longitude,
                elevation=self.elevation, depth=self.depth,
                azimuth="\tAzimuth: %.2f degrees from north, clockwise\n" %
                self.azimuth if self.azimuth is not None else "",
                dip="\tDip: %.2f degrees down from horizontal\n" %
                self.dip if self.dip is not None else "",
                channel_types="\tChannel types: %s\n" % ", ".join(self.types)
                    if self.types else "",
                sampling_rate=self.sample_rate, sensor=self.sensor.type,
                response="\tResponse information available"
                    if self.response else "")
        return ret

    @property
    def location_code(self):
        return self._location_code

    @location_code.setter
    def location_code(self, value):
        self._location_code = value.strip()

    @property
    def longitude(self):
        return self._longitude

    @longitude.setter
    def longitude(self, value):
        if isinstance(value, Longitude):
            self._longitude = value
        else:
            self._longitude = Longitude(value)

    @property
    def latitude(self):
        return self._latitude

    @latitude.setter
    def latitude(self, value):
        if isinstance(value, Latitude):
            self._latitude = value
        else:
            self._latitude = Latitude(value)


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
