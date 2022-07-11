#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Provides the Channel class.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2013
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""
import warnings

from obspy.core.util.decorator import deprecated_keywords
from obspy.core.util.deprecation_helpers import ObsPyDeprecationWarning
from obspy.core.util.obspy_types import (
    FloatWithUncertainties, FloatWithUncertaintiesAndUnit)
from . import BaseNode
from .util import (Azimuth, ClockDrift, Dip, Distance, Latitude, Longitude,
                   Equipment)


class Channel(BaseNode):
    """
    From the StationXML definition:
        Equivalent to SEED blockette 52 and parent element for the related
        response blockettes.
    """
    @deprecated_keywords({"equipment": "equipments"})
    def __init__(self, code, location_code, latitude, longitude,
                 elevation, depth, azimuth=None, dip=None, types=None,
                 external_references=None, sample_rate=None,
                 sample_rate_ratio_number_samples=None,
                 sample_rate_ratio_number_seconds=None, storage_format=None,
                 clock_drift_in_seconds_per_sample=None,
                 calibration_units=None, calibration_units_description=None,
                 sensor=None, pre_amplifier=None, data_logger=None,
                 equipments=None, response=None, description=None,
                 comments=None, start_date=None, end_date=None,
                 restricted_status=None, alternate_code=None,
                 historical_code=None, data_availability=None,
                 identifiers=None, water_level=None, source_id=None):
        """
        :type code: str
        :param code: The SEED channel code for this channel
        :type location_code: str
        :param location_code: The SEED location code for this channel
        :type latitude: :class:`~obspy.core.inventory.util.Latitude`
        :param latitude: Latitude coordinate of this channel's sensor.
        :type longitude: :class:`~obspy.core.inventory.util.Longitude`
        :param longitude: Longitude coordinate of this channel's sensor.
        :type elevation: float
        :param elevation: Elevation of the sensor.
        :type depth: float
        :param depth: The local depth or overburden of the instrument's
            location. For downhole instruments, the depth of the instrument
            under the surface ground level. For underground vaults, the
            distance from the instrument to the local ground level above.
        :type azimuth: float
        :param azimuth: Azimuth of the sensor in degrees from North, clockwise.
        :type dip: float
        :param dip: Dip of the instrument in degrees, down from horizontal.
        :type types: list[str]
        :param types: The type of data this channel collects. Corresponds to
            channel flags in SEED blockette 52. The SEED volume producer could
            use the first letter of an Output value as the SEED channel flag.
            Possible values: TRIGGERED, CONTINUOUS, HEALTH, GEOPHYSICAL,
            WEATHER, FLAG, SYNTHESIZED, INPUT, EXPERIMENTAL, MAINTENANCE, BEAM
        :type external_references: list of
            :class:`~obspy.core.inventory.util.ExternalReference`
        :param external_references: URI of any type of external report, such as
            data quality reports.
        :type sample_rate: float
        :param sample_rate: This is a group of elements that represent sample
            rate. If this group is included, then SampleRate, which is the
            sample rate in samples per second, is required. SampleRateRatio,
            which is expressed as a ratio of number of samples in a number of
            seconds, is optional. If both are included, SampleRate should be
            considered more definitive.
        :type sample_rate_ratio_number_samples: int
        :param sample_rate_ratio_number_samples: The sample rate expressed as
            number of samples in a number of seconds. This is the number of
            samples.
        :type sample_rate_ratio_number_seconds: int
        :param sample_rate_ratio_number_seconds: The sample rate expressed as
            number of samples in a number of seconds. This is the number of
            seconds.
        :param storage_format: Ignored. Removed with ObsPy version 1.2.0 in
            accordance with StationXML 1.1
        :type clock_drift_in_seconds_per_sample: float
        :param clock_drift_in_seconds_per_sample: A tolerance value, measured
            in seconds per sample, used as a threshold for time error detection
            in data from the channel.
        :type calibration_units: str
        :param calibration_units: Name of units , e.g. "M/S", "M", ...
        :type calibration_units_description: str
        :param calibration_units_description: Description of units, e.g.
            "Velocity in meters per second", ...
        :type sensor: :class:`~obspy.core.inventory.util.Equipment`
        :param sensor: The sensor
        :type pre_amplifier: :class:`~obspy.core.inventory.util.Equipment`
        :param pre_amplifier: The pre-amplifier
        :type data_logger: :class:`~obspy.core.inventory.util.Equipment`
        :param data_logger: The data-logger
        :type equipments: list of :class:`~obspy.core.inventory.util.Equipment`
        :param equipments: Other station equipment
        :type response: :class:`~obspy.core.inventory.response.Response`
        :param response: The response of the channel
        :type description: str
        :param description: A description of the resource
        :type comments: list of :class:`~obspy.core.inventory.util.Comment`
        :param comments: An arbitrary number of comments to the resource
        :type start_date: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param start_date: The start date of the resource
        :type end_date: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param end_date: The end date of the resource
        :type restricted_status: str
        :param restricted_status: The restriction status
        :type alternate_code: str
        :param alternate_code: A code used for display or association,
            alternate to the SEED-compliant code.
        :type historical_code: str
        :param historical_code: A previously used code if different from the
            current code.
        :type data_availability:
            :class:`~obspy.core.inventory.util.DataAvailability`
        :param data_availability: Information about time series availability
            for the channel.
        :type identifiers: list[str], optional
        :param identifiers: Persistent identifiers for network/station/channel
            (schema version >=1.1). URIs are in general composed of a 'scheme'
            and a 'path' (optionally with additional components), the two of
            which separated by a colon.
        :type water_level: float, optional
        :param water_level: Elevation of the water surface in meters for
            underwater sites, where 0 is sea level. (schema version >=1.1)
        :type source_id: str, optional
        :param source_id: A data source identifier in URI form
            (schema version >=1.1). URIs are in general composed of a 'scheme'
            and a 'path' (optionally with additional components), the two of
            which separated by a colon.
        """
        self.location_code = location_code
        self.latitude = latitude
        self.longitude = longitude
        self.elevation = elevation
        self.depth = depth
        self.azimuth = azimuth
        self.dip = dip
        self.water_level = water_level
        self.types = types or []
        self.external_references = external_references or []
        self.sample_rate = sample_rate
        self.sample_rate_ratio_number_samples = \
            sample_rate_ratio_number_samples
        self.sample_rate_ratio_number_seconds = \
            sample_rate_ratio_number_seconds
        if storage_format is not None:
            msg = ("Attribute 'storage_format' was removed in accordance with "
                   "StationXML 1.1, ignoring.")
            warnings.warn(msg, ObsPyDeprecationWarning)
        self.clock_drift_in_seconds_per_sample = \
            clock_drift_in_seconds_per_sample
        self.calibration_units = calibration_units
        self.calibration_units_description = calibration_units_description
        self.sensor = sensor
        self.pre_amplifier = pre_amplifier
        self.data_logger = data_logger
        if isinstance(equipments, Equipment):
            equipments = [equipments]
        self.equipments = equipments or []
        self.response = response
        super(Channel, self).__init__(
            code=code, description=description, comments=comments,
            start_date=start_date, end_date=end_date,
            restricted_status=restricted_status, alternate_code=alternate_code,
            historical_code=historical_code,
            data_availability=data_availability, identifiers=identifiers,
            source_id=source_id)

    @property
    def storage_format(self):
        msg = ("Attribute 'storage_format' was removed in accordance with "
               "StationXML 1.1, returning None.")
        warnings.warn(msg, ObsPyDeprecationWarning)
        return None

    @storage_format.setter
    def storage_format(self, value):
        msg = ("Attribute 'storage_format' was removed in accordance with "
               "StationXML 1.1, ignoring.")
        warnings.warn(msg, ObsPyDeprecationWarning)

    @property
    def equipment(self):
        msg = ("Attribute 'equipment' (holding a single Equipment) is "
               "deprecated in favor of 'equipments' which now holds a list "
               "of Equipment objects (following changes in StationXML 1.1) "
               "and might be removed in the future. Returning the first entry "
               "found in 'equipments'.")
        warnings.warn(msg, ObsPyDeprecationWarning)
        equipments = self.equipments
        if not equipments:
            return None
        return equipments[0]

    @equipment.setter
    def equipment(self, value):
        msg = ("Attribute 'equipment' (holding a single Equipment) is "
               "deprecated in favor of 'equipments' which now holds a list "
               "of Equipment objects (following changes in StationXML 1.1) "
               "and might be removed in the future. Setting 'equipments' "
               "attribute with a list with the given item as only entry.")
        warnings.warn(msg, ObsPyDeprecationWarning)
        self.equipments = [value]

    def __str__(self):
        ret = (
            "Channel '{id}', Location '{location}' {description}\n"
            "{availability}"
            "\tTime range: {start_date} - {end_date}\n"
            "\tLatitude: {latitude:.4f}, Longitude: {longitude:.4f}, "
            "Elevation: {elevation:.1f} m, Local Depth: {depth:.1f} m\n"
            "{azimuth}"
            "{dip}"
            "{channel_types}"
            "{sampling_rate}"
            "{sensor}"
            "{response}")\
            .format(
                id=self.code, location=self.location_code,
                latitude=self.latitude, longitude=self.longitude,
                elevation=self.elevation, depth=self.depth,
                availability=("\t%s\n" % str(self.data_availability)
                              if self.data_availability else ""),
                description=("(%s)" % self.description
                             if self.description else ""),
                start_date=str(self.start_date) if self.start_date else "--",
                end_date=str(self.end_date) if self.end_date else "--",
                azimuth=("\tAzimuth: %.2f degrees from north, clockwise\n" %
                         self.azimuth if self.azimuth is not None else ""),
                dip=("\tDip: %.2f degrees down from horizontal\n" %
                     self.dip if self.dip is not None else ""),
                channel_types=("\tChannel types: %s\n" % ", ".join(self.types)
                               if self.types else ""),
                sampling_rate=("\tSampling Rate: %.2f Hz\n" %
                               self.sample_rate if self.sample_rate else ""),
                sensor=("\tSensor (Description): %s (%s)\n" % (
                        self.sensor.type, self.sensor.description)
                        if self.sensor else ""),
                response=("\tResponse information available"
                          if self.response else ""))
        return ret

    def _repr_pretty_(self, p, cycle):
        p.text(str(self))

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

    @property
    def elevation(self):
        return self._elevation

    @elevation.setter
    def elevation(self, value):
        if isinstance(value, Distance):
            self._elevation = value
        else:
            self._elevation = Distance(value)

    @property
    def depth(self):
        return self._depth

    @depth.setter
    def depth(self, value):
        if isinstance(value, Distance):
            self._depth = value
        else:
            self._depth = Distance(value)

    @property
    def azimuth(self):
        return self._azimuth

    @azimuth.setter
    def azimuth(self, value):
        if value is None:
            self._azimuth = None
        elif isinstance(value, Azimuth):
            self._azimuth = value
        else:
            self._azimuth = Azimuth(value)

    @property
    def dip(self):
        return self._dip

    @dip.setter
    def dip(self, value):
        if value is None:
            self._dip = None
        elif isinstance(value, Dip):
            self._dip = value
        else:
            self._dip = Dip(value)

    @property
    def water_level(self):
        return self._water_level

    @water_level.setter
    def water_level(self, value):
        if value is None:
            self._water_level = None
        elif isinstance(value, FloatWithUncertaintiesAndUnit):
            self._water_level = value
        else:
            self._water_level = FloatWithUncertaintiesAndUnit(value)

    @property
    def sample_rate(self):
        return self._sample_rate

    @sample_rate.setter
    def sample_rate(self, value):
        if value is None:
            self._sample_rate = None
        elif isinstance(value, FloatWithUncertainties):
            self._sample_rate = value
        else:
            self._sample_rate = FloatWithUncertainties(value)

    @property
    def clock_drift_in_seconds_per_sample(self):
        return self._clock_drift_in_seconds_per_sample

    @clock_drift_in_seconds_per_sample.setter
    def clock_drift_in_seconds_per_sample(self, value):
        if value is None:
            self._clock_drift_in_seconds_per_sample = None
        elif isinstance(value, ClockDrift):
            self._clock_drift_in_seconds_per_sample = value
        else:
            self._clock_drift_in_seconds_per_sample = ClockDrift(value)

    @property
    def equipments(self):
        return self._equipments

    @equipments.setter
    def equipments(self, value):
        if not hasattr(value, "__iter__"):
            msg = "equipments needs to be an iterable, e.g. a list."
            raise ValueError(msg)
        # make sure to unwind actual iterators, or the just might get exhausted
        # at some point
        equipments = [equipment for equipment in value]
        if any([not isinstance(x, Equipment) for x in equipments]):
            msg = "equipments can only contain Equipment objects."
            raise ValueError(msg)
        self._equipments = equipments

    def plot(self, min_freq, output="VEL", start_stage=None, end_stage=None,
             label=None, axes=None, unwrap_phase=False, plot_degrees=False,
             show=True, outfile=None):
        """
        Show bode plot of the channel's instrument response.

        :type min_freq: float
        :param min_freq: Lowest frequency to plot.
        :type output: str
        :param output: Output units. One of:

            ``"DISP"``
                displacement, output unit is meters
            ``"VEL"``
                velocity, output unit is meters/second
            ``"ACC"``
                acceleration, output unit is meters/second**2
            ``"DEF"``
                default units, the response is calculated in
                output units/input units (last stage/first stage).
                Useful if the units for a particular type of sensor (e.g., a
                pressure sensor) cannot be converted to displacement, velocity
                or acceleration.

        :type start_stage: int
        :param start_stage: Sequence number of first stage that will be used
            (disregarding all earlier stages).
        :type end_stage: int
        :param end_stage: Sequence number of last stage that will be used
            (disregarding all later stages).
        :type label: str
        :param label: Label string for legend.
        :type axes: list[:class:`matplotlib.axes.Axes`,
            :class:`matplotlib.axes.Axes`]
        :param axes: List/tuple of two axes instances on which to plot the
            amplitude/phase spectrum. If not specified, a new figure is opened.
        :type unwrap_phase: bool
        :param unwrap_phase: Set optional phase unwrapping using NumPy.
        :type plot_degrees: bool
        :param plot_degrees: if ``True`` plot bode in degrees
        :type show: bool
        :param show: Whether to show the figure after plotting or not. Can be
            used to do further customization of the plot before showing it.
        :type outfile: str
        :param outfile: Output file path to directly save the resulting image
            (e.g. ``"/tmp/image.png"``). Overrides the ``show`` option; image
            will not be displayed interactively. The given path/file name is
            also used to automatically determine the output format. Supported
            file formats depend on your matplotlib backend.  Most backends
            support png, pdf, ps, eps and svg. Defaults to ``None``.

        .. rubric:: Basic Usage

        >>> from obspy import read_inventory
        >>> cha = read_inventory()[0][0][0]
        >>> cha.plot(0.001, output="VEL")  # doctest: +SKIP

        .. plot::

            from obspy import read_inventory
            cha = read_inventory()[0][0][0]
            cha.plot(0.001, output="VEL")
        """
        return self.response.plot(
            min_freq=min_freq, output=output,
            start_stage=start_stage, end_stage=end_stage, label=label,
            axes=axes, sampling_rate=self.sample_rate,
            unwrap_phase=unwrap_phase, plot_degrees=plot_degrees, show=show,
            outfile=outfile)


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
