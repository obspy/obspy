#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Classes related to instrument responses.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2013
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""


class ResponseStage(object):
    """
    From the StationXML Definition:
        This complex type represents channel response and covers SEED
        blockettes 53 to 56.
    """
    def __init__(self, stage_sequence_number, stage_gain_value,
            stage_gain_frequency, input_units_name, output_units_name,
            input_units_description=None, output_units_description=None,
            resource_id=None, decimation=None, name=None, description=None):
        """
        :type stage_sequence_number: integer greater or equal to zero
        :param stage_sequence_number: Stage sequence number. This is used in
            all the response SEED blockettes.
        :type stage_gain_value: float, optional
        :param stage_gain_value: Complex type for sensitivity and frequency
            ranges. This complex type can be used to represent both overall
            sensitivities and individual stage gains.  A scalar that, when
            applied to the data values, converts the data to different units
            (e.g. Earth units).
        :type stage_gain_frequency: float, optional
        :param stage_gain_frequency: Complex type for sensitivity and frequency
            ranges. This complex type can be used to represent both overall
            sensitivities and individual stage gains. The frequency (in Hertz)
            at which the Value is valid.
        :param decimation:
        :param input_units_name: string
        :param input_units_name: The units of the data as input from the
            perspective of data acquisition. After correcting data for this
            response, these would be the resulting units.
            Name of units, e.g. "M/S", "V", "PA".
        :param input_units_description: string, optional
        :param input_units_description: The units of the data as input from the
            perspective of data acquisition. After correcting data for this
            response, these would be the resulting units.
            Description of units, e.g. "Velocity in meters per second",
            "Volts", "Pascals".
        :param output_units_name: string
        :param output_units_name: The units of the data as output from the
            perspective of data acquisition. These would be the units of the
            data prior to correcting for this response.
            Name of units, e.g. "M/S", "V", "PA".
        :type output_units_description: string, optional
        :param output_units_description: The units of the data as output from
            the perspective of data acquisition. These would be the units of
            the data prior to correcting for this response.
            Description of units, e.g. "Velocity in meters per second",
            "Volts", "Pascals".
        :type resource_id: string, optional
        :param resource_id: This field contains a string that should serve as a
            unique resource identifier. This identifier can be interpreted
            differently depending on the datacenter/software that generated the
            document. Also, we recommend to use something like
            GENERATOR:Meaningful ID. As a common behaviour equipment with the
            same ID should contains the same information/be derived from the
            same base instruments.
        :type name: string, optional
        :param name: A name given to this filter.
        :type description: string, optional
        :param description: A short description of of the filter.

        .. note::
            The stage gain (or stage sensitivity) is the gain at the stage of
            the encapsulating response element and corresponds to SEED
            blockette 58. In the SEED convention, stage 0 gain represents the
            overall sensitivity of the channel.  In this schema, stage 0 gains
            are allowed but are considered deprecated.  Overall sensitivity
            should be specified in the InstrumentSensitivity element.
        """
        self.stage_sequence_number = stage_sequence_number
        self.input_units_name = input_units_name
        self.output_units_name = output_units_name
        self.input_units_description = input_units_description
        self.output_units_description = output_units_description
        self.resource_id = resource_id
        self.stage_gain_value = stage_gain_value
        self.stage_gain_frequency = stage_gain_frequency
        self.decimation = decimation
        self.name = name
        self.description = description


class PolesZerosResponseStage(ResponseStage):
    """
    From the StationXML Definition:
        Response: complex poles and zeros. Corresponds to SEED blockette 53.
    """
    def __init__(self, stage_sequence_number, resource_id=None,
            stage_gain=None, decimation=None):
        super(PolesZerosResponseStage, self).__init__()


class CoefficientsTypeResponseStage(ResponseStage):
    """
    """
    def __init__(self, stage_sequence_number, resource_id=None,
            stage_gain=None, decimation=None):
        super(CoefficientsTypeResponseStage, self).__init__()


class ResponseListResponseStage(ResponseStage):
    """
    """
    def __init__(self, stage_sequence_number, resource_id=None,
            stage_gain=None, decimation=None):
        super(ResponseListResponseStage, self).__init__()


class FIRResponseStage(ResponseStage):
    """
    """
    def __init__(self, stage_sequence_number, resource_id=None,
            stage_gain=None, decimation=None):
        super(FIRResponseStage, self).__init__()


class PolynomialResponseStage(ResponseStage):
    """
    """
    def __init__(self, stage_sequence_number, resource_id=None,
            stage_gain=None, decimation=None):
        super(PolynomialResponseStage, self).__init__()


class Response(object):
    """
    The root response object.
    """
    def __init__(self, resource_id=None, instrument_sensitivity=None,
            response_stages=None):
        """
        :type resource_id: string
        :param resource_id: This field contains a string that should serve as a
            unique resource identifier. This identifier can be interpreted
            differently depending on the datacenter/software that generated the
            document. Also, we recommend to use something like
            GENERATOR:Meaningful ID. As a common behaviour equipment with the
            same ID should contains the same information/be derived from the
            same base instruments.
        :type instrument_sensitivity:
            :class:`~obspy.station.response.InstrumentSensitivity`
        :param instrument_sensitivity: The total sensitivity for the given
            channel.
        :type response_stages: List of
            :class:`~obspy.station.response.ResponseStage` objects
        :param response_stages: A list of the response stages. Covers SEED
            blockettes 53 to 56.
        """
        self.resource_id = resource_id
        self.instrument_sensitivity = instrument_sensitivity
        if response_stages is None:
            self.response_stages = []
        elif hasattr(response_stages, "__iter__"):
            self.response_stages = response_stages
        else:
            msg = "response_stages must be an iterable."
            raise ValueError(msg)


class InstrumentSensitivity(Response):
    """
    From the StationXML Definition:
        The total sensitivity for a channel, representing the complete
        acquisition system expressed as a scalar. Equivalent to SEED stage 0
        gain with (blockette 58) with the ability to specify a frequency range.

    Sensitivity and frequency ranges. The FrequencyRangeGroup is an optional
    construct that defines a pass band in Hertz (FrequencyStart and
    FrequencyEnd) in which the SensitivityValue is valid within the number of
    decibels specified in FrequencyDBVariation.
    """
    def __init__(self, value, frequency, input_units_name,
            output_units_name, input_units_description=None,
            output_units_description=None, frequency_range_start=None,
            frequency_range_end=None, frequency_range_DB_variation=None):
        """
        :type value: float
        :param value: Complex type for sensitivity and frequency ranges.
            This complex type can be used to represent both overall
            sensitivities and individual stage gains. The FrequencyRangeGroup
            is an optional construct that defines a pass band in Hertz (
            FrequencyStart and FrequencyEnd) in which the SensitivityValue is
            valid within the number of decibels specified in
            FrequencyDBVariation.
        :type frequency: float
        :param frequency: Complex type for sensitivity and frequency
            ranges.  This complex type can be used to represent both overall
            sensitivities and individual stage gains. The FrequencyRangeGroup
            is an optional construct that defines a pass band in Hertz (
            FrequencyStart and FrequencyEnd) in which the SensitivityValue is
            valid within the number of decibels specified in
            FrequencyDBVariation.
        :param input_units_name: string
        :param input_units_name: The units of the data as input from the
            perspective of data acquisition. After correcting data for this
            response, these would be the resulting units.
            Name of units, e.g. "M/S", "V", "PA".
        :param input_units_description: string, optional
        :param input_units_description: The units of the data as input from the
            perspective of data acquisition. After correcting data for this
            response, these would be the resulting units.
            Description of units, e.g. "Velocity in meters per second",
            "Volts", "Pascals".
        :param output_units_name: string
        :param output_units_name: The units of the data as output from the
            perspective of data acquisition. These would be the units of the
            data prior to correcting for this response.
            Name of units, e.g. "M/S", "V", "PA".
        :type output_units_description: string, optional
        :param output_units_description: The units of the data as output from
            the perspective of data acquisition. These would be the units of
            the data prior to correcting for this response.
            Description of units, e.g. "Velocity in meters per second",
            "Volts", "Pascals".
        :type frequency_range_start: float, optional
        :param frequency_range_start: Start of the frequency range for which
            the SensitivityValue is valid within the dB variation specified.
        :type frequency_range_end: float, optional
        :param frequency_range_end: End of the frequency range for which the
            SensitivityValue is valid within the dB variation specified.
        :type frequency_range_DB_variation: float, optional
        :param frequency_range_DB_variation: Variation in decibels within the
            specified range.
        """
        self.value = value
        self.frequency = frequency
        self.input_units_name = input_units_name
        self.input_units_description = input_units_description
        self.output_units_name = output_units_name
        self.output_units_description = output_units_description
        self.frequency_range_start = frequency_range_start
        self.frequency_range_end = frequency_range_end
        self.frequency_range_DB_variation = frequency_range_DB_variation
        super(InstrumentSensitivity, self).__init__()


class InstrumentPolynomial(Response):
    """
    From the StationXML Definition:
        The total sensitivity for a channel, representing the complete
        acquisition system expressed as a polynomial. Equivalent to SEED stage
        0 polynomial (blockette 62).
    """
    def __init__(self):
        super(InstrumentPolynomial, self).__init__()
