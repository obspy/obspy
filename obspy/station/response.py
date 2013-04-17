#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Instrument Response class.

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
    def __init__(self, stage_sequence_number, input_units, output_units,
            resource_id=None, stage_gain=None, decimation=None, name=None,
            description=None):

        """
        :type stage_sequence_number: integer greater or equal to zero
        :param stage_sequence_number: Stage sequence number. This is used in
            all the response SEED blockettes.
        :type resource_id: string
        :param resource_id: This field contains a string that should serve as a
            unique resource identifier. This identifier can be interpreted
            differently depending on the datacenter/software that generated the
            document. Also, we recommend to use something like
            GENERATOR:Meaningful ID. As a common behaviour equipment with the
            same ID should contains the same information/be derived from the
            same base instruments.
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
        :type input_units:
        :param input_units: The units of the data as input from the
            perspective of data acquisition. After correcting data for this
            response, these would be the resulting units.
        :type output_units:
        :param output_units: The units of the data as output from the
            perspective of data acquisition. These would be the units of the
            data prior to correcting for this response.
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
        self.input_units = input_units
        self.output_units = output_units
        self.resource_id = resource_id
        self.stage_gain = stage_gain
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
    From the StationXML Definition:
        Instrument sensitivities, or the complete system sensitivity, can be
        expressed using either a sensitivity value or a polynomial. The
        information can be used to convert raw data to Earth at a specified
        frequency or within a range of frequencies.
    """
    def __init__(self, resource_id=None, response_stages=[]):
        """
        :type resource_id: string
        :param resource_id: This field contains a string that should serve as a
            unique resource identifier. This identifier can be interpreted
            differently depending on the datacenter/software that generated the
            document. Also, we recommend to use something like
            GENERATOR:Meaningful ID. As a common behaviour equipment with the
            same ID should contains the same information/be derived from the
            same base instruments.
        :type response_stages: List of
            :class:`~obspy.station.response.ResponseStage` objects
        :param response_stages:
        """
        self.resource_id = resource_id
        self.response_stages = []


class InstrumentSensitivity(Response):
    """
    From the StationXML Definition:
        The total sensitivity for a channel, representing the complete
        acquisition system expressed as a scalar. Equivalent to SEED stage 0
        gain with (blockette 58) with the ability to specify a frequency range.
    """
    def __init__(self):
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
