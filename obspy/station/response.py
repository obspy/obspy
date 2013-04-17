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
    def __init__(self, stage_sequence_number, resource_id=None,
            stage_gain=None, decimation=None):
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
        :param stage_gain: StageSensitivity is the gain at the stage of the
            encapsulating response element and corresponds to SEED blockette
            58. In the SEED convention, stage 0 gain represents the overall
            sensitivity of the channel.  In this schema, stage 0 gains are
            allowed but are considered deprecated.  Overall sensitivity should
            be specified in the InstrumentSensitivity element.
        :param decimation:
        """
        self.stage_sequence_number = stage_sequence_number
        self.resource_id = resource_id
        self.stage_gain = stage_gain
        self.decimation = decimation


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
