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
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import warnings
import ctypes as C
import numpy as np
from math import pi
from copy import deepcopy
from collections import defaultdict

from obspy.core.util.base import ComparingObject
from obspy.core.util.obspy_types import CustomComplex, \
    FloatWithUncertaintiesAndUnit, CustomFloat, FloatWithUncertainties
from obspy.core.util.base import getMatplotlibVersion
from obspy.station.util import Frequency, Angle


MATPLOTLIB_VERSION = getMatplotlibVersion()


class ResponseStage(ComparingObject):
    """
    From the StationXML Definition:
        This complex type represents channel response and covers SEED
        blockettes 53 to 56.
    """
    def __init__(self, stage_sequence_number, stage_gain,
                 stage_gain_frequency, input_units, output_units,
                 resource_id=None, resource_id2=None, name=None,
                 input_units_description=None,
                 output_units_description=None, description=None,
                 decimation_input_sample_rate=None, decimation_factor=None,
                 decimation_offset=None, decimation_delay=None,
                 decimation_correction=None):
        """
        :type stage_sequence_number: int
        :param stage_sequence_number: Stage sequence number, greater or equal
            to zero. This is used in all the response SEED blockettes.
        :type stage_gain: float
        :param stage_gain: Value of stage gain.
        :type stage_gain_frequency: float
        :param stage_gain_frequency: Frequency of stage gain.
        :type input_units: str
        :param input_units: The units of the data as input from the
            perspective of data acquisition. After correcting data for this
            response, these would be the resulting units.
            Name of units, e.g. "M/S", "V", "PA".
        :type output_units: str
        :param output_units: The units of the data as output from the
            perspective of data acquisition. These would be the units of the
            data prior to correcting for this response.
            Name of units, e.g. "M/S", "V", "PA".
        :type resource_id: str
        :param resource_id: This field contains a string that should serve as a
            unique resource identifier. This identifier can be interpreted
            differently depending on the data center/software that generated
            the document. Also, we recommend to use something like
            GENERATOR:Meaningful ID. As a common behavior equipment with the
            same ID should contains the same information/be derived from the
            same base instruments.
        :type resource_id2: str
        :param resource_id2: This field contains a string that should serve as
            a unique resource identifier. Resource identifier of the subgroup
            of the response stage that varies across different response stage
            types (e.g. the poles and zeros part or the FIR part).
        :type name: str
        :param name: A name given to the filter stage.
        :type input_units_description: str, optional
        :param input_units_description: The units of the data as input from the
            perspective of data acquisition. After correcting data for this
            response, these would be the resulting units.
            Description of units, e.g. "Velocity in meters per second",
            "Volts", "Pascals".
        :type output_units_description: str, optional
        :param output_units_description: The units of the data as output from
            the perspective of data acquisition. These would be the units of
            the data prior to correcting for this response.
            Description of units, e.g. "Velocity in meters per second",
            "Volts", "Pascals".
        :type description: str, optional
        :param description: A short description of of the filter.
        :type decimation_input_sample_rate:  float, optional
        :param decimation_input_sample_rate: The sampling rate before the
            decimation in samples per second.
        :type decimation_factor: int, optional
        :param decimation_factor: The applied decimation factor.
        :type decimation_offset: int, optional
        :param decimation_offset: The sample chosen for use. 0 denotes the
            first sample, 1 the second, and so forth.
        :type decimation_delay: float, optional
        :param decimation_delay: The estimated pure delay from the decimation.
        :type decimation_correction: float, optional
        :param decimation_correction: The time shift applied to correct for the
            delay at this stage.

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
        self.input_units_description = input_units_description
        self.output_units_description = output_units_description
        self.resource_id = resource_id
        self.resource_id2 = resource_id2
        self.stage_gain = stage_gain
        self.stage_gain_frequency = stage_gain_frequency
        self.name = name
        self.description = description
        self.decimation_input_sample_rate = decimation_input_sample_rate
        self.decimation_factor = decimation_factor
        self.decimation_offset = decimation_offset
        self.decimation_delay = decimation_delay
        self.decimation_correction = decimation_correction

    def __str__(self):
        ret = (
            "Response type: {response_type}, Stage Sequence Number: "
            "{response_stage}\n"
            "{name_desc}"
            "{resource_id}"
            "\tFrom {input_units}{input_desc} to {output_units}{output_desc}\n"
            "\tStage gain: {gain}, defined at {gain_freq:.2f} Hz\n"
            "{decimation}").format(
            response_type=self.__class__.__name__,
            response_stage=self.stage_sequence_number,
            gain=self.stage_gain,
            gain_freq=self.stage_gain_frequency,
            name_desc="\t%s %s\n" % (
                self.name, "(%s)" % self.description
                if self.description else "") if self.name else "",
            resource_id=("\tResource Id: %s" % self.resource_id
                         if self.resource_id else ""),
            input_units=self.input_units if self.input_units else "UNKNOWN",
            input_desc=(" (%s)" % self.input_units_description
                        if self.input_units_description else ""),
            output_units=self.output_units if self.output_units else "UNKNOWN",
            output_desc=(" (%s)" % self.output_units_description
                         if self.output_units_description else ""),
            decimation=(
                "\tDecimation:\n\t\tInput Sample Rate: %.2f Hz\n\t\t"
                "Decimation Factor: %i\n\t\tDecimation Offset: %i\n\t\t"
                "Decimation Delay: %.2f\n\t\tDecimation Correction: %.2f" % (
                    self.decimation_input_sample_rate, self.decimation_factor,
                    self.decimation_offset, self.decimation_delay,
                    self.decimation_correction)
                if self.decimation_input_sample_rate is not None else ""))
        return ret.strip()

    def _repr_pretty_(self, p, cycle):
        p.text(str(self))


class PolesZerosResponseStage(ResponseStage):
    """
    From the StationXML Definition:
        Response: complex poles and zeros. Corresponds to SEED blockette 53.

    The response stage is used for the analog stages of the filter system and
    the description of infinite impulse response (IIR) digital filters.

    Has all the arguments of the parent class
    :class:`~obspy.station.response.ResponseStage` and the following:

    :type pz_transfer_function_type: str
    :param pz_transfer_function_type: A string describing the type of transfer
        function. Can be one of:

        * ``LAPLACE (RADIANS/SECOND)``
        * ``LAPLACE (HERTZ)``
        * ``DIGITAL (Z-TRANSFORM)``

        The function tries to match inputs to one of three types if it can.
    :type normalization_frequency: float
    :param normalization_frequency: The frequency at which the normalization
        factor is normalized.
    :type zeros: list of complex
    :param zeros: All zeros of the stage.
    :type poles: list of complex
    :param poles: All poles of the stage.
    :type normalization_factor: float, optional
    :param normalization_factor:
    """
    def __init__(self, stage_sequence_number, stage_gain,
                 stage_gain_frequency, input_units, output_units,
                 pz_transfer_function_type,
                 normalization_frequency, zeros, poles,
                 normalization_factor=1.0, resource_id=None, resource_id2=None,
                 name=None, input_units_description=None,
                 output_units_description=None, description=None,
                 decimation_input_sample_rate=None, decimation_factor=None,
                 decimation_offset=None, decimation_delay=None,
                 decimation_correction=None):
        # Set the Poles and Zeros specific attributes. Special cases are
        # handled by properties.
        self.pz_transfer_function_type = pz_transfer_function_type
        self.normalization_frequency = normalization_frequency
        self.normalization_factor = float(normalization_factor)
        self.zeros = zeros
        self.poles = poles
        super(PolesZerosResponseStage, self).__init__(
            stage_sequence_number=stage_sequence_number,
            input_units=input_units,
            output_units=output_units,
            input_units_description=input_units_description,
            output_units_description=output_units_description,
            resource_id=resource_id, resource_id2=resource_id2,
            stage_gain=stage_gain,
            stage_gain_frequency=stage_gain_frequency, name=name,
            description=description,
            decimation_input_sample_rate=decimation_input_sample_rate,
            decimation_factor=decimation_factor,
            decimation_offset=decimation_offset,
            decimation_delay=decimation_delay,
            decimation_correction=decimation_correction)

    def __str__(self):
        ret = super(PolesZerosResponseStage, self).__str__()
        ret += (
            "\n"
            "\tTransfer function type: {transfer_fct_type}\n"
            "\tNormalization factor: {norm_fact:g}, "
            "Normalization frequency: {norm_freq:.2f} Hz\n"
            "\tPoles: {poles}\n"
            "\tZeros: {zeros}").format(
            transfer_fct_type=self.pz_transfer_function_type,
            norm_fact=self.normalization_factor,
            norm_freq=self.normalization_frequency,
            poles=", ".join(map(str, self.poles)),
            zeros=", ".join(map(str, self.zeros)),
            )
        return ret

    def _repr_pretty_(self, p, cycle):
        p.text(str(self))

    @property
    def zeros(self):
        return self._zeros

    @zeros.setter
    def zeros(self, value):
        for x in value:
            if not isinstance(x, CustomComplex):
                msg = "Zeros must be of CustomComplex type."
                raise TypeError(msg)
        self._zeros = value

    @property
    def poles(self):
        return self._poles

    @poles.setter
    def poles(self, value):
        for x in value:
            if not isinstance(x, CustomComplex):
                msg = "Poles must be of CustomComplex type."
                raise TypeError(msg)
        self._poles = value

    @property
    def pz_transfer_function_type(self):
        return self._pz_transfer_function_type

    @pz_transfer_function_type.setter
    def pz_transfer_function_type(self, value):
        """
        Setter for the transfer function type.

        Rather permissive but should make it less awkward to use.
        """
        msg = ("'%s' is not a valid value for 'pz_transfer_function_type'. "
               "Valid one are:\n"
               "\tLAPLACE (RADIANS/SECOND)\n"
               "\tLAPLACE (HERTZ)\n"
               "\tDIGITAL (Z-TRANSFORM)") % value
        value = value.lower()
        if "laplace" in value:
            if "radian" in value:
                self._pz_transfer_function_type = "LAPLACE (RADIANS/SECOND)"
            elif "hertz" in value or "hz" in value:
                self._pz_transfer_function_type = "LAPLACE (HERTZ)"
            else:
                raise ValueError(msg)
        elif "digital" in value:
            self._pz_transfer_function_type = "DIGITAL (Z-TRANSFORM)"
        else:
            raise ValueError(msg)


class CoefficientsTypeResponseStage(ResponseStage):
    """
    This response type can describe coefficients for FIR filters. Laplace
    transforms and IIR filters can also be expressed using this type but should
    rather be described using the PolesZerosResponseStage class. Effectively
    corresponds to SEED blockette 54.

    Has all the arguments of the parent class
    :class:`~obspy.station.response.ResponseStage` and the following:

    :type cf_transfer_function_type: str
    :param cf_transfer_function_type: A string describing the type of transfer
        function. Can be one of:

        * ``ANALOG (RADIANS/SECOND)``
        * ``ANALOG (HERTZ)``
        * ``DIGITAL``

        The function tries to match inputs to one of three types if it can.
    :type numerator: list of
        :class:`~obspy.core.util.obspy_types.FloatWithUncertaintiesAndUnit`
    :param numerator: Numerator of the coefficient response stage.
    :type denominator: list of
        :class:`~obspy.core.util.obspy_types.FloatWithUncertaintiesAndUnit`
    :param denominator: Denominator of the coefficient response stage.
    """
    def __init__(self, stage_sequence_number, stage_gain,
                 stage_gain_frequency, input_units, output_units,
                 cf_transfer_function_type, resource_id=None,
                 resource_id2=None, name=None, numerator=None,
                 denominator=None, input_units_description=None,
                 output_units_description=None, description=None,
                 decimation_input_sample_rate=None, decimation_factor=None,
                 decimation_offset=None, decimation_delay=None,
                 decimation_correction=None):
        # Set the Coefficients type specific attributes. Special cases are
        # handled by properties.
        self.cf_transfer_function_type = cf_transfer_function_type
        self.numerator = numerator
        self.denominator = denominator
        super(CoefficientsTypeResponseStage, self).__init__(
            stage_sequence_number=stage_sequence_number,
            input_units=input_units,
            output_units=output_units,
            input_units_description=input_units_description,
            output_units_description=output_units_description,
            resource_id=resource_id, resource_id2=resource_id2,
            stage_gain=stage_gain,
            stage_gain_frequency=stage_gain_frequency, name=name,
            description=description,
            decimation_input_sample_rate=decimation_input_sample_rate,
            decimation_factor=decimation_factor,
            decimation_offset=decimation_offset,
            decimation_delay=decimation_delay,
            decimation_correction=decimation_correction)

    def __str__(self):
        ret = super(CoefficientsTypeResponseStage, self).__str__()
        ret += (
            "\n"
            "\tTransfer function type: {transfer_fct_type}\n"
            "\tContains {num_count} numerators and {den_count} denominators")\
            .format(
                transfer_fct_type=self.cf_transfer_function_type,
                num_count=len(self.numerator), den_count=len(self.denominator))
        return ret

    def _repr_pretty_(self, p, cycle):
        p.text(str(self))

    @property
    def numerator(self):
        return self._numerator

    @numerator.setter
    def numerator(self, value):
        for x in value:
            if not isinstance(x, FloatWithUncertaintiesAndUnit):
                msg = ("Numerator elements must be of "
                       "FloatWithUncertaintiesAndUnit type.")
                raise TypeError(msg)
        self._numerator = value

    @property
    def denominator(self):
        return self._denominator

    @denominator.setter
    def denominator(self, value):
        for x in value:
            if not isinstance(x, FloatWithUncertaintiesAndUnit):
                msg = ("Denominator elements must be of "
                       "FloatWithUncertaintiesAndUnit type.")
                raise TypeError(msg)
        self._denominator = value

    @property
    def cf_transfer_function_type(self):
        return self._cf_transfer_function_type

    @cf_transfer_function_type.setter
    def cf_transfer_function_type(self, value):
        """
        Setter for the transfer function type.

        Rather permissive but should make it less awkward to use.
        """
        msg = ("'%s' is not a valid value for 'cf_transfer_function_type'. "
               "Valid one are:\n"
               "\tANALOG (RADIANS/SECOND)\n"
               "\tANALOG (HERTZ)\n"
               "\tDIGITAL") % value
        value = value.lower()
        if "analog" in value:
            if "rad" in value:
                self._cf_transfer_function_type = "ANALOG (RADIANS/SECOND)"
            elif "hertz" in value or "hz" in value:
                self._cf_transfer_function_type = "ANALOG (HERTZ)"
            else:
                raise ValueError(msg)
        elif "digital" in value:
            self._cf_transfer_function_type = "DIGITAL"
        else:
            raise ValueError(msg)


class ResponseListResponseStage(ResponseStage):
    """
    This response type gives a list of frequency, amplitude and phase value
    pairs. Effectively corresponds to SEED blockette 55.

    Has all the arguments of the parent class
    :class:`~obspy.station.response.ResponseStage` and the following:

    :type response_list_elements: list of
        :class:`~obspy.station.response.ResponseListElement`
    :param response_list_elements: A list of single discrete frequency,
        amplitude and phase response values.
    """
    def __init__(self, stage_sequence_number, stage_gain,
                 stage_gain_frequency, input_units, output_units,
                 resource_id=None, resource_id2=None, name=None,
                 response_list_elements=None,
                 input_units_description=None, output_units_description=None,
                 description=None, decimation_input_sample_rate=None,
                 decimation_factor=None, decimation_offset=None,
                 decimation_delay=None, decimation_correction=None):
        self.response_list_elements = response_list_elements or []
        super(ResponseListResponseStage, self).__init__(
            stage_sequence_number=stage_sequence_number,
            input_units=input_units,
            output_units=output_units,
            input_units_description=input_units_description,
            output_units_description=output_units_description,
            resource_id=resource_id, resource_id2=resource_id2,
            stage_gain=stage_gain,
            stage_gain_frequency=stage_gain_frequency, name=name,
            description=description,
            decimation_input_sample_rate=decimation_input_sample_rate,
            decimation_factor=decimation_factor,
            decimation_offset=decimation_offset,
            decimation_delay=decimation_delay,
            decimation_correction=decimation_correction)


class ResponseListElement(ComparingObject):
    """
    Describes the amplitude and phase response value for a discrete frequency
    value.
    """
    def __init__(self, frequency, amplitude, phase):
        """
        :type frequency: float
        :param frequency: The frequency for which the response is valid.
        :type amplitude: float
        :param amplitude: The value for the amplitude response at this
            frequency.
        :type phase: float
        :param phase: The value for the phase response at this frequency.
        """
        self.frequency = frequency
        self.amplitude = amplitude
        self.phase = phase

    @property
    def frequency(self):
        return self._frequency

    @frequency.setter
    def frequency(self, value):
        if not isinstance(value, Frequency):
            value = Frequency(value)
        self._frequency = value

    @property
    def amplitude(self):
        return self._amplitude

    @amplitude.setter
    def amplitude(self, value):
        if not isinstance(value, FloatWithUncertaintiesAndUnit):
            value = FloatWithUncertaintiesAndUnit(value)
        self._amplitude = value

    @property
    def phase(self):
        return self._phase

    @phase.setter
    def phase(self, value):
        if not isinstance(value, Angle):
            value = Angle(value)
        self._phase = value


class FIRResponseStage(ResponseStage):
    """
    From the StationXML Definition:
        Response: FIR filter. Corresponds to SEED blockette 61. FIR filters are
        also commonly documented using the CoefficientsType element.

    Has all the arguments of the parent class
    :class:`~obspy.station.response.ResponseStage` and the following:

    :type symmetry: str
    :param symmetry: A string describing the symmetry. Can be one of:

            * ``NONE``
            * ``EVEN``
            * ``ODD``

    :type coefficients: list of floats
    :param coefficients: List of FIR coefficients.
    """
    def __init__(self, stage_sequence_number, stage_gain,
                 stage_gain_frequency, input_units, output_units,
                 symmetry="NONE", resource_id=None, resource_id2=None,
                 name=None,
                 coefficients=None, input_units_description=None,
                 output_units_description=None, description=None,
                 decimation_input_sample_rate=None, decimation_factor=None,
                 decimation_offset=None, decimation_delay=None,
                 decimation_correction=None):
        self._symmetry = symmetry
        self.coefficients = coefficients or []
        super(FIRResponseStage, self).__init__(
            stage_sequence_number=stage_sequence_number,
            input_units=input_units,
            output_units=output_units,
            input_units_description=input_units_description,
            output_units_description=output_units_description,
            resource_id=resource_id, resource_id2=resource_id2,
            stage_gain=stage_gain,
            stage_gain_frequency=stage_gain_frequency, name=name,
            description=description,
            decimation_input_sample_rate=decimation_input_sample_rate,
            decimation_factor=decimation_factor,
            decimation_offset=decimation_offset,
            decimation_delay=decimation_delay,
            decimation_correction=decimation_correction)

    @property
    def symmetry(self):
        return self._symmetry

    @symmetry.setter
    def symmetry(self, value):
        value = str(value).upper()
        allowed = ("NONE", "EVEN", "ODD")
        if value not in allowed:
            msg = ("Value '%s' for FIR Response symmetry not allowed. "
                   "Possible values are: '%s'")
            msg = msg % (value, "', '".join(allowed))
            raise ValueError(msg)
        self._symmetry = value

    @property
    def coefficients(self):
        return self._coefficients

    @coefficients.setter
    def coefficients(self, value):
        new_values = []
        for x in value:
            if not isinstance(x, FilterCoefficient):
                x = FilterCoefficient(x)
            new_values.append(x)
        self._coefficients = new_values


class PolynomialResponseStage(ResponseStage):
    """
    From the StationXML Definition:
        Response: expressed as a polynomial (allows non-linear sensors to be
        described). Corresponds to SEED blockette 62. Can be used to describe a
        stage of acquisition or a complete system.

    Has all the arguments of the parent class
    :class:`~obspy.station.response.ResponseStage` and the following:

    :type approximation_type: str
    :param approximation_type: Approximation type. Currently restricted to
        'MACLAURIN' by StationXML definition.
    :type frequency_lower_bound: float
    :param frequency_lower_bound: Lower frequency bound.
    :type frequency_upper_bound: float
    :param frequency_upper_bound: Upper frequency bound.
    :type approximation_lower_bound: float
    :param approximation_lower_bound: Lower bound of approximation.
    :type approximation_upper_bound: float
    :param approximation_upper_bound: Upper bound of approximation.
    :type maximum_error: float
    :param maximum_error: Maximum error.
    :type coefficients: list of floats
    :param coefficients: List of polynomial coefficients.
    """
    def __init__(self, stage_sequence_number, stage_gain,
                 stage_gain_frequency, input_units, output_units,
                 frequency_lower_bound,
                 frequency_upper_bound, approximation_lower_bound,
                 approximation_upper_bound, maximum_error, coefficients,
                 approximation_type='MACLAURIN', resource_id=None,
                 resource_id2=None, name=None,
                 input_units_description=None,
                 output_units_description=None, description=None,
                 decimation_input_sample_rate=None, decimation_factor=None,
                 decimation_offset=None, decimation_delay=None,
                 decimation_correction=None):
        self._approximation_type = approximation_type
        self.frequency_lower_bound = frequency_lower_bound
        self.frequency_upper_bound = frequency_upper_bound
        self.approximation_lower_bound = approximation_lower_bound
        self.approximation_upper_bound = approximation_upper_bound
        self.maximum_error = maximum_error
        self.coefficients = coefficients
        super(PolynomialResponseStage, self).__init__(
            stage_sequence_number=stage_sequence_number,
            input_units=input_units,
            output_units=output_units,
            input_units_description=input_units_description,
            output_units_description=output_units_description,
            resource_id=resource_id, resource_id2=resource_id2,
            stage_gain=stage_gain,
            stage_gain_frequency=stage_gain_frequency, name=name,
            description=description,
            decimation_input_sample_rate=decimation_input_sample_rate,
            decimation_factor=decimation_factor,
            decimation_offset=decimation_offset,
            decimation_delay=decimation_delay,
            decimation_correction=decimation_correction)

    @property
    def approximation_type(self):
        return self._approximation_type

    @approximation_type.setter
    def approximation_type(self, value):
        value = str(value).upper()
        allowed = ("MACLAURIN",)
        if value not in allowed:
            msg = ("Value '%s' for polynomial response approximation type not "
                   "allowed. Possible values are: '%s'")
            msg = msg % (value, "', '".join(allowed))
            raise ValueError(msg)
        self._approximation_type = value

    @property
    def coefficients(self):
        return self._coefficients

    @coefficients.setter
    def coefficients(self, value):
        new_values = []
        for x in value:
            if not isinstance(x, CoefficientWithUncertainties):
                x = CoefficientWithUncertainties(x)
            new_values.append(x)
        self._coefficients = new_values


class Response(ComparingObject):
    """
    The root response object.
    """
    def __init__(self, resource_id=None, instrument_sensitivity=None,
                 instrument_polynomial=None, response_stages=None):
        """
        :type resource_id: str
        :param resource_id: This field contains a string that should serve as a
            unique resource identifier. This identifier can be interpreted
            differently depending on the data center/software that generated
            the document. Also, we recommend to use something like
            GENERATOR:Meaningful ID. As a common behavior equipment with the
            same ID should contains the same information/be derived from the
            same base instruments.
        :type instrument_sensitivity:
            :class:`~obspy.station.response.InstrumentSensitivity`
        :param instrument_sensitivity: The total sensitivity for the given
            channel, representing the complete acquisition system expressed as
            a scalar.
        :type instrument_polynomial:
            :class:`~obspy.station.response.InstrumentPolynomial`
        :param instrument_polynomial: The total sensitivity for the given
            channel, representing the complete acquisition system expressed as
            a polynomial.
        :type response_stages: list of
            :class:`~obspy.station.response.ResponseStage` objects
        :param response_stages: A list of the response stages. Covers SEED
            blockettes 53 to 56.
        """
        self.resource_id = resource_id
        self.instrument_sensitivity = instrument_sensitivity
        self.instrument_polynomial = instrument_polynomial
        if response_stages is None:
            self.response_stages = []
        elif hasattr(response_stages, "__iter__"):
            self.response_stages = response_stages
        else:
            msg = "response_stages must be an iterable."
            raise ValueError(msg)

    def get_evalresp_response(self, t_samp, nfft, output="VEL",
                              start_stage=None, end_stage=None):
        """
        Returns frequency response and corresponding frequencies using
        evalresp.

        :type t_samp: float
        :param t_samp: time resolution (inverse frequency resolution)
        :type nfft: int
        :param nfft: Number of FFT points to use
        :type output: str
        :param output: Output units. One of:

            ``"DISP"``
                displacement, output unit is meters
            ``"VEL"``
                velocity, output unit is meters/second
            ``"ACC"``
                acceleration, output unit is meters/second**2

        :type start_stage: int, optional
        :param start_stage: Stage sequence number of first stage that will be
            used (disregarding all earlier stages).
        :type end_stage: int, optional
        :param end_stage: Stage sequence number of last stage that will be
            used (disregarding all later stages).
        :rtype: tuple of two arrays
        :returns: frequency response and corresponding frequencies
        """
        import obspy.signal.evrespwrapper as ew
        from obspy.signal.headers import clibevresp

        out_units = output.upper()
        if out_units not in ("DISP", "VEL", "ACC"):
            msg = ("requested output is '%s' but must be one of 'DISP', 'VEL' "
                   "or 'ACC'") % output
            raise ValueError(msg)

        # Whacky. Evalresp uses a global variable and uses that to scale the
        # response if it encounters any unit that is not SI.
        scale_factor = [1.0]

        def get_unit_mapping(key):
            try:
                key = key.upper()
            except:
                pass
            units_mapping = {
                "M": ew.ENUM_UNITS["DIS"],
                "NM": ew.ENUM_UNITS["DIS"],
                "CM": ew.ENUM_UNITS["DIS"],
                "MM": ew.ENUM_UNITS["DIS"],
                "M/S": ew.ENUM_UNITS["VEL"],
                "M/SEC": ew.ENUM_UNITS["VEL"],
                "NM/S": ew.ENUM_UNITS["VEL"],
                "NM/SEC": ew.ENUM_UNITS["VEL"],
                "CM/S": ew.ENUM_UNITS["VEL"],
                "CM/SEC": ew.ENUM_UNITS["VEL"],
                "MM/S": ew.ENUM_UNITS["VEL"],
                "MM/SEC": ew.ENUM_UNITS["VEL"],
                "M/S**2": ew.ENUM_UNITS["ACC"],
                "M/(S**2)": ew.ENUM_UNITS["ACC"],
                "M/SEC**2": ew.ENUM_UNITS["ACC"],
                "M/(SEC**2)": ew.ENUM_UNITS["ACC"],
                "NM/S**2": ew.ENUM_UNITS["ACC"],
                "NM/(S**2)": ew.ENUM_UNITS["ACC"],
                "NM/SEC**2": ew.ENUM_UNITS["ACC"],
                "NM/(SEC**2)": ew.ENUM_UNITS["ACC"],
                "CM/S**2": ew.ENUM_UNITS["ACC"],
                "CM/(S**2)": ew.ENUM_UNITS["ACC"],
                "CM/SEC**2": ew.ENUM_UNITS["ACC"],
                "CM/(SEC**2)": ew.ENUM_UNITS["ACC"],
                "MM/S**2": ew.ENUM_UNITS["ACC"],
                "MM/(S**2)": ew.ENUM_UNITS["ACC"],
                "MM/SEC**2": ew.ENUM_UNITS["ACC"],
                "MM/(SEC**2)": ew.ENUM_UNITS["ACC"],
                "V": ew.ENUM_UNITS["VOLTS"],
                "VOLT": ew.ENUM_UNITS["VOLTS"],
                "VOLTS": ew.ENUM_UNITS["VOLTS"],
                # This is weird, but evalresp appears to do the same.
                "V/M": ew.ENUM_UNITS["VOLTS"],
                "COUNTS": ew.ENUM_UNITS["COUNTS"],
                "T": ew.ENUM_UNITS["TESLA"],
                "PA": ew.ENUM_UNITS["PRESSURE"],
                "MBAR": ew.ENUM_UNITS["PRESSURE"]}
            if key not in units_mapping:
                if key is not None:
                    msg = ("The unit '%s' is not known to ObsPy. Raw evalresp "
                           "would refuse to calculate a response for this "
                           "channel. Proceed with caution.") % key
                    warnings.warn(msg)
                value = ew.ENUM_UNITS["UNDEF_UNITS"]
            else:
                value = units_mapping[key]

            # Scale factor with the same logic as evalresp.
            if key in ["CM/S**2", "CM/S", "CM/SEC", "CM"]:
                scale_factor[0] = 1.0E2
            elif key in ["MM/S**2", "MM/S", "MM/SEC", "MM"]:
                scale_factor[0] = 1.0E3
            elif key in ["NM/S**2", "NM/S", "NM/SEC", "NM"]:
                scale_factor[0] = 1.0E9

            return value

        all_stages = defaultdict(list)

        for stage in self.response_stages:
            # optionally select only stages as requested by user
            if start_stage is not None:
                if stage.stage_sequence_number < start_stage:
                    continue
            if end_stage is not None:
                if stage.stage_sequence_number > end_stage:
                    continue
            all_stages[stage.stage_sequence_number].append(stage)

        stage_lengths = set(map(len, all_stages.values()))
        if len(stage_lengths) != 1 or stage_lengths.pop() != 1:
            msg = "Each stage can only appear once."
            raise ValueError(msg)

        stage_list = sorted(all_stages.keys())

        stage_objects = []

        for stage_number in stage_list:
            st = ew.stage()
            st.sequence_no = stage_number

            stage_blkts = []

            blockette = all_stages[stage_number][0]

            # Write the input and output units.
            st.input_units = get_unit_mapping(blockette.input_units)
            st.output_units = get_unit_mapping(blockette.output_units)

            if isinstance(blockette, PolesZerosResponseStage):
                blkt = ew.blkt()
                # Map the transfer function type.
                transfer_fct_mapping = {
                    "LAPLACE (RADIANS/SECOND)": "LAPLACE_PZ",
                    "LAPLACE (HERTZ)": "ANALOG_PZ",
                    "DIGITAL (Z-TRANSFORM)": "IIR_PZ"}
                blkt.type = ew.ENUM_FILT_TYPES[transfer_fct_mapping[
                    blockette.pz_transfer_function_type]]

                # The blockette is a pole zero blockette.
                pz = blkt.blkt_info.pole_zero

                pz.nzeros = len(blockette.zeros)
                pz.npoles = len(blockette.poles)
                pz.a0 = blockette.normalization_factor
                pz.a0_freq = blockette.normalization_frequency

                # XXX: Find a better way to do this.
                poles = (ew.complex_number * len(blockette.poles))()
                for i, value in enumerate(blockette.poles):
                    poles[i].real = value.real
                    poles[i].imag = value.imag

                zeros = (ew.complex_number * len(blockette.zeros))()
                for i, value in enumerate(blockette.zeros):
                    zeros[i].real = value.real
                    zeros[i].imag = value.imag

                pz.poles = C.cast(C.pointer(poles),
                                  C.POINTER(ew.complex_number))
                pz.zeros = C.cast(C.pointer(zeros),
                                  C.POINTER(ew.complex_number))
            elif isinstance(blockette, CoefficientsTypeResponseStage):
                blkt = ew.blkt()
                # This type can have either an FIR or an IIR response. If
                # the number of denominators is 0, it is a FIR. Otherwise
                # an IIR.

                # FIR
                if len(blockette.denominator) == 0:
                    if blockette.cf_transfer_function_type.lower() \
                            != "digital":
                        msg = ("When no denominators are given it must "
                               "be a digital FIR filter.")
                        raise ValueError(msg)
                    # Set the type to an asymmetric FIR blockette.
                    blkt.type = ew.ENUM_FILT_TYPES["FIR_ASYM"]
                    fir = blkt.blkt_info.fir
                    fir.h0 = 1.0
                    fir.ncoeffs = len(blockette.numerator)
                    # XXX: Find a better way to do this.
                    coeffs = (C.c_double * len(blockette.numerator))()
                    for i, value in enumerate(blockette.numerator):
                        coeffs[i] = float(value)
                    fir.coeffs = C.cast(C.pointer(coeffs),
                                        C.POINTER(C.c_double))
                # IIR
                else:
                    blkt.type = ew.ENUM_FILT_TYPES["IIR_COEFFS"]
                    coeff = blkt.blkt_info.coeff

                    coeff.h0 = 1.0
                    coeff.nnumer = len(blockette.numerator)
                    coeff.ndenom = len(blockette.denominator)

                    # XXX: Find a better way to do this.
                    coeffs = (C.c_double * len(blockette.numerator))()
                    for i, value in enumerate(blockette.numerator):
                        coeffs[i] = float(value)
                    coeff.numer = C.cast(C.pointer(coeffs),
                                         C.POINTER(C.c_double))
                    coeffs = (C.c_double * len(blockette.denominator))()
                    for i, value in enumerate(blockette.denominator):
                        coeffs[i] = float(value)
                    coeff.denom = C.cast(C.pointer(coeffs),
                                         C.POINTER(C.c_double))
            elif isinstance(blockette, ResponseListResponseStage):
                msg = ("ResponseListResponseStage not yet implemented due to "
                       "missing example data. Please contact the developers "
                       "with a test data set (waveforms and StationXML "
                       "metadata).")
                raise NotImplementedError(msg)
            elif isinstance(blockette, FIRResponseStage):
                blkt = ew.blkt()

                if blockette.symmetry == "NONE":
                    blkt.type = ew.ENUM_FILT_TYPES["FIR_ASYM"]
                if blockette.symmetry == "ODD":
                    blkt.type = ew.ENUM_FILT_TYPES["FIR_SYM_1"]
                if blockette.symmetry == "EVEN":
                    blkt.type = ew.ENUM_FILT_TYPES["FIR_SYM_2"]

                # The blockette is a fir blockette
                fir = blkt.blkt_info.fir
                fir.h0 = 1.0
                fir.ncoeffs = len(blockette.coefficients)

                # XXX: Find a better way to do this.
                coeffs = (C.c_double * len(blockette.coefficients))()
                for i, value in enumerate(blockette.coefficients):
                    coeffs[i] = float(value)
                fir.coeffs = C.cast(C.pointer(coeffs),
                                    C.POINTER(C.c_double))
            elif isinstance(blockette, PolynomialResponseStage):
                msg = ("PolynomialResponseStage not yet implemented. "
                       "Please contact the developers.")
                raise NotImplementedError(msg)
            else:
                # Otherwise it could be a gain only stage.
                if blockette.stage_gain is not None and \
                        blockette.stage_gain_frequency is not None:
                    blkt = None
                else:
                    msg = "Type: %s." % str(type(blockette))
                    raise NotImplementedError(msg)

            if blkt is not None:
                stage_blkts.append(blkt)

            # Parse the decimation if is given.
            decimation_values = set([
                blockette.decimation_correction,
                blockette.decimation_delay, blockette.decimation_factor,
                blockette.decimation_input_sample_rate,
                blockette.decimation_offset])
            if None in decimation_values:
                if len(decimation_values) != 1:
                    msg = ("If a decimation is given, all values must "
                           "be specified.")
                    raise ValueError(msg)
            else:
                blkt = ew.blkt()
                blkt.type = ew.ENUM_FILT_TYPES["DECIMATION"]
                decimation_blkt = blkt.blkt_info.decimation

                # Evalresp does the same!
                if blockette.decimation_input_sample_rate == 0:
                    decimation_blkt.sample_int = 0.0
                else:
                    decimation_blkt.sample_int = \
                        1.0 / blockette.decimation_input_sample_rate

                decimation_blkt.deci_fact = blockette.decimation_factor
                decimation_blkt.deci_offset = blockette.decimation_offset
                decimation_blkt.estim_delay = blockette.decimation_delay
                decimation_blkt.applied_corr = \
                    blockette.decimation_correction
                stage_blkts.append(blkt)

            # Add the gain if it is available.
            if blockette.stage_gain is not None and \
                    blockette.stage_gain_frequency is not None:
                blkt = ew.blkt()
                blkt.type = ew.ENUM_FILT_TYPES["GAIN"]
                gain_blkt = blkt.blkt_info.gain
                gain_blkt.gain = blockette.stage_gain
                gain_blkt.gain_freq = blockette.stage_gain_frequency
                stage_blkts.append(blkt)

            if not stage_blkts:
                msg = "At least one blockette is needed for the stage."
                raise ValueError(msg)

            # Attach the blockette chain to the stage.
            st.first_blkt = C.pointer(stage_blkts[0])
            for _i in range(1, len(stage_blkts)):
                stage_blkts[_i - 1].next_blkt = C.pointer(stage_blkts[_i])

            stage_objects.append(st)

        # Attach the instrument sensitivity as stage 0 at the end.
        st = ew.stage()
        st.sequence_no = 0
        st.input_units = 0
        st.output_units = 0
        blkt = ew.blkt()
        blkt.type = ew.ENUM_FILT_TYPES["GAIN"]
        gain_blkt = blkt.blkt_info.gain
        gain_blkt.gain = self.instrument_sensitivity.value
        gain_blkt.gain_freq = self.instrument_sensitivity.frequency
        st.first_blkt = C.pointer(blkt)
        stage_objects.append(st)

        chan = ew.channel()
        if not stage_objects:
            msg = "At least one stage is needed."
            raise ValueError(msg)

        # Attach the stage chain to the channel.
        chan.first_stage = C.pointer(stage_objects[0])
        for _i in range(1, len(stage_objects)):
            stage_objects[_i - 1].next_stage = C.pointer(stage_objects[_i])

        chan.nstages = len(stage_objects)

        # Evalresp will take care of setting it to the overall sensitivity.
        chan.sensit = 0.0
        chan.sensfreq = 0.0

        fy = 1 / (t_samp * 2.0)
        # start at zero to get zero for offset/ DC of fft
        freqs = np.linspace(0, fy, nfft // 2 + 1).astype(np.float64)

        output = np.empty(len(freqs), dtype=np.complex128)
        out_units = C.c_char_p(out_units.encode('ascii', 'strict'))

        # Set global variables
        if self.resource_id:
            clibevresp.curr_file.value = self.resource_id.encode('utf-8')
        else:
            clibevresp.curr_file.value = None

        try:
            rc = clibevresp._obspy_check_channel(C.byref(chan))
            if rc:
                e, m = ew.ENUM_ERROR_CODES[rc]
                raise e('check_channel: ' + m)

            rc = clibevresp._obspy_norm_resp(C.byref(chan), -1, 0)
            if rc:
                e, m = ew.ENUM_ERROR_CODES[rc]
                raise e('norm_resp: ' + m)

            rc = clibevresp._obspy_calc_resp(C.byref(chan), freqs, len(freqs),
                                             output, out_units, -1, 0, 0)
            if rc:
                e, m = ew.ENUM_ERROR_CODES[rc]
                raise e('calc_resp: ' + m)

            # XXX: Check if this is really not needed.
            # output *= scale_factor[0]

        finally:
            clibevresp.curr_file.value = None

        return output, freqs

    def __str__(self):
        i_s = self.instrument_sensitivity
        if i_s:
            input_units = i_s.input_units \
                if i_s.input_units else "UNKNOWN"
            input_units_description = i_s.input_units_description \
                if i_s.input_units_description else ""
            output_units = i_s.output_units \
                if i_s.output_units else "UNKNOWN"
            output_units_description = i_s.output_units_description \
                if i_s.output_units_description else ""
            sensitivity = ("%g" % i_s.value) if i_s.value else "UNKNOWN"
            freq = ("%.3f" % i_s.frequency) if i_s.frequency else "UNKNOWN"
        else:
            input_units = "UNKNOWN"
            input_units_description = ""
            output_units = "UNKNOWN"
            output_units_description = ""
            sensitivity = "UNKNOWN"
            freq = "UNKNOWN"

        ret = (
            "Channel Response\n"
            "\tFrom {input_units} ({input_units_description}) to "
            "{output_units} ({output_units_description})\n"
            "\tOverall Sensitivity: {sensitivity} defined at {freq} Hz\n"
            "\t{stages} stages:\n{stage_desc}").format(
            input_units=input_units,
            input_units_description=input_units_description,
            output_units=output_units,
            output_units_description=output_units_description,
            sensitivity=sensitivity,
            freq=freq,
            stages=len(self.response_stages),
            stage_desc="\n".join(
                ["\t\tStage %i: %s from %s to %s,"
                 " gain: %s" % (
                     i.stage_sequence_number, i.__class__.__name__,
                     i.input_units, i.output_units,
                     ("%g" % i.stage_gain) if i.stage_gain else "UNKNOWN")
                 for i in self.response_stages]))
        return ret

    def _repr_pretty_(self, p, cycle):
        p.text(str(self))

    def plot(self, min_freq, output="VEL", start_stage=None,
             end_stage=None, label=None, axes=None, sampling_rate=None,
             unwrap_phase=False, show=True, outfile=None):
        """
        Show bode plot of instrument response.

        :type min_freq: float
        :param min_freq: Lowest frequency to plot.
        :type output: str
        :param output: Output units. One of:

                ``"DISP"``
                    displacement
                ``"VEL"``
                    velocity
                ``"ACC"``
                    acceleration

        :type start_stage: int, optional
        :param start_stage: Stage sequence number of first stage that will be
            used (disregarding all earlier stages).
        :type end_stage: int, optional
        :param end_stage: Stage sequence number of last stage that will be
            used (disregarding all later stages).
        :type label: str
        :param label: Label string for legend.
        :type axes: list of 2 :class:`matplotlib.axes.Axes`
        :param axes: List/tuple of two axes instances to plot the
            amplitude/phase spectrum into. If not specified, a new figure is
            opened.
        :type sampling_rate: float
        :param sampling_rate: Manually specify sampling rate of time series.
            If not given it is attempted to determine it from the information
            in the individual response stages.  Does not influence the spectra
            calculation, if it is not known, just provide the highest frequency
            that should be plotted times two.
        :type unwrap_phase: bool
        :param unwrap_phase: Set optional phase unwrapping using NumPy.
        :type show: bool
        :param show: Whether to show the figure after plotting or not. Can be
            used to do further customization of the plot before showing it.
        :type outfile: str
        :param outfile: Output file path to directly save the resulting image
            (e.g. ``"/tmp/image.png"``). Overrides the ``show`` option, image
            will not be displayed interactively. The given path/filename is
            also used to automatically determine the output format. Supported
            file formats depend on your matplotlib backend.  Most backends
            support png, pdf, ps, eps and svg. Defaults to ``None``.

        .. rubric:: Basic Usage

        >>> from obspy import read_inventory
        >>> resp = read_inventory()[0][0][0].response
        >>> resp.plot(0.001, output="VEL")  # doctest: +SKIP

        .. plot::

            from obspy import read_inventory
            resp = read_inventory()[0][0][0].response
            resp.plot(0.001, output="VEL")
        """
        import matplotlib.pyplot as plt
        from matplotlib.transforms import blended_transform_factory

        # detect sampling rate from response stages
        if sampling_rate is None:
            for stage in self.response_stages[::-1]:
                if (stage.decimation_input_sample_rate is not None
                        and stage.decimation_factor is not None):
                    sampling_rate = (stage.decimation_input_sample_rate /
                                     stage.decimation_factor)
                    break
            else:
                msg = ("Failed to autodetect sampling rate of channel from "
                       "response stages. Please manually specify parameter "
                       "`sampling_rate`")
                raise Exception(msg)

        t_samp = 1.0 / sampling_rate
        nyquist = sampling_rate / 2.0
        nfft = sampling_rate / min_freq

        cpx_response, freq = self.get_evalresp_response(
            t_samp=t_samp, nfft=nfft, output=output, start_stage=start_stage,
            end_stage=end_stage)

        if axes:
            ax1, ax2 = axes
            fig = ax1.figure
        else:
            fig = plt.figure()
            ax1 = fig.add_subplot(211)
            ax2 = fig.add_subplot(212, sharex=ax1)

        label_kwarg = {}
        if label is not None:
            label_kwarg['label'] = label

        # plot amplitude response
        lw = 1.5
        lines = ax1.loglog(freq, abs(cpx_response), lw=lw, **label_kwarg)
        color = lines[0].get_color()
        # Cannot be plotted with matplotlib < 1.0.0
        if MATPLOTLIB_VERSION < [1, 0, 0]:
            warnings.warn("Cannot plot the instrument sensitivity. Your "
                          "matplotlib version is too old. Please update.")
        if self.instrument_sensitivity and MATPLOTLIB_VERSION >= [1, 0, 0]:
            trans_above = blended_transform_factory(ax1.transData,
                                                    ax1.transAxes)
            trans_right = blended_transform_factory(ax1.transAxes,
                                                    ax1.transData)
            arrowprops = dict(
                arrowstyle="wedge,tail_width=1.4,shrink_factor=0.8", fc=color)
            bbox = dict(boxstyle="round", fc="w")
            ax1.annotate("%.1g" % self.instrument_sensitivity.frequency,
                         (self.instrument_sensitivity.frequency, 1.0),
                         xytext=(self.instrument_sensitivity.frequency, 1.1),
                         xycoords=trans_above, textcoords=trans_above,
                         ha="center", va="bottom",
                         arrowprops=arrowprops, bbox=bbox)
            ax1.annotate("%.1e" % self.instrument_sensitivity.value,
                         (1.0, self.instrument_sensitivity.value),
                         xytext=(1.05, self.instrument_sensitivity.value),
                         xycoords=trans_right, textcoords=trans_right,
                         ha="left", va="center",
                         arrowprops=arrowprops, bbox=bbox)

        # plot phase response
        phase = np.angle(cpx_response)
        if unwrap_phase:
            phase = np.unwrap(phase)
        ax2.semilogx(freq, phase, color=color, lw=lw)

        # plot nyquist frequency
        for ax in (ax1, ax2):
            ax.axvline(nyquist, ls="--", color=color, lw=lw)

        # only do adjustments if we initialized the figure in here
        if not axes:
            _adjust_bode_plot_figure(fig, show=False)

        if outfile:
            fig.savefig(outfile)
        else:
            if show:
                plt.show()

        return fig

    def get_paz(self):
        """
        Get Poles and Zeros stage.

        Prints a warning if more than one poles and zeros stage is found.
        Raises if no poles and zeros stage is found.

        :rtype: :class:`PolesZerosResponseStage`
        :returns: Poles and Zeros response stage.
        """
        paz = [deepcopy(stage) for stage in self.response_stages
               if isinstance(stage, PolesZerosResponseStage)]
        if len(paz) == 0:
            msg = "No PolesZerosResponseStage found."
            raise Exception(msg)
        elif len(paz) > 1:
            msg = ("More than one PolesZerosResponseStage encountered. "
                   "Returning first one found.")
            warnings.warn(msg)
        return paz[0]

    def get_sacpz(self):
        """
        Returns SACPZ ASCII text representation of Response.

        :rtype: str
        :returns: Textual SACPZ representation of response.
        """
        # extract paz
        paz = self.get_paz()
        sensitivity = self.instrument_sensitivity.value
        return paz_to_sacpz_string(paz, sensitivity)


def paz_to_sacpz_string(paz, instrument_sensitivity):
    """
    Returns SACPZ ASCII text representation of Response.

    :type paz: :class:`PolesZerosResponseStage`
    :param paz: Poles and Zeros response information
    :type instrument_sensitivity: :class:`InstrumentSensitivity`
    :param paz: Overall instrument sensitivity of response
    :rtype: str
    :returns: Textual SACPZ representation of poles and zeros response stage.
    """
    # assemble output string
    out = []
    out.append("ZEROS %i" % len(paz.zeros))
    for c in paz.zeros:
        out.append(" %+.6e %+.6e" % (c.real, c.imag))
    out.append("POLES %i" % len(paz.poles))
    for c in paz.poles:
        out.append(" %+.6e %+.6e" % (c.real, c.imag))
    constant = paz.normalization_factor * instrument_sensitivity.value
    out.append("CONSTANT %.6e" % constant)
    return "\n".join(out)


class InstrumentSensitivity(ComparingObject):
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
    def __init__(self, value, frequency, input_units,
                 output_units, input_units_description=None,
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
        :param input_units: string
        :param input_units: The units of the data as input from the
            perspective of data acquisition. After correcting data for this
            response, these would be the resulting units.
            Name of units, e.g. "M/S", "V", "PA".
        :param input_units_description: string, optional
        :param input_units_description: The units of the data as input from the
            perspective of data acquisition. After correcting data for this
            response, these would be the resulting units.
            Description of units, e.g. "Velocity in meters per second",
            "Volts", "Pascals".
        :param output_units: string
        :param output_units: The units of the data as output from the
            perspective of data acquisition. These would be the units of the
            data prior to correcting for this response.
            Name of units, e.g. "M/S", "V", "PA".
        :type output_units_description: str, optional
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
        self.input_units = input_units
        self.input_units_description = input_units_description
        self.output_units = output_units
        self.output_units_description = output_units_description
        self.frequency_range_start = frequency_range_start
        self.frequency_range_end = frequency_range_end
        self.frequency_range_DB_variation = frequency_range_DB_variation


# XXX duplicated code, PolynomialResponseStage could probably be implemented by
# XXX inheriting from both InstrumentPolynomial and ResponseStage
class InstrumentPolynomial(ComparingObject):
    """
    From the StationXML Definition:
        The total sensitivity for a channel, representing the complete
        acquisition system expressed as a polynomial. Equivalent to SEED stage
        0 polynomial (blockette 62).
    """
    def __init__(self, input_units, output_units,
                 frequency_lower_bound,
                 frequency_upper_bound, approximation_lower_bound,
                 approximation_upper_bound, maximum_error, coefficients,
                 approximation_type='MACLAURIN', resource_id=None, name=None,
                 input_units_description=None,
                 output_units_description=None, description=None):
        """
        :type approximation_type: str
        :param approximation_type: Approximation type. Currently restricted to
            'MACLAURIN' by StationXML definition.
        :type frequency_lower_bound: float
        :param frequency_lower_bound: Lower frequency bound.
        :type frequency_upper_bound: float
        :param frequency_upper_bound: Upper frequency bound.
        :type approximation_lower_bound: float
        :param approximation_lower_bound: Lower bound of approximation.
        :type approximation_upper_bound: float
        :param approximation_upper_bound: Upper bound of approximation.
        :type maximum_error: float
        :param maximum_error: Maximum error.
        :type coefficients: list of floats
        :param coefficients: List of polynomial coefficients.
        :param input_units: string
        :param input_units: The units of the data as input from the
            perspective of data acquisition. After correcting data for this
            response, these would be the resulting units.
            Name of units, e.g. "M/S", "V", "PA".
        :param output_units: string
        :param output_units: The units of the data as output from the
            perspective of data acquisition. These would be the units of the
            data prior to correcting for this response.
            Name of units, e.g. "M/S", "V", "PA".
        :type resource_id: str
        :param resource_id: This field contains a string that should serve as a
            unique resource identifier. This identifier can be interpreted
            differently depending on the data center/software that generated
            the document. Also, we recommend to use something like
            GENERATOR:Meaningful ID. As a common behavior equipment with the
            same ID should contains the same information/be derived from the
            same base instruments.
        :type name: str
        :param name: A name given to the filter stage.
        :param input_units_description: string, optional
        :param input_units_description: The units of the data as input from the
            perspective of data acquisition. After correcting data for this
            response, these would be the resulting units.
            Description of units, e.g. "Velocity in meters per second",
            "Volts", "Pascals".
        :type output_units_description: str, optional
        :param output_units_description: The units of the data as output from
            the perspective of data acquisition. These would be the units of
            the data prior to correcting for this response.
            Description of units, e.g. "Velocity in meters per second",
            "Volts", "Pascals".
        :type description: str, optional
        :param description: A short description of of the filter.
        """
        self.input_units = input_units
        self.output_units = output_units
        self.input_units_description = input_units_description
        self.output_units_description = output_units_description
        self.resource_id = resource_id
        self.name = name
        self.description = description
        self._approximation_type = approximation_type
        self.frequency_lower_bound = frequency_lower_bound
        self.frequency_upper_bound = frequency_upper_bound
        self.approximation_lower_bound = approximation_lower_bound
        self.approximation_upper_bound = approximation_upper_bound
        self.maximum_error = maximum_error
        self.coefficients = coefficients

    @property
    def approximation_type(self):
        return self._approximation_type

    @approximation_type.setter
    def approximation_type(self, value):
        value = str(value).upper()
        allowed = ("MACLAURIN",)
        if value not in allowed:
            msg = ("Value '%s' for polynomial response approximation type not "
                   "allowed. Possible values are: '%s'")
            msg = msg % (value, "', '".join(allowed))
            raise ValueError(msg)
        self._approximation_type = value


class FilterCoefficient(CustomFloat):
    """
    A filter coefficient.
    """
    def __init__(self, value, number=None):
        """
        :type value: float
        :param value: The actual value of the coefficient
        :type number: int, optional
        :param number: Number to indicate the position of the coefficient.
        """
        super(FilterCoefficient, self).__init__(value)
        self.number = number

    @property
    def number(self):
        return self._number

    @number.setter
    def number(self, value):
        if value is not None:
            value = int(value)
        self._number = value


class CoefficientWithUncertainties(FloatWithUncertainties):
    """
    A coefficient with optional uncertainties.
    """
    def __init__(self, value, number=None, lower_uncertainty=None,
                 upper_uncertainty=None):
        """
        :type value: float
        :param value: The actual value of the coefficient
        :type number: int, optional
        :param number: Number to indicate the position of the coefficient.
        :type lower_uncertainty: float
        :param lower_uncertainty: Lower uncertainty (aka minusError)
        :type upper_uncertainty: float
        :param upper_uncertainty: Upper uncertainty (aka plusError)
        """
        super(CoefficientWithUncertainties, self).__init__(
            value, lower_uncertainty=lower_uncertainty,
            upper_uncertainty=upper_uncertainty)
        self.number = number

    @property
    def number(self):
        return self._number

    @number.setter
    def number(self, value):
        if value is not None:
            value = int(value)
        self._number = value


def _adjust_bode_plot_figure(fig, grid=True, show=True):
    """
    Helper function to do final adjustments to Bode plot figure.
    """
    import matplotlib.pyplot as plt
    # make more room in between subplots for the ylabel of right plot
    fig.subplots_adjust(hspace=0.02, top=0.87, right=0.82)
    ax1, ax2 = fig.axes[:2]
    # workaround for older matplotlib versions
    try:
        ax1.legend(loc="lower center", ncol=3, fontsize='small')
    except TypeError:
        leg_ = ax1.legend(loc="lower center", ncol=3)
        leg_.prop.set_size("small")
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax2.get_yticklabels()[-1], visible=False)
    ax1.set_ylabel('Amplitude')
    minmax1 = ax1.get_ylim()
    ax1.set_ylim(top=minmax1[1] * 5)
    ax1.grid(True)
    ax2.set_xlabel('Frequency [Hz]')
    ax2.set_ylabel('Phase [rad]')
    minmax2 = ax2.yaxis.get_data_interval()
    yticks2 = np.arange(minmax2[0] - minmax2[0] % (pi / 2),
                        minmax2[1] - minmax2[1] % (pi / 2) + pi, pi / 2)
    ax2.set_yticks(yticks2)
    ax2.set_yticklabels([_pitick2latex(x) for x in yticks2])
    ax2.grid(True)
    if show:
        plt.show()


def _pitick2latex(x):
    """
    Helper function to convert a float that is a multiple of pi/2
    to a latex string.
    """
    # safety check, if no multiple of pi/2 return normal representation
    if x % (pi / 2) != 0:
        return "%#.3g" % x
    string = "$"
    if x < 0:
        string += "-"
    if x / pi % 1 == 0:
        x = abs(int(x / pi))
        if x == 0:
            return "$0$"
        elif x == 1:
            x = ""
        string += r"%s\pi$" % x
    else:
        x = abs(int(2 * x / pi))
        if x == 1:
            x = ""
        string += r"\frac{%s\pi}{2}$" % x
    return string


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
