# -*- coding: utf-8 -*-
"""
obspy.core.event.magnitude - The Magnitude class definition
===========================================================
This module provides a class hierarchy to consistently handle event metadata.
This class hierarchy is closely modelled after the de-facto standard format
`QuakeML <https://quake.ethz.ch/quakeml/>`_.

.. note::

    For handling additional information not covered by the QuakeML standard and
    how to output it to QuakeML see the :ref:`ObsPy Tutorial <quakeml-extra>`.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

from obspy import UTCDateTime
from obspy.core.event.base import (
    _event_type_class_factory, CreationInfo,
    WaveformStreamID, TimeWindow)
from obspy.core.event import ResourceIdentifier
from obspy.core.event.header import (
    AmplitudeCategory, AmplitudeUnit, EvaluationMode, EvaluationStatus,
    ATTRIBUTE_HAS_ERRORS)


__Magnitude = _event_type_class_factory(
    "__Magnitude",
    class_attributes=[("resource_id", ResourceIdentifier),
                      ("mag", float, ATTRIBUTE_HAS_ERRORS),
                      ("magnitude_type", str),
                      ("origin_id", ResourceIdentifier),
                      ("method_id", ResourceIdentifier),
                      ("station_count", int),
                      ("azimuthal_gap", float),
                      ("evaluation_mode", EvaluationMode),
                      ("evaluation_status", EvaluationStatus),
                      ("creation_info", CreationInfo)],
    class_contains=["comments", "station_magnitude_contributions"])


class Magnitude(__Magnitude):
    """
    Describes a magnitude which can, but does not need to be associated with an
    origin.

    Association with an origin is expressed with the optional attribute
    originID. It is either a combination of different magnitude estimations, or
    it represents the reported magnitude for the given event.

    :type resource_id: :class:`~obspy.core.event.resourceid.ResourceIdentifier`
    :param resource_id: Resource identifier of Magnitude.
    :type force_resource_id: bool, optional
    :param force_resource_id: If set to False, the automatic initialization of
        `resource_id` attribute in case it is not specified will be skipped.
    :type mag: float
    :param mag: Resulting magnitude value from combining values of type
        :class:`~obspy.core.event.magnitude.StationMagnitude`.
        If no estimations are available, this value can represent the
        reported magnitude.
    :type mag_errors: :class:`~obspy.core.event.base.QuantityError`
    :param mag_errors: AttribDict containing error quantities.
    :type magnitude_type: str, optional
    :param magnitude_type: Describes the type of magnitude. This is a free-text
        field because it is impossible to cover all existing magnitude type
        designations with an enumeration. Possible values are:

        * unspecified magnitude (``'M'``),
        * local magnitude (``'ML'``),
        * body wave magnitude (``'Mb'``),
        * surface wave magnitude (``'MS'``),
        * moment magnitude (``'Mw'``),
        * duration magnitude (``'Md'``)
        * coda magnitude (``'Mc'``)
        * ``'MH'``, ``'Mwp'``, ``'M50'``, ``'M100'``, etc.

    :type origin_id: :class:`~obspy.core.event.resourceid.ResourceIdentifier`,
        optional
    :param origin_id: Reference to an origin’s resource_id if the magnitude has
        an associated Origin.
    :type method_id: :class:`~obspy.core.event.resourceid.ResourceIdentifier`,
        optional
    :param method_id: Identifies the method of magnitude estimation. Users
        should avoid to give contradictory information in method_id and
        magnitude_type.
    :type station_count: int, optional
    :param station_count: Number of used stations for this magnitude
        computation.
    :type azimuthal_gap: float, optional
    :param azimuthal_gap: Azimuthal gap for this magnitude computation.
        Unit: deg
    :type evaluation_mode: str, optional
    :param evaluation_mode: Evaluation mode of Magnitude.
        See :class:`~obspy.core.event.header.EvaluationMode` for allowed
        values.
    :type evaluation_status: str, optional
    :param evaluation_status: Evaluation status of Magnitude.
        See :class:`~obspy.core.event.header.EvaluationStatus` for allowed
        values.
    :type comments: list of :class:`~obspy.core.event.base.Comment`, optional
    :param comments: Additional comments.
    :type station_magnitude_contributions: list of
        :class:`~obspy.core.event.magnitude.StationMagnitudeContribution`.
    :param station_magnitude_contributions: StationMagnitudeContribution
        instances associated with the Magnitude.
    :type creation_info: :class:`~obspy.core.event.base.CreationInfo`, optional
    :param creation_info: Creation information used to describe author,
        version, and creation time.

    .. note::

        For handling additional information not covered by the QuakeML
        standard and how to output it to QuakeML see the
        :ref:`ObsPy Tutorial <quakeml-extra>`.
    """


__StationMagnitude = _event_type_class_factory(
    "__StationMagnitude",
    class_attributes=[("resource_id", ResourceIdentifier),
                      ("origin_id", ResourceIdentifier),
                      ("mag", float, ATTRIBUTE_HAS_ERRORS),
                      ("station_magnitude_type", str),
                      ("amplitude_id", ResourceIdentifier),
                      ("method_id", ResourceIdentifier),
                      ("waveform_id", WaveformStreamID),
                      ("creation_info", CreationInfo)],
    class_contains=["comments"])


class StationMagnitude(__StationMagnitude):
    """
    This class describes the magnitude derived from a single waveform stream.

    :type resource_id:
        :class:`~obspy.core.event.resourceid.ResourceIdentifier`
    :param resource_id: Resource identifier of StationMagnitude.
    :type force_resource_id: bool, optional
    :param force_resource_id: If set to False, the automatic initialization of
        `resource_id` attribute in case it is not specified will be skipped.
    :type origin_id: :class:`~obspy.core.event.resourceid.ResourceIdentifier`
    :param origin_id: Reference to an origin’s ``resource_id`` if the
        StationMagnitude has an associated :class:`~obspy.core.event.Origin`.
    :type mag: float
    :param mag: Estimated magnitude.
    :type mag_errors: :class:`~obspy.core.event.base.QuantityError`
    :param mag_errors: AttribDict containing error quantities.
    :type station_magnitude_type: str, optional
    :param station_magnitude_type: See
        :class:`~obspy.core.event.magnitude.Magnitude`
    :type amplitude_id:
        :class:`~obspy.core.event.resourceid.ResourceIdentifier`
    :param amplitude_id: Identifies the data source of the StationMagnitude.
        For magnitudes derived from amplitudes in waveforms (e.g., local
        magnitude ML), amplitudeID points to publicID in class Amplitude.
    :type method_id: :class:`~obspy.core.event.resourceid.ResourceIdentifier`
    :param method_id: See :class:`~obspy.core.event.magnitude.Magnitude`
    :type waveform_id: :class:`~obspy.core.event.base.WaveformStreamID`,
        optional
    :param waveform_id: Identifies the waveform stream. This element can be
        helpful if no amplitude is referenced, or the amplitude is not
        available in the context. Otherwise, it would duplicate the waveform_id
        provided there and can be omitted.
    :type comments: list of :class:`~obspy.core.event.base.Comment`, optional
    :param comments: Additional comments.
    :type creation_info: :class:`~obspy.core.event.base.CreationInfo`, optional
    :param creation_info: Creation information used to describe author,
        version, and creation time.

    .. note::

        For handling additional information not covered by the QuakeML
        standard and how to output it to QuakeML see the
        :ref:`ObsPy Tutorial <quakeml-extra>`.
    """


__StationMagnitudeContribution = _event_type_class_factory(
    "__StationMagnitudeContribution",
    class_attributes=[("station_magnitude_id", ResourceIdentifier),
                      ("residual", float),
                      ("weight", float)])


class StationMagnitudeContribution(__StationMagnitudeContribution):
    """
    This class describes the weighting of magnitude values from several
    StationMagnitude objects for computing a network magnitude estimation.

    :type station_magnitude_id:
        :class:`~obspy.core.event.resourceid.ResourceIdentifier`
    :param station_magnitude_id: Refers to the resource_id of a
        StationMagnitude object.
    :type residual: float, optional
    :param residual: Residual of magnitude computation.
    :type weight: float, optional
    :param weight: Weight of the magnitude value from class StationMagnitude
        for computing the magnitude value in class Magnitude. Note that there
        is no rule for the sum of the weights of all station magnitude
        contributions to a specific network magnitude. In particular, the
        weights are not required to sum up to unity.

    .. note::

        For handling additional information not covered by the QuakeML
        standard and how to output it to QuakeML see the
        :ref:`ObsPy Tutorial <quakeml-extra>`.
    """


__Amplitude = _event_type_class_factory(
    "__Amplitude",
    class_attributes=[("resource_id", ResourceIdentifier),
                      ("generic_amplitude", float, ATTRIBUTE_HAS_ERRORS),
                      ("type", str),
                      ("category", AmplitudeCategory),
                      ("unit", AmplitudeUnit),
                      ("method_id", ResourceIdentifier),
                      ("period", float, ATTRIBUTE_HAS_ERRORS),
                      ("snr", float),
                      ("time_window", TimeWindow),
                      ("pick_id", ResourceIdentifier),
                      ("waveform_id", WaveformStreamID),
                      ("filter_id", ResourceIdentifier),
                      ("scaling_time", UTCDateTime, ATTRIBUTE_HAS_ERRORS),
                      ("magnitude_hint", str),
                      ("evaluation_mode", EvaluationMode),
                      ("evaluation_status", EvaluationStatus),
                      ("creation_info", CreationInfo)],
    class_contains=["comments"])


class Amplitude(__Amplitude):
    """
    This class represents a quantification of the waveform anomaly, usually a
    single amplitude measurement or a measurement of the visible signal
    duration for duration magnitudes.

    :type resource_id: :class:`~obspy.core.event.resourceid.ResourceIdentifier`
    :param resource_id: Resource identifier of Amplitude.
    :type force_resource_id: bool, optional
    :param force_resource_id: If set to False, the automatic initialization of
        `resource_id` attribute in case it is not specified will be skipped.
    :type generic_amplitude: float
    :param generic_amplitude: Measured amplitude value for the given
        waveformID. Note that this attribute can describe different physical
        quantities, depending on the type and category of the amplitude. These
        can be, e.g., displacement, velocity, or a period. If the only
        amplitude information is a period, it has to specified here, not in the
        period attribute. The latter can be used if the amplitude measurement
        contains information on, e.g., displacement and an additional period.
        Since the physical quantity described by this attribute is not fixed,
        the unit of measurement cannot be defined in advance. However, the
        quantity has to be specified in SI base units. The enumeration given in
        attribute unit provides the most likely units that could be needed
        here. For clarity, using the optional unit attribute is highly
        encouraged.
    :type generic_amplitude_errors:
        :class:`~obspy.core.event.base.QuantityError`
    :param generic_amplitude_errors: AttribDict containing error quantities.
    :type type: str, optional
    :param type: Describes the type of amplitude using the nomenclature from
        Storchak et al. (2003). Possible values are:

        * unspecified amplitude reading (``'A'``),
        * amplitude reading for local magnitude (``'AML'``),
        * amplitude reading for body wave magnitude (``'AMB'``),
        * amplitude reading for surface wave magnitude (``'AMS'``), and
        * time of visible end of record for duration magnitude (``'END'``).

    :type category: str, optional
    :param category:  Amplitude category.  This attribute describes the way the
        waveform trace is evaluated to derive an amplitude value. This can be
        just reading a single value for a given point in time (point), taking a
        mean value over a time interval (mean), integrating the trace over a
        time interval (integral), specifying just a time interval (duration),
        or evaluating a period (period).
        See :class:`~obspy.core.event.header.AmplitudeCategory` for allowed
        values.
    :type unit: str, optional
    :param unit: Amplitude unit. This attribute provides the most likely
        measurement units for the physical quantity described in the
        genericAmplitude attribute. Possible values are specified as
        combinations of SI base units.
        See :class:`~obspy.core.event.header.AmplitudeUnit` for allowed
        values.
    :type method_id: :class:`~obspy.core.event.resourceid.ResourceIdentifier`
    :param method_id: Describes the method of amplitude determination.
    :type period: float, optional
    :param period: Dominant period in the timeWindow in case of amplitude
        measurements. Not used for duration magnitude.  Unit: s
    :type snr: float, optional
    :param snr: Signal-to-noise ratio of the spectrogram at the location the
        amplitude was measured.
    :type time_window: :class:`~obspy.core.event.base.TimeWindow`, optional
    :param time_window: Description of the time window used for amplitude
        measurement. Recommended for duration magnitudes.
    :type pick_id: :class:`~obspy.core.event.resourceid.ResourceIdentifier`
    :param pick_id: Refers to the ``resource_id`` of an associated
        :class:`~obspy.core.event.origin.Pick` object.
    :type waveform_id: :class:`~obspy.core.event.base.WaveformStreamID`,
    :param waveform_id: Identifies the waveform stream on which the amplitude
        was measured.
    :type filter_id: :class:`~obspy.core.event.resourceid.ResourceIdentifier`
    :param filter_id: Identifies the filter or filter setup used for filtering
        the waveform stream referenced by ``waveform_id``.
    :type scaling_time: :class:`~obspy.core.utcdatetime.UTCDateTime`, optional
    :param scaling_time: Scaling time for amplitude measurement.
    :type scaling_time_errors: :class:`~obspy.core.event.base.QuantityError`
    :param scaling_time_errors: AttribDict containing error quantities.
    :type magnitude_hint: str, optional
    :param magnitude_hint: Type of magnitude the amplitude measurement is used
        for.  This is a free-text field because it is impossible to cover all
        existing magnitude type designations with an enumeration. Possible
        values are:

        * unspecified magnitude (``'M'``),
        * local magnitude (``'ML'``),
        * body wave magnitude (``'Mb'``),
        * surface wave magnitude (``'MS'``),
        * moment magnitude (``'Mw'``),
        * duration magnitude (``'Md'``)
        * coda magnitude (``'Mc'``)
        * ``'MH'``, ``'Mwp'``, ``'M50'``, ``'M100'``, etc.

    :type evaluation_mode: str, optional
    :param evaluation_mode: Evaluation mode of Amplitude.
        See :class:`~obspy.core.event.header.EvaluationMode` for allowed
        values.
    :type evaluation_status: str, optional
    :param evaluation_status: Evaluation status of Amplitude.
        See :class:`~obspy.core.event.header.EvaluationStatus` for allowed
        values.
    :type comments: list of :class:`~obspy.core.event.base.Comment`, optional
    :param comments: Additional comments.
    :type creation_info: :class:`~obspy.core.event.base.CreationInfo`, optional
    :param creation_info: CreationInfo for the Amplitude object.

    .. note::

        For handling additional information not covered by the QuakeML
        standard and how to output it to QuakeML see the
        :ref:`ObsPy Tutorial <quakeml-extra>`.
    """


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
