# -*- coding: utf-8 -*-
"""
obspy.core.event.origin - The Origin class definition
=====================================================
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
    (https://www.gnu.org/copyleft/lesser.html)
"""
from obspy import UTCDateTime
from obspy.core.event.base import (
    _event_type_class_factory, CreationInfo,
    WaveformStreamID, ConfidenceEllipsoid)
from obspy.core.event import ResourceIdentifier
from obspy.core.event.header import (
    EvaluationMode, EvaluationStatus, OriginDepthType, OriginType,
    OriginUncertaintyDescription, PickOnset, PickPolarity,
    ATTRIBUTE_HAS_ERRORS)


__OriginUncertainty = _event_type_class_factory(
    "__OriginUncertainty",
    class_attributes=[("horizontal_uncertainty", float),
                      ("min_horizontal_uncertainty", float),
                      ("max_horizontal_uncertainty", float),
                      ("azimuth_max_horizontal_uncertainty", float),
                      ("confidence_ellipsoid", ConfidenceEllipsoid),
                      ("preferred_description", OriginUncertaintyDescription),
                      ("confidence_level", float)])


class OriginUncertainty(__OriginUncertainty):
    """
    This class describes the location uncertainties of an origin.

    The uncertainty can be described either as a simple circular horizontal
    uncertainty, an uncertainty ellipse according to IMS1.0, or a confidence
    ellipsoid. If multiple uncertainty models are given, the preferred variant
    can be specified in the attribute ``preferred_description``.

    :type horizontal_uncertainty: float, optional
    :param horizontal_uncertainty: Circular confidence region, given by single
        value of horizontal uncertainty. Unit: m
    :type min_horizontal_uncertainty: float, optional
    :param min_horizontal_uncertainty: Semi-minor axis of confidence ellipse.
        Unit: m
    :type max_horizontal_uncertainty: float, optional
    :param max_horizontal_uncertainty: Semi-major axis of confidence ellipse.
        Unit: m
    :type azimuth_max_horizontal_uncertainty: float, optional
    :param azimuth_max_horizontal_uncertainty: Azimuth of major axis of
        confidence ellipse. Measured clockwise from South-North direction at
        epicenter. Unit: deg
    :type confidence_ellipsoid:
        :class:`~obspy.core.event.base.ConfidenceEllipsoid`, optional
    :param confidence_ellipsoid: Confidence ellipsoid
    :type preferred_description: str, optional
    :param preferred_description: Preferred uncertainty description.
        See :class:`~obspy.core.event.header.OriginUncertaintyDescription` for
        allowed values.
    :type confidence_level: float, optional
    :param confidence_level: Confidence level of the uncertainty, given in
        percent.

    .. note::

        For handling additional information not covered by the QuakeML
        standard and how to output it to QuakeML see the
        :ref:`ObsPy Tutorial <quakeml-extra>`.
    """


__OriginQuality = _event_type_class_factory(
    "__OriginQuality",
    class_attributes=[("associated_phase_count", int),
                      ("used_phase_count", int),
                      ("associated_station_count", int),
                      ("used_station_count", int),
                      ("depth_phase_count", int),
                      ("standard_error", float),
                      ("azimuthal_gap", float),
                      ("secondary_azimuthal_gap", float),
                      ("ground_truth_level", str),
                      ("minimum_distance", float),
                      ("maximum_distance", float),
                      ("median_distance", float)])


class OriginQuality(__OriginQuality):
    """
    This type contains various attributes commonly used to describe the quality
    of an origin, e. g., errors, azimuthal coverage, etc. Origin objects have
    an optional attribute of the type OriginQuality.

    :type associated_phase_count: int, optional
    :param associated_phase_count: Number of associated phases, regardless of
        their use for origin computation.
    :type used_phase_count: int, optional
    :param used_phase_count: Number of defining phases, i. e., phase
        observations that were actually used for computing the origin. Note
        that there may be more than one defining phase per station.
    :type associated_station_count: int, optional
    :param associated_station_count: Number of stations at which the event was
        observed.
    :type used_station_count: int, optional
    :param used_station_count: Number of stations from which data was used for
        origin computation.
    :type depth_phase_count: int, optional
    :param depth_phase_count: Number of depth phases (typically pP, sometimes
        sP) used in depth computation.
    :type standard_error: float, optional
    :param standard_error: RMS of the travel time residuals of the arrivals
        used for the origin computation. Unit: s
    :type azimuthal_gap: float, optional
    :param azimuthal_gap: Largest azimuthal gap in station distribution as seen
        from epicenter. For an illustration of azimuthal gap and secondary
        azimuthal gap (see below), see Fig. 5 of Bond ́ar et al. (2004).
        Unit: deg
    :type secondary_azimuthal_gap: float, optional
    :param secondary_azimuthal_gap: Secondary azimuthal gap in station
        distribution, i. e., the largest azimuthal gap a station closes.
        Unit: deg
    :type ground_truth_level: str, optional
    :param ground_truth_level: String describing ground-truth level, e. g. GT0,
        GT5, etc.
    :type minimum_distance: float, optional
    :param minimum_distance: Epicentral distance of station closest to the
        epicenter.  Unit: deg
    :type maximum_distance: float, optional
    :param maximum_distance: Epicentral distance of station farthest from the
        epicenter.  Unit: deg
    :type median_distance: float, optional
    :param median_distance: Median epicentral distance of used stations.
        Unit: deg

    .. note::

        For handling additional information not covered by the QuakeML
        standard and how to output it to QuakeML see the
        :ref:`ObsPy Tutorial <quakeml-extra>`.
    """


__Origin = _event_type_class_factory(
    "__Origin",
    class_attributes=[("resource_id", ResourceIdentifier),
                      ("time", UTCDateTime, ATTRIBUTE_HAS_ERRORS),
                      ("longitude", float, ATTRIBUTE_HAS_ERRORS),
                      ("latitude", float, ATTRIBUTE_HAS_ERRORS),
                      ("depth", float, ATTRIBUTE_HAS_ERRORS),
                      ("depth_type", OriginDepthType),
                      ("time_fixed", bool),
                      ("epicenter_fixed", bool),
                      ("reference_system_id", ResourceIdentifier),
                      ("method_id", ResourceIdentifier),
                      ("earth_model_id", ResourceIdentifier),
                      ("quality", OriginQuality),
                      ("origin_type", OriginType),
                      ("origin_uncertainty", OriginUncertainty),
                      ("region", str),
                      ("evaluation_mode", EvaluationMode),
                      ("evaluation_status", EvaluationStatus),
                      ("creation_info", CreationInfo)],
    class_contains=["comments", "arrivals", "composite_times"])


class Origin(__Origin):
    """
    This class represents the focal time and geographical location of an
    earthquake hypocenter, as well as additional meta-information. Origin can
    have objects of type OriginUncertainty and Arrival as child elements.

    :type resource_id: :class:`~obspy.core.event.resourceid.ResourceIdentifier`
    :param resource_id: Resource identifier of Origin.
    :type force_resource_id: bool, optional
    :param force_resource_id: If set to False, the automatic initialization of
        `resource_id` attribute in case it is not specified will be skipped.
    :type time: :class:`~obspy.core.utcdatetime.UTCDateTime`
    :param time: Focal time.
    :type time_errors: :class:`~obspy.core.event.base.QuantityError`
    :param time_errors: AttribDict containing error quantities.
    :type longitude: float
    :param longitude: Hypocenter longitude, with respect to the World Geodetic
        System 1984 (WGS84) reference system. Unit: deg
    :type longitude_errors: :class:`~obspy.core.event.base.QuantityError`
    :param longitude_errors: AttribDict containing error quantities.
    :type latitude: float
    :param latitude: Hypocenter latitude, with respect to the WGS84 reference
        system. Unit: deg
    :type latitude_errors: :class:`~obspy.core.event.base.QuantityError`
    :param latitude_errors: AttribDict containing error quantities.
    :type depth: float, optional
    :param depth: Depth of hypocenter with respect to the nominal sea level
        given by the WGS84 geoid. Positive values indicate hypocenters below
        sea level. For shallow hypocenters, the depth value can be negative.
        Note: Other standards use different conventions for depth measurement.
        As an example, GSE2.0, defines depth with respect to the local surface.
        If event data is converted from other formats to QuakeML, depth values
        may have to be modified accordingly. Unit: m
    :type depth_errors: :class:`~obspy.core.event.base.QuantityError`
    :param depth_errors: AttribDict containing error quantities.
    :type depth_type: str, optional
    :param depth_type: Type of depth determination.
        See :class:`~obspy.core.event.header.OriginDepthType` for allowed
        values.
    :type time_fixed: bool, optional
    :param time_fixed: True if focal time was kept fixed for computation of the
        Origin.
    :type epicenter_fixed: bool, optional
    :param epicenter_fixed: True if epicenter was kept fixed for computation of
        Origin.
    :type reference_system_id:
        :class:`~obspy.core.event.resourceid.ResourceIdentifier`
    :param reference_system_id: Identifies the reference system used for
        hypocenter determination. This is only necessary if a modified version
        of the standard (with local extensions) is used that provides a
        non-standard coordinate system.
    :type method_id: :class:`~obspy.core.event.resourceid.ResourceIdentifier`
    :param method_id: Identifies the method used for locating the event.
    :type earth_model_id:
        :class:`~obspy.core.event.resourceid.ResourceIdentifier`
    :param earth_model_id: Identifies the earth model used in method_id.
    :type arrivals: list of :class:`~obspy.core.event.origin.Arrival`, optional
    :param arrivals: List of arrivals associated with the origin.
    :type composite_times: list of
        :class:`~obspy.core.event.base.CompositeTime`, optional
    :param composite_times: Supplementary information on time of rupture start.
        Complex descriptions of focal times of historic events are possible,
        see description of the CompositeTime type. Note that even if
        compositeTime is used, the mandatory time attribute has to be set, too.
        It has to be set to the single point in time (with uncertainties
        allowed) that is most characteristic for the event.
    :type quality: :class:`~obspy.core.event.origin.OriginQuality`, optional
    :param quality: Additional parameters describing the quality of an Origin
        determination.
    :type origin_type: str, optional
    :param origin_type: Describes the origin type.
        See :class:`~obspy.core.event.header.OriginType` for allowed values.
    :type origin_uncertainty:
        :class:`~obspy.core.event.origin.OriginUncertainty`, optional
    :param origin_uncertainty: Describes the location uncertainties of an
        origin.
    :type region: str, optional
    :param region: Can be used to describe the geographical region of the
        epicenter location. Useful if an event has multiple origins from
        different agencies, and these have different region designations. Note
        that an event-wide region can be defined in the description attribute
        of an Event object. The user has to take care that this information
        corresponds to the region attribute of the preferred Origin.
    :type evaluation_mode: str, optional
    :param evaluation_mode: Evaluation mode of Origin.
        See :class:`~obspy.core.event.header.EvaluationMode` for allowed
        values.
    :type evaluation_status: str, optional
    :param evaluation_status: Evaluation status of Origin.
        See :class:`~obspy.core.event.header.EvaluationStatus` for allowed
        values.
    :type comments: list of :class:`~obspy.core.event.base.Comment`, optional
    :param comments: Additional comments.
    :type creation_info: :class:`~obspy.core.event.base.CreationInfo`, optional
    :param creation_info: Creation information used to describe author,
        version, and creation time.

    .. rubric:: Example

    >>> from obspy.core.event import Origin
    >>> origin = Origin()
    >>> origin.resource_id = 'smi:ch.ethz.sed/origin/37465'
    >>> origin.time = UTCDateTime(0)
    >>> origin.latitude = 12
    >>> origin.latitude_errors.uncertainty = 0.01
    >>> origin.latitude_errors.confidence_level = 95.0
    >>> origin.longitude = 42
    >>> origin.depth_type = 'from location'
    >>> print(origin)  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    Origin
        resource_id: ResourceIdentifier(id="smi:ch.ethz.sed/...")
               time: UTCDateTime(1970, 1, 1, 0, 0)
          longitude: 42.0
           latitude: 12.0 [confidence_level=95.0, uncertainty=0.01]
         depth_type: ...'from location'

    .. note::

        For handling additional information not covered by the QuakeML
        standard and how to output it to QuakeML see the
        :ref:`ObsPy Tutorial <quakeml-extra>`.
    """


__Pick = _event_type_class_factory(
    "__Pick",
    class_attributes=[("resource_id", ResourceIdentifier),
                      ("time", UTCDateTime, ATTRIBUTE_HAS_ERRORS),
                      ("waveform_id", WaveformStreamID),
                      ("filter_id", ResourceIdentifier),
                      ("method_id", ResourceIdentifier),
                      ("horizontal_slowness", float, ATTRIBUTE_HAS_ERRORS),
                      ("backazimuth", float, ATTRIBUTE_HAS_ERRORS),
                      ("slowness_method_id", ResourceIdentifier),
                      ("onset", PickOnset),
                      ("phase_hint", str),
                      ("polarity", PickPolarity),
                      ("evaluation_mode", EvaluationMode),
                      ("evaluation_status", EvaluationStatus),
                      ("creation_info", CreationInfo)],
    class_contains=["comments"])


class Pick(__Pick):
    """
    A pick is the observation of an amplitude anomaly in a seismogram at a
    specific point in time. It is not necessarily related to a seismic event.

    :type resource_id: :class:`~obspy.core.event.resourceid.ResourceIdentifier`
    :param resource_id: Resource identifier of Pick.
    :type force_resource_id: bool, optional
    :param force_resource_id: If set to False, the automatic initialization of
        `resource_id` attribute in case it is not specified will be skipped.
    :type time: :class:`~obspy.core.utcdatetime.UTCDateTime`
    :param time: Observed onset time of signal (“pick time”).
    :type time_errors: :class:`~obspy.core.event.base.QuantityError`
    :param time_errors: AttribDict containing error quantities.
    :type waveform_id: :class:`~obspy.core.event.base.WaveformStreamID`
    :param waveform_id: Identifies the waveform stream.
    :type filter_id: :class:`~obspy.core.event.resourceid.ResourceIdentifier`
    :param filter_id: Identifies the filter or filter setup used for filtering
        the waveform stream referenced by waveform_id.
    :type method_id: :class:`~obspy.core.event.resourceid.ResourceIdentifier`
    :param method_id: Identifies the picker that produced the pick. This can be
        either a detection software program or a person.
    :type horizontal_slowness: float, optional
    :param horizontal_slowness: Observed horizontal slowness of the signal.
        Most relevant in array measurements. Unit: s·deg^(−1)
    :type horizontal_slowness_errors:
        :class:`~obspy.core.event.base.QuantityError`
    :param horizontal_slowness_errors: AttribDict containing error quantities.
    :type backazimuth: float, optional
    :param backazimuth: Observed backazimuth of the signal. Most relevant in
        array measurements. Unit: deg
    :type backazimuth_errors: :class:`~obspy.core.event.base.QuantityError`
    :param backazimuth_errors: AttribDict containing error quantities.
    :type slowness_method_id:
        :class:`~obspy.core.event.resourceid.ResourceIdentifier`
    :param slowness_method_id: Identifies the method that was used to determine
        the slowness.
    :type onset: str, optional
    :param onset: Flag that roughly categorizes the sharpness of the onset.
        See :class:`~obspy.core.event.header.PickOnset` for allowed values.
    :type phase_hint: str, optional
    :param phase_hint: Tentative phase identification as specified by the
        picker.
    :type polarity: str, optional
    :param polarity: Indicates the polarity of first motion, usually from
        impulsive onsets.
        See :class:`~obspy.core.event.header.PickPolarity` for allowed values.
    :type evaluation_mode: str, optional
    :param evaluation_mode: Evaluation mode of Pick.
        See :class:`~obspy.core.event.header.EvaluationMode` for allowed
        values.
    :type evaluation_status: str, optional
    :param evaluation_status: Evaluation status of Pick.
        See :class:`~obspy.core.event.header.EvaluationStatus` for allowed
        values.
    :type comments: list of :class:`~obspy.core.event.base.Comment`, optional
    :param comments: Additional comments.
    :type creation_info: :class:`~obspy.core.event.base.CreationInfo`, optional
    :param creation_info: CreationInfo for the Pick object.

    .. note::

        For handling additional information not covered by the QuakeML
        standard and how to output it to QuakeML see the
        :ref:`ObsPy Tutorial <quakeml-extra>`.
    """


__Arrival = _event_type_class_factory(
    "__Arrival",
    class_attributes=[("resource_id", ResourceIdentifier),
                      ("pick_id", ResourceIdentifier),
                      ("phase", str),
                      ("time_correction", float),
                      ("azimuth", float),
                      ("distance", float),
                      ("takeoff_angle", float, ATTRIBUTE_HAS_ERRORS),
                      ("time_residual", float),
                      ("horizontal_slowness_residual", float),
                      ("backazimuth_residual", float),
                      ("time_weight", float),
                      ("horizontal_slowness_weight", float),
                      ("backazimuth_weight", float),
                      ("earth_model_id", ResourceIdentifier),
                      ("creation_info", CreationInfo)],
    class_contains=["comments"])


class Arrival(__Arrival):
    """
    Successful association of a pick with an origin qualifies this pick as an
    arrival. An arrival thus connects a pick with an origin and provides
    additional attributes that describe this relationship. Usually
    qualification of a pick as an arrival for a given origin is a hypothesis,
    which is based on assumptions about the type of arrival (phase) as well as
    observed and (on the basis of an earth model) computed arrival times, or
    the residual, respectively. Additional pick attributes like the horizontal
    slowness and backazimuth of the observed wave—especially if derived from
    array data—may further constrain the nature of the arrival.

    :type resource_id: :class:`~obspy.core.event.resourceid.ResourceIdentifier`
    :param resource_id: Resource identifier of Arrival.
    :type force_resource_id: bool, optional
    :param force_resource_id: If set to False, the automatic initialization of
        `resource_id` attribute in case it is not specified will be skipped.
    :type pick_id: :class:`~obspy.core.event.resourceid.ResourceIdentifier`
    :param pick_id: Refers to the resource_id of a Pick.
    :type phase: str
    :param phase: Phase identification. For possible values, please refer to
        the description of the Phase object.
    :type time_correction: float, optional
    :param time_correction: Time correction value. Usually, a value
        characteristic for the station at which the pick was detected,
        sometimes also characteristic for the phase type or the slowness.
        Unit: s
    :type azimuth: float, optional
    :param azimuth: Azimuth of station as seen from the epicenter. Unit: deg
    :type distance: float, optional
    :param distance: Epicentral distance. Unit: deg
    :type takeoff_angle: float, optional
    :param takeoff_angle: Angle of emerging ray at the source, measured against
        the downward normal direction. Unit: deg
    :type takeoff_angle_errors: :class:`~obspy.core.event.base.QuantityError`
    :param takeoff_angle_errors: AttribDict containing error quantities.
    :type time_residual: float, optional
    :param time_residual: Residual between observed and expected arrival time
        assuming proper phase identification and given the earth_model_ID of
        the Origin, taking into account the time_correction. Unit: s
    :type horizontal_slowness_residual: float, optional
    :param horizontal_slowness_residual: Residual of horizontal slowness and
        the expected slowness given the current origin (refers to attribute
        horizontal_slowness of class Pick).
    :type backazimuth_residual: float, optional
    :param backazimuth_residual: Residual of backazimuth and the backazimuth
        computed for the current origin (refers to attribute backazimuth of
        class Pick).
    :type time_weight: float, optional
    :param time_weight: Weight of the arrival time for computation of the
        associated Origin. Note that the sum of all weights is not required to
        be unity.
    :type horizontal_slowness_weight: float, optional
    :param horizontal_slowness_weight: Weight of the horizontal slowness for
        computation of the associated Origin. Note that the sum of all weights
        is not required to be unity.
    :type backazimuth_weight: float, optional
    :param backazimuth_weight: Weight of the backazimuth for computation of the
        associated Origin. Note that the sum of all weights is not required to
        be unity.
    :type earth_model_id:
        :class:`~obspy.core.event.resourceid.ResourceIdentifier`
    :param earth_model_id: Earth model which is used for the association of
        Arrival to Pick and computation of the residuals.
    :type comments: list of :class:`~obspy.core.event.base.Comment`, optional
    :param comments: Additional comments.
    :type creation_info: :class:`~obspy.core.event.base.CreationInfo`, optional
    :param creation_info: CreationInfo for the Arrival object.

    .. note::

        For handling additional information not covered by the QuakeML
        standard and how to output it to QuakeML see the
        :ref:`ObsPy Tutorial <quakeml-extra>`.
    """


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
