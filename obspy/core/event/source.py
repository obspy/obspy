# -*- coding: utf-8 -*-
"""
obspy.core.event.source - Classes for handling seismic source characteristics
=============================================================================
This module provides a class hierarchy to consistently handle event metadata.
This class hierarchy is closely modelled after the de-facto standard format
`QuakeML <https://quake.ethz.ch/quakeml/>`_.

.. figure:: /_images/Event.png

.. note::

    For handling additional information not covered by the QuakeML standard and
    how to output it to QuakeML see the :ref:`ObsPy Tutorial <quakeml-extra>`.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""
import numpy as np

from obspy.core.event.base import (
    _event_type_class_factory, CreationInfo)
from obspy.core.event import ResourceIdentifier
from obspy.core.event.header import (
    EvaluationMode, EvaluationStatus, MomentTensorCategory, MTInversionType,
    SourceTimeFunctionType, ATTRIBUTE_HAS_ERRORS)


__Axis = _event_type_class_factory(
    "__Axis",
    class_attributes=[("azimuth", float, ATTRIBUTE_HAS_ERRORS),
                      ("plunge", float, ATTRIBUTE_HAS_ERRORS),
                      ("length", float, ATTRIBUTE_HAS_ERRORS)])


class Axis(__Axis):
    """
    This class describes an eigenvector of a moment tensor expressed in its
    principal-axes system. It uses the angles azimuth, plunge, and the
    eigenvalue length.

    :type azimuth: float
    :param azimuth: Azimuth of eigenvector of moment tensor expressed in
        principal-axes system. Measured clockwise from South-North direction at
        epicenter. Unit: deg
    :type azimuth_errors: :class:`~obspy.core.event.base.QuantityError`
    :param azimuth_errors: AttribDict containing error quantities.
    :type plunge: float
    :param plunge: Plunge of eigenvector of moment tensor expressed in
        principal-axes system. Measured against downward vertical direction at
        epicenter. Unit: deg
    :type plunge_errors: :class:`~obspy.core.event.base.QuantityError`
    :param plunge_errors: AttribDict containing error quantities.
    :type length: float
    :param length: Eigenvalue of moment tensor expressed in principal-axes
        system. Unit: Nm
    :type length_errors: :class:`~obspy.core.event.base.QuantityError`
    :param length_errors: AttribDict containing error quantities.

    .. note::

        For handling additional information not covered by the QuakeML
        standard and how to output it to QuakeML see the
        :ref:`ObsPy Tutorial <quakeml-extra>`.
    """


__NodalPlane = _event_type_class_factory(
    "__NodalPlane",
    class_attributes=[("strike", float, ATTRIBUTE_HAS_ERRORS),
                      ("dip", float, ATTRIBUTE_HAS_ERRORS),
                      ("rake", float, ATTRIBUTE_HAS_ERRORS)])


class NodalPlane(__NodalPlane):
    """
    This class describes a nodal plane using the attributes strike, dip, and
    rake. For a definition of the angles see Aki & Richards (1980).

    :type strike: float
    :param strike: Strike angle of nodal plane. Unit: deg
    :type strike_errors: :class:`~obspy.core.event.base.QuantityError`
    :param strike_errors: AttribDict containing error quantities.
    :type dip: float
    :param dip: Dip angle of nodal plane. Unit: deg
    :type dip_errors: :class:`~obspy.core.event.base.QuantityError`
    :param dip_errors: AttribDict containing error quantities.
    :type rake: float
    :param rake: Rake angle of nodal plane. Unit: deg
    :type rake_errors: :class:`~obspy.core.event.base.QuantityError`
    :param rake_errors: AttribDict containing error quantities.

    .. note::

        For handling additional information not covered by the QuakeML
        standard and how to output it to QuakeML see the
        :ref:`ObsPy Tutorial <quakeml-extra>`.
    """


__NodalPlanes = _event_type_class_factory(
    "__NodalPlanes",
    class_attributes=[("nodal_plane_1", NodalPlane),
                      ("nodal_plane_2", NodalPlane),
                      ("preferred_plane", int)])


class NodalPlanes(__NodalPlanes):
    """
    This class describes the nodal planes of a double-couple moment-tensor
    solution. The attribute ``preferred_plane`` can be used to define which
    plane is the preferred one.

    :type nodal_plane_1: :class:`~obspy.core.event.source.NodalPlane`, optional
    :param nodal_plane_1: First nodal plane of double-couple moment tensor
        solution.
    :type nodal_plane_2: :class:`~obspy.core.event.source.NodalPlane`, optional
    :param nodal_plane_2: Second nodal plane of double-couple moment tensor
        solution.
    :type preferred_plane: int, optional
    :param preferred_plane: Indicator for preferred nodal plane of moment
        tensor solution. It can take integer values ``1`` or ``2``.

    .. note::

        For handling additional information not covered by the QuakeML
        standard and how to output it to QuakeML see the
        :ref:`ObsPy Tutorial <quakeml-extra>`.
    """


__PrincipalAxes = _event_type_class_factory(
    "__PrincipalAxes",
    class_attributes=[("t_axis", Axis),
                      ("p_axis", Axis),
                      ("n_axis", Axis)])


class PrincipalAxes(__PrincipalAxes):
    """
    This class describes the principal axes of a double-couple moment tensor
    solution. t_axis and p_axis are required, while n_axis is optional.

    :type t_axis: :class:`~obspy.core.event.source.Axis`
    :param t_axis: T (tension) axis of a double-couple moment tensor solution.
    :type p_axis: :class:`~obspy.core.event.source.Axis`
    :param p_axis: P (pressure) axis of a double-couple moment tensor solution.
    :type n_axis: :class:`~obspy.core.event.source.Axis`, optional
    :param n_axis: N (neutral) axis of a double-couple moment tensor solution.

    .. note::

        For handling additional information not covered by the QuakeML
        standard and how to output it to QuakeML see the
        :ref:`ObsPy Tutorial <quakeml-extra>`.
    """


__Tensor = _event_type_class_factory(
    "__Tensor",
    class_attributes=[("m_rr", float, ATTRIBUTE_HAS_ERRORS),
                      ("m_tt", float, ATTRIBUTE_HAS_ERRORS),
                      ("m_pp", float, ATTRIBUTE_HAS_ERRORS),
                      ("m_rt", float, ATTRIBUTE_HAS_ERRORS),
                      ("m_rp", float, ATTRIBUTE_HAS_ERRORS),
                      ("m_tp", float, ATTRIBUTE_HAS_ERRORS)])


class Tensor(__Tensor):
    """
    The Tensor class represents the six moment-tensor elements Mrr, Mtt, Mpp,
    Mrt, Mrp, Mtp in the spherical coordinate system defined by local upward
    vertical (r), North-South (t), and West-East (p) directions.

    :type m_rr: float
    :param m_rr: Moment-tensor element Mrr. Unit: Nm
    :type m_rr_errors: :class:`~obspy.core.event.base.QuantityError`
    :param m_rr_errors: AttribDict containing error quantities.
    :type m_tt: float
    :param m_tt: Moment-tensor element Mtt. Unit: Nm
    :type m_tt_errors: :class:`~obspy.core.event.base.QuantityError`
    :param m_tt_errors: AttribDict containing error quantities.
    :type m_pp: float
    :param m_pp: Moment-tensor element Mpp. Unit: Nm
    :type m_pp_errors: :class:`~obspy.core.event.base.QuantityError`
    :param m_pp_errors: AttribDict containing error quantities.
    :type m_rt: float
    :param m_rt: Moment-tensor element Mrt. Unit: Nm
    :type m_rt_errors: :class:`~obspy.core.event.base.QuantityError`
    :param m_rt_errors: AttribDict containing error quantities.
    :type m_rp: float
    :param m_rp: Moment-tensor element Mrp. Unit: Nm
    :type m_rp_errors: :class:`~obspy.core.event.base.QuantityError`
    :param m_rp_errors: AttribDict containing error quantities.
    :type m_tp: float
    :param m_tp: Moment-tensor element Mtp. Unit: Nm
    :type m_tp_errors: :class:`~obspy.core.event.base.QuantityError`
    :param m_tp_errors: AttribDict containing error quantities.

    .. note::

        For handling additional information not covered by the QuakeML
        standard and how to output it to QuakeML see the
        :ref:`ObsPy Tutorial <quakeml-extra>`.
    """


__SourceTimeFunction = _event_type_class_factory(
    "__SourceTimeFunction",
    class_attributes=[("type", SourceTimeFunctionType),
                      ("duration", float),
                      ("rise_time", float),
                      ("decay_time", float)])


class SourceTimeFunction(__SourceTimeFunction):
    """
    Source time function used in moment-tensor inversion.

    :type type: str
    :param type: Type of source time function.
        See :class:`~obspy.core.event.header.SourceTimeFunctionType` for
        allowed values.
    :type duration: float
    :param duration: Source time function duration. Unit: s
    :type rise_time: float, optional
    :param rise_time: Source time function rise time. Unit: s
    :type decay_time: float, optional
    :param decay_time: Source time function decay time. Unit: s

    .. note::

        For handling additional information not covered by the QuakeML
        standard and how to output it to QuakeML see the
        :ref:`ObsPy Tutorial <quakeml-extra>`.
    """


__MomentTensor = _event_type_class_factory(
    "__MomentTensor",
    class_attributes=[("resource_id", ResourceIdentifier),
                      ("derived_origin_id", ResourceIdentifier),
                      ("moment_magnitude_id", ResourceIdentifier),
                      ("scalar_moment", float, ATTRIBUTE_HAS_ERRORS),
                      ("tensor", Tensor),
                      ("variance", float),
                      ("variance_reduction", float),
                      ("double_couple", float),
                      ("clvd", float),
                      ("iso", float),
                      ("greens_function_id", ResourceIdentifier),
                      ("filter_id", ResourceIdentifier),
                      ("source_time_function", SourceTimeFunction),
                      ("method_id", ResourceIdentifier),
                      ("category", MomentTensorCategory),
                      ("inversion_type", MTInversionType),
                      ("creation_info", CreationInfo)],
    class_contains=["comments", "data_used"])


class MomentTensor(__MomentTensor):
    """
    This class represents a moment tensor solution for an event. It is an
    optional part of a FocalMechanism description.

    :type resource_id: :class:`~obspy.core.event.resourceid.ResourceIdentifier`
    :param resource_id: Resource identifier of MomentTensor.
    :type force_resource_id: bool, optional
    :param force_resource_id: If set to False, the automatic initialization of
        `resource_id` attribute in case it is not specified will be skipped.
    :type derived_origin_id:
        :class:`~obspy.core.event.resourceid.ResourceIdentifier`
    :param derived_origin_id: Refers to the resource_id of the Origin derived
        in the moment tensor inversion.
    :type moment_magnitude_id:
        :class:`~obspy.core.event.resourceid.ResourceIdentifier`
    :param moment_magnitude_id: Refers to the publicID of the Magnitude object
        which represents the derived moment magnitude.
    :type scalar_moment: float, optional
    :param scalar_moment: Scalar moment as derived in moment tensor inversion.
        Unit: Nm
    :type scalar_moment_errors: :class:`~obspy.core.event.base.QuantityError`
    :param scalar_moment_errors: AttribDict containing error quantities.
    :type tensor: :class:`~obspy.core.event.source.Tensor`, optional
    :param tensor: Tensor object holding the moment tensor elements.
    :type variance: float, optional
    :param variance: Variance of moment tensor inversion.
    :type variance_reduction: float, optional
    :param variance_reduction: Variance reduction of moment tensor inversion,
        given in percent (Dreger 2003). This is a goodness-of-fit measure.
    :type double_couple: float, optional
    :param double_couple: Double couple parameter obtained from moment tensor
        inversion (decimal fraction between 0 and 1).
    :type clvd: float, optional
    :param clvd: CLVD (compensated linear vector dipole) parameter obtained
        from moment tensor inversion (decimal fraction between 0 and 1).
    :type iso: float, optional
    :param iso: Isotropic part obtained from moment tensor inversion (decimal
        fraction between 0 and 1).
    :type greens_function_id:
        :class:`~obspy.core.event.resourceid.ResourceIdentifier`
    :param greens_function_id: Resource identifier of the Green’s function used
        in moment tensor inversion.
    :type filter_id: :class:`~obspy.core.event.resourceid.ResourceIdentifier`
    :param filter_id: Resource identifier of the filter setup used in moment
        tensor inversion.
    :type source_time_function:
        :class:`~obspy.core.event.source.SourceTimeFunction`, optional
    :param source_time_function: Source time function used in moment-tensor
        inversion.
    :type data_used: list of :class:`~obspy.core.event.base.DataUsed`, optional
    :param data_used: Describes waveform data used for moment-tensor inversion.
    :type method_id: :class:`~obspy.core.event.resourceid.ResourceIdentifier`
    :param method_id: Resource identifier of the method used for moment-tensor
        inversion.
    :type category: str, optional
    :param category: Moment tensor category.
        See :class:`~obspy.core.event.header.MomentTensorCategory` for allowed
        values.
    :type inversion_type: str, optional
    :param inversion_type: Moment tensor inversion type. Users should avoid to
        give contradictory information in inversion_type and method_id.
        See :class:`~obspy.core.event.header.MTInversionType` for allowed
        values.
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


__FocalMechanism = _event_type_class_factory(
    "__FocalMechanism",
    class_attributes=[("resource_id", ResourceIdentifier),
                      ("triggering_origin_id", ResourceIdentifier),
                      ("nodal_planes", NodalPlanes),
                      ("principal_axes", PrincipalAxes),
                      ("azimuthal_gap", float),
                      ("station_polarity_count", int),
                      ("misfit", float),
                      ("station_distribution_ratio", float),
                      ("method_id", ResourceIdentifier),
                      ("evaluation_mode", EvaluationMode),
                      ("evaluation_status", EvaluationStatus),
                      ("moment_tensor", MomentTensor),
                      ("creation_info", CreationInfo)],
    class_contains=['waveform_id', 'comments'])


class FocalMechanism(__FocalMechanism):
    """
    This class describes the focal mechanism of an event. It includes different
    descriptions like nodal planes, principal axes, and a moment tensor. The
    moment tensor description is provided by objects of the class MomentTensor
    which can be specified as child elements of FocalMechanism.

    :type resource_id: :class:`~obspy.core.event.resourceid.ResourceIdentifier`
    :param resource_id: Resource identifier of FocalMechanism.
    :type force_resource_id: bool, optional
    :param force_resource_id: If set to False, the automatic initialization of
        `resource_id` attribute in case it is not specified will be skipped.
    :type triggering_origin_id:
        :class:`~obspy.core.event.resourceid.ResourceIdentifier`
    :param triggering_origin_id: Refers to the resource_id of the triggering
        origin.
    :type nodal_planes: :class:`~obspy.core.event.source.NodalPlanes`, optional
    :param nodal_planes: Nodal planes of the focal mechanism.
    :type principal_axes: :class:`~obspy.core.event.source.PrincipalAxes`,
        optional
    :param principal_axes: Principal axes of the focal mechanism.
    :type azimuthal_gap: float, optional
    :param azimuthal_gap: Largest azimuthal gap in distribution of stations
        used for determination of focal mechanism. Unit: deg
    :type station_polarity_count: int, optional
    :param station_polarity_count:
    :type misfit: float, optional
    :param misfit: Fraction of misfit polarities in a first-motion focal
        mechanism determination. Decimal fraction between 0 and 1.
    :type station_distribution_ratio: float, optional
    :param station_distribution_ratio: Station distribution ratio (STDR)
        parameter. Indicates how the stations are distributed about the focal
        sphere (Reasenberg and Oppenheimer 1985). Decimal fraction between 0
        and 1.
    :type method_id: :class:`~obspy.core.event.resourceid.ResourceIdentifier`
    :param method_id: Resource identifier of the method used for determination
        of the focal mechanism.
    :type waveform_id: list of
        :class:`~obspy.core.event.base.WaveformStreamID`, optional
    :param waveform_id: Refers to a set of waveform streams from which the
        focal mechanism was derived.
    :type evaluation_mode: str, optional
    :param evaluation_mode: Evaluation mode of FocalMechanism.
        See :class:`~obspy.core.event.header.EvaluationMode` for allowed
        values.
    :type evaluation_status: str, optional
    :param evaluation_status: Evaluation status of FocalMechanism.
        See :class:`~obspy.core.event.header.EvaluationStatus` for allowed
        values.
    :type moment_tensor: :class:`~obspy.core.event.source.MomentTensor`,
        optional
    :param moment_tensor: Moment tensor description for this focal mechanism.
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


def farfield(mt, points, type):
    """
    Returns the P/S farfield radiation pattern
    based on [Aki1980]_ eq. 4.29.

    :param mt: Focal mechanism NM x 6 (Mxx, Myy, Mzz, Mxy, Mxz, Myz - the
               six independent components of the moment tensor)

    :param points: 3D vector array with shape [3,npts] (x,y,z) or [2,npts]
                   (theta,phi) The normalized displacement of the moment
                   tensor source is computed at these points.
    :type type: str
    :param type: 'P' or 'S' (P or S wave).

    :return: 3D vector array with shape [3,npts] that contains the
             displacement vector for each grid point
    """
    type = type.upper()
    if type not in ("P", "S"):
        msg = ("type must be 'P' or 'S'")
        raise ValueError(msg)
    is_p_wave = type == "P"

    ndim, npoints = points.shape
    if ndim == 2:
        # points are given as theta,phi
        _points = np.empty((3, npoints))
        _points[0] = np.sin(points[0]) * np.cos(points[1])
        _points[1] = np.sin(points[0]) * np.sin(points[1])
        _points[2] = np.cos(points[0])
        points = _points
        ndim = 3
    elif ndim == 3:
        # points are given as x,y,z, (same system as the moment tensor)
        pass
    else:
        raise ValueError('points should have shape 2 x npoints or 3 x npoints')
    m_pq = _fullmt(mt)

    # precompute directional cosine array
    dists = np.sqrt(points[0] * points[0] + points[1] * points[1] +
                    points[2] * points[2])
    gammas = points / dists

    # initialize displacement array
    disp = np.empty((ndim, npoints))

    # loop through points
    if is_p_wave:
        for ipoint in range(npoints):
            # loop through displacement component [n index]
            gamma = gammas[:, ipoint]
            gammapq = np.outer(gamma, gamma)
            gammatimesmt = gammapq * m_pq
            for n in range(ndim):
                disp[n, ipoint] = gamma[n] * np.sum(gammatimesmt.flatten())
    else:
        for ipoint in range(npoints):
            # loop through displacement component [n index]
            gamma = gammas[:, ipoint]
            m_p = np.dot(m_pq, gamma)
            for n in range(ndim):
                psum = 0.0
                for p in range(ndim):
                    deltanp = int(n == p)
                    psum += (gamma[n] * gamma[p] - deltanp) * m_p[p]
                disp[n, ipoint] = psum

    return disp


def _fullmt(mt):
    """takes 6 comp moment tensor and returns full 3x3 moment tensor"""
    mt_full = np.array(([[mt[0], mt[3], mt[4]],
                         [mt[3], mt[1], mt[5]],
                         [mt[4], mt[5], mt[2]]]))
    return mt_full


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
