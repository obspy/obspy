# -*- coding: utf-8 -*-
"""
QuakeML read and write support.

QuakeML is a flexible, extensible and modular XML representation of
seismological data which is intended to cover a broad range of fields of
application in modern seismology. QuakeML is an open standard and is developed
by a distributed team in a transparent collaborative manner.

.. seealso:: https://quake.ethz.ch/quakeml/

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""

from obspy.core.event import Catalog, Event, Origin, CreationInfo, Magnitude, \
    EventDescription, OriginUncertainty, OriginQuality, CompositeTime, \
    ConfidenceEllipsoid, StationMagnitude, Comment, WaveformStreamID, Pick, \
    QuantityError, Arrival, FocalMechanism, MomentTensor, NodalPlanes, \
    PrincipalAxes, Axis, NodalPlane, SourceTimeFunction, Tensor, DataUsed, \
    ResourceIdentifier, StationMagnitudeContribution
from obspy.core.utcdatetime import UTCDateTime
from obspy.core.util.xmlwrapper import XMLParser, tostring, etree
import StringIO


def isQuakeML(filename):
    """
    Checks whether a file is QuakeML format.

    :type filename: str
    :param filename: Name of the QuakeML file to be checked.
    :rtype: bool
    :return: ``True`` if QuakeML file.

    .. rubric:: Example

    >>> isQuakeML('/path/to/quakeml.xml')  # doctest: +SKIP
    True
    """
    try:
        xml_doc = XMLParser(filename)
    except:
        return False
    # check if node "*/eventParameters/event" for the global namespace exists
    try:
        namespace = xml_doc._getFirstChildNamespace()
        xml_doc.xpath('eventParameters', namespace=namespace)[0]
    except:
        return False
    return True


class Unpickler(object):
    """
    De-serializes a QuakeML string into an ObsPy Catalog object.
    """
    def __init__(self, parser=None):
        self.parser = parser

    def load(self, file):
        """
        Reads QuakeML file into ObsPy catalog object.

        :type file: str
        :param file: File name to read.
        :rtype: :class:`~obspy.core.event.Catalog`
        :returns: ObsPy Catalog object.
        """
        self.parser = XMLParser(file)
        return self._deserialize()

    def loads(self, string):
        """
        Parses QuakeML string into ObsPy catalog object.

        :type string: str
        :param string: QuakeML string to parse.
        :rtype: :class:`~obspy.core.event.Catalog`
        :returns: ObsPy Catalog object.
        """
        self.parser = XMLParser(StringIO.StringIO(string))
        return self._deserialize()

    def _xpath2obj(self, *args, **kwargs):
        return self.parser.xpath2obj(*args, **kwargs)

    def _xpath(self, *args, **kwargs):
        return self.parser.xpath(*args, **kwargs)

    def _comments(self, element):
        obj = []
        for el in self._xpath('comment', element):
            comment = Comment()
            comment.text = self._xpath2obj('text', el)
            temp = el.get('id', None)
            if temp is not None:
                comment.resource_id = temp
            comment.creation_info = self._creation_info(el)
            obj.append(comment)
        return obj

    def _station_magnitude_contributions(self, element):
        obj = []
        for el in self._xpath("stationMagnitudeContribution", element):
            contrib = StationMagnitudeContribution()
            contrib.weight = self._xpath2obj("weight", el, float)
            contrib.residual = self._xpath2obj("residual", el, float)
            contrib.station_magnitude_id = \
                self._xpath2obj("stationMagnitudeID", el, str)
            obj.append(contrib)
        return obj

    def _creation_info(self, element):
        has_creation_info = False
        for child in element:
            if 'creationInfo' in child.tag:
                has_creation_info = True
                break
        if not has_creation_info:
            return None
        obj = CreationInfo()
        obj.agency_uri = self._xpath2obj('creationInfo/agencyURI', element)
        obj.author_uri = self._xpath2obj('creationInfo/authorURI', element)
        obj.agency_id = self._xpath2obj('creationInfo/agencyID', element)
        obj.author = self._xpath2obj('creationInfo/author', element)
        obj.creation_time = self._xpath2obj('creationInfo/creationTime',
            element, UTCDateTime)
        obj.version = self._xpath2obj('creationInfo/version', element)
        return obj

    def _origin_quality(self, element):
        obj = OriginQuality()
        obj.associated_phase_count = self._xpath2obj(
            'quality/associatedPhaseCount', element, int)
        obj.used_phase_count = self._xpath2obj(
            'quality/usedPhaseCount', element, int)
        obj.associated_station_count = self._xpath2obj(
            'quality/associatedStationCount', element, int)
        obj.used_station_count = self._xpath2obj(
            'quality/usedStationCount', element, int)
        obj.depth_phase_count = self._xpath2obj(
            'quality/depthPhaseCount', element, int)
        obj.standard_error = self._xpath2obj(
            'quality/standardError', element, float)
        obj.azimuthal_gap = self._xpath2obj(
            'quality/azimuthalGap', element, float)
        obj.secondary_azimuthal_gap = self._xpath2obj(
            'quality/secondaryAzimuthalGap', element, float)
        obj.ground_truth_level = self._xpath2obj(
            'quality/groundTruthLevel', element)
        obj.minimum_distance = self._xpath2obj(
            'quality/minimumDistance', element, float)
        obj.maximum_distance = self._xpath2obj(
            'quality/maximumDistance', element, float)
        obj.median_distance = self._xpath2obj(
            'quality/medianDistance', element, float)
        return obj

    def _event_description(self, element):
        out = []
        for el in self._xpath('description', element):
            text = self._xpath2obj('text', el)
            type = self._xpath2obj('type', el)
            out.append(EventDescription(text=text, type=type))
        return out

    def _value(self, element, name, quantity_type=float):
        try:
            el = self._xpath(name, element)[0]
        except:
            return None, None

        value = self._xpath2obj('value', el, quantity_type)
        # All errors are QuantityError.
        error = QuantityError()
        # Don't set the errors if they are not set.
        confidence_level = self._xpath2obj('confidenceLevel', el, float)
        if confidence_level is not None:
            error.confidence_level = confidence_level
        if quantity_type != int:
            uncertainty = self._xpath2obj('uncertainty', el, float)
            if uncertainty is not None:
                error.uncertainty = uncertainty
            lower_uncertainty = self._xpath2obj('lowerUncertainty', el, float)
            if lower_uncertainty is not None:
                error.lower_uncertainty = lower_uncertainty
            upper_uncertainty = self._xpath2obj('upperUncertainty', el, float)
            if upper_uncertainty is not None:
                error.upper_uncertainty = upper_uncertainty
        else:
            uncertainty = self._xpath2obj('uncertainty', el, int)
            if uncertainty is not None:
                error.uncertainty = uncertainty
            lower_uncertainty = self._xpath2obj('lowerUncertainty', el, int)
            if lower_uncertainty is not None:
                error.lower_uncertainty = lower_uncertainty
            upper_uncertainty = self._xpath2obj('upperUncertainty', el, int)
            if upper_uncertainty is not None:
                error.upper_uncertainty = upper_uncertainty
        return value, error

    def _float_value(self, element, name):
        return self._value(element, name, float)

    def _int_value(self, element, name):
        return self._value(element, name, int)

    def _time_value(self, element, name):
        return self._value(element, name, UTCDateTime)

    def _composite_times(self, element):
        obj = []
        for el in self._xpath('compositeTime', element):
            ct = CompositeTime()
            ct.year, ct.year_errors = self._int_value(el, 'year')
            ct.month, ct.month_errors = self._int_value(el, 'month')
            ct.day, ct.day_errors = self._int_value(el, 'day')
            ct.hour, ct.hour_errors = self._int_value(el, 'hour')
            ct.minute, ct.minute_errors = self._int_value(el, 'minute')
            ct.second, ct.second_errors = self._float_value(el, 'second')
            obj.append(ct)
        return obj

    def _confidence_ellipsoid(self, element):
        obj = ConfidenceEllipsoid()
        obj.semi_major_axis_length = self._xpath2obj(
            'semiMajorAxisLength', element, float)
        obj.semi_minor_axis_length = self._xpath2obj(
            'semiMinorAxisLength', element, float)
        obj.semi_intermediate_axis_length = self._xpath2obj(
            'semiIntermediateAxisLength', element, float)
        obj.major_axis_plunge = self._xpath2obj(
            'majorAxisPlunge', element, float)
        obj.major_axis_azimuth = self._xpath2obj(
            'majorAxisAzimuth', element, float)
        obj.major_axis_rotation = self._xpath2obj(
            'majorAxisRotation', element, float)
        return obj

    def _origin_uncertainty(self, element):
        obj = OriginUncertainty()
        obj.preferred_description = self._xpath2obj(
            'originUncertainty/preferredDescription', element)
        obj.horizontal_uncertainty = self._xpath2obj(
            'originUncertainty/horizontalUncertainty', element, float)
        obj.min_horizontal_uncertainty = self._xpath2obj(
            'originUncertainty/minHorizontalUncertainty', element, float)
        obj.max_horizontal_uncertainty = self._xpath2obj(
            'originUncertainty/maxHorizontalUncertainty', element, float)
        obj.azimuth_max_horizontal_uncertainty = self._xpath2obj(
            'originUncertainty/azimuthMaxHorizontalUncertainty', element,
            float)
        try:
            ce_el = self._xpath('originUncertainty/confidenceEllipsoid',
                                element)
            obj.confidence_ellipsoid = self._confidence_ellipsoid(ce_el[0])
        except:
            obj.confidence_ellipsoid = ConfidenceEllipsoid()
        return obj

    def _waveform_id(self, element):
        obj = WaveformStreamID()
        try:
            wid_el = self._xpath('waveformID', element)[0]
        except:
            return obj
        obj.network_code = wid_el.get('networkCode') or ''
        obj.station_code = wid_el.get('stationCode') or ''
        obj.location_code = wid_el.get('locationCode')
        obj.channel_code = wid_el.get('channelCode')
        obj.resource_uri = wid_el.text
        return obj

    def _arrival(self, element):
        """
        Converts an etree.Element into an Arrival object.

        :type element: etree.Element
        :rtype: :class:`~obspy.core.event.Arrival`
        """
        obj = Arrival()
        # required parameter
        obj.resource_id = element.get('publicID')
        obj.pick_id = self._xpath2obj('pickID', element) or ''
        obj.phase = self._xpath2obj('phase', element) or ''
        # optional parameter
        obj.time_correction = self._xpath2obj('timeCorrection', element, float)
        obj.azimuth = self._xpath2obj('azimuth', element, float)
        obj.distance = self._xpath2obj('distance', element, float)
        obj.takeoff_angle, obj.takeoff_angle_errors = \
            self._float_value(element, 'takeoffAngle')
        obj.time_residual = self._xpath2obj('timeResidual', element, float)
        obj.horizontal_slowness_residual = \
            self._xpath2obj('horizontalSlownessResidual', element, float)
        obj.backazimuth_residual = \
            self._xpath2obj('backazimuthResidual', element, float)
        obj.time_weight = self._xpath2obj('timeWeight', element, float)
        obj.horizontal_slowness_weight = \
            self._xpath2obj('horizontalSlownessWeight', element, float)
        obj.backazimuth_weight = \
            self._xpath2obj('backazimuthWeight', element, float)
        obj.earth_model_id = self._xpath2obj('earthModelID', element)
        obj.comments = self._comments(element)
        obj.creation_info = self._creation_info(element)
        return obj

    def _pick(self, element):
        """
        Converts an etree.Element into a Pick object.

        :type element: etree.Element
        :rtype: :class:`~obspy.core.event.Pick`
        """
        obj = Pick()
        # required parameter
        obj.resource_id = element.get('publicID')
        obj.time, obj.time_errors = self._time_value(element, 'time')
        obj.waveform_id = self._waveform_id(element)
        # optional parameter
        obj.filter_id = self._xpath2obj('filterID', element)
        obj.method_id = self._xpath2obj('methodID', element)
        obj.horizontal_slowness, obj.horizontal_slowness_errors = \
            self._float_value(element, 'horizontalSlowness')
        obj.backazimuth, obj.backazimuth_errors = \
            self._float_value(element, 'backazimuth')
        obj.slowness_method_id = self._xpath2obj('slownessMethodID', element)
        obj.onset = self._xpath2obj('onset', element)
        obj.phase_hint = self._xpath2obj('phaseHint', element)
        obj.polarity = self._xpath2obj('polarity', element)
        obj.evaluation_mode = self._xpath2obj('evaluationMode', element)
        obj.evaluation_status = self._xpath2obj('evaluationStatus', element)
        obj.comments = self._comments(element)
        obj.creation_info = self._creation_info(element)
        return obj

    def _origin(self, element):
        """
        Converts an etree.Element into an Origin object.

        :type element: etree.Element
        :rtype: :class:`~obspy.core.event.Origin`

        .. rubric:: Example

        >>> from obspy.core.util import XMLParser
        >>> XML = '''<?xml version="1.0" encoding="UTF-8"?>
        ... <origin>
        ...   <latitude><value>34.23</value></latitude>
        ... </origin>'''
        >>> parser = XMLParser(XML)
        >>> unpickler = Unpickler(parser)
        >>> origin = unpickler._origin(parser.xml_root)
        >>> print(origin.latitude)
        34.23
        """
        obj = Origin()
        # required parameter
        obj.resource_id = element.get('publicID')
        obj.time, obj.time_errors = self._time_value(element, 'time')
        obj.latitude, obj.latitude_errors = \
            self._float_value(element, 'latitude')
        obj.longitude, obj.longitude_errors = \
            self._float_value(element, 'longitude')
        # optional parameter
        obj.depth, obj.depth_errors = self._float_value(element, 'depth')
        obj.depth_type = self._xpath2obj('depthType', element)
        obj.time_fixed = self._xpath2obj('timeFixed', element, bool)
        obj.epicenter_fixed = self._xpath2obj('epicenterFixed', element, bool)
        obj.reference_system_id = self._xpath2obj('referenceSystemID', element)
        obj.method_id = self._xpath2obj('methodID', element)
        obj.earth_model_id = self._xpath2obj('earthModelID', element)
        obj.composite_times = self._composite_times(element)
        obj.quality = self._origin_quality(element)
        obj.origin_type = self._xpath2obj('type', element)
        obj.evaluation_mode = self._xpath2obj('evaluationMode', element)
        obj.evaluation_status = self._xpath2obj('evaluationStatus', element)
        obj.creation_info = self._creation_info(element)
        obj.comments = self._comments(element)
        obj.origin_uncertainty = self._origin_uncertainty(element)
        return obj

    def _magnitude(self, element):
        """
        Converts an etree.Element into a Magnitude object.

        :type element: etree.Element
        :rtype: :class:`~obspy.core.event.Magnitude`

        .. rubric:: Example

        >>> from obspy.core.util import XMLParser
        >>> XML = '''<?xml version="1.0" encoding="UTF-8"?>
        ... <magnitude>
        ...   <mag><value>3.2</value></mag>
        ... </magnitude>'''
        >>> parser = XMLParser(XML)
        >>> unpickler = Unpickler(parser)
        >>> magnitude = unpickler._magnitude(parser.xml_root)
        >>> print(magnitude.mag)
        3.2
        """
        obj = Magnitude()
        # required parameter
        obj.resource_id = element.get('publicID')
        obj.mag, obj.mag_errors = self._float_value(element, 'mag')
        # optional parameter
        obj.magnitude_type = self._xpath2obj('type', element)
        obj.origin_id = self._xpath2obj('originID', element)
        obj.method_id = self._xpath2obj('methodID', element)
        obj.station_count = self._xpath2obj('stationCount', element, int)
        obj.azimuthal_gap = self._xpath2obj('azimuthalGap', element, float)
        obj.evaluation_status = self._xpath2obj('evaluationStatus', element)
        obj.creation_info = self._creation_info(element)
        obj.station_magnitude_contributions = \
            self._station_magnitude_contributions(element)
        obj.comments = self._comments(element)
        return obj

    def _station_magnitude(self, element):
        """
        Converts an etree.Element into a StationMagnitude object.

        :type element: etree.Element
        :rtype: :class:`~obspy.core.event.StationMagnitude`

        .. rubric:: Example

        >>> from obspy.core.util import XMLParser
        >>> XML = '''<?xml version="1.0" encoding="UTF-8"?>
        ... <stationMagnitude>
        ...   <mag><value>3.2</value></mag>
        ... </stationMagnitude>'''
        >>> parser = XMLParser(XML)
        >>> unpickler = Unpickler(parser)
        >>> station_mag = unpickler._station_magnitude(parser.xml_root)
        >>> print(station_mag.mag)
        3.2
        """
        obj = StationMagnitude()
        # required parameter
        obj.resource_id = element.get('publicID')
        obj.origin_id = self._xpath2obj('originID', element) or ''
        obj.mag, obj.mag_errors = self._float_value(element, 'mag')
        # optional parameter
        obj.station_magnitude_type = self._xpath2obj('type', element)
        obj.amplitude_id = self._xpath2obj('amplitudeID', element)
        obj.method_id = self._xpath2obj('methodID', element)
        obj.waveform_id = self._waveform_id(element)
        obj.creation_info = self._creation_info(element)
        obj.comments = self._comments(element)
        return obj

    def _axis(self, element, name):
        """
        Converts an etree.Element into an Axis object.

        :type element: etree.Element
        :type name: tag name of axis
        :rtype: :class:`~obspy.core.event.Axis`
        """
        obj = Axis()
        try:
            sub_el = self._xpath(name, element)[0]
        except:
            return obj
        # required parameter
        obj.azimuth, obj.azimuth_errors = self._float_value(sub_el, 'azimuth')
        obj.plunge, obj.plunge_errors = self._float_value(sub_el, 'plunge')
        obj.length, obj.length_errors = self._float_value(sub_el, 'length')
        return obj

    def _principal_axes(self, element):
        """
        Converts an etree.Element into an PrincipalAxes object.

        :type element: etree.Element
        :rtype: :class:`~obspy.core.event.PrincipalAxes`
        """
        try:
            sub_el = self._xpath('principalAxes', element)[0]
        except:
            return None
        obj = PrincipalAxes()
        # required parameter
        obj.t_axis = self._axis(sub_el, 'tAxis')
        obj.p_axis = self._axis(sub_el, 'pAxis')
        # optional parameter
        obj.n_axis = self._axis(sub_el, 'nAxis')
        return obj

    def _nodal_plane(self, element, name):
        """
        Converts an etree.Element into an NodalPlane object.

        :type element: etree.Element
        :type name: tag name of sub nodal plane
        :rtype: :class:`~obspy.core.event.NodalPlane`
        """
        obj = NodalPlane()
        try:
            sub_el = self._xpath(name, element)[0]
        except:
            return obj
        # required parameter
        obj.strike, obj.strike_errors = self._float_value(sub_el, 'strike')
        obj.dip, obj.dip_errors = self._float_value(sub_el, 'dip')
        obj.rake, obj.rake_errors = self._float_value(sub_el, 'rake')
        return obj

    def _nodal_planes(self, element):
        """
        Converts an etree.Element into an NodalPlanes object.

        :type element: etree.Element
        :rtype: :class:`~obspy.core.event.NodalPlanes`
        """
        obj = NodalPlanes()
        try:
            sub_el = self._xpath('nodalPlanes', element)[0]
        except:
            return obj
        # optional parameter
        obj.nodal_plane_1 = self._nodal_plane(sub_el, 'nodalPlane1')
        obj.nodal_plane_2 = self._nodal_plane(sub_el, 'nodalPlane2')
        # optional attribute
        try:
            obj.preferred_plane = int(sub_el.get('preferredPlane'))
        except:
            obj.preferred_plane = None
        return obj

    def _source_time_function(self, element):
        """
        Converts an etree.Element into an SourceTimeFunction object.

        :type element: etree.Element
        :rtype: :class:`~obspy.core.event.SourceTimeFunction`
        """
        obj = SourceTimeFunction()
        try:
            sub_el = self._xpath('sourceTimeFunction', element)[0]
        except:
            return obj
        # required parameters
        obj.type = self._xpath2obj('type', sub_el)
        obj.duration = self._xpath2obj('duration', sub_el, float)
        # optional parameter
        obj.rise_time = self._xpath2obj('riseTime', sub_el, float)
        obj.decay_time = self._xpath2obj('decayTime', sub_el, float)
        return obj

    def _tensor(self, element):
        """
        Converts an etree.Element into an Tensor object.

        :type element: etree.Element
        :rtype: :class:`~obspy.core.event.Tensor`
        """
        obj = Tensor()
        try:
            sub_el = self._xpath('tensor', element)[0]
        except:
            return obj
        # required parameters
        obj.m_rr, obj.m_rr_errors = self._float_value(sub_el, 'Mrr')
        obj.m_tt, obj.m_tt_errors = self._float_value(sub_el, 'Mtt')
        obj.m_pp, obj.m_pp_errors = self._float_value(sub_el, 'Mpp')
        obj.m_rt, obj.m_rt_errors = self._float_value(sub_el, 'Mrt')
        obj.m_rp, obj.m_rp_errors = self._float_value(sub_el, 'Mrp')
        obj.m_tp, obj.m_tp_errors = self._float_value(sub_el, 'Mtp')
        return obj

    def _data_used(self, element):
        """
        Converts an etree.Element into an DataUsed object.

        :type element: etree.Element
        :rtype: :class:`~obspy.core.event.DataUsed`
        """
        obj = DataUsed()
        try:
            sub_el = self._xpath('dataUsed', element)[0]
        except:
            return obj
        # required parameters
        obj.wave_type = self._xpath2obj('waveType', sub_el)
        # optional parameter
        obj.station_count = self._xpath2obj('stationCount', sub_el, int)
        obj.component_count = self._xpath2obj('componentCount', sub_el, int)
        obj.shortest_period = self._xpath2obj('shortestPeriod', sub_el, float)
        obj.longest_period = self._xpath2obj('longestPeriod', sub_el, float)
        return obj

    def _moment_tensor(self, element):
        """
        Converts an etree.Element into an MomentTensor object.

        :type element: etree.Element
        :rtype: :class:`~obspy.core.event.MomentTensor`
        """
        obj = MomentTensor()
        try:
            mt_el = self._xpath('momentTensor', element)[0]
        except:
            return obj
        # required parameters
        obj.resource_id = mt_el.get('publicID')
        obj.derived_origin_id = self._xpath2obj('derivedOriginID', mt_el)
        # optional parameter
        obj.data_used = self._data_used(mt_el)
        obj.moment_magnitude_id = self._xpath2obj('momentMagnitudeID', mt_el)
        obj.scalar_moment, obj.scalar_moment_errors = \
            self._float_value(mt_el, 'scalarMoment')
        obj.tensor = self._tensor(mt_el)
        obj.variance = self._xpath2obj('variance', mt_el, float)
        obj.variance_reduction = \
            self._xpath2obj('varianceReduction', mt_el, float)
        obj.double_couple = self._xpath2obj('doubleCouple', mt_el, float)
        obj.clvd = self._xpath2obj('clvd', mt_el, float)
        obj.iso = self._xpath2obj('iso', mt_el, float)
        obj.greens_function_id = self._xpath2obj('greensFunctionID', mt_el)
        obj.filter_id = self._xpath2obj('filterID', mt_el)
        obj.source_time_function = self._source_time_function(mt_el)
        obj.method_id = self._xpath2obj('MethodID', mt_el)
        obj.category = self._xpath2obj('category', mt_el)
        obj.inversion_type = self._xpath2obj('inversionType', mt_el)
        obj.evaluation_mode = self._xpath2obj('evaluationMode', mt_el)
        obj.evaluation_status = self._xpath2obj('evaluationStatus', mt_el)
        obj.creation_info = self._creation_info(mt_el)
        obj.comments = self._comments(mt_el)
        return obj

    def _focal_mechanism(self, element):
        """
        Converts an etree.Element into a FocalMechanism object.

        :type element: etree.Element
        :rtype: :class:`~obspy.core.event.FocalMechanism`

        .. rubric:: Example

        >>> from obspy.core.util import XMLParser
        >>> XML = '''<?xml version="1.0" encoding="UTF-8"?>
        ... <focalMechanism>
        ...   <methodID>smi:ISC/methodID=Best_double_couple</methodID>
        ... </focalMechanism>'''
        >>> parser = XMLParser(XML)
        >>> unpickler = Unpickler(parser)
        >>> fm = unpickler._focal_mechanism(parser.xml_root)
        >>> print(fm.method_id)
        smi:ISC/methodID=Best_double_couple
        """
        obj = FocalMechanism()
        # required parameter
        obj.resource_id = element.get('publicID')
        # optional parameter
        obj.waveform_id = self._waveform_id(element)
        obj.triggering_origin_id = \
            self._xpath2obj('triggeringOriginID', element)
        obj.azimuthal_gap = self._xpath2obj('azimuthalGap', element, float)
        obj.station_polarity_count = \
            self._xpath2obj('stationPolarityCount', element, int)
        obj.misfit = self._xpath2obj('misfit', element, float)
        obj.station_distribution_ratio = \
            self._xpath2obj('stationDistributionRatio', element, float)
        obj.method_id = self._xpath2obj('methodID', element)
        obj.moment_tensor = self._moment_tensor(element)
        obj.nodal_planes = self._nodal_planes(element)
        obj.principal_axes = self._principal_axes(element)
        obj.creation_info = self._creation_info(element)
        obj.comments = self._comments(element)
        return obj

    def _deserialize(self):
        # check node "quakeml/eventParameters" for global namespace
        try:
            namespace = self.parser._getFirstChildNamespace()
            catalog_el = self._xpath('eventParameters', namespace=namespace)[0]
        except:
            raise Exception("Not a QuakeML compatible file or string")
        # set default namespace for parser
        self.parser.namespace = self.parser._getElementNamespace(catalog_el)
        # create catalog
        catalog = Catalog()
        # optional catalog attributes
        catalog.resource_id = catalog_el.get('publicID')
        catalog.description = self._xpath2obj('description', catalog_el)
        catalog.comments = self._comments(catalog_el)
        catalog.creation_info = self._creation_info(catalog_el)
        # loop over all events
        for event_el in self._xpath('event', catalog_el):
            # create new Event object
            resource_id = event_el.get('publicID')
            event = Event(resource_id)
            # optional event attributes
            event.preferred_origin_id = \
                self._xpath2obj('preferredOriginID', event_el)
            event.preferred_magnitude_id = \
                self._xpath2obj('preferredMagnitudeID', event_el)
            event.preferred_focal_mechanism_id = \
                self._xpath2obj('preferredFocalMechanismID', event_el)
            event_type = self._xpath2obj('type', event_el)
            # Change for QuakeML 1.2RC4. 'null' is no longer acceptable as an
            # event type. Will be replaced with 'not reported'.
            if event_type == "null":
                event_type = "not reported"
            event.event_type = event_type
            event.event_type_certainty = self._xpath2obj('typeCertainty',
                    event_el)
            event.creation_info = self._creation_info(event_el)
            event.event_descriptions = self._event_description(event_el)
            event.comments = self._comments(event_el)
            # origins
            event.origins = []
            for origin_el in self._xpath('origin', event_el):
                origin = self._origin(origin_el)
                # arrivals
                origin.arrivals = []
                for arrival_el in self._xpath('arrival', origin_el):
                    arrival = self._arrival(arrival_el)
                    origin.arrivals.append(arrival)
                # append origin with arrivals
                event.origins.append(origin)
            # magnitudes
            event.magnitudes = []
            for magnitude_el in self._xpath('magnitude', event_el):
                magnitude = self._magnitude(magnitude_el)
                event.magnitudes.append(magnitude)
            # station magnitudes
            event.station_magnitudes = []
            for magnitude_el in self._xpath('stationMagnitude', event_el):
                magnitude = self._station_magnitude(magnitude_el)
                event.station_magnitudes.append(magnitude)
            # picks
            event.picks = []
            for pick_el in self._xpath('pick', event_el):
                pick = self._pick(pick_el)
                event.picks.append(pick)
            # focal mechanisms
            event.focal_mechanisms = []
            for fm_el in self._xpath('focalMechanism', event_el):
                fm = self._focal_mechanism(fm_el)
                event.focal_mechanisms.append(fm)
            # finally append newly created event to catalog
            catalog.append(event)
        return catalog


class Pickler(object):
    """
    Serializes an ObsPy Catalog object into QuakeML format.
    """
    def dump(self, catalog, file):
        """
        Writes ObsPy Catalog into given file.

        :type catalog: :class:`~obspy.core.event.Catalog`
        :param catalog: ObsPy Catalog object.
        :type file: str
        :param file: File name.
        """
        fh = open(file, 'wt')
        fh.write(self._serialize(catalog))
        fh.close()

    def dumps(self, catalog):
        """
        Returns QuakeML string of given ObsPy Catalog object.

        :type catalog: :class:`~obspy.core.event.Catalog`
        :param catalog: ObsPy Catalog object.
        :rtype: str
        :returns: QuakeML formated string.
        """
        return self._serialize(catalog)

    def _id(self, obj):
        try:
            return obj.getQuakeMLURI()
        except:
            return ResourceIdentifier().getQuakeMLURI()

    def _str(self, value, root, tag, always_create=False):
        if always_create is False and value is None:
            return
        etree.SubElement(root, tag).text = "%s" % (value)

    def _bool(self, value, root, tag, always_create=False):
        if always_create is False and value is None:
            return
        etree.SubElement(root, tag).text = str(bool(value)).lower()

    def _time(self, value, root, tag, always_create=False):
        if always_create is False and value is None:
            return
        dt = value.strftime("%Y-%m-%dT%H:%M:%S+00:00")
        etree.SubElement(root, tag).text = "%s" % (dt)

    def _value(self, quantity, error, element, tag, always_create=False):
        if always_create is False and quantity is None:
            return
        subelement = etree.Element(tag)
        self._str(quantity, subelement, 'value')
        self._str(error.uncertainty, subelement, 'uncertainty')
        self._str(error.lower_uncertainty, subelement, 'lowerUncertainty')
        self._str(error.upper_uncertainty, subelement, 'upperUncertainty')
        self._str(error.confidence_level, subelement, 'confidenceLevel')
        element.append(subelement)

    def _waveform_id(self, obj, element, required=False):
        attrib = {}
        if obj is None:
            return
        if obj.network_code:
            attrib['networkCode'] = obj.network_code
        if obj.station_code:
            attrib['stationCode'] = obj.station_code
        if obj.location_code is not None:
            attrib['locationCode'] = obj.location_code
        if obj.channel_code:
            attrib['channelCode'] = obj.channel_code
        subelement = etree.Element('waveformID', attrib=attrib)
        # WaveformStreamID has a non-mandatory resource_id
        if obj.resource_uri is None or obj.resource_uri == "":
            subelement.text = ""
        else:
            subelement.text = self._id(obj.resource_uri)

        if len(subelement.attrib) > 0 or required:
            element.append(subelement)

    def _creation_info(self, creation_info, element):
        if creation_info is None:
            return
        subelement = etree.Element('creationInfo')
        self._str(creation_info.agency_id, subelement, 'agencyID')
        self._str(creation_info.agency_uri, subelement, 'agencyURI')
        self._str(creation_info.author, subelement, 'author')
        self._str(creation_info.author_uri, subelement, 'authorURI')
        self._time(creation_info.creation_time, subelement, 'creationTime')
        self._str(creation_info.version, subelement, 'version')
        # append only if at least one sub-element is set
        if len(subelement) > 0:
            element.append(subelement)

    def _station_magnitude_contributions(self, stat_contrib, element):
        for contrib in stat_contrib:
            contrib_el = etree.Element('stationMagnitudeContribution')
            etree.SubElement(contrib_el, 'stationMagnitudeID').text = \
                contrib.station_magnitude_id.resource_id
            if contrib.weight:
                etree.SubElement(contrib_el, 'weight').text = \
                    str(contrib.weight)
            if contrib.residual:
                etree.SubElement(contrib_el, 'residual').text = \
                    str(contrib.residual)
            element.append(contrib_el)

    def _comments(self, comments, element):
        for comment in comments:
            attrib = {}
            if comment.resource_id:
                attrib['id'] = self._id(comment.resource_id)
            comment_el = etree.Element('comment', attrib=attrib)
            etree.SubElement(comment_el, 'text').text = comment.text
            self._creation_info(comment.creation_info, comment_el)
            element.append(comment_el)

    def _arrival(self, arrival):
        """
        Converts an Arrival into etree.Element object.

        :type arrival: :class:`~obspy.core.event.Arrival`
        :rtype: etree.Element
        """
        attrib = {'publicID': self._id(arrival.resource_id)}
        element = etree.Element('arrival', attrib=attrib)
        # required parameter
        self._str(arrival.pick_id, element, 'pickID', True)
        self._str(arrival.phase, element, 'phase', True)
        # optional parameter
        self._str(arrival.time_correction, element, 'timeCorrection')
        self._str(arrival.azimuth, element, 'azimuth')
        self._str(arrival.distance, element, 'distance')
        self._value(arrival.takeoff_angle, arrival.takeoff_angle_errors,
                    element, 'takeoffAngle')
        self._str(arrival.time_residual, element, 'timeResidual')
        self._str(arrival.horizontal_slowness_residual, element,
                  'horizontalSlownessResidual')
        self._str(arrival.backazimuth_residual, element, 'backazimuthResidual')
        self._str(arrival.time_weight, element, 'timeWeight')
        self._str(arrival.horizontal_slowness_weight, element,
                  'horizontalSlownessWeight')
        self._str(arrival.backazimuth_weight, element, 'backazimuthWeight')
        self._str(arrival.earth_model_id, element, 'earthModelID')
        self._comments(arrival.comments, element)
        self._creation_info(arrival.creation_info, element)
        return element

    def _magnitude(self, magnitude):
        """
        Converts an Magnitude into etree.Element object.

        :type magnitude: :class:`~obspy.core.event.Magnitude`
        :rtype: etree.Element

        .. rubric:: Example

        >>> from obspy.core.quakeml import Pickler
        >>> from obspy.core.event import Magnitude
        >>> from obspy.core.util import tostring
        >>> magnitude = Magnitude()
        >>> magnitude.mag = 3.2
        >>> el = Pickler()._magnitude(magnitude)
        >>> print(tostring(el))  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
        <?xml version='1.0' encoding='utf-8'?>
        <magnitude ...<mag><value>3.2</value></mag>...</magnitude>
        """
        element = etree.Element('magnitude',
            attrib={'publicID': self._id(magnitude.resource_id)})
        self._value(magnitude.mag, magnitude.mag_errors, element, 'mag', True)
        # optional parameter
        self._str(magnitude.magnitude_type, element, 'type')
        self._str(magnitude.origin_id, element, 'originID')
        self._str(magnitude.method_id, element, 'methodID')
        self._str(magnitude.station_count, element, 'stationCount')
        self._str(magnitude.azimuthal_gap, element, 'azimuthalGap')
        self._str(magnitude.evaluation_status, element, 'evaluationStatus')
        self._station_magnitude_contributions(
            magnitude.station_magnitude_contributions, element)
        self._comments(magnitude.comments, element)
        self._creation_info(magnitude.creation_info, element)
        return element

    def _station_magnitude(self, magnitude):
        """
        Converts an StationMagnitude into etree.Element object.

        :type magnitude: :class:`~obspy.core.event.StationMagnitude`
        :rtype: etree.Element

        .. rubric:: Example

        >>> from obspy.core.quakeml import Pickler
        >>> from obspy.core.event import StationMagnitude
        >>> from obspy.core.util import tostring
        >>> station_mag = StationMagnitude()
        >>> station_mag.mag = 3.2
        >>> el = Pickler()._station_magnitude(station_mag)
        >>> print(tostring(el))  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
        <?xml version='1.0' encoding='utf-8'?>
        <stationMagnitude ...<value>3.2</value>...</stationMagnitude>
        """
        element = etree.Element('stationMagnitude',
            attrib={'publicID': self._id(magnitude.resource_id)})
        self._str(magnitude.origin_id, element, 'originID', True)
        self._value(magnitude.mag, magnitude.mag_errors, element, 'mag', True)
        # optional parameter
        self._str(magnitude.station_magnitude_type, element, 'type')
        self._str(magnitude.amplitude_id, element, 'amplitudeID')
        self._str(magnitude.method_id, element, 'methodID')
        self._waveform_id(magnitude.waveform_id, element)
        self._comments(magnitude.comments, element)
        self._creation_info(magnitude.creation_info, element)
        return element

    def _origin(self, origin):
        """
        Converts an Origin into etree.Element object.

        :type origin: :class:`~obspy.core.event.Origin`
        :rtype: etree.Element

        .. rubric:: Example

        >>> from obspy.core.quakeml import Pickler
        >>> from obspy.core.event import Origin
        >>> from obspy.core.util import tostring
        >>> origin = Origin()
        >>> origin.latitude = 34.23
        >>> el = Pickler()._origin(origin)
        >>> print(tostring(el))  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
        <?xml version='1.0' encoding='utf-8'?>
        <origin ...<latitude><value>34.23</value></latitude>...</origin>
        """
        element = etree.Element('origin',
            attrib={'publicID': self._id(origin.resource_id)})
        self._value(origin.time, origin.time_errors, element, 'time', True)
        self._value(origin.latitude, origin.latitude_errors, element,
                    'latitude', True)
        self._value(origin.longitude, origin.longitude_errors, element,
                    'longitude', True)
        # optional parameter
        self._value(origin.depth, origin.depth_errors, element, 'depth')
        self._str(origin.depth_type, element, 'depthType')
        self._bool(origin.time_fixed, element, 'timeFixed')
        self._bool(origin.epicenter_fixed, element, 'epicenterFixed')
        self._str(origin.reference_system_id, element, 'referenceSystemID')
        self._str(origin.method_id, element, 'methodID')
        self._str(origin.earth_model_id, element, 'earthModelID')
        # compositeTime
        for ctime in origin.composite_times:
            ct_el = etree.Element('compositeTime')
            self._value(ctime.year, ctime.year_errors, ct_el, 'year')
            self._value(ctime.month, ctime.month_errors, ct_el, 'month')
            self._value(ctime.day, ctime.day_errors, ct_el, 'day')
            self._value(ctime.hour, ctime.hour_errors, ct_el, 'hour')
            self._value(ctime.minute, ctime.minute_errors, ct_el, 'minute')
            self._value(ctime.second, ctime.second_errors, ct_el, 'second')
            if len(ct_el) > 0:
                element.append(ct_el)
        # quality
        qu = origin.quality
        if qu:
            qu_el = etree.Element('quality')
            self._str(qu.associated_phase_count, qu_el, 'associatedPhaseCount')
            self._str(qu.used_phase_count, qu_el, 'usedPhaseCount')
            self._str(qu.associated_station_count, qu_el,
                      'associatedStationCount')
            self._str(qu.used_station_count, qu_el, 'usedStationCount')
            self._str(qu.depth_phase_count, qu_el, 'depthPhaseCount')
            self._str(qu.standard_error, qu_el, 'standardError')
            self._str(qu.azimuthal_gap, qu_el, 'azimuthalGap')
            self._str(qu.secondary_azimuthal_gap, qu_el,
                      'secondaryAzimuthalGap')
            self._str(qu.ground_truth_level, qu_el, 'groundTruthLevel')
            self._str(qu.minimum_distance, qu_el, 'minimumDistance')
            self._str(qu.maximum_distance, qu_el, 'maximumDistance')
            self._str(qu.median_distance, qu_el, 'medianDistance')
            if len(qu_el) > 0:
                element.append(qu_el)
        self._str(origin.origin_type, element, 'type')
        self._str(origin.evaluation_mode, element, 'evaluationMode')
        self._str(origin.evaluation_status, element, 'evaluationStatus')
        self._comments(origin.comments, element)
        self._creation_info(origin.creation_info, element)
        # origin uncertainty
        ou = origin.origin_uncertainty
        if ou is not None:
            ou_el = etree.Element('originUncertainty')
            self._str(ou.preferred_description, ou_el, 'preferredDescription')
            self._str(ou.horizontal_uncertainty, ou_el,
                      'horizontalUncertainty')
            self._str(ou.min_horizontal_uncertainty, ou_el,
                      'minHorizontalUncertainty')
            self._str(ou.max_horizontal_uncertainty, ou_el,
                      'maxHorizontalUncertainty')
            self._str(ou.azimuth_max_horizontal_uncertainty, ou_el,
                      'azimuthMaxHorizontalUncertainty')
            ce = ou.confidence_ellipsoid
            if ce is not None:
                ce_el = etree.Element('confidenceEllipsoid')
                self._str(ce.semi_major_axis_length, ce_el,
                          'semiMajorAxisLength')
                self._str(ce.semi_minor_axis_length, ce_el,
                          'semiMinorAxisLength')
                self._str(ce.semi_intermediate_axis_length, ce_el,
                          'semiIntermediateAxisLength')
                self._str(ce.major_axis_plunge, ce_el, 'majorAxisPlunge')
                self._str(ce.major_axis_azimuth, ce_el, 'majorAxisAzimuth')
                self._str(ce.major_axis_rotation, ce_el, 'majorAxisRotation')
                # add confidence ellipsoid to origin uncertainty only if set
                if len(ce_el) > 0:
                    ou_el.append(ce_el)
            # add origin uncertainty to origin only if anything is set
            if len(ou_el) > 0:
                element.append(ou_el)
        # arrivals
        for ar in origin.arrivals:
            element.append(self._arrival(ar))
        return element

    def _pick(self, pick):
        """
        Converts a Pick into etree.Element object.

        :type pick: :class:`~obspy.core.event.Pick`
        :rtype: etree.Element
        """
        element = etree.Element('pick',
        attrib={'publicID': self._id(pick.resource_id)})
        # required parameter
        self._value(pick.time, pick.time_errors, element, 'time', True)
        self._waveform_id(pick.waveform_id, element, True)
        # optional parameter
        self._str(pick.filter_id, element, 'filterID')
        self._str(pick.method_id, element, 'methodID')
        self._value(pick.horizontal_slowness, pick.horizontal_slowness_errors,
                    element, 'horizontalSlowness')
        self._value(pick.backazimuth, pick.backazimuth_errors, element,
                    'backazimuth')
        self._str(pick.slowness_method_id, element, 'slownessMethodID')
        self._str(pick.onset, element, 'onset')
        self._str(pick.phase_hint, element, 'phaseHint')
        self._str(pick.polarity, element, 'polarity')
        self._str(pick.evaluation_mode, element, 'evaluationMode')
        self._str(pick.evaluation_status, element, 'evaluationStatus')
        self._comments(pick.comments, element)
        self._creation_info(pick.creation_info, element)
        return element

    def _nodal_planes(self, obj, element):
        """
        Converts a NodalPlanes into etree.Element object.

        :type pick: :class:`~obspy.core.event.NodalPlanes`
        :rtype: etree.Element
        """
        subelement = etree.Element('nodalPlanes')
        # optional
        if obj.nodal_plane_1:
            el = etree.Element('nodalPlane1')
            self._value(obj.nodal_plane_1.strike,
                        obj.nodal_plane_1.strike_errors, el, 'strike')
            self._value(obj.nodal_plane_1.dip,
                        obj.nodal_plane_1.dip_errors, el, 'dip')
            self._value(obj.nodal_plane_1.rake,
                        obj.nodal_plane_1.rake_errors, el, 'rake')
            subelement.append(el)
        if obj.nodal_plane_2:
            el = etree.Element('nodalPlane2')
            self._value(obj.nodal_plane_2.strike,
                        obj.nodal_plane_2.strike_errors, el, 'strike')
            self._value(obj.nodal_plane_2.dip,
                        obj.nodal_plane_2.dip_errors, el, 'dip')
            self._value(obj.nodal_plane_2.rake,
                        obj.nodal_plane_2.rake_errors, el, 'rake')
            subelement.append(el)
        if obj.preferred_plane:
            subelement.attrib['preferredPlane'] = str(obj.preferred_plane)
        # append only if at least one sub-element is set
        if len(subelement) > 0:
            element.append(subelement)

    def _principal_axes(self, obj, element):
        """
        Converts a PrincipalAxes into etree.Element object.

        :type pick: :class:`~obspy.core.event.PrincipalAxes`
        :rtype: etree.Element
        """
        if obj is None:
            return
        subelement = etree.Element('principalAxes')
        # tAxis
        el = etree.Element('tAxis')
        self._value(obj.t_axis.azimuth,
                    obj.t_axis.azimuth_errors, el, 'azimuth')
        self._value(obj.t_axis.plunge,
                    obj.t_axis.plunge_errors, el, 'plunge')
        self._value(obj.t_axis.length,
                    obj.t_axis.length_errors, el, 'length')
        subelement.append(el)
        # pAxis
        el = etree.Element('pAxis')
        self._value(obj.p_axis.azimuth,
                    obj.p_axis.azimuth_errors, el, 'azimuth')
        self._value(obj.p_axis.plunge,
                    obj.p_axis.plunge_errors, el, 'plunge')
        self._value(obj.p_axis.length,
                    obj.p_axis.length_errors, el, 'length')
        subelement.append(el)
        # nAxis (optional)
        if obj.n_axis:
            el = etree.Element('nAxis')
            self._value(obj.n_axis.azimuth,
                        obj.n_axis.azimuth_errors, el, 'azimuth')
            self._value(obj.n_axis.plunge,
                        obj.n_axis.plunge_errors, el, 'plunge')
            self._value(obj.n_axis.length,
                        obj.n_axis.length_errors, el, 'length')
            subelement.append(el)
        element.append(subelement)

    def _moment_tensor(self, moment_tensor, element):
        """
        Converts a MomentTensor into etree.Element object.

        :type pick: :class:`~obspy.core.event.MomentTensor`
        :rtype: etree.Element
        """
        if moment_tensor is None:
            return
        mt_el = etree.Element('momentTensor')
        if moment_tensor.resource_id:
            mt_el.attrib['publicID'] = self._id(moment_tensor.resource_id)
        # required parameters
        self._str(moment_tensor.derived_origin_id, mt_el, 'derivedOriginID')
        # optional parameter
        # Data Used
        if moment_tensor.data_used:
            sub_el = etree.Element('dataUsed')
            sub = moment_tensor.data_used
            self._str(sub.wave_type, sub_el, 'waveType')
            self._str(sub.station_count, sub_el, 'stationCount')
            self._str(sub.component_count, sub_el, 'componentCount')
            self._str(sub.shortest_period, sub_el, 'shortestPeriod')
            self._str(sub.longest_period, sub_el, 'longestPeriod')
            mt_el.append(sub_el)
        self._str(moment_tensor.moment_magnitude_id, mt_el,
            'momentMagnitudeID')
        self._value(moment_tensor.scalar_moment,
            moment_tensor.scalar_moment_errors, mt_el, 'scalarMoment')
        # Tensor
        if moment_tensor.tensor:
            sub_el = etree.Element('tensor')
            sub = moment_tensor.tensor
            self._value(sub.m_rr, sub.m_rr_errors, sub_el, 'Mrr')
            self._value(sub.m_tt, sub.m_tt_errors, sub_el, 'Mtt')
            self._value(sub.m_pp, sub.m_pp_errors, sub_el, 'Mpp')
            self._value(sub.m_rt, sub.m_rt_errors, sub_el, 'Mrt')
            self._value(sub.m_rp, sub.m_rp_errors, sub_el, 'Mrp')
            self._value(sub.m_tp, sub.m_tp_errors, sub_el, 'Mtp')
            mt_el.append(sub_el)
        self._str(moment_tensor.variance, mt_el, 'variance')
        self._str(moment_tensor.variance_reduction, mt_el, 'varianceReduction')
        self._str(moment_tensor.double_couple, mt_el, 'doubleCouple')
        self._str(moment_tensor.clvd, mt_el, 'clvd')
        self._str(moment_tensor.iso, mt_el, 'iso')
        self._str(moment_tensor.greens_function_id, mt_el, 'greensFunctionID')
        self._str(moment_tensor.filter_id, mt_el, 'filterID')
        # SourceTimeFunction
        if moment_tensor.source_time_function:
            sub_el = etree.Element('sourceTimeFunction')
            sub = moment_tensor.source_time_function
            self._str(sub.type, sub_el, 'type')
            self._str(sub.duration, sub_el, 'duration')
            self._str(sub.rise_time, sub_el, 'riseTime')
            self._str(sub.decay_time, sub_el, 'decayTime')
            mt_el.append(sub_el)
        self._str(moment_tensor.method_id, mt_el, 'MethodID')
        self._str(moment_tensor.category, mt_el, 'category')
        self._str(moment_tensor.inversion_type, mt_el, 'inversionType')
        self._str(moment_tensor.evaluation_mode, mt_el, 'evaluationMode')
        self._str(moment_tensor.evaluation_status, mt_el, 'evaluationStatus')
        self._comments(moment_tensor.comments, mt_el)
        self._creation_info(moment_tensor.creation_info, mt_el)
        element.append(mt_el)

    def _focal_mechanism(self, focal_mechanism):
        """
        Converts a FocalMechanism into etree.Element object.

        :type pick: :class:`~obspy.core.event.FocalMechanism`
        :rtype: etree.Element
        """
        element = etree.Element('focalMechanism',
            attrib={'publicID': self._id(focal_mechanism.resource_id)})
        # optional parameter
        self._waveform_id(focal_mechanism.waveform_id, element)
        self._str(focal_mechanism.triggering_origin_id, element,
                  'triggeringOriginID')
        self._str(focal_mechanism.azimuthal_gap, element,
                  'azimuthalGap')
        self._str(focal_mechanism.station_polarity_count, element,
                  'stationPolarityCount')
        self._str(focal_mechanism.misfit, element, 'misfit')
        self._str(focal_mechanism.station_distribution_ratio, element,
                  'stationDistributionRatio')
        self._nodal_planes(focal_mechanism.nodal_planes, element)
        self._principal_axes(focal_mechanism.principal_axes, element)
        self._str(focal_mechanism.method_id, element, 'methodID')
        self._moment_tensor(focal_mechanism.moment_tensor, element)
        self._comments(focal_mechanism.comments, element)
        self._creation_info(focal_mechanism.creation_info, element)
        return element

    def _serialize(self, catalog, pretty_print=True):
        """
        Converts a Catalog object into XML string.
        """
        root_el = etree.Element(
            '{http://quakeml.org/xmlns/quakeml/1.2}quakeml',
            attrib={'xmlns': "http://quakeml.org/xmlns/bed/1.2"})
        catalog_el = etree.Element('eventParameters',
            attrib={'publicID': self._id(catalog.resource_id)})
        # optional catalog parameters
        if catalog.description:
            self._str(catalog.description, catalog_el, 'description')
        self._comments(catalog.comments, catalog_el)
        self._creation_info(catalog.creation_info, catalog_el)
        root_el.append(catalog_el)
        for event in catalog:
            # create event node
            event_el = etree.Element('event',
                attrib={'publicID': self._id(event.resource_id)})
            # optional event attributes
            if hasattr(event, "preferred_origin_id"):
                self._str(event.preferred_origin_id, event_el,
                        'preferredOriginID')
            if hasattr(event, "preferred_magnitude_id"):
                self._str(event.preferred_magnitude_id, event_el,
                         'preferredMagnitudeID')
            if hasattr(event, "preferred_focal_mechanism_id"):
                self._str(event.preferred_focal_mechanism_id, event_el,
                         'preferredFocalMechanismID')
            # event type and event type certainty also are optional attributes.
            if hasattr(event, "event_type"):
                self._str(event.event_type, event_el, 'type')
            if hasattr(event, "event_type_certainty"):
                self._str(event.event_type_certainty, event_el,
                    'typeCertainty')
            # event descriptions
            for description in event.event_descriptions:
                el = etree.Element('description')
                self._str(description.text, el, 'text', True)
                self._str(description.type, el, 'type')
                event_el.append(el)
            self._comments(event.comments, event_el)
            self._creation_info(event.creation_info, event_el)
            # origins
            for origin in event.origins:
                event_el.append(self._origin(origin))
            # magnitudes
            for magnitude in event.magnitudes:
                event_el.append(self._magnitude(magnitude))
            # station magnitudes
            for magnitude in event.station_magnitudes:
                event_el.append(self._station_magnitude(magnitude))
            # picks
            for pick in event.picks:
                event_el.append(self._pick(pick))
            # focal mechanisms
            for focal_mechanism in event.focal_mechanisms:
                event_el.append(self._focal_mechanism(focal_mechanism))
            # add event node to catalog
            catalog_el.append(event_el)
        return tostring(root_el, pretty_print=pretty_print)


def readQuakeML(filename):
    """
    Reads a QuakeML file and returns a ObsPy Catalog object.

    .. warning::
        This function should NOT be called directly, it registers via the
        ObsPy :func:`~obspy.core.event.readEvents` function, call this instead.

    :type filename: str
    :param filename: QuakeML file to be read.
    :rtype: :class:`~obspy.core.event.Catalog`
    :return: A ObsPy Catalog object.

    .. rubric:: Example

    >>> from obspy.core.event import readEvents
    >>> cat = readEvents('/path/to/iris_events.xml')
    >>> print cat
    2 Event(s) in Catalog:
    2011-03-11T05:46:24.120000Z | +38.297, +142.373 | 9.1 MW
    2006-09-10T04:26:33.610000Z |  +9.614, +121.961 | 9.8 MS
    """
    return Unpickler().load(filename)


def writeQuakeML(catalog, filename, **kwargs):  # @UnusedVariable
    """
    Writes a QuakeML file.

    .. warning::
        This function should NOT be called directly, it registers via the
        the :meth:`~obspy.core.event.Catalog.write` method of an
        ObsPy :class:`~obspy.core.event.Catalog` object, call this instead.

    :type catalog: :class:`~obspy.core.stream.Catalog`
    :param catalog: The ObsPy Catalog object to write.
    :type filename: str
    :param filename: Name of file to write.
    """
    # Open filehandler or use an existing file like object.
    if not hasattr(filename, 'write'):
        fh = open(filename, 'wt')
    else:
        fh = filename

    xml_doc = Pickler().dumps(catalog)
    fh.write(xml_doc)
    fh.close()
    # Close if its a file handler.
    if isinstance(fh, file):
        fh.close()


def readSeisHubEventXML(filename):
    """
    Reads a single SeisHub event XML file and returns a ObsPy Catalog object.
    """
    # XXX: very ugly way to add new root tags without parsing
    lines = open(filename, 'rt').readlines()
    lines.insert(2, '<quakeml xmlns="http://quakeml.org/xmlns/quakeml/1.0">\n')
    lines.insert(3, '  <eventParameters>')
    lines.append('  </eventParameters>\n')
    lines.append('</quakeml>\n')
    temp = StringIO.StringIO(''.join(lines))
    return readQuakeML(temp)


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
