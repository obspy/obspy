# -*- coding: utf-8 -*-
"""
QuakeML read and write support.

QuakeML is a flexible, extensible and modular XML representation of
seismological data which is intended to cover a broad range of fields of
application in modern seismology. QuakeML is an open standard and is developed
by a distributed team in a transparent collaborative manner.

.. seealso:: https://quake.ethz.ch/quakeml/

.. note::

    For handling additional information not covered by the QuakeML standard and
    how to output it to QuakeML see the :ref:`ObsPy Tutorial <quakeml-extra>`.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""
import inspect
import io
import os
import re
import warnings

from lxml import etree

from obspy.core import compatibility
from obspy.core.event import (Amplitude, Arrival, Axis, Catalog, Comment,
                              CompositeTime, ConfidenceEllipsoid, CreationInfo,
                              DataUsed, Event, EventDescription,
                              FocalMechanism, Magnitude, MomentTensor,
                              NodalPlane, NodalPlanes, Origin, OriginQuality,
                              OriginUncertainty, Pick, PrincipalAxes,
                              QuantityError, ResourceIdentifier,
                              SourceTimeFunction, StationMagnitude,
                              StationMagnitudeContribution, Tensor, TimeWindow,
                              WaveformStreamID)
from obspy.core.utcdatetime import UTCDateTime
from obspy.core.util import AttribDict, Enum

QUAKEML_ROOTTAG_REGEX = r'^{(http://quakeml.org/xmlns/quakeml/([^}]*))}quakeml'
NS_QUAKEML_PATTERN = 'http://quakeml.org/xmlns/quakeml/{version}'
NS_QUAKEML_BED_PATTERN = 'http://quakeml.org/xmlns/bed/{version}'
NSMAP_QUAKEML = {None: NS_QUAKEML_BED_PATTERN.format(version="1.2"),
                 'q': NS_QUAKEML_PATTERN.format(version="1.2")}


def _get_first_child_namespace(element):
    """
    Helper function extracting the namespace of an element.
    """
    try:
        element = element[0]
    except IndexError:
        return None
    return etree.QName(element.tag).namespace


def _xml_doc_from_anything(source):
    """
    Helper function attempting to create an xml etree element from either a
    filename, a file-like object, or a (byte)string.

    Will raise a ValueError if it fails.
    """
    try:
        xml_doc = etree.parse(source).getroot()
    except Exception:
        try:
            xml_doc = etree.fromstring(source)
        except Exception:
            try:
                xml_doc = etree.fromstring(source.encode())
            except Exception:
                raise ValueError("Could not parse '%s' to an etree element." %
                                 source)
    return xml_doc


def _is_quakeml(filename):
    """
    Checks whether a file is QuakeML format.

    :type filename: str
    :param filename: Name of the QuakeML file to be checked.
    :rtype: bool
    :return: ``True`` if QuakeML file.

    .. rubric:: Example

    >>> _is_quakeml('/path/to/quakeml.xml')  # doctest: +SKIP
    True
    """
    if hasattr(filename, "tell") and hasattr(filename, "seek") and \
            hasattr(filename, "read"):
        file_like_object = True
        position = filename.tell()
    else:
        file_like_object = False

    try:
        xml_doc = _xml_doc_from_anything(filename)
    except Exception:
        return False
    finally:
        if file_like_object:
            filename.seek(position, 0)

    # check if node "*/eventParameters/event" for the global namespace exists
    try:
        if hasattr(xml_doc, "getroot"):
            namespace = _get_first_child_namespace(xml_doc.getroot())
        else:
            namespace = _get_first_child_namespace(xml_doc)
        xml_doc.xpath('q:eventParameters', namespaces={"q": namespace})[0]
    except Exception:
        return False
    return True


class Unpickler(object):
    """
    De-serializes a QuakeML string into an ObsPy Catalog object.
    """
    def __init__(self, xml_doc=None):
        self.xml_doc = xml_doc

    @property
    def xml_root(self):
        try:
            return self.xml_doc.getroot()
        except AttributeError:
            return self.xml_doc

    def load(self, file):
        """
        Reads QuakeML file into ObsPy catalog object.

        :type file: str
        :param file: File name to read.
        :rtype: :class:`~obspy.core.event.Catalog`
        :returns: ObsPy Catalog object.
        """
        self.xml_doc = _xml_doc_from_anything(file)
        return self._deserialize()

    def loads(self, string):
        """
        Parses QuakeML string into ObsPy catalog object.

        :type string: str
        :param string: QuakeML string to parse.
        :rtype: :class:`~obspy.core.event.Catalog`
        :returns: ObsPy Catalog object.
        """
        self.xml_doc = etree.parse(io.BytesIO(string))
        return self._deserialize()

    def _xpath2obj(self, xpath, element=None, convert_to=str, namespace=None):
        q = self._xpath(xpath, element=element, namespace=namespace)
        if not q:
            return None
        text = q[0].text
        if text is None or text == '':
            return None
        if convert_to == bool:
            if text.lower() in ["true", "1"]:
                return True
            elif text.lower() in ["false", "0"]:
                return False
            return None
        try:
            return convert_to(text)
        except Exception:
            msg = "Could not convert %s to type %s. Returning None."
            warnings.warn(msg % (text, convert_to))
        return None

    def _set_enum(self, xpath, element, obj, key):
        obj_type = obj._property_dict[key]
        if not isinstance(obj_type, Enum):  # pragma: no cover
            raise ValueError
        value = self._xpath2obj(xpath, element)
        try:
            setattr(obj, key, value)
        except ValueError as e:
            msg = ('%s. The attribute "%s" will not be set and will be missing'
                   ' in the resulting object.' % (e.args[0], key))
            warnings.warn(msg)

    def _xpath(self, xpath, element=None, namespace=None):
        if element is None:
            element = self.xml_root

        namespaces = None
        if namespace:
            xpath = "b:%s" % xpath
            namespaces = {"b": namespace}
        elif hasattr(element, "nsmap") and None in element.nsmap:
            xpath = "b:%s" % xpath
            namespaces = {"b": element.nsmap[None]}
        elif hasattr(self, "nsmap") and None in self.nsmap:
            xpath = "b:%s" % xpath
            namespaces = {"b": self.nsmap[None]}

        return element.xpath(xpath, namespaces=namespaces)

    def _comments(self, parent):
        obj = []
        for el in self._xpath('comment', parent):
            comment = Comment(force_resource_id=False)
            comment.text = self._xpath2obj('text', el)
            comment.creation_info = self._creation_info(el)
            comment.resource_id = el.get('id', None)
            self._extra(el, comment)
            obj.append(comment)
        return obj

    def _station_magnitude_contributions(self, parent):
        obj = []
        for el in self._xpath("stationMagnitudeContribution", parent):
            contrib = StationMagnitudeContribution()
            contrib.weight = self._xpath2obj("weight", el, float)
            contrib.residual = self._xpath2obj("residual", el, float)
            contrib.station_magnitude_id = \
                self._xpath2obj("stationMagnitudeID", el, str)
            self._extra(el, contrib)
            obj.append(contrib)
        return obj

    def _creation_info(self, parent):
        elements = self._xpath("creationInfo", parent)
        if len(elements) > 1:
            raise NotImplementedError("Only one CreationInfo allowed.")
        elif len(elements) == 0:
            return None
        element = elements[0]
        obj = CreationInfo()
        obj.agency_uri = self._xpath2obj('agencyURI', element)
        obj.author_uri = self._xpath2obj('authorURI', element)
        obj.agency_id = self._xpath2obj('agencyID', element)
        obj.author = self._xpath2obj('author', element)
        obj.creation_time = self._xpath2obj(
            'creationTime', element, UTCDateTime)
        obj.version = self._xpath2obj('version', element)
        self._extra(element, obj)
        return obj

    def _origin_quality(self, parent):
        elements = self._xpath("quality", parent)
        if len(elements) > 1:
            raise NotImplementedError("Only one OriginQuality allowed.")
        # Do not add an element if it is not given in the XML file.
        elif len(elements) == 0:
            return None
        element = elements[0]
        obj = OriginQuality()
        obj.associated_phase_count = self._xpath2obj(
            'associatedPhaseCount', element, int)
        obj.used_phase_count = self._xpath2obj(
            'usedPhaseCount', element, int)
        obj.associated_station_count = self._xpath2obj(
            'associatedStationCount', element, int)
        obj.used_station_count = self._xpath2obj(
            'usedStationCount', element, int)
        obj.depth_phase_count = self._xpath2obj(
            'depthPhaseCount', element, int)
        obj.standard_error = self._xpath2obj(
            'standardError', element, float)
        obj.azimuthal_gap = self._xpath2obj(
            'azimuthalGap', element, float)
        obj.secondary_azimuthal_gap = self._xpath2obj(
            'secondaryAzimuthalGap', element, float)
        obj.ground_truth_level = self._xpath2obj(
            'groundTruthLevel', element)
        obj.minimum_distance = self._xpath2obj(
            'minimumDistance', element, float)
        obj.maximum_distance = self._xpath2obj(
            'maximumDistance', element, float)
        obj.median_distance = self._xpath2obj(
            'medianDistance', element, float)
        self._extra(element, obj)
        return obj

    def _event_description(self, parent):
        out = []
        for el in self._xpath('description', parent):
            desc = EventDescription()
            desc.text = self._xpath2obj('text', el)
            self._set_enum('type', el, desc, 'type')
            out.append(desc)
            self._extra(el, out[-1])
        return out

    def _value(self, parent, name, quantity_type=float):
        try:
            el = self._xpath(name, parent)[0]
        except IndexError:
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

    def _composite_times(self, parent):
        obj = []
        for el in self._xpath('compositeTime', parent):
            ct = CompositeTime()
            ct.year, ct.year_errors = self._int_value(el, 'year')
            ct.month, ct.month_errors = self._int_value(el, 'month')
            ct.day, ct.day_errors = self._int_value(el, 'day')
            ct.hour, ct.hour_errors = self._int_value(el, 'hour')
            ct.minute, ct.minute_errors = self._int_value(el, 'minute')
            ct.second, ct.second_errors = self._float_value(el, 'second')
            self._extra(el, ct)
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
        self._extra(element, obj)
        return obj

    def _origin_uncertainty(self, parent):
        elements = self._xpath("originUncertainty", parent)
        if len(elements) > 1:
            raise NotImplementedError("Only one OriginUncertainty allowed.")
        # Do not add an element if it is not given in the XML file.
        elif len(elements) == 0:
            return None
        element = elements[0]
        obj = OriginUncertainty()
        self._set_enum('preferredDescription', element,
                       obj, 'preferred_description')
        obj.horizontal_uncertainty = self._xpath2obj(
            'horizontalUncertainty', element, float)
        obj.min_horizontal_uncertainty = self._xpath2obj(
            'minHorizontalUncertainty', element, float)
        obj.max_horizontal_uncertainty = self._xpath2obj(
            'maxHorizontalUncertainty', element, float)
        obj.azimuth_max_horizontal_uncertainty = self._xpath2obj(
            'azimuthMaxHorizontalUncertainty', element, float)
        obj.confidence_level = self._xpath2obj(
            'confidenceLevel', element, float)
        ce_el = self._xpath('confidenceEllipsoid', element)
        try:
            ce_el = ce_el[0]
        except IndexError:
            obj.confidence_ellipsoid = ConfidenceEllipsoid()
        else:
            obj.confidence_ellipsoid = self._confidence_ellipsoid(ce_el)
        self._extra(element, obj)
        return obj

    def _waveform_ids(self, parent):
        objs = []
        for wid_el in self._xpath('waveformID', parent):
            obj = WaveformStreamID()
            obj.network_code = wid_el.get('networkCode') or ''
            obj.station_code = wid_el.get('stationCode') or ''
            obj.location_code = wid_el.get('locationCode')
            obj.channel_code = wid_el.get('channelCode')
            obj.resource_uri = wid_el.text
            objs.append(obj)
        return objs

    def _waveform_id(self, parent):
        try:
            return self._waveform_ids(parent)[0]
        except IndexError:
            return None

    def _arrival(self, element):
        """
        Converts an etree.Element into an Arrival object.

        :type element: etree.Element
        :rtype: :class:`~obspy.core.event.Arrival`
        """
        obj = Arrival(force_resource_id=False)
        # required parameter
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
        obj.resource_id = element.get('publicID')
        self._extra(element, obj)
        return obj

    def _pick(self, element):
        """
        Converts an etree.Element into a Pick object.

        :type element: etree.Element
        :rtype: :class:`~obspy.core.event.Pick`
        """
        obj = Pick(force_resource_id=False)
        # required parameter
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
        self._set_enum('onset', element, obj, 'onset')
        obj.phase_hint = self._xpath2obj('phaseHint', element)
        self._set_enum('polarity', element, obj, 'polarity')
        self._set_enum('evaluationMode', element, obj, 'evaluation_mode')
        self._set_enum('evaluationStatus', element, obj, 'evaluation_status')
        obj.comments = self._comments(element)
        obj.creation_info = self._creation_info(element)
        obj.resource_id = element.get('publicID')
        self._extra(element, obj)
        return obj

    def _time_window(self, element):
        """
        Converts an etree.Element into a TimeWindow object.

        :type element: etree.Element
        :rtype: :class:`~obspy.core.event.TimeWindow`
        """
        obj = TimeWindow(force_resource_id=False)
        # required parameter
        obj.begin = self._xpath2obj('begin', element, convert_to=float)
        obj.end = self._xpath2obj('end', element, convert_to=float)
        obj.reference = self._xpath2obj('reference', element,
                                        convert_to=UTCDateTime)
        self._extra(element, obj)
        return obj

    def _amplitude(self, element):
        """
        Converts an etree.Element into a Amplitude object.

        :type element: etree.Element
        :rtype: :class:`~obspy.core.event.Amplitude`
        """
        obj = Amplitude(force_resource_id=False)
        # required parameter
        obj.generic_amplitude, obj.generic_amplitude_errors = \
            self._float_value(element, 'genericAmplitude')
        # optional parameter
        obj.type = self._xpath2obj('type', element)
        self._set_enum('category', element, obj, 'category')
        self._set_enum('unit', element, obj, 'unit')
        obj.method_id = self._xpath2obj('methodID', element)
        obj.period, obj.period_errors = self._float_value(element, 'period')
        obj.snr = self._xpath2obj('snr', element)
        time_window_el = self._xpath('timeWindow', element) or None
        if time_window_el is not None:
            obj.time_window = self._time_window(time_window_el[0])
        obj.pick_id = self._xpath2obj('pickID', element)
        obj.waveform_id = self._waveform_id(element)
        obj.filter_id = self._xpath2obj('filterID', element)
        obj.scaling_time, obj.scaling_time_errors = \
            self._time_value(element, 'scalingTime')
        obj.magnitude_hint = self._xpath2obj('magnitudeHint', element)
        self._set_enum('evaluationMode', element, obj, 'evaluation_mode')
        self._set_enum('evaluationStatus', element, obj, 'evaluation_status')
        obj.comments = self._comments(element)
        obj.creation_info = self._creation_info(element)
        obj.resource_id = element.get('publicID')
        self._extra(element, obj)
        return obj

    def _origin(self, element, arrivals):
        """
        Converts an etree.Element into an Origin object.

        :type element: etree.Element
        :rtype: :class:`~obspy.core.event.Origin`

        .. rubric:: Example

        >>> from lxml import etree
        >>> XML = b'''<?xml version="1.0" encoding="UTF-8"?>
        ... <origin>
        ...   <latitude><value>34.23</value></latitude>
        ... </origin>'''
        >>> xml_doc = etree.fromstring(XML)
        >>> unpickler = Unpickler(xml_doc)
        >>> origin = unpickler._origin(xml_doc, arrivals=[])
        >>> print(origin.latitude)
        34.23
        """
        obj = Origin(force_resource_id=False)
        # required parameter
        obj.time, obj.time_errors = self._time_value(element, 'time')
        obj.latitude, obj.latitude_errors = \
            self._float_value(element, 'latitude')
        obj.longitude, obj.longitude_errors = \
            self._float_value(element, 'longitude')
        # optional parameter
        obj.depth, obj.depth_errors = self._float_value(element, 'depth')
        self._set_enum('depthType', element, obj, 'depth_type')
        obj.time_fixed = self._xpath2obj('timeFixed', element, bool)
        obj.epicenter_fixed = self._xpath2obj('epicenterFixed', element, bool)
        obj.reference_system_id = self._xpath2obj('referenceSystemID', element)
        obj.method_id = self._xpath2obj('methodID', element)
        obj.earth_model_id = self._xpath2obj('earthModelID', element)
        obj.composite_times = self._composite_times(element)
        obj.quality = self._origin_quality(element)
        self._set_enum('type', element, obj, 'origin_type')
        obj.region = self._xpath2obj('region', element)
        self._set_enum('evaluationMode', element, obj, 'evaluation_mode')
        self._set_enum('evaluationStatus', element, obj, 'evaluation_status')
        obj.creation_info = self._creation_info(element)
        obj.comments = self._comments(element)
        obj.origin_uncertainty = self._origin_uncertainty(element)
        obj.arrivals = arrivals
        obj.resource_id = element.get('publicID')
        self._extra(element, obj)
        return obj

    def _magnitude(self, element):
        """
        Converts an etree.Element into a Magnitude object.

        :type element: etree.Element
        :rtype: :class:`~obspy.core.event.Magnitude`

        .. rubric:: Example

        >>> from lxml import etree
        >>> XML = b'''<?xml version="1.0" encoding="UTF-8"?>
        ... <magnitude>
        ...   <mag><value>3.2</value></mag>
        ... </magnitude>'''
        >>> xml_doc = etree.fromstring(XML)
        >>> unpickler = Unpickler(xml_doc)
        >>> magnitude = unpickler._magnitude(xml_doc)
        >>> print(magnitude.mag)
        3.2
        """
        obj = Magnitude(force_resource_id=False)
        # required parameter
        obj.mag, obj.mag_errors = self._float_value(element, 'mag')
        # optional parameter
        obj.magnitude_type = self._xpath2obj('type', element)
        obj.origin_id = self._xpath2obj('originID', element)
        obj.method_id = self._xpath2obj('methodID', element)
        obj.station_count = self._xpath2obj('stationCount', element, int)
        obj.azimuthal_gap = self._xpath2obj('azimuthalGap', element, float)
        self._set_enum('evaluationMode', element, obj, 'evaluation_mode')
        self._set_enum('evaluationStatus', element, obj, 'evaluation_status')
        obj.creation_info = self._creation_info(element)
        obj.station_magnitude_contributions = \
            self._station_magnitude_contributions(element)
        obj.comments = self._comments(element)
        obj.resource_id = element.get('publicID')
        self._extra(element, obj)
        return obj

    def _station_magnitude(self, element):
        """
        Converts an etree.Element into a StationMagnitude object.

        :type element: etree.Element
        :rtype: :class:`~obspy.core.event.StationMagnitude`

        .. rubric:: Example

        >>> from lxml import etree
        >>> XML = b'''<?xml version="1.0" encoding="UTF-8"?>
        ... <stationMagnitude>
        ...   <mag><value>3.2</value></mag>
        ... </stationMagnitude>'''
        >>> xml_doc = etree.fromstring(XML)
        >>> unpickler = Unpickler(xml_doc)
        >>> station_mag = unpickler._station_magnitude(xml_doc)
        >>> print(station_mag.mag)
        3.2
        """
        obj = StationMagnitude(force_resource_id=False)
        # required parameter
        obj.origin_id = self._xpath2obj('originID', element) or ''
        obj.mag, obj.mag_errors = self._float_value(element, 'mag')
        # optional parameter
        obj.station_magnitude_type = self._xpath2obj('type', element)
        obj.amplitude_id = self._xpath2obj('amplitudeID', element)
        obj.method_id = self._xpath2obj('methodID', element)
        obj.waveform_id = self._waveform_id(element)
        obj.creation_info = self._creation_info(element)
        obj.comments = self._comments(element)
        obj.resource_id = element.get('publicID')
        self._extra(element, obj)
        return obj

    def _axis(self, parent, name):
        """
        Converts an etree.Element into an Axis object.

        :type parent: etree.Element
        :type name: str
        :param name: tag name of axis
        :rtype: :class:`~obspy.core.event.Axis`
        """
        obj = Axis()
        try:
            sub_el = self._xpath(name, parent)[0]
        except IndexError:
            return obj
        # required parameter
        obj.azimuth, obj.azimuth_errors = self._float_value(sub_el, 'azimuth')
        obj.plunge, obj.plunge_errors = self._float_value(sub_el, 'plunge')
        obj.length, obj.length_errors = self._float_value(sub_el, 'length')
        self._extra(sub_el, obj)
        return obj

    def _principal_axes(self, parent):
        """
        Converts an etree.Element into an PrincipalAxes object.

        :type parent: etree.Element
        :rtype: :class:`~obspy.core.event.PrincipalAxes`
        """
        try:
            sub_el = self._xpath('principalAxes', parent)[0]
        except IndexError:
            return None
        obj = PrincipalAxes()
        # required parameter
        obj.t_axis = self._axis(sub_el, 'tAxis')
        obj.p_axis = self._axis(sub_el, 'pAxis')
        # optional parameter
        obj.n_axis = self._axis(sub_el, 'nAxis')
        self._extra(sub_el, obj)
        return obj

    def _nodal_plane(self, parent, name):
        """
        Converts an etree.Element into an NodalPlane object.

        :type parent: etree.Element
        :type name: str
        :param name: tag name of sub nodal plane
        :rtype: :class:`~obspy.core.event.NodalPlane`
        """
        try:
            sub_el = self._xpath(name, parent)[0]
        except IndexError:
            return None
        obj = NodalPlane()
        # required parameter
        obj.strike, obj.strike_errors = self._float_value(sub_el, 'strike')
        obj.dip, obj.dip_errors = self._float_value(sub_el, 'dip')
        obj.rake, obj.rake_errors = self._float_value(sub_el, 'rake')
        self._extra(sub_el, obj)
        return obj

    def _nodal_planes(self, parent):
        """
        Converts an etree.Element into an NodalPlanes object.

        :type parent: etree.Element
        :rtype: :class:`~obspy.core.event.NodalPlanes`
        """
        try:
            sub_el = self._xpath('nodalPlanes', parent)[0]
        except IndexError:
            return None
        obj = NodalPlanes()
        # optional parameter
        obj.nodal_plane_1 = self._nodal_plane(sub_el, 'nodalPlane1')
        obj.nodal_plane_2 = self._nodal_plane(sub_el, 'nodalPlane2')
        # optional attribute
        try:
            obj.preferred_plane = int(sub_el.get('preferredPlane'))
        except Exception:
            obj.preferred_plane = None
        self._extra(sub_el, obj)
        return obj

    def _source_time_function(self, parent):
        """
        Converts an etree.Element into an SourceTimeFunction object.

        :type parent: etree.Element
        :rtype: :class:`~obspy.core.event.SourceTimeFunction`
        """
        try:
            sub_el = self._xpath('sourceTimeFunction', parent)[0]
        except IndexError:
            return None
        obj = SourceTimeFunction()
        # required parameters
        self._set_enum('type', sub_el, obj, 'type')
        obj.duration = self._xpath2obj('duration', sub_el, float)
        # optional parameter
        obj.rise_time = self._xpath2obj('riseTime', sub_el, float)
        obj.decay_time = self._xpath2obj('decayTime', sub_el, float)
        self._extra(sub_el, obj)
        return obj

    def _tensor(self, parent):
        """
        Converts an etree.Element into an Tensor object.

        :type parent: etree.Element
        :rtype: :class:`~obspy.core.event.Tensor`
        """
        try:
            sub_el = self._xpath('tensor', parent)[0]
        except IndexError:
            return None
        obj = Tensor()
        # required parameters
        obj.m_rr, obj.m_rr_errors = self._float_value(sub_el, 'Mrr')
        obj.m_tt, obj.m_tt_errors = self._float_value(sub_el, 'Mtt')
        obj.m_pp, obj.m_pp_errors = self._float_value(sub_el, 'Mpp')
        obj.m_rt, obj.m_rt_errors = self._float_value(sub_el, 'Mrt')
        obj.m_rp, obj.m_rp_errors = self._float_value(sub_el, 'Mrp')
        obj.m_tp, obj.m_tp_errors = self._float_value(sub_el, 'Mtp')
        self._extra(sub_el, obj)
        return obj

    def _data_used(self, parent):
        """
        Converts an etree.Element into a list of DataUsed objects.

        :type parent: etree.Element
        :rtype: list of :class:`~obspy.core.event.DataUsed`
        """
        obj = []
        for el in self._xpath('dataUsed', parent):
            data_used = DataUsed()
            # required parameters
            self._set_enum('waveType', el, data_used, 'wave_type')
            # optional parameter
            data_used.station_count = \
                self._xpath2obj('stationCount', el, int)
            data_used.component_count = \
                self._xpath2obj('componentCount', el, int)
            data_used.shortest_period = \
                self._xpath2obj('shortestPeriod', el, float)
            data_used.longest_period = \
                self._xpath2obj('longestPeriod', el, float)

            self._extra(el, data_used)
            obj.append(data_used)
        return obj

    def _moment_tensor(self, parent):
        """
        Converts an etree.Element into an MomentTensor object.

        :type parent: etree.Element
        :rtype: :class:`~obspy.core.event.MomentTensor`
        """
        try:
            mt_el = self._xpath('momentTensor', parent)[0]
        except IndexError:
            return None
        obj = MomentTensor(force_resource_id=False)
        # required parameters
        obj.derived_origin_id = self._xpath2obj('derivedOriginID', mt_el)
        # optional parameter
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
        obj.data_used = self._data_used(mt_el)
        obj.method_id = self._xpath2obj('methodID', mt_el)
        self._set_enum('category', mt_el, obj, 'category')
        self._set_enum('inversionType', mt_el, obj, 'inversion_type')
        obj.creation_info = self._creation_info(mt_el)
        obj.comments = self._comments(mt_el)
        obj.resource_id = mt_el.get('publicID')
        self._extra(mt_el, obj)
        return obj

    def _focal_mechanism(self, element):
        """
        Converts an etree.Element into a FocalMechanism object.

        :type element: etree.Element
        :rtype: :class:`~obspy.core.event.FocalMechanism`

        .. rubric:: Example

        >>> from lxml import etree
        >>> XML = b'''<?xml version="1.0" encoding="UTF-8"?>
        ... <focalMechanism>
        ...   <methodID>smi:ISC/methodID=Best_double_couple</methodID>
        ... </focalMechanism>'''
        >>> xml_doc = etree.fromstring(XML)
        >>> unpickler = Unpickler(xml_doc)
        >>> fm = unpickler._focal_mechanism(xml_doc)
        >>> print(fm.method_id)
        smi:ISC/methodID=Best_double_couple
        """
        obj = FocalMechanism(force_resource_id=False)
        # required parameter
        # optional parameter
        obj.waveform_id = self._waveform_ids(element)
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
        self._set_enum('evaluationMode', element, obj, 'evaluation_mode')
        self._set_enum('evaluationStatus', element, obj, 'evaluation_status')
        obj.creation_info = self._creation_info(element)
        obj.comments = self._comments(element)
        obj.resource_id = element.get('publicID')
        self._extra(element, obj)
        return obj

    def _deserialize(self):
        # check node "quakeml/eventParameters" for global namespace
        try:
            namespace = _get_first_child_namespace(self.xml_root)
            catalog_el = self._xpath('eventParameters', namespace=namespace)[0]
        except IndexError:
            raise Exception("Not a QuakeML compatible file or string")
        root_namespace, quakeml_version = re.match(
            QUAKEML_ROOTTAG_REGEX, self.xml_root.tag).groups()
        self._quakeml_namespaces = [
            root_namespace,
            NS_QUAKEML_BED_PATTERN.format(version=quakeml_version)]
        # create catalog
        catalog = Catalog(force_resource_id=False)
        # add any custom namespace abbreviations of root element to Catalog
        catalog.nsmap = self.xml_root.nsmap.copy()
        # optional catalog attributes
        catalog.description = self._xpath2obj('description', catalog_el)
        catalog.comments = self._comments(catalog_el)
        catalog.creation_info = self._creation_info(catalog_el)
        # loop over all events
        for event_el in self._xpath('event', catalog_el):
            # create new Event object
            event = Event(force_resource_id=False)
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
            # USGS event types contain '_' which is not compliant with
            # the QuakeML standard
            if isinstance(event_type, str):
                event_type = event_type.replace("_", " ")
            try:
                event.event_type = event_type
            except ValueError:
                msg = "Event type '%s' does not comply " % event_type
                msg += "with QuakeML standard -- event will be ignored."
                warnings.warn(msg, UserWarning)
                continue
            self._set_enum('typeCertainty', event_el,
                           event, 'event_type_certainty')
            event.creation_info = self._creation_info(event_el)
            event.event_descriptions = self._event_description(event_el)
            event.comments = self._comments(event_el)
            # origins
            event.origins = []
            for origin_el in self._xpath('origin', event_el):
                # Have to be created before the origin is created to avoid a
                # rare issue where a warning is read when the same event is
                # read twice - the warnings does not occur if two referred
                # to objects compare equal - for this the arrivals have to
                # be bound to the event before the resource id is assigned.
                arrivals = []
                for arrival_el in self._xpath('arrival', origin_el):
                    arrival = self._arrival(arrival_el)
                    arrivals.append(arrival)

                origin = self._origin(origin_el, arrivals=arrivals)

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
            # amplitudes
            event.amplitudes = []
            for el in self._xpath('amplitude', event_el):
                amp = self._amplitude(el)
                event.amplitudes.append(amp)
            # focal mechanisms
            event.focal_mechanisms = []
            for fm_el in self._xpath('focalMechanism', event_el):
                fm = self._focal_mechanism(fm_el)
                event.focal_mechanisms.append(fm)
            # finally append newly created event to catalog
            event.resource_id = event_el.get('publicID')
            self._extra(event_el, event)
            # bind event scoped resource IDs to this event
            event.scope_resource_ids()
            catalog.append(event)

        catalog.resource_id = catalog_el.get('publicID')
        self._extra(catalog_el, catalog)
        return catalog

    def _extra(self, element, obj):
        """
        Add information stored in custom tags/attributes in obj.extra.
        """
        # search all namespaces in current scope
        for ns in element.nsmap.values():
            # skip the two top-level quakeml namespaces,
            # we're not interested in quakeml defined tags here
            if ns in self._quakeml_namespaces:
                continue
            # process all elements of this custom namespace, if any
            for el in element.iterfind("{%s}*" % ns):
                # remove namespace from tag name
                _, name = el.tag.split("}")
                # check if element has children (nested tags)
                if len(el):
                    sub_obj = AttribDict()
                    self._extra(el, sub_obj)
                    value = sub_obj.extra
                else:
                    value = el.text
                try:
                    extra = obj.setdefault("extra", AttribDict())
                # Catalog object is not based on AttribDict..
                except AttributeError:
                    if not isinstance(obj, Catalog):
                        raise
                    if hasattr(obj, "extra"):
                        extra = obj.extra
                    else:
                        extra = AttribDict()
                        obj.extra = extra
                extra[name] = {'value': value,
                               'namespace': '%s' % ns}
                if el.attrib:
                    extra[name]['attrib'] = el.attrib
        # process all attributes of custom namespaces, if any
        for key, value in element.attrib.items():
            # no custom namespace
            if "}" not in key:
                continue
            # separate namespace from tag name
            ns, name = key.lstrip("{").split("}")
            try:
                extra = obj.setdefault("extra", AttribDict())
            # Catalog object is not based on AttribDict..
            except AttributeError:
                if not isinstance(obj, Catalog):
                    raise
                if hasattr(obj, "extra"):
                    extra = obj.extra
                else:
                    extra = AttribDict()
                    obj.extra = extra
            extra[name] = {'value': str(value),
                           'namespace': '%s' % ns,
                           'type': 'attribute'}


class Pickler(object):
    """
    Serializes an ObsPy Catalog object into QuakeML format.
    """
    def __init__(self, nsmap=None):
        # set of namespace urls without given abbreviation
        self.ns_set = set()
        # dictionary of namespace/namespace urls
        self.ns_dict = nsmap
        if self.ns_dict is None:
            self.ns_dict = {}
        self.ns_dict.update(NSMAP_QUAKEML.copy())

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
        :returns: QuakeML formatted string.
        """
        return self._serialize(catalog)

    def _id(self, obj):
        try:
            return obj.get_quakeml_uri_str()
        except Exception:
            msg = ("'%s' is not a valid QuakeML URI. It will be in the final "
                   "file but note that the file will not be a valid QuakeML "
                   "file.")
            warnings.warn(msg % obj.id)
            return obj.id

    def _str(self, value, root, tag, always_create=False, attrib=None):
        if isinstance(value, ResourceIdentifier):
            value = self._id(value)
        if always_create is False and value is None:
            return
        etree.SubElement(root, tag, attrib=attrib).text = "%s" % value

    def _bool(self, value, root, tag, always_create=False, attrib=None):
        if always_create is False and value is None:
            return
        etree.SubElement(root, tag, attrib=attrib).text = \
            str(bool(value)).lower()

    def _time(self, value, root, tag, always_create=False):
        if always_create is False and value is None:
            return
        etree.SubElement(root, tag).text = str(value)

    def _value(self, quantity, error, element, tag, always_create=False):
        if always_create is False and quantity is None:
            return
        subelement = etree.Element(tag)
        self._str(quantity, subelement, 'value')
        if error is not None:
            self._str(error.uncertainty, subelement, 'uncertainty')
            self._str(error.lower_uncertainty, subelement, 'lowerUncertainty')
            self._str(error.upper_uncertainty, subelement, 'upperUncertainty')
            self._str(error.confidence_level, subelement, 'confidenceLevel')
        element.append(subelement)

    def _waveform_id(self, obj, element, required=False):
        if obj is None:
            return
        attrib = {}
        if obj.network_code is not None:
            attrib['networkCode'] = obj.network_code
        if obj.station_code is not None:
            attrib['stationCode'] = obj.station_code
        if obj.location_code is not None:
            attrib['locationCode'] = obj.location_code
        if obj.channel_code is not None:
            attrib['channelCode'] = obj.channel_code
        subelement = etree.Element('waveformID', attrib=attrib)
        # WaveformStreamID has a non-mandatory resource_id
        if obj.resource_uri is None or obj.resource_uri == "":
            subelement.text = ""
        else:
            subelement.text = self._id(obj.resource_uri)

        if len(subelement.attrib) > 0 or required:
            element.append(subelement)

    def _waveform_ids(self, objs, element, required=False):
        for obj in objs:
            self._waveform_id(obj, element, required=required)

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
        self._extra(creation_info, subelement)
        # append only if at least one sub-element is set
        if len(subelement) > 0:
            element.append(subelement)

    def _station_magnitude_contributions(self, stat_contrib, element):
        for contrib in stat_contrib:
            contrib_el = etree.Element('stationMagnitudeContribution')
            self._str(contrib.station_magnitude_id.id, contrib_el,
                      'stationMagnitudeID')
            self._str(contrib.weight, contrib_el, 'weight')
            self._str(contrib.residual, contrib_el, 'residual')
            self._extra(contrib, contrib_el)
            element.append(contrib_el)

    def _comments(self, comments, element):
        for comment in comments:
            attrib = {}
            if comment.resource_id:
                attrib['id'] = self._id(comment.resource_id)
            comment_el = etree.Element('comment', attrib=attrib)
            etree.SubElement(comment_el, 'text').text = comment.text
            self._creation_info(comment.creation_info, comment_el)
            self._extra(comment, comment_el)
            element.append(comment_el)

    def _extra(self, obj, element):
        """
        Add information stored in obj.extra as custom tags/attributes in
        non-quakeml namespace.
        """
        if not hasattr(obj, "extra"):
            return
        self._custom(obj.extra, element)

    def _custom(self, obj, element):
        for key, item in obj.items():
            value = item["value"]
            ns = item["namespace"]
            attrib = item.get("attrib", {})
            type_ = item.get("type", "element")
            self._add_namespace(ns)
            tag = "{%s}%s" % (ns, key)
            # add either as subelement or attribute
            if type_.lower() in ("attribute", "attrib"):
                element.attrib[tag] = str(value)
            elif type_.lower() == "element":
                # check if value is dictionary-like
                if isinstance(value, compatibility.collections_abc.Mapping):
                    subelement = etree.SubElement(element, tag, attrib=attrib)
                    self._custom(value, subelement)
                elif isinstance(value, bool):
                    self._bool(value, element, tag, attrib=attrib)
                else:
                    self._str(value, element, tag, attrib=attrib)
            else:
                msg = ("Invalid 'type' for custom QuakeML item: '%s'. "
                       "Should be either 'element' or 'attribute' or "
                       "left empty.") % type_
                raise ValueError(msg)

    def _get_namespace_map(self):
        nsmap = self.ns_dict.copy()
        _i = 0
        for ns in self.ns_set:
            if ns in nsmap.values():
                continue
            ns_abbrev = "ns%d" % _i
            while ns_abbrev in nsmap:
                _i += 1
                ns_abbrev = "ns%d" % _i
            nsmap[ns_abbrev] = ns
        return nsmap

    def _add_namespace(self, ns):
        self.ns_set.add(ns)

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
        self._extra(arrival, element)
        return element

    def _magnitude(self, magnitude):
        """
        Converts an Magnitude into etree.Element object.

        :type magnitude: :class:`~obspy.core.event.Magnitude`
        :rtype: etree.Element

        .. rubric:: Example

        >>> from obspy.io.quakeml.core import Pickler
        >>> from obspy.core.event import Magnitude
        >>> from lxml.etree import tostring
        >>> magnitude = Magnitude()
        >>> magnitude.mag = 3.2
        >>> el = Pickler()._magnitude(magnitude)
        >>> print(tostring(el, encoding="utf-8",
        ...                xml_declaration=True).decode()) \
                # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
        <?xml version='1.0' encoding='utf-8'?>
        <magnitude ...<mag><value>3.2</value></mag>...</magnitude>
        """
        element = etree.Element(
            'magnitude', attrib={'publicID': self._id(magnitude.resource_id)})
        self._value(magnitude.mag, magnitude.mag_errors, element, 'mag', True)
        # optional parameter
        self._str(magnitude.magnitude_type, element, 'type')
        self._str(magnitude.origin_id, element, 'originID')
        self._str(magnitude.method_id, element, 'methodID')
        self._str(magnitude.station_count, element, 'stationCount')
        self._str(magnitude.azimuthal_gap, element, 'azimuthalGap')
        self._str(magnitude.evaluation_mode, element, 'evaluationMode')
        self._str(magnitude.evaluation_status, element, 'evaluationStatus')
        self._station_magnitude_contributions(
            magnitude.station_magnitude_contributions, element)
        self._comments(magnitude.comments, element)
        self._creation_info(magnitude.creation_info, element)
        self._extra(magnitude, element)
        return element

    def _station_magnitude(self, magnitude):
        """
        Converts an StationMagnitude into etree.Element object.

        :type magnitude: :class:`~obspy.core.event.StationMagnitude`
        :rtype: etree.Element

        .. rubric:: Example

        >>> from obspy.io.quakeml.core import Pickler
        >>> from obspy.core.event import StationMagnitude
        >>> from lxml.etree import tostring
        >>> station_mag = StationMagnitude()
        >>> station_mag.mag = 3.2
        >>> el = Pickler()._station_magnitude(station_mag)
        >>> print(tostring(el, encoding="utf-8",
        ...       xml_declaration=True).decode()) \
                # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
        <?xml version='1.0' encoding='utf-8'?>
        <stationMagnitude ...<value>3.2</value>...</stationMagnitude>
        """
        element = etree.Element(
            'stationMagnitude',
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
        self._extra(magnitude, element)
        return element

    def _origin(self, origin):
        """
        Converts an Origin into etree.Element object.

        :type origin: :class:`~obspy.core.event.Origin`
        :rtype: etree.Element

        .. rubric:: Example

        >>> from obspy.io.quakeml.core import Pickler
        >>> from obspy.core.event import Origin
        >>> from lxml.etree import tostring
        >>> origin = Origin()
        >>> origin.latitude = 34.23
        >>> el = Pickler()._origin(origin)
        >>> print(tostring(el, encoding="utf-8",
        ...                xml_declaration=True).decode()) \
                # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
        <?xml version='1.0' encoding='utf-8'?>
        <origin ...<latitude><value>34.23</value></latitude>...</origin>
        """
        element = etree.Element(
            'origin', attrib={'publicID': self._id(origin.resource_id)})
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
            self._extra(ctime, ct_el)
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
            self._extra(qu, qu_el)
            if len(qu_el) > 0:
                element.append(qu_el)
        self._str(origin.origin_type, element, 'type')
        self._str(origin.region, element, 'region')
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
            self._str(ou.confidence_level, ou_el,
                      'confidenceLevel')
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
                self._extra(ce, ce_el)
                # add confidence ellipsoid to origin uncertainty only if set
                if len(ce_el) > 0:
                    ou_el.append(ce_el)
            self._extra(ou, ou_el)
            # add origin uncertainty to origin only if anything is set
            if len(ou_el) > 0:
                element.append(ou_el)
        # arrivals
        for ar in origin.arrivals:
            element.append(self._arrival(ar))
        self._extra(origin, element)
        return element

    def _time_window(self, time_window, element):
        el = etree.Element('timeWindow')
        self._time(time_window.reference, el, 'reference')
        self._str(time_window.begin, el, 'begin')
        self._str(time_window.end, el, 'end')
        self._extra(time_window, element)
        element.append(el)

    def _amplitude(self, amp):
        """
        Converts an Amplitude into etree.Element object.

        :type amp: :class:`~obspy.core.event.Amplitude`
        :rtype: etree.Element
        """
        element = etree.Element(
            'amplitude', attrib={'publicID': self._id(amp.resource_id)})
        # required parameter
        self._value(amp.generic_amplitude, amp.generic_amplitude_errors,
                    element, 'genericAmplitude', True)
        # optional parameter
        self._str(amp.type, element, 'type')
        self._str(amp.category, element, 'category')
        self._str(amp.unit, element, 'unit')
        self._str(amp.method_id, element, 'methodID')
        self._value(amp.period, amp.period_errors, element, 'period')
        self._str(amp.snr, element, 'snr')
        if amp.time_window is not None:
            self._time_window(amp.time_window, element)
        self._str(amp.pick_id, element, 'pickID')
        self._waveform_id(amp.waveform_id, element, required=False)
        self._str(amp.filter_id, element, 'filterID')
        self._value(amp.scaling_time, amp.scaling_time_errors, element,
                    'scalingTime')
        self._str(amp.magnitude_hint, element, 'magnitudeHint')
        self._str(amp.evaluation_mode, element, 'evaluationMode')
        self._str(amp.evaluation_status, element, 'evaluationStatus')
        self._comments(amp.comments, element)
        self._creation_info(amp.creation_info, element)
        self._extra(amp, element)
        return element

    def _pick(self, pick):
        """
        Converts a Pick into etree.Element object.

        :type pick: :class:`~obspy.core.event.Pick`
        :rtype: etree.Element
        """
        element = etree.Element(
            'pick', attrib={'publicID': self._id(pick.resource_id)})
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
        self._extra(pick, element)
        return element

    def _nodal_planes(self, obj, element):
        """
        Converts a NodalPlanes into etree.Element object.

        :type obj: :class:`~obspy.core.event.NodalPlanes`
        :rtype: etree.Element
        """
        if obj is None:
            return
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
            self._extra(obj.nodal_plane_1, el)
            subelement.append(el)
        if obj.nodal_plane_2:
            el = etree.Element('nodalPlane2')
            self._value(obj.nodal_plane_2.strike,
                        obj.nodal_plane_2.strike_errors, el, 'strike')
            self._value(obj.nodal_plane_2.dip,
                        obj.nodal_plane_2.dip_errors, el, 'dip')
            self._value(obj.nodal_plane_2.rake,
                        obj.nodal_plane_2.rake_errors, el, 'rake')
            self._extra(obj.nodal_plane_2, el)
            subelement.append(el)
        if obj.preferred_plane:
            subelement.attrib['preferredPlane'] = str(obj.preferred_plane)
        # append only if at least one sub-element is set
        self._extra(obj, subelement)
        if len(subelement) > 0:
            element.append(subelement)

    def _principal_axes(self, obj, element):
        """
        Converts a PrincipalAxes into etree.Element object.

        :type obj: :class:`~obspy.core.event.PrincipalAxes`
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
        self._extra(obj.t_axis, el)
        subelement.append(el)
        # pAxis
        el = etree.Element('pAxis')
        self._value(obj.p_axis.azimuth,
                    obj.p_axis.azimuth_errors, el, 'azimuth')
        self._value(obj.p_axis.plunge,
                    obj.p_axis.plunge_errors, el, 'plunge')
        self._value(obj.p_axis.length,
                    obj.p_axis.length_errors, el, 'length')
        self._extra(obj.p_axis, el)
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
            self._extra(obj.n_axis, el)
            subelement.append(el)
        self._extra(obj, subelement)
        element.append(subelement)

    def _moment_tensor(self, moment_tensor, element):
        """
        Converts a MomentTensor into etree.Element object.

        :type moment_tensor: :class:`~obspy.core.event.MomentTensor`
        :rtype: etree.Element
        """
        if moment_tensor is None:
            return
        mt_el = etree.Element(
            'momentTensor',
            attrib={'publicID': self._id(moment_tensor.resource_id)})
        # required parameters
        self._str(moment_tensor.derived_origin_id, mt_el, 'derivedOriginID')
        # optional parameter
        # Data Used
        for sub in moment_tensor.data_used:
            sub_el = etree.Element('dataUsed')
            self._str(sub.wave_type, sub_el, 'waveType')
            self._str(sub.station_count, sub_el, 'stationCount')
            self._str(sub.component_count, sub_el, 'componentCount')
            self._str(sub.shortest_period, sub_el, 'shortestPeriod')
            self._str(sub.longest_period, sub_el, 'longestPeriod')
            self._extra(sub, sub_el)
            mt_el.append(sub_el)
        self._str(moment_tensor.moment_magnitude_id,
                  mt_el, 'momentMagnitudeID')
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
            self._extra(sub, sub_el)
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
            self._extra(sub, sub_el)
            mt_el.append(sub_el)
        self._str(moment_tensor.method_id, mt_el, 'methodID')
        self._str(moment_tensor.category, mt_el, 'category')
        self._str(moment_tensor.inversion_type, mt_el, 'inversionType')
        self._comments(moment_tensor.comments, mt_el)
        self._creation_info(moment_tensor.creation_info, mt_el)
        self._extra(moment_tensor, mt_el)
        element.append(mt_el)

    def _focal_mechanism(self, focal_mechanism):
        """
        Converts a FocalMechanism into etree.Element object.

        :type focal_mechanism: :class:`~obspy.core.event.FocalMechanism`
        :rtype: etree.Element
        """
        if focal_mechanism is None:
            return
        element = etree.Element(
            'focalMechanism',
            attrib={'publicID': self._id(focal_mechanism.resource_id)})
        # optional parameter
        self._waveform_ids(focal_mechanism.waveform_id, element)
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
        self._str(focal_mechanism.evaluation_mode, element, 'evaluationMode')
        self._str(focal_mechanism.evaluation_status, element,
                  'evaluationStatus')
        self._comments(focal_mechanism.comments, element)
        self._creation_info(focal_mechanism.creation_info, element)
        self._extra(focal_mechanism, element)
        return element

    def _serialize(self, catalog, pretty_print=True):
        """
        Converts a Catalog object into XML string.
        """
        catalog_el = etree.Element('eventParameters', attrib={'publicID':
                                   self._id(catalog.resource_id)})
        # optional catalog parameters
        if catalog.description:
            self._str(catalog.description, catalog_el, 'description')
        self._comments(catalog.comments, catalog_el)
        self._creation_info(catalog.creation_info, catalog_el)
        for event in catalog:
            # create event node
            event_el = etree.Element(
                'event', attrib={'publicID': self._id(event.resource_id)})
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
                self._str(description.text, el, 'text')
                self._str(description.type, el, 'type')
                self._extra(description, el)
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
            # amplitudes
            for amp in event.amplitudes:
                event_el.append(self._amplitude(amp))
            # focal mechanisms
            for focal_mechanism in event.focal_mechanisms:
                event_el.append(self._focal_mechanism(focal_mechanism))
            self._extra(event, event_el)
            # add event node to catalog
            catalog_el.append(event_el)
        self._extra(catalog, catalog_el)
        nsmap = self._get_namespace_map()
        root_el = etree.Element('{%s}quakeml' % NSMAP_QUAKEML['q'],
                                nsmap=nsmap)
        root_el.append(catalog_el)
        return etree.tostring(root_el, pretty_print=pretty_print,
                              encoding="utf-8", xml_declaration=True)


def _read_quakeml(filename):
    """
    Reads a QuakeML file and returns an ObsPy Catalog object.

    .. warning::
        This function should NOT be called directly, it registers via the
        ObsPy :func:`~obspy.core.event.read_events` function, call this
        instead.

    :type filename: str
    :param filename: QuakeML file to be read.
    :rtype: :class:`~obspy.core.event.Catalog`
    :return: An ObsPy Catalog object.

    .. rubric:: Example

    >>> from obspy.core.event import read_events
    >>> cat = read_events('/path/to/iris_events.xml')
    >>> print(cat)
    2 Event(s) in Catalog:
    2011-03-11T05:46:24.120000Z | +38.297, +142.373 | 9.1  MW
    2006-09-10T04:26:33.610000Z |  +9.614, +121.961 | 9.8  MS
    """
    return Unpickler().load(filename)


def _write_quakeml(catalog, filename, validate=False, nsmap=None,
                   **kwargs):  # @UnusedVariable
    """
    Writes a QuakeML file.

    .. warning::
        This function should NOT be called directly, it registers via the
        the :meth:`~obspy.core.event.Catalog.write` method of an
        ObsPy :class:`~obspy.core.event.Catalog` object, call this instead.

    :type catalog: :class:`~obspy.core.event.catalog.Catalog`
    :param catalog: The ObsPy Catalog object to write.
    :type filename: str or file
    :param filename: Filename to write or open file-like object.
    :type validate: bool, optional
    :param validate: If True, the final QuakeML file will be validated against
        the QuakeML schema file. Raises an AssertionError if the validation
        fails.
    :type nsmap: dict, optional
    :param nsmap: Additional custom namespace abbreviation mappings
        (e.g. `{"edb": "http://erdbeben-in-bayern.de/xmlns/0.1"}`).
    """
    nsmap_ = getattr(catalog, "nsmap", {})
    if nsmap:
        nsmap_.update(nsmap)
    xml_doc = Pickler(nsmap=nsmap_).dumps(catalog)

    if validate is True and not _validate(io.BytesIO(xml_doc)):
        raise AssertionError(
            "The final QuakeML file did not pass validation.")

    # Open filehandler or use an existing file like object
    try:
        with open(filename, 'wb') as fh:
            fh.write(xml_doc)
    except TypeError:
        filename.write(xml_doc)


def _read_seishub_event_xml(filename):
    """
    Reads a single SeisHub event XML file and returns an ObsPy Catalog object.
    """
    # XXX: very ugly way to add new root tags without parsing
    lines = open(filename, 'rb').readlines()
    lines.insert(2,
                 b'<quakeml xmlns="http://quakeml.org/xmlns/quakeml/1.0">\n')
    lines.insert(3, b'  <eventParameters>')
    lines.append(b'  </eventParameters>\n')
    lines.append(b'</quakeml>\n')
    temp = io.BytesIO(b''.join(lines))
    return _read_quakeml(temp)


def _validate(xml_file, verbose=False):
    """
    Validates a QuakeML file against the QuakeML 1.2 RelaxNG Schema. Returns
    either True or False.
    """
    try:
        from lxml.etree import RelaxNG
    except ImportError:
        msg = "Could not validate QuakeML - try using a newer lxml version"
        warnings.warn(msg, UserWarning)
        return True
    # Get the schema location.
    schema_location = os.path.dirname(inspect.getfile(inspect.currentframe()))
    schema_location = os.path.join(schema_location, "data", "QuakeML-1.2.rng")

    try:
        relaxng = RelaxNG(etree.parse(schema_location))
    except TypeError:
        msg = "Could not validate QuakeML - try using a newer lxml version"
        warnings.warn(msg, UserWarning)
        return True
    xmldoc = etree.parse(xml_file)

    valid = relaxng.validate(xmldoc)

    # Pretty error printing if the validation fails.
    if verbose and valid is not True:
        print("Error validating QuakeML file:")
        for entry in relaxng.error_log:
            print("\t%s" % entry)
    return valid


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
