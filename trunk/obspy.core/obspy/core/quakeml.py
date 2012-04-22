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
    IntegerQuantity, FloatQuantity, TimeQuantity, ConfidenceEllipsoid, \
    StationMagnitude, Comment, WaveformStreamID, Arrival, Pick
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

    >>> isSLIST('/path/to/quakeml.xml')  # doctest: +SKIP
    True
    """
    try:
        p = XMLParser(filename)
    except:
        False
    # check node "*/eventParameters/event" for the global namespace exists
    try:
        namespace = p._getFirstChildNamespace()
        p.xpath('eventParameters/event', namespace=namespace)[0]
    except:
        return False
    return True


def __xmlStr(value, root, tag, always_create=False):
    if always_create is False and value is None:
        return
    etree.SubElement(root, tag).text = "%s" % (value)


def __xmlBool(value, root, tag, always_create=False):
    if always_create is False and value is None:
        return
    etree.SubElement(root, tag).text = str(bool(value)).lower()


def __xmlTime(value, root, tag, always_create=False):
    if always_create is False and value is None:
        return
    dt = value.strftime("%Y-%m-%dT%H:%M:%S+00:00")
    etree.SubElement(root, tag).text = dt


def __toCreationInfo(parser, element):
    obj = CreationInfo()
    obj.agency_uri = parser.xpath2obj('creationInfo/agencyURI', element)
    obj.author_uri = parser.xpath2obj('creationInfo/authorURI', element)
    obj.agency_id = parser.xpath2obj('creationInfo/agencyID', element)
    obj.author = parser.xpath2obj('creationInfo/author', element)
    obj.creation_time = parser.xpath2obj('creationInfo/creationTime',
        element, UTCDateTime)
    obj.version = parser.xpath2obj('creationInfo/version', element)
    return obj


def __xmlCreationInfo(creation_info, element):
    subelement = etree.Element('creationInfo')
    __xmlStr(creation_info.agency_id, subelement, 'agencyID')
    __xmlStr(creation_info.agency_uri, subelement, 'agencyURI')
    __xmlStr(creation_info.author, subelement, 'author')
    __xmlStr(creation_info.author_uri, subelement, 'authorURI')
    __xmlTime(creation_info.creation_time, subelement, 'creationTime')
    __xmlStr(creation_info.version, subelement, 'version')
    # append only if any information is set
    if len(subelement) > 0:
        element.append(subelement)


def __toOriginQuality(parser, element):
    obj = OriginQuality()
    obj.associated_phase_count = parser.xpath2obj(
        'quality/associatedPhaseCount', element, int)
    obj.used_phase_count = parser.xpath2obj(
        'quality/usedPhaseCount', element, int)
    obj.associated_station_count = parser.xpath2obj(
        'quality/associatedStationCount', element, int)
    obj.used_station_count = parser.xpath2obj(
        'quality/usedStationCount', element, int)
    obj.depth_phase_count = parser.xpath2obj(
        'quality/depthPhaseCount', element, int)
    obj.standard_error = parser.xpath2obj(
        'quality/standardError', element, float)
    obj.azimuthal_gap = parser.xpath2obj(
        'quality/azimuthalGap', element, float)
    obj.secondary_azimuthal_gap = parser.xpath2obj(
        'quality/secondaryAzimuthalGap', element, float)
    obj.ground_truth_level = parser.xpath2obj(
        'quality/groundTruthLevel', element)
    obj.minimum_distance = parser.xpath2obj(
        'quality/minimumDistance', element, float)
    obj.maximum_distance = parser.xpath2obj(
        'quality/maximumDistance', element, float)
    obj.median_distance = parser.xpath2obj(
        'quality/medianDistance', element, float)
    return obj


def __toEventDescription(parser, element):
    out = []
    for el in parser.xpath('description', element):
        text = parser.xpath2obj('text', el)
        type = parser.xpath2obj('type', el)
        out.append(EventDescription({'text': text, 'type': type}))
    return out


def __toComments(parser, element):
    obj = []
    for el in parser.xpath('comment', element):
        comment = Comment()
        comment.text = parser.xpath2obj('text', el)
        comment.id = el.get('id')
        comment.creation_info = __toCreationInfo(parser, el)
        obj.append(comment)
    return obj


def __xmlComments(comments, element):
    for comment in comments:
        attrib = {}
        if comment.id:
            attrib['id'] = comment.id
        comment_el = etree.Element('comment', attrib=attrib)
        etree.SubElement(comment_el, 'text').text = comment.text
        __xmlCreationInfo(comment.creation_info, comment_el)
        element.append(comment_el)


def __toValueQuantity(parser, element, name, quantity_type=FloatQuantity):
    obj = quantity_type()
    try:
        el = parser.xpath(name, element)[0]
    except:
        return obj
    obj.value = parser.xpath2obj('value', el, quantity_type._value_type)
    obj.uncertainty = parser.xpath2obj('uncertainty', el, float)
    obj.lower_uncertainty = parser.xpath2obj('lowerUncertainty', el, float)
    obj.upper_uncertainty = parser.xpath2obj('upperUncertainty', el, float)
    obj.confidence_level = parser.xpath2obj('confidenceLevel', el, float)
    return obj


def __xmlValueQuantity(quantity, element, tag, always_create=False):
    if always_create is False and quantity.value is None:
        return
    subelement = etree.Element(tag)
    __xmlStr(quantity.value, subelement, 'value')
    __xmlStr(quantity.uncertainty, subelement, 'uncertainty')
    __xmlStr(quantity.lower_uncertainty, subelement, 'lowerUncertainty')
    __xmlStr(quantity.upper_uncertainty, subelement, 'upperUncertainty')
    __xmlStr(quantity.confidence_level, subelement, 'confidenceLevel')
    element.append(subelement)


def __toFloatQuantity(parser, element, name):
    return __toValueQuantity(parser, element, name, FloatQuantity)


def __toIntegerQuantity(parser, element, name):
    return __toValueQuantity(parser, element, name, IntegerQuantity)


def __toTimeQuantity(parser, element, name):
    return __toValueQuantity(parser, element, name, TimeQuantity)


def __toCompositeTimes(parser, element):
    obj = []
    for el in parser.xpath('compositeTime', element):
        ct = CompositeTime()
        ct.year = __toIntegerQuantity(parser, el, 'year')
        ct.month = __toIntegerQuantity(parser, el, 'month')
        ct.day = __toIntegerQuantity(parser, el, 'day')
        ct.hour = __toIntegerQuantity(parser, el, 'hour')
        ct.minute = __toIntegerQuantity(parser, el, 'minute')
        ct.second = __toFloatQuantity(parser, el, 'second')
        obj.append(ct)
    return obj


def __toConfidenceEllipsoid(parser, element):
    obj = ConfidenceEllipsoid()
    obj.semi_major_axis_length = parser.xpath2obj(
        'semiMajorAxisLength', element, float)
    obj.semi_minor_axis_length = parser.xpath2obj(
        'semiMinorAxisLength', element, float)
    obj.semi_intermediate_axis_length = parser.xpath2obj(
        'semiIntermediateAxisLength', element, float)
    obj.major_axis_plunge = parser.xpath2obj(
        'majorAxisPlunge', element, float)
    obj.major_axis_azimuth = parser.xpath2obj(
        'majorAxisAzimuth', element, float)
    obj.major_axis_rotation = parser.xpath2obj(
        'majorAxisRotation', element, float)
    return obj


def __toOriginUncertainty(parser, element):
    obj = OriginUncertainty()
    obj.preferred_description = parser.xpath2obj(
        'originUncertainty/preferredDescription', element)
    obj.horizontal_uncertainty = parser.xpath2obj(
        'originUncertainty/horizontalUncertainty', element, float)
    obj.min_horizontal_uncertainty = parser.xpath2obj(
        'originUncertainty/minHorizontalUncertainty', element, float)
    obj.max_horizontal_uncertainty = parser.xpath2obj(
        'originUncertainty/maxHorizontalUncertainty', element, float)
    obj.azimuth_max_horizontal_uncertainty = parser.xpath2obj(
        'originUncertainty/azimuthMaxHorizontalUncertainty', element, float)
    try:
        ce_el = parser.xpath('originUncertainty/confidenceEllipsoid', element)
        obj.confidence_ellipsoid = __toConfidenceEllipsoid(parser, ce_el[0])
    except:
        obj.confidence_ellipsoid = ConfidenceEllipsoid()
    return obj


def __toWaveformStreamID(parser, element):
    obj = WaveformStreamID()
    try:
        wid_el = parser.xpath('waveformID', element)[0]
    except:
        return obj
    obj.network = wid_el.get('networkCode') or ''
    obj.station = wid_el.get('stationCode') or ''
    obj.location = wid_el.get('locationCode')
    obj.channel = wid_el.get('channelCode')
    obj.resource_uri = wid_el.text
    return obj


def __xmlWaveformStreamID(obj, element, required=True):  # @UnusedVariable
    attrib = {}
    if obj.network or obj.resource_uri is None:
        attrib['networkCode'] = obj.network or ''
    if obj.station or obj.resource_uri is None:
        attrib['stationCode'] = obj.station or ''
    if obj.location:
        attrib['locationCode'] = obj.location
    if obj.channel:
        attrib['channelCode'] = obj.channel
    subelement = etree.Element('waveformID', attrib=attrib)
    if obj.resource_uri:
        subelement.text = obj.resource_uri
    element.append(subelement)


def _toArrival(parser, element):
    """
    Converts an etree.Element into an Arrival object.

    :type parser: :class:`~obspy.core.util.xmlwrapper.XMLParser`
    :type element: etree.Element
    :rtype: :class:`~obspy.core.event.Arrival`
    """
    obj = Arrival()
    # required parameter
    obj.pick_id = parser.xpath2obj('pickID', element) or ''
    obj.phase = parser.xpath2obj('phase', element) or ''
    # optional parameter
    obj.time_correction = parser.xpath2obj('timeCorrection', element, float)
    obj.azimuth = parser.xpath2obj('azimuth', element, float)
    obj.distance = parser.xpath2obj('distance', element, float)
    obj.time_residual = parser.xpath2obj('timeResidual', element, float)
    obj.horizontal_slowness_residual = \
        parser.xpath2obj('horizontalSlownessResidual', element, float)
    obj.backazimuth_residual = \
        parser.xpath2obj('backazimuthResidual', element, float)
    obj.time_used = parser.xpath2obj('timeUsed', element, bool)
    obj.horizontal_slowness_used = \
        parser.xpath2obj('horizontalSlownessUsed', element, bool)
    obj.backazimuth_used = parser.xpath2obj('backazimuthUsed', element, bool)
    obj.time_weight = parser.xpath2obj('timeWeight', element, float)
    obj.earth_model_id = parser.xpath2obj('earthModelID', element)
    obj.preliminary = element.get('preliminary')
    obj.comments = __toComments(parser, element)
    obj.creation_info = __toCreationInfo(parser, element)
    return obj


def _xmlArrival(arrival):
    """
    Converts an Arrival into etree.Element object.

    :type arrival: :class:`~obspy.core.event.Arrival`
    :rtype: etree.Element
    """
    attrib = {}
    if arrival.preliminary:
        attrib['preliminary'] = arrival.preliminary
    element = etree.Element('arrival', attrib=attrib)
    # required parameter
    __xmlStr(arrival.pick_id, element, 'pickID', True)
    __xmlStr(arrival.phase, element, 'phase', True)
    # optional parameter
    __xmlStr(arrival.time_correction, element, 'timeCorrection')
    __xmlStr(arrival.azimuth, element, 'azimuth')
    __xmlStr(arrival.distance, element, 'distance')
    __xmlStr(arrival.time_residual, element, 'timeResidual')
    __xmlStr(arrival.horizontal_slowness_residual, element,
             'horizontalSlownessResidual')
    __xmlStr(arrival.backazimuth_residual, element, 'backazimuthResidual')
    __xmlBool(arrival.time_used, element, 'timeUsed')
    __xmlBool(arrival.horizontal_slowness_used, element,
              'horizontalSlownessUsed')
    __xmlBool(arrival.backazimuth_used, element, 'backazimuthUsed')
    __xmlStr(arrival.time_weight, element, 'timeWeight')
    __xmlStr(arrival.earth_model_id, element, 'earthModelID')
    __xmlComments(arrival.comments, element)
    __xmlCreationInfo(arrival.creation_info, element)
    return element


def _toPick(parser, element):
    """
    Converts an etree.Element into a Pick object.

    :type parser: :class:`~obspy.core.util.xmlwrapper.XMLParser`
    :type element: etree.Element
    :rtype: :class:`~obspy.core.event.Pick`
    """
    obj = Pick()
    # required parameter
    obj.public_id = element.get('publicID')
    obj.time = __toTimeQuantity(parser, element, 'time')
    obj.waveform_id = __toWaveformStreamID(parser, element)
    # optional parameter
    obj.filter_id = parser.xpath2obj('filterID', element)
    obj.method_id = parser.xpath2obj('methodID', element)
    obj.horizontal_slowness = \
        __toFloatQuantity(parser, element, 'horizontalSlowness')
    obj.backazimuth = __toFloatQuantity(parser, element, 'backazimuth')
    obj.slowness_method_id = parser.xpath2obj('slownessMethodID', element)
    obj.onset = parser.xpath2obj('onset', element)
    obj.phase_hint = parser.xpath2obj('phaseHint', element)
    obj.polarity = parser.xpath2obj('polarity', element)
    obj.evaluation_mode = parser.xpath2obj('evaluationMode', element)
    obj.evaluation_status = parser.xpath2obj('evaluationStatus', element)
    obj.comments = __toComments(parser, element)
    obj.creation_info = __toCreationInfo(parser, element)
    return obj


def _xmlPick(pick):
    """
    Converts a Pick into etree.Element object.

    :type pick: :class:`~obspy.core.event.Pick`
    :rtype: etree.Element
    """
    element = etree.Element('pick', attrib={'publicID': pick.public_id or ''})
    # required parameter
    __xmlValueQuantity(pick.time, element, 'time', True)
    __xmlWaveformStreamID(pick.waveform_id, element, True)
    # optional parameter
    __xmlStr(pick.filter_id, element, 'filterID')
    __xmlStr(pick.method_id, element, 'methodID')
    __xmlValueQuantity(pick.horizontal_slowness, element, 'horizontalSlowness')
    __xmlValueQuantity(pick.backazimuth, element, 'backazimuth')
    __xmlStr(pick.slowness_method_id, element, 'slownessMethodID')
    __xmlStr(pick.onset, element, 'onset')
    __xmlStr(pick.phase_hint, element, 'phaseHint')
    __xmlStr(pick.polarity, element, 'polarity')
    __xmlStr(pick.evaluation_mode, element, 'evaluationMode')
    __xmlStr(pick.evaluation_status, element, 'evaluationStatus')
    __xmlComments(pick.comments, element)
    __xmlCreationInfo(pick.creation_info, element)
    return element


def _toOrigin(parser, element):
    """
    Converts an etree.Element into an Origin object.

    :type parser: :class:`~obspy.core.util.xmlwrapper.XMLParser`
    :type element: etree.Element
    :rtype: :class:`~obspy.core.event.Origin`

    .. rubric:: Example

    >>> from obspy.core.util import XMLParser
    >>> XML = '<?xml version="1.0" encoding="UTF-8"?>'
    >>> XML += '<origin><latitude><value>34.23</value></latitude></origin>'
    >>> parser = XMLParser(XML)
    >>> origin = _toOrigin(parser, parser.xml_root)
    >>> print(origin.latitude.value)
    34.23
    """
    obj = Origin()
    # required parameter
    obj.public_id = element.get('publicID')
    obj.time = __toTimeQuantity(parser, element, 'time')
    obj.latitude = __toFloatQuantity(parser, element, 'latitude')
    obj.longitude = __toFloatQuantity(parser, element, 'longitude')
    # optional parameter
    obj.depth = __toFloatQuantity(parser, element, 'depth')
    obj.depth_type = parser.xpath2obj('depthType', element)
    obj.time_fixed = parser.xpath2obj('timeFixed', element, bool)
    obj.epicenter_fixed = parser.xpath2obj('epicenterFixed', element, bool)
    obj.reference_system_id = parser.xpath2obj('referenceSystemID', element)
    obj.method_id = parser.xpath2obj('methodID', element)
    obj.earth_model_id = parser.xpath2obj('earthModelID', element)
    obj.composite_times = __toCompositeTimes(parser, element)
    obj.quality = __toOriginQuality(parser, element)
    obj.type = parser.xpath2obj('type', element)
    obj.evaluation_mode = parser.xpath2obj('evaluationMode', element)
    obj.evaluation_status = parser.xpath2obj('evaluationStatus', element)
    obj.creation_info = __toCreationInfo(parser, element)
    obj.comments = __toComments(parser, element)
    obj.origin_uncertainty = __toOriginUncertainty(parser, element)
    return obj


def _xmlOrigin(origin):
    """
    Converts an Origin into etree.Element object.

    :type origin: :class:`~obspy.core.event.Origin`
    :rtype: etree.Element

    .. rubric:: Example

    >>> from obspy.core.event import Origin
    >>> from obspy.core.util import tostring
    >>> origin = Origin()
    >>> origin.latitude.value = 34.23
    >>> el = _xmlOrigin(origin)
    >>> print(tostring(el))  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    <?xml version='1.0' encoding='utf-8'?>
    <origin ...<latitude><value>34.23</value></latitude>...</origin>
    """
    element = etree.Element('origin',
                            attrib={'publicID': origin.public_id or ''})
    __xmlValueQuantity(origin.time, element, 'time', True)
    __xmlValueQuantity(origin.latitude, element, 'latitude', True)
    __xmlValueQuantity(origin.longitude, element, 'longitude', True)
    # optional parameter
    __xmlValueQuantity(origin.depth, element, 'depth')
    __xmlStr(origin.depth_type, element, 'depthType')
    __xmlBool(origin.time_fixed, element, 'timeFixed')
    __xmlBool(origin.epicenter_fixed, element, 'epicenterFixed')
    __xmlStr(origin.reference_system_id, element, 'referenceSystemID')
    __xmlStr(origin.method_id, element, 'methodID')
    __xmlStr(origin.earth_model_id, element, 'earthModelID')
    # compositeTime
    for ctime in origin.composite_times:
        ct_el = etree.Element('compositeTime')
        __xmlValueQuantity(ctime.year, ct_el, 'year')
        __xmlValueQuantity(ctime.month, ct_el, 'month')
        __xmlValueQuantity(ctime.day, ct_el, 'day')
        __xmlValueQuantity(ctime.hour, ct_el, 'hour')
        __xmlValueQuantity(ctime.minute, ct_el, 'minute')
        __xmlValueQuantity(ctime.second, ct_el, 'second')
        if len(ct_el) > 0:
            element.append(ct_el)
    # quality
    quality = origin.quality
    qu_el = etree.Element('quality')
    __xmlStr(quality.associated_phase_count, qu_el, 'associatedPhaseCount')
    __xmlStr(quality.used_phase_count, qu_el, 'usedPhaseCount')
    __xmlStr(quality.associated_station_count, qu_el, 'associatedStationCount')
    __xmlStr(quality.used_station_count, qu_el, 'usedStationCount')
    __xmlStr(quality.depth_phase_count, qu_el, 'depthPhaseCount')
    __xmlStr(quality.standard_error, qu_el, 'standardError')
    __xmlStr(quality.azimuthal_gap, qu_el, 'azimuthalGap')
    __xmlStr(quality.secondary_azimuthal_gap, qu_el, 'secondaryAzimuthalGap')
    __xmlStr(quality.ground_truth_level, qu_el, 'groundTruthLevel')
    __xmlStr(quality.minimum_distance, qu_el, 'minimumDistance')
    __xmlStr(quality.maximum_distance, qu_el, 'maximumDistance')
    __xmlStr(quality.median_distance, qu_el, 'medianDistance')
    if len(qu_el) > 0:
        element.append(qu_el)
    __xmlStr(origin.type, element, 'type')
    __xmlStr(origin.evaluation_mode, element, 'evaluationMode')
    __xmlStr(origin.evaluation_status, element, 'evaluationStatus')
    __xmlComments(origin.comments, element)
    __xmlCreationInfo(origin.creation_info, element)
    # origin uncertainty
    ou = origin.origin_uncertainty
    ou_el = etree.Element('originUncertainty')
    __xmlStr(ou.preferred_description, ou_el, 'preferredDescription')
    __xmlStr(ou.horizontal_uncertainty, ou_el, 'horizontalUncertainty')
    __xmlStr(ou.min_horizontal_uncertainty, ou_el, 'minHorizontalUncertainty')
    __xmlStr(ou.max_horizontal_uncertainty, ou_el, 'maxHorizontalUncertainty')
    __xmlStr(ou.azimuth_max_horizontal_uncertainty, ou_el,
             'azimuthMaxHorizontalUncertainty')
    ce = ou.confidence_ellipsoid
    ce_el = etree.Element('confidenceEllipsoid')
    __xmlStr(ce.semi_major_axis_length, ce_el, 'semiMajorAxisLength')
    __xmlStr(ce.semi_minor_axis_length, ce_el, 'semiMinorAxisLength')
    __xmlStr(ce.semi_intermediate_axis_length, ce_el,
             'semiIntermediateAxisLength')
    __xmlStr(ce.major_axis_plunge, ce_el, 'majorAxisPlunge')
    __xmlStr(ce.major_axis_azimuth, ce_el, 'majorAxisAzimuth')
    __xmlStr(ce.major_axis_rotation, ce_el, 'majorAxisRotation')
    # add confidence ellipsoid to origin uncertainty only if anything is set
    if len(ce_el) > 0:
        ou_el.append(ce_el)
    # add origin uncertainty to origin only if anything is set
    if len(ou_el) > 0:
        element.append(ou_el)
    # arrivals
    for ar in origin.arrivals:
        element.append(_xmlArrival(ar))
    return element


def _toMagnitude(parser, element):
    """
    Converts an etree.Element into a Magnitude object.

    :type parser: :class:`~obspy.core.util.xmlwrapper.XMLParser`
    :type element: etree.Element
    :rtype: :class:`~obspy.core.event.Magnitude`

    .. rubric:: Example

    >>> from obspy.core.util import XMLParser
    >>> XML = '<?xml version="1.0" encoding="UTF-8"?>'
    >>> XML += '<magnitude><mag><value>3.2</value></mag></magnitude>'
    >>> parser = XMLParser(XML)
    >>> magnitude = _toMagnitude(parser, parser.xml_root)
    >>> print(magnitude.mag.value)
    3.2
    """
    obj = Magnitude()
    # required parameter
    obj.public_id = element.get('publicID')
    obj.mag = __toFloatQuantity(parser, element, 'mag')
    # optional parameter
    obj.type = parser.xpath2obj('type', element)
    obj.origin_id = parser.xpath2obj('originID', element)
    obj.method_id = parser.xpath2obj('methodID', element)
    obj.station_count = parser.xpath2obj('stationCount', element, int)
    obj.azimuthal_gap = parser.xpath2obj('azimuthalGap', element, float)
    obj.evaluation_status = parser.xpath2obj('evaluationStatus', element)
    obj.creation_info = __toCreationInfo(parser, element)
    obj.comments = __toComments(parser, element)
    return obj


def _xmlMagnitude(magnitude):
    """
    Converts an Magnitude into etree.Element object.

    :type magnitude: :class:`~obspy.core.event.Magnitude`
    :rtype: etree.Element

    .. rubric:: Example

    >>> from obspy.core.event import Magnitude
    >>> from obspy.core.util import tostring
    >>> magnitude = Magnitude()
    >>> magnitude.mag.value = 3.2
    >>> el = _xmlMagnitude(magnitude)
    >>> print(tostring(el))  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    <?xml version='1.0' encoding='utf-8'?>
    <magnitude ...<mag><value>3.2</value></mag>...</magnitude>
    """
    element = etree.Element('magnitude',
                            attrib={'publicID': magnitude.public_id or ''})
    __xmlValueQuantity(magnitude.mag, element, 'mag', True)
    # optional parameter
    __xmlStr(magnitude.type, element, 'type')
    __xmlStr(magnitude.origin_id, element, 'originID')
    __xmlStr(magnitude.method_id, element, 'methodID')
    __xmlStr(magnitude.station_count, element, 'stationCount')
    __xmlStr(magnitude.azimuthal_gap, element, 'azimuthalGap')
    __xmlStr(magnitude.evaluation_status, element, 'evaluationStatus')
    __xmlComments(magnitude.comments, element)
    __xmlCreationInfo(magnitude.creation_info, element)
    return element


def _toStationMagnitude(parser, element):
    """
    Converts an etree.Element into a StationMagnitude object.

    :type parser: :class:`~obspy.core.util.xmlwrapper.XMLParser`
    :type element: etree.Element
    :rtype: :class:`~obspy.core.event.StationMagnitude`

    .. rubric:: Example

    >>> from obspy.core.util import XMLParser
    >>> XML = '<?xml version="1.0" encoding="UTF-8"?><stationMagnitude>'
    >>> XML += '<mag><value>3.2</value></mag></stationMagnitude>'
    >>> parser = XMLParser(XML)
    >>> station_mag = _toStationMagnitude(parser, parser.xml_root)
    >>> print(station_mag.mag.value)
    3.2
    """
    obj = StationMagnitude()
    # required parameter
    obj.public_id = element.get('publicID')
    obj.origin_id = parser.xpath2obj('originID', element) or ''
    obj.mag = __toFloatQuantity(parser, element, 'mag')
    # optional parameter
    obj.type = parser.xpath2obj('type', element)
    obj.amplitude_id = parser.xpath2obj('amplitudeID', element)
    obj.method_id = parser.xpath2obj('methodID', element)
    obj.waveform_id = __toWaveformStreamID(parser, element)
    obj.creation_info = __toCreationInfo(parser, element)
    obj.comments = __toComments(parser, element)
    return obj


def _xmlStationMagnitude(magnitude):
    """
    Converts an StationMagnitude into etree.Element object.

    :type magnitude: :class:`~obspy.core.event.StationMagnitude`
    :rtype: etree.Element

    .. rubric:: Example

    >>> from obspy.core.event import StationMagnitude
    >>> from obspy.core.util import tostring
    >>> station_mag = StationMagnitude()
    >>> station_mag.mag.value = 3.2
    >>> el = _xmlStationMagnitude(station_mag)
    >>> print(tostring(el))  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    <?xml version='1.0' encoding='utf-8'?>
    <stationMagnitude ...<mag><value>3.2</value></mag>...</stationMagnitude>
    """
    element = etree.Element('stationMagnitude',
                            attrib={'publicID': magnitude.public_id or ''})
    __xmlStr(magnitude.origin_id, element, 'originID', True)
    __xmlValueQuantity(magnitude.mag, element, 'mag', True)
    # optional parameter
    __xmlStr(magnitude.type, element, 'type')
    __xmlStr(magnitude.amplitude_id, element, 'amplitudeID')
    __xmlStr(magnitude.method_id, element, 'methodID')
    __xmlWaveformStreamID(magnitude.waveform_id, element)
    __xmlComments(magnitude.comments, element)
    __xmlCreationInfo(magnitude.creation_info, element)
    return element


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
    p = XMLParser(filename)
    # check node "quakeml/eventParameters" for global namespace
    try:
        namespace = p._getFirstChildNamespace()
        catalog_el = p.xpath('eventParameters', namespace=namespace)[0]
    except:
        raise Exception("Not a QuakeML compatible file: %s." % filename)
    # set default namespace for parser
    p.namespace = p._getElementNamespace(catalog_el)
    # create catalog
    catalog = Catalog()
    # optional catalog attributes
    catalog.public_id = catalog_el.get('publicID')
    catalog.description = p.xpath2obj('description', catalog_el)
    catalog.comments = __toComments(p, catalog_el)
    catalog.creation_info = __toCreationInfo(p, catalog_el)
    # loop over all events
    for event_el in p.xpath('event', catalog_el):
        # create new Event object
        public_id = event_el.get('publicID')
        event = Event(public_id)
        # optional event attributes
        event.preferred_origin_id = p.xpath2obj('preferredOriginID', event_el)
        event.preferred_magnitude_id = \
            p.xpath2obj('preferredMagnitudeID', event_el)
        event.preferred_focal_mechanism_id = \
            p.xpath2obj('preferredFocalMechanismID', event_el)
        event.type = p.xpath2obj('type', event_el)
        event.type_certainty = p.xpath2obj('typeCertainty', event_el)
        event.creation_info = __toCreationInfo(p, event_el)
        event.descriptions = __toEventDescription(p, event_el)
        event.comments = __toComments(p, event_el)
        # origins
        event.origins = []
        for origin_el in p.xpath('origin', event_el):
            origin = _toOrigin(p, origin_el)
            # arrivals
            origin.arrivals = []
            for arrival_el in p.xpath('arrival', origin_el):
                arrival = _toArrival(p, arrival_el)
                origin.arrivals.append(arrival)
            # append origin with arrivals
            event.origins.append(origin)
        # magnitudes
        event.magnitudes = []
        for magnitude_el in p.xpath('magnitude', event_el):
            magnitude = _toMagnitude(p, magnitude_el)
            event.magnitudes.append(magnitude)
        # station magnitudes
        event.station_magnitudes = []
        for magnitude_el in p.xpath('stationMagnitude', event_el):
            magnitude = _toStationMagnitude(p, magnitude_el)
            event.station_magnitudes.append(magnitude)
        # picks
        event.picks = []
        for pick_el in p.xpath('pick', event_el):
            pick = _toPick(p, pick_el)
            event.picks.append(pick)
        # add current event to catalog
        catalog.append(event)
    return catalog


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
    raise NotImplementedError


def _xmlCatalog(catalog, pretty_print=True):
    """
    Converts a Catalog object into XML string.
    """
    root_el = etree.Element('{http://quakeml.org/xmlns/quakeml/1.2}quakeml',
        attrib={'xmlns': "http://quakeml.org/xmlns/bed/1.2"})
    catalog_el = etree.Element('eventParameters',
                              attrib={'publicID': catalog.public_id})
    # optional catalog parameters
    __xmlStr(catalog.description, catalog_el, 'description')
    __xmlComments(catalog.comments, catalog_el)
    __xmlCreationInfo(catalog.creation_info, catalog_el)
    root_el.append(catalog_el)
    for event in catalog:
        # create event node
        event_el = etree.Element('event', attrib={'publicID': event.public_id})
        # optional event attributes
        __xmlStr(event.preferred_origin_id, event_el, 'preferredOriginID')
        __xmlStr(event.preferred_magnitude_id, event_el,
                 'preferredMagnitudeID')
        __xmlStr(event.preferred_focal_mechanism_id, event_el,
                 'preferredFocalMechanismID')
        __xmlStr(event.type, event_el, 'type')
        __xmlStr(event.type_certainty, event_el, 'typeCertainty')
        # event descriptions
        for description in event.descriptions:
            el = etree.Element('description')
            __xmlStr(description.text, el, 'text', True)
            __xmlStr(description.type, el, 'type')
            event_el.append(el)
        __xmlComments(event.comments, event_el)
        __xmlCreationInfo(event.creation_info, event_el)
        # origins
        for origin in event.origins:
            event_el.append(_xmlOrigin(origin))
        # magnitudes
        for magnitude in event.magnitudes:
            event_el.append(_xmlMagnitude(magnitude))
        # station magnitudes
        for magnitude in event.station_magnitudes:
            event_el.append(_xmlStationMagnitude(magnitude))
        # picks
        for pick in event.picks:
            event_el.append(_xmlPick(pick))
        # add event node to catalog
        catalog_el.append(event_el)
    return tostring(root_el, pretty_print=pretty_print)


def readSeisHubEventXML(filename):
    """
    Reads a single SeisHub event XML file and returns a ObsPy Catalog object.
    """
    # XXX: very ugly way to add new root tags
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
