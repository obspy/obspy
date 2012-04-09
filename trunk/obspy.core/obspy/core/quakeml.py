# -*- coding: utf-8 -*-
"""
Module for handling ObsPy Catalog and Event objects.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""

from obspy.core.event import Catalog, Event, Origin, CreationInfo, Magnitude, \
    EventDescription, OriginUncertainty, OriginQuality, CompositeTime, \
    IntegerQuantity, FloatQuantity, TimeQuantity, ConfidenceEllipsoid, \
    StationMagnitude, Comment, WaveformStreamID
from obspy.core.utcdatetime import UTCDateTime
from obspy.core.util.xmlwrapper import XMLParser, tostring, etree, \
    register_namespace
import StringIO


# global QuakeML namespace
register_namespace('q', 'http://quakeml.org/xmlns/quakeml/1.2')


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
        comment.id = parser.xpath2obj('id', el) or None
        comment.creation_info = __toCreationInfo(parser, el)
        obj.append(comment)
    return obj


def __xmlComments(comments, element):
    for comment in comments:
        comment_el = etree.Element('comment')
        etree.SubElement(comment_el, 'text').text = comment.text
        if comment.id:
            etree.SubElement(comment_el, 'id').text = comment.id
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
    obj.network = parser.xpath2obj('waveformID/networkCode', element) or ''
    obj.station = parser.xpath2obj('waveformID/stationCode', element) or ''
    obj.location = parser.xpath2obj('waveformID/locationCode', element)
    obj.channel = parser.xpath2obj('waveformID/channelCode', element)
    obj.resource_uri = parser.xpath2obj('waveformID/resourceURI', element)
    return obj


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
    origin = Origin()
    # required parameter
    origin.public_id = element.get('publicID')
    origin.time = __toTimeQuantity(parser, element, 'time')
    origin.latitude = __toFloatQuantity(parser, element, 'latitude')
    origin.longitude = __toFloatQuantity(parser, element, 'longitude')
    # optional parameter
    origin.depth = __toFloatQuantity(parser, element, 'depth')
    origin.depth_type = parser.xpath2obj('depthType', element)
    origin.time_fixed = parser.xpath2obj('timeFixed', element, bool)
    origin.epicenter_fixed = parser.xpath2obj('epicenterFixed', element, bool)
    origin.reference_system_id = parser.xpath2obj('referenceSystemID', element)
    origin.method_id = parser.xpath2obj('methodID', element)
    origin.earth_model_id = parser.xpath2obj('earthModelID', element)
    origin.composite_times = __toCompositeTimes(parser, element)
    origin.quality = __toOriginQuality(parser, element)
    origin.type = parser.xpath2obj('type', element)
    origin.evaluation_mode = parser.xpath2obj('evaluationMode', element)
    origin.evaluation_status = parser.xpath2obj('evaluationStatus', element)
    origin.creation_info = __toCreationInfo(parser, element)
    origin.comments = __toComments(parser, element)
    origin.origin_uncertainty = __toOriginUncertainty(parser, element)
    return origin


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
    return element


def _toMagnitude(parser, element):
    """
    Converts an etree.Element into an Magnitude object.

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
    mag = Magnitude()
    # required parameter
    mag.public_id = element.get('publicID')
    mag.mag = __toFloatQuantity(parser, element, 'mag')
    # optional parameter
    mag.type = parser.xpath2obj('type', element)
    mag.origin_id = parser.xpath2obj('originID', element)
    mag.method_id = parser.xpath2obj('methodID', element)
    mag.station_count = parser.xpath2obj('stationCount', element, int)
    mag.azimuthal_gap = parser.xpath2obj('azimuthalGap', element, float)
    mag.evaluation_status = parser.xpath2obj('evaluationStatus', element)
    mag.creation_info = __toCreationInfo(parser, element)
    mag.comments = __toComments(parser, element)
    return mag


def _xmlMagnitude(magnitude):
    """
    Converts an Magnitude into etree.Element object.

    :type origin: :class:`~obspy.core.event.Magnitude`
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
    Converts an etree.Element into an StationMagnitude object.

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
    mag = StationMagnitude()
    # required parameter
    mag.public_id = element.get('publicID')
    mag.origin_id = parser.xpath2obj('originID', element) or ''
    mag.mag = __toFloatQuantity(parser, element, 'mag')
    # optional parameter
    mag.type = parser.xpath2obj('type', element)
    mag.amplitude_id = parser.xpath2obj('amplitudeID', element)
    mag.method_id = parser.xpath2obj('methodID', element)
    mag.waveform_id = __toWaveformStreamID(parser, element)
    mag.creation_info = __toCreationInfo(parser, element)
    mag.comments = __toComments(parser, element)
    return mag


def _xmlStationMagnitude(magnitude):
    """
    Converts an StationMagnitude into etree.Element object.

    :type origin: :class:`~obspy.core.event.StationMagnitude`
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
    # waveform_id
    wid = magnitude.waveform_id
    wid_el = etree.Element('waveformID')
    __xmlStr(wid.network, wid_el, 'networkCode')
    __xmlStr(wid.station, wid_el, 'stationCode')
    __xmlStr(wid.location, wid_el, 'locationCode')
    __xmlStr(wid.channel, wid_el, 'channelCode')
    __xmlStr(wid.resource_uri, wid_el, 'resourceURI')
    if len(wid_el) > 0:
        element.append(wid_el)
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
            # add preferred origin to front
            if origin.public_id == event.preferred_origin_id:
                event.origins.insert(0, origin)
            else:
                event.origins.append(origin)
        # magnitudes
        event.magnitudes = []
        for magnitude_el in p.xpath('magnitude', event_el):
            magnitude = _toMagnitude(p, magnitude_el)
            # add preferred magnitude to front
            if magnitude.public_id == event.preferred_magnitude_id:
                event.magnitudes.insert(0, magnitude)
            else:
                event.magnitudes.append(magnitude)
        # station magnitudes
        event.station_magnitudes = []
        for magnitude_el in p.xpath('stationMagnitude', event_el):
            magnitude = _toStationMagnitude(p, magnitude_el)
            event.station_magnitudes.append(magnitude)
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
        # origins
        for magnitude in event.magnitudes:
            event_el.append(_xmlMagnitude(magnitude))
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
