# -*- coding: utf-8 -*-
"""
Module for handling ObsPy Catalog and Event objects.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""

from obspy.core.event import Catalog, Event, Origin
from obspy.core.utcdatetime import UTCDateTime
from obspy.core.util.attribdict import AttribDict
from obspy.core.util.xmlwrapper import XMLParser
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
    p = XMLParser(filename)
    # check node "quakeml/eventParameters" for global namespace
    try:
        namespace = p._getFirstChildNamespace()
        p.xpath('eventParameters', namespace=namespace)[0]
    except:
        return False
    return True


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
    >>> st = readEvents('/path/to/iris_events.xml')
    >>> print st
    2 Event(s) in Catalog:
    2011-03-11T05:46:24.120000Z | +38.297, +142.373 | 9.1 MW
    2006-09-10T04:26:33.610000Z |  +9.614, +121.961 | 9.8 MS
    """
    p = XMLParser(filename)
    catalog = Catalog()
    # check node "quakeml/eventParameters" for global namespace
    try:
        namespace = p._getFirstChildNamespace()
        obj = p.xpath('eventParameters', namespace=namespace)[0]
    except:
        raise Exception("Not a QuakeML compatible file: %s." % filename)
    # set default namespace for parser
    p.namespace = p._getElementNamespace(obj)
    for event_xml in p.xpath('eventParameters/event'):
        # create new Event object
        event = Event()
        event.public_id = event_xml.get('publicID')
        # preferred items
        preferred_origin = p.xml2obj('preferredOriginID', event_xml)
        preferred_magnitude = p.xml2obj('preferredMagnitudeID', event_xml)
        preferred_focal_mechanism = \
            p.xml2obj('preferredFocalMechanismID', event_xml)
        # event type
        event.type = p.xml2obj('type', event_xml)
        # creation info
        event.creation_info = AttribDict()
        event.creation_info.agency_uri = \
            p.xml2obj('creationInfo/agencyURI', event_xml)
        event.creation_info.author_uri = \
            p.xml2obj('creationInfo/authorURI', event_xml)
        event.creation_info.creation_time = \
            p.xml2obj('creationInfo/creationTime', event_xml, UTCDateTime)
        event.creation_info.version = \
            p.xml2obj('creationInfo/version', event_xml)
        # description
        event.description = AttribDict()
        event.description.type = p.xml2obj('description/type', event_xml)
        event.description.text = p.xml2obj('description/text', event_xml)
        # origins
        event.origins = []
        for origin_xml in p.xpath('origin', event_xml):
            origin = Origin()
            origin.public_id = origin_xml.get('publicID')
            origin.time = p.xml2obj('time/value', origin_xml, UTCDateTime)
            origin.time_uncertainty = \
                p.xml2obj('time/uncertainty', origin_xml, float)
            origin.latitude = p.xml2obj('latitude/value', origin_xml, float)
            origin.latitude_uncertainty = \
                p.xml2obj('latitude/uncertainty', origin_xml, float)
            origin.longitude = p.xml2obj('longitude/value', origin_xml, float)
            origin.longitude_uncertainty = \
                p.xml2obj('longitude/uncertainty', origin_xml, float)
            origin.depth = p.xml2obj('depth/value', origin_xml, float)
            origin.depth_uncertainty = \
                p.xml2obj('depth/uncertainty', origin_xml, float)
            origin.depth_type = p.xml2obj('depthType', origin_xml)
            origin.method_id = p.xml2obj('depthType', origin_xml)
            # quality
            origin.quality.used_station_count = \
                p.xml2obj('quality/usedStationCount', origin_xml, int)
            origin.quality.standard_error = \
                p.xml2obj('quality/standardError', origin_xml, float)
            origin.quality.azimuthal_gap = \
                p.xml2obj('quality/azimuthalGap', origin_xml, float)
            origin.quality.maximum_distance = \
                p.xml2obj('quality/maximumDistance', origin_xml, float)
            origin.quality.minimum_distance = \
                p.xml2obj('quality/minimumDistance', origin_xml, float)
            origin.type = p.xml2obj('type', origin_xml)
            origin.evaluation_mode = p.xml2obj('evaluationMode', origin_xml)
            origin.evaluation_status = \
                p.xml2obj('evaluationStatus', origin_xml)
            origin.comment = p.xml2obj('comment', origin_xml)
            # creationInfo
            origin.creation_info.agency_uri = \
                p.xml2obj('creationInfo/agencyURI', origin_xml)
            origin.creation_info.author_uri = \
                p.xml2obj('creationInfo/authorURI', origin_xml)
            # originUncertainty
            origin.origin_uncertainty.min_horizontal_uncertainty = \
                p.xml2obj('originUncertainty/minHorizontalUncertainty',
                          origin_xml, float)
            origin.origin_uncertainty.max_horizontal_uncertainty = \
                p.xml2obj('originUncertainty/maxHorizontalUncertainty',
                          origin_xml, float)
            origin.origin_uncertainty.azimuth_max_horizontal_uncertainty = \
                p.xml2obj('originUncertainty/azimuthMaxHorizontalUncertainty',
                          origin_xml, float)
            # add preferred origin to front
            if origin.public_id == preferred_origin:
                event.origins.insert(0, origin)
            else:
                event.origins.append(origin)
        # magnitudes
        event.magnitudes = []
        for mag_xml in p.xpath('magnitude', event_xml):
            magnitude = Origin()
            magnitude.public_id = mag_xml.get('publicID')
            magnitude.magnitude = p.xml2obj('mag/value', mag_xml, float)
            magnitude.magnitude_uncertainty = \
                p.xml2obj('mag/uncertainty', mag_xml, float)
            magnitude.type = p.xml2obj('type', mag_xml)
            magnitude.origin_id = p.xml2obj('originID', mag_xml)
            magnitude.station_count = p.xml2obj('stationCount', mag_xml, int)
            # creationInfo
            magnitude.creation_info.agency_uri = \
                p.xml2obj('creationInfo/agencyURI', origin_xml)
            magnitude.creation_info.author_uri = \
                p.xml2obj('creationInfo/authorURI', origin_xml)
            # add preferred magnitude to front
            if magnitude.public_id == preferred_magnitude:
                event.magnitudes.insert(0, magnitude)
            else:
                event.magnitudes.append(magnitude)
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
