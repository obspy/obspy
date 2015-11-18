# -*- coding: utf-8 -*-
"""
Keyhole Markup Language (KML) output support in ObsPy

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

from math import log

from lxml.etree import Element, SubElement, tostring
from matplotlib.cm import get_cmap

from obspy import UTCDateTime
from obspy.core.event import Catalog
from obspy.core.inventory.inventory import Inventory


def inventory_to_kml_string(
        inventory,
        icon_url="https://maps.google.com/mapfiles/kml/shapes/triangle.png",
        icon_size=1.5, label_size=1.0, cmap="Paired", encoding="UTF-8",
        timespans=True, strip_far_future_end_times=True):
    """
    Convert an :class:`~obspy.core.inventory.inventory.Inventory` to a KML
    string representation.

    :type inventory: :class:`~obspy.core.inventory.inventory.Inventory`
    :param inventory: Input station metadata.
    :type icon_url: str
    :param icon_url: Internet URL of icon to use for station (e.g. PNG image).
    :type icon_size: float
    :param icon_size: Icon size.
    :type label_size: float
    :param label_size: Label size.
    :type encoding: str
    :param encoding: Encoding used for XML string.
    :type timespans: bool
    :param timespans: Whether to add timespan information to the single station
        elements in the KML or not. If timespans are used, the displayed
        information in e.g. Google Earth will represent a snapshot in time,
        such that using the time slider different states of the inventory in
        time can be visualized. If timespans are not used, any station active
        at any point in time is always shown.
    :type strip_far_future_end_times: bool
    :param strip_far_future_end_times: Leave out likely fictitious end times of
        stations (more than twenty years after current time). Far future end
        times may produce time sliders with bad overall time span in third
        party applications viewing the KML file.
    :rtype: byte string
    :return: Encoded byte string containing KML information of the station
        metadata.
    """
    twenty_years_from_now = UTCDateTime() + 3600 * 24 * 365 * 20
    # construct the KML file
    kml = Element("kml")
    kml.set("xmlns", "http://www.opengis.net/kml/2.2")

    document = SubElement(kml, "Document")
    SubElement(document, "name").text = "Inventory"

    # style definition
    cmap = get_cmap(name=cmap, lut=len(inventory.networks))
    for i in range(len(inventory.networks)):
        color = _rgba_tuple_to_kml_color_code(cmap(i))
        style = SubElement(document, "Style")
        style.set("id", "station_%i" % i)

        iconstyle = SubElement(style, "IconStyle")
        SubElement(iconstyle, "color").text = color
        SubElement(iconstyle, "scale").text = str(icon_size)
        icon = SubElement(iconstyle, "Icon")
        SubElement(icon, "href").text = icon_url
        hotspot = SubElement(iconstyle, "hotSpot")
        hotspot.set("x", "0.5")
        hotspot.set("y", "0.5")
        hotspot.set("xunits", "fraction")
        hotspot.set("yunits", "fraction")

        labelstyle = SubElement(style, "LabelStyle")
        SubElement(labelstyle, "color").text = color
        SubElement(labelstyle, "scale").text = str(label_size)

    for i, net in enumerate(inventory):
        folder = SubElement(document, "Folder")
        SubElement(folder, "name").text = str(net.code)
        SubElement(folder, "open").text = "1"

        SubElement(folder, "description").text = str(net)

        style = SubElement(folder, "Style")
        liststyle = SubElement(style, "ListStyle")
        SubElement(liststyle, "listItemType").text = "check"
        SubElement(liststyle, "bgColor").text = "00ffff"
        SubElement(liststyle, "maxSnippetLines").text = "5"

        # add one marker per station code
        for sta in net:
            placemark = SubElement(folder, "Placemark")
            SubElement(placemark, "name").text = ".".join((net.code, sta.code))
            SubElement(placemark, "styleUrl").text = "#station_%i" % i
            SubElement(placemark, "color").text = color
            if sta.longitude is not None and sta.latitude is not None:
                point = SubElement(placemark, "Point")
                SubElement(point, "coordinates").text = "%.6f,%.6f,0" % \
                    (sta.longitude, sta.latitude)

            SubElement(placemark, "description").text = str(sta)

            if timespans:
                start = sta.start_date
                end = sta.end_date
                if start is not None or end is not None:
                    timespan = SubElement(placemark, "TimeSpan")
                    if start is not None:
                        SubElement(timespan, "begin").text = str(start)
                    if end is not None:
                        if not strip_far_future_end_times or \
                                end < twenty_years_from_now:
                            SubElement(timespan, "end").text = str(end)
        if timespans:
            start = net.start_date
            end = net.end_date
            if start is not None or end is not None:
                timespan = SubElement(folder, "TimeSpan")
                if start is not None:
                    SubElement(timespan, "begin").text = str(start)
                if end is not None:
                    if not strip_far_future_end_times or \
                            end < twenty_years_from_now:
                        SubElement(timespan, "end").text = str(end)

    # generate and return KML string
    return tostring(kml, pretty_print=True, xml_declaration=True,
                    encoding=encoding)


def catalog_to_kml_string(
        catalog,
        icon_url="https://maps.google.com/mapfiles/kml/shapes/earthquake.png",
        label_func=None, icon_size_func=None, encoding="UTF-8",
        timestamps=True):
    """
    Convert a :class:`~obspy.core.event.Catalog` to a KML string
    representation.

    :type catalog: :class:`~obspy.core.event.Catalog`
    :param catalog: Input catalog data.
    :type icon_url: str
    :param icon_url: Internet URL of icon to use for events (e.g. PNG image).
    :type label_func: func
    :type label_func: Custom function to use for determining each event's
        label. User provided function is supposed to take an
        :class:`~obspy.core.event.Event` object as single argument, e.g. for
        empty labels use `label_func=lambda x: ""`.
    :type icon_size_func: func
    :type icon_size_func: Custom function to use for determining each
        event's icon size. User provided function should take an
        :class:`~obspy.core.event.Event` object as single argument and return a
        float.
    :type encoding: str
    :param encoding: Encoding used for XML string.
    :type timestamps: bool
    :param timestamps: Whether to add timestamp information to the event
        elements in the KML or not. If timestamps are used, the displayed
        information in e.g. Google Earth will represent a snapshot in time,
        such that using the time slider different states of the catalog in time
        can be visualized. If timespans are not used, any event happening at
        any point in time is always shown.
    :rtype: byte string
    :return: Encoded byte string containing KML information of the event
        metadata.
    """
    # default label and size functions
    if not label_func:
        def label_func(event):
            origin = (event.preferred_origin() or
                      event.origins and event.origins[0] or
                      None)
            mag = (event.preferred_magnitude() or
                   event.magnitudes and event.magnitudes[0] or
                   None)
            label = origin.time and str(origin.time.date) or ""
            if mag:
                label += " %.1f" % mag.mag
            return label
    if not icon_size_func:
        def icon_size_func(event):
            mag = (event.preferred_magnitude() or
                   event.magnitudes and event.magnitudes[0] or
                   None)
            if mag:
                try:
                    icon_size = 1.2 * log(1.5 + mag.mag)
                except ValueError:
                    icon_size = 0.1
            else:
                icon_size = 0.5
            return icon_size

    # construct the KML file
    kml = Element("kml")
    kml.set("xmlns", "http://www.opengis.net/kml/2.2")

    document = SubElement(kml, "Document")
    SubElement(document, "name").text = "Catalog"

    # style definitions for earthquakes
    style = SubElement(document, "Style")
    style.set("id", "earthquake")

    iconstyle = SubElement(style, "IconStyle")
    SubElement(iconstyle, "scale").text = "0.5"
    icon = SubElement(iconstyle, "Icon")
    SubElement(icon, "href").text = icon_url
    hotspot = SubElement(iconstyle, "hotSpot")
    hotspot.set("x", "0.5")
    hotspot.set("y", "0.5")
    hotspot.set("xunits", "fraction")
    hotspot.set("yunits", "fraction")

    labelstyle = SubElement(style, "LabelStyle")
    SubElement(labelstyle, "color").text = "ff0000ff"
    SubElement(labelstyle, "scale").text = "0.8"

    folder = SubElement(document, "Folder")
    SubElement(folder, "name").text = "Catalog"
    SubElement(folder, "open").text = "1"

    SubElement(folder, "description").text = str(catalog)

    style = SubElement(folder, "Style")
    liststyle = SubElement(style, "ListStyle")
    SubElement(liststyle, "listItemType").text = "check"
    SubElement(liststyle, "bgColor").text = "00ffffff"
    SubElement(liststyle, "maxSnippetLines").text = "5"

    # add one marker per event
    for event in catalog:
        origin = (event.preferred_origin() or
                  event.origins and event.origins[0] or
                  None)

        placemark = SubElement(folder, "Placemark")
        SubElement(placemark, "name").text = label_func(event)
        SubElement(placemark, "styleUrl").text = "#earthquake"
        style = SubElement(placemark, "Style")
        icon_style = SubElement(style, "IconStyle")
        liststyle = SubElement(style, "ListStyle")
        SubElement(liststyle, "maxSnippetLines").text = "5"
        SubElement(icon_style, "scale").text = "%.5f" % icon_size_func(event)
        if origin:
            if origin.longitude is not None and origin.latitude is not None:
                point = SubElement(placemark, "Point")
                SubElement(point, "coordinates").text = "%.6f,%.6f,0" % \
                    (origin.longitude, origin.latitude)

        SubElement(placemark, "description").text = str(event)

        if timestamps:
            time = _get_event_timestamp(event)
            if time is not None:
                SubElement(placemark, "TimeStamp").text = str(time)

    # generate and return KML string
    return tostring(kml, pretty_print=True, xml_declaration=True,
                    encoding=encoding)


def _write_kml(obj, filename, **kwargs):
    """
    Write :class:`~obspy.core.inventory.inventory.Inventory` or
    :class:`~obspy.core.event.Catalog` object to a KML file.
    For additional parameters see :meth:`inventory_to_kml_string` and
    :meth:`catalog_to_kml_string`.

    :type obj: :class:`~obspy.core.event.Catalog` or
        :class:`~obspy.core.inventory.Inventory`
    :param obj: ObsPy object for KML output
    :type filename: str
    :param filename: Filename to write to. Suffix ".kml" will be appended if
        not already present.
    """
    if isinstance(obj, Catalog):
        kml_string = catalog_to_kml_string(obj, **kwargs)
    elif isinstance(obj, Inventory):
        kml_string = inventory_to_kml_string(obj, **kwargs)
    else:
        msg = ("Object for KML output must be "
               "a Catalog or Inventory.")
        raise TypeError(msg)
    if not filename.endswith(".kml"):
        filename += ".kml"
    with open(filename, "wb") as fh:
        fh.write(kml_string)


def _rgba_tuple_to_kml_color_code(rgba):
    """
    Convert tuple of (red, green, blue, alpha) float values (0.0-1.0) to KML
    hex color code string "aabbggrr".
    """
    try:
        r, g, b, a = rgba
    except:
        r, g, b = rgba
        a = 1.0
    return "".join(["%02x" % int(x * 255) for x in (a, b, g, r)])


def _get_event_timestamp(event):
    """
    Get timestamp information for the event. Search is perfomed in the
    following order:

     - origin time of preferred origin
     - origin time of first origin found that has a origin time
     - minimum of all found pick times
     - `None` if no time is found in the above search
    """
    origin = event.preferred_origin()
    if origin is not None and origin.time is not None:
        return origin.time
    for origin in event.origins:
        if origin.time is not None:
            return origin.time
    pick_times = [pick.time for pick in event.picks
                  if pick.time is not None]
    if pick_times:
        return min(pick_times)
    return None


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
