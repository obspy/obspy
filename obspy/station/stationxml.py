#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DESCRIPTION

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2013
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""
import inspect
from lxml import etree
from lxml.builder import E
import os

import obspy


# Define some constants for writing StationXML files.
SOFTWARE_MODULE = "ObsPy %s" % obspy.__version__
SOFTWARE_URI = "http://www.obspy.org"
SCHEMA_VERSION = "1.0"


def is_StationXML(path_or_file_object):
    """
    Simple function checking if the passed object contains a valid StationXML
    1.0 file. Returns True of False.

    This is simply done by validating against the StationXML schema.

    :param path_of_file_object: Filename or file like object.
    """
    return validate_StationXML(path_or_file_object)[0]


def validate_StationXML(path_or_file_object):
    """
    Checks if the given path is a valid StationXML file.

    Returns a tuple. The first item is a boolean describing if the validation
    was successful or not. The second item is a list of all found validation
    errors, if existant.

    :path_or_file_object: Filename of file like object.
    """
    # Get the schema location.
    schema_location = os.path.dirname(inspect.getfile(inspect.currentframe()))
    schema_location = os.path.join(schema_location, "docs",
        "fdsn-station-1.0.xsd")

    xmlschema = etree.XMLSchema(etree.parse(schema_location))
    xmldoc = etree.parse(path_or_file_object)

    valid = xmlschema.validate(xmldoc)

    # Pretty error printing if the validation fails.
    if valid is not True:
        return (False, xmlschema.error_log)
    return (True, ())


def read_StationXML(path_or_file_object):
    """
    Function reading a StationXML file.

    :path_or_file_object: Filename of file like object.
    """
    root = etree.parse(path_or_file_object).getroot()
    namespace = root.nsmap.itervalues().next()

    _ns = lambda tagname: "{%s}%s" % (namespace, tagname)

    # Source and Created field must exist in a StationXML.
    source = root.find(_ns("Source")).text
    created = obspy.UTCDateTime(root.find(_ns("Created")).text)

    networks = []
    for network in root.findall(_ns("Network")):
        network = obspy.station.SeismicNetwork(network.get("code"))
        networks.append(network)

    inv = obspy.station.SeismicInventory(networks=networks, source=source,
        created=created)
    return inv


def write_StationXML(inventory, buffer):
    """
    Writes an inventory object to a buffer.

    :type inventory: :class:`~obspy.station.inventory.SeismicInventory`
    :param inventory: The inventory instance to be written.
    :type buffer: Open file or file-like object.
    :param buffer: The file buffer object the StationXML will be written to.
    """
    # Use the etree factory to create the very basic structure.
    xml_doc = E.FSDNStationXML(
        E.Source(str(inventory.source)))
