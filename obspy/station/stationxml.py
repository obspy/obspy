#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
File dealing with the StationXML format.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2013
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""
import inspect
from io import BytesIO
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


def validate_StationXML(path_or_object):
    """
    Checks if the given path is a valid StationXML file.

    Returns a tuple. The first item is a boolean describing if the validation
    was successful or not. The second item is a list of all found validation
    errors, if existant.

    :path_or_object: Filename of file like object. Can also be an etree
        element.
    """
    # Get the schema location.
    schema_location = os.path.dirname(inspect.getfile(inspect.currentframe()))
    schema_location = os.path.join(schema_location, "docs",
        "fdsn-station-1.0.xsd")

    xmlschema = etree.XMLSchema(etree.parse(schema_location))

    if isinstance(path_or_object, etree._Element):
        xmldoc = path_or_object
    else:
        try:
            xmldoc = etree.parse(path_or_object)
        except etree.XMLSyntaxError:
            return (False, ("Not a XML file.",))

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


def write_StationXML(inventory, file_or_file_object, validate=False):
    """
    Writes an inventory object to a buffer.

    :type inventory: :class:`~obspy.station.inventory.SeismicInventory`
    :param inventory: The inventory instance to be written.
    :param file_or_file_object: The file or file-like object to be written to.
    :type validate: Boolean
    :type validate: If True, the created document will be validated with the
        StationXML schema before being written. Useful for debugging or if you
        don't trust ObsPy. Defaults to False.
    """
    root = etree.Element(
        "FDSNStationXML",
        attrib={
            "xmlns": "http://www.fdsn.org/xml/station/1",
            "schemaVersion": SCHEMA_VERSION}
    )
    etree.SubElement(root, "Source").text = inventory.source
    etree.SubElement(root, "Created").text = _format_time(inventory.created)

    for network in inventory.networks:
        _write_network(root, network)

    str_repr = etree.tostring(root, pretty_print=True, xml_declaration=True,
        encoding="UTF-8")

    # The validation has to be done after parsing once again so that the
    # namespaces are correctly assembled.
    if validate is True:
        buf = BytesIO(str_repr)
        validates, errors = validate_StationXML(buf)
        buf.close()
        if validates is False:
            msg = "The created file fails to validate.\n"
            for err in errors:
                msg += "\t%s\n" % err
            raise Exception(msg)

    if hasattr(file_or_file_object, "write") and \
            hasattr(file_or_file_object.write, "__call__"):
        file_or_file_object.write(str_repr)
        return
    with open(file_or_file_object, "wt") as fh:
        fh.write(str_repr)


def _format_time(value):
    return value.strftime("%Y-%m-%dT%H:%M:%S+00:00")


def _write_network(parent, network):
    """
    Helper function converting a SeismicNetwork instance to an etree.Element.
    """
    elem = etree.SubElement(parent, "Network", {"code": network.code})
