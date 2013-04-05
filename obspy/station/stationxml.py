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

    # These are optional
    sender = _tag2obj(root, _ns("Sender"), str)
    module = _tag2obj(root, _ns("Module"), str)
    module_uri = _tag2obj(root, _ns("ModuleURI"), str)

    networks = []
    for network in root.findall(_ns("Network")):
        networks.append(_read_network(network, _ns))

    inv = obspy.station.SeismicInventory(networks=networks, source=source,
        sender=sender, created=created, module=module, module_uri=module_uri)
    return inv


def _read_network(net_element, _ns):
    network = obspy.station.SeismicNetwork(net_element.get("code"))
    network.start_date = _attr2obj(net_element, "startDate", obspy.UTCDateTime)
    network.end_date = _attr2obj(net_element, "endDate", obspy.UTCDateTime)
    network.restricted_status = \
        _attr2obj(net_element, "restrictedStatus", str)
    network.alternate_code = _attr2obj(net_element, "alternateCode", str)
    network.historical_code = _attr2obj(net_element, "historicalCode", str)
    network.description = _tag2obj(net_element, _ns("Description"), str)
    network.comments = []
    for comment in net_element.findall(_ns("Comment")):
        network.comments.append(_read_comment(comment, _ns))
    return network


def _read_comment(comment_element, _ns):
    value = _tag2obj(comment_element, _ns("Value"), str)
    begin_effective_time = _tag2obj(comment_element, _ns("BeginEffectiveTime"),
        obspy.UTCDateTime)
    end_effective_time = _tag2obj(comment_element, _ns("EndEffectiveTime"),
        obspy.UTCDateTime)
    authors = []
    for author in comment_element.findall(_ns("Author")):
        authors.append(_read_person(author, _ns))
    return obspy.station.Comment(value=value,
        begin_effective_time=begin_effective_time,
        end_effective_time=end_effective_time,
        authors=authors)


def _read_person(person_element, _ns):
    names = _tags2obj(person_element, _ns("Name"), str)
    agencies = _tags2obj(person_element, _ns("Agency"), str)
    emails = _tags2obj(person_element, _ns("Email"), str)
    phones = []
    for phone in person_element.findall(_ns("Phone")):
        phones.append(_read_phone(phone, _ns))
    return obspy.station.Person(names=names, agencies=agencies, emails=emails,
        phones=phones)


def _read_phone(phone_element, _ns):
    country_code = _tag2obj(phone_element, _ns("CountryCode"), int)
    area_code = _tag2obj(phone_element, _ns("AreaCode"), int)
    phone_number = _tag2obj(phone_element, _ns("PhoneNumber"), str)
    description = phone_element.get("description")
    return obspy.station.PhoneNumber(country_code=country_code,
        area_code=area_code, phone_number=phone_number,
        description=description)


def write_StationXML(inventory, file_or_file_object, validate=False, **kwargs):
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
    if inventory.sender:
        etree.SubElement(root, "Sender").text = inventory.sender

    # Undocumented flag that does not write the module flags. Useful for
    # testing. It is undocumented because it should not be used publicly.
    if not kwargs.get("_suppress_module_tags", False):
        etree.SubElement(root, "Module").text = SOFTWARE_MODULE
        etree.SubElement(root, "ModuleURI").text = SOFTWARE_URI

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


def _tag2obj(element, tag, convert):
    try:
        return convert(element.find(tag).text)
    except:
        None


def _tags2obj(element, tag, convert):
    values = []
    for elem in element.findall(tag):
        values.append(convert(elem.text))
    return values


def _attr2obj(element, attr, convert):
    attribute = element.get(attr)
    if attribute is None:
        return None
    try:
        return convert(attribute)
    except:
        None


def _format_time(value):
    return value.strftime("%Y-%m-%dT%H:%M:%S+00:00")


def _write_network(parent, network):
    """
    Helper function converting a SeismicNetwork instance to an etree.Element.
    """
    attribs = {"code": network.code}
    if network.start_date:
        attribs["startDate"] = _format_time(network.start_date)
    if network.end_date:
        attribs["endDate"] = _format_time(network.end_date)
    if network.restricted_status:
        attribs["restrictedStatus"] = network.restricted_status
    if network.alternate_code:
        attribs["alternateCode"] = network.alternate_code
    if network.historical_code:
        attribs["historicalCode"] = network.historical_code
    network_elem = etree.SubElement(parent, "Network", attribs)
    if network.description:
        etree.SubElement(network_elem, "Description").text = \
            network.description
    for comment in network.comments:
        _write_comment(network_elem, comment)


def _write_comment(parent, comment):
    comment_elem = etree.SubElement(parent, "Comment")
    etree.SubElement(comment_elem, "Value").text = comment.value
    if comment.begin_effective_time:
        etree.SubElement(comment_elem, "BeginEffectiveTime").text = \
            _format_time(comment.begin_effective_time)
    if comment.end_effective_time:
        etree.SubElement(comment_elem, "EndEffectiveTime").text = \
            _format_time(comment.end_effective_time)
    for author in comment.authors:
        _write_person(comment_elem, author, "Author")


def _write_person(parent, person, tag_name):
    person_elem = etree.SubElement(parent, tag_name)
    for name in person.names:
        etree.SubElement(person_elem, "Name").text = name
    for agency in person.agencies:
        etree.SubElement(person_elem, "Agency").text = agency
    for email in person.emails:
        etree.SubElement(person_elem, "Email").text = email
    for phone in person.phones:
        _write_phone(person_elem, phone)


def _write_phone(parent, phone):
    attribs = {}
    if phone.description:
        attribs["description"] = phone.description
    phone_elem = etree.SubElement(parent, "Phone", attribs)
    if phone.country_code:
        etree.SubElement(phone_elem, "CountryCode").text = \
            str(phone.country_code)
    etree.SubElement(phone_elem, "AreaCode").text = str(phone.area_code)
    etree.SubElement(phone_elem, "PhoneNumber").text = phone.phone_number
