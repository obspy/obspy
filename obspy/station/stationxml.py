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
import warnings

import obspy
from obspy.station.util import Longitude, Latitude


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
    namespace = root.nsmap[None]

    _ns = lambda tagname: "{%s}%s" % (namespace, tagname)

    # Source and Created field must exist in a StationXML.
    source = root.find(_ns("Source")).text
    created = obspy.UTCDateTime(root.find(_ns("Created")).text)

    # These are optional
    sender = _tag2obj(root, _ns("Sender"), unicode)
    module = _tag2obj(root, _ns("Module"), unicode)
    module_uri = _tag2obj(root, _ns("ModuleURI"), unicode)

    networks = []
    for network in root.findall(_ns("Network")):
        networks.append(_read_network(network, _ns))

    inv = obspy.station.Inventory(networks=networks, source=source,
                                  sender=sender, created=created,
                                  module=module, module_uri=module_uri)
    return inv


def _read_base_node(element, object_to_write_to, _ns):
    """
    Reads the base node structure from element and saves it in
    object_to_write_to.

    Reads everything except the 'code' attribute.
    """
    object_to_write_to.start_date = \
        _attr2obj(element, "startDate", obspy.UTCDateTime)
    object_to_write_to.end_date = \
        _attr2obj(element, "endDate", obspy.UTCDateTime)
    object_to_write_to.restricted_status = \
        _attr2obj(element, "restrictedStatus", unicode)
    object_to_write_to.alternate_code = \
        _attr2obj(element, "alternateCode", unicode)
    object_to_write_to.historical_code = \
        _attr2obj(element, "historicalCode", unicode)
    object_to_write_to.description = \
        _tag2obj(element, _ns("Description"), unicode)
    object_to_write_to.comments = []
    for comment in element.findall(_ns("Comment")):
        object_to_write_to.comments.append(_read_comment(comment, _ns))


def _read_network(net_element, _ns):
    network = obspy.station.Network(net_element.get("code"))
    _read_base_node(net_element, network, _ns)
    network.total_number_of_stations = \
        _tag2obj(net_element, _ns("TotalNumberStations"), int)
    network.selected_number_of_stations = \
        _tag2obj(net_element, _ns("SelectedNumberStations"), int)
    stations = []
    for station in net_element.findall(_ns("Station")):
        stations.append(_read_station(station, _ns))
    network.stations = stations
    return network


def _read_station(sta_element, _ns):
    longitude, latitude = _read_lonlat(sta_element, _ns)
    elevation = _tag2obj(sta_element, _ns("Elevation"), float)
    station = obspy.station.Station(code=sta_element.get("code"),
                                    latitude=latitude, longitude=longitude,
                                    elevation=elevation)
    station.site = _read_site(sta_element.find(_ns("Site")), _ns)
    _read_base_node(sta_element, station, _ns)
    station.vault = _tag2obj(sta_element, _ns("Vault"), unicode)
    station.geology = _tag2obj(sta_element, _ns("Geology"), unicode)
    for equipment in sta_element.findall(_ns("Equipment")):
        station.equipments.append(_read_equipment(equipment, _ns))
    for operator in sta_element.findall(_ns("Operator")):
        station.operators.append(_read_operator(operator, _ns))
    station.creation_date = \
        _tag2obj(sta_element, _ns("CreationDate"), obspy.UTCDateTime)
    station.termination_date = \
        _tag2obj(sta_element, _ns("TerminationDate"), obspy.UTCDateTime)
    station.selected_number_of_channels = \
        _tag2obj(sta_element, _ns("SelectedNumberChannels"), int)
    station.total_number_of_channels = \
        _tag2obj(sta_element, _ns("TotalNumberChannels"), int)
    for ref in sta_element.findall(_ns("ExternalReference")):
        station.external_references.append(_read_external_reference(ref, _ns))
    channels = []
    for channel in sta_element.findall(_ns("Channel")):
        channels.append(_read_channel(channel, _ns))
    station.channels = channels
    return station


def _read_lonlat(parent, _ns):
    lon_elem = parent.find(_ns("Longitude"))
    lat_elem = parent.find(_ns("Latitude"))
    lon = Longitude(_tag2obj(parent, _ns("Longitude"), float))
    lat = Latitude(_tag2obj(parent, _ns("Latitude"), float))
    for obj_, elem_ in zip((lon, lat), (lon_elem, lat_elem)):
        obj_.unit = elem_.attrib.get("unit")
        obj_.datum = elem_.attrib.get("datum")
        obj_.lower_uncertainty = elem_.attrib.get("minusError")
        obj_.upper_uncertainty = elem_.attrib.get("plusError")
    return lon, lat


def _read_channel(cha_element, _ns):
    longitude, latitude = _read_lonlat(cha_element, _ns)
    elevation = _tag2obj(cha_element, _ns("Elevation"), float)
    depth = _tag2obj(cha_element, _ns("Depth"), float)
    code = cha_element.get("code")
    location_code = cha_element.get("locationCode")
    channel = obspy.station.Channel(
        code=code, location_code=location_code, latitude=latitude,
        longitude=longitude, elevation=elevation, depth=depth)
    _read_base_node(cha_element, channel, _ns)
    channel.azimuth = _tag2obj(cha_element, _ns("Azimuth"), float)
    channel.dip = _tag2obj(cha_element, _ns("Dip"), float)
    # Add all types.
    for type_element in cha_element.findall(_ns("Type")):
        channel.types.append(type_element.text)
    # Add all external references.
    channel.external_references = \
        [_read_external_reference(ext_ref, _ns)
         for ext_ref in cha_element.findall(_ns("ExternalReference"))]
    channel.sample_rate = _tag2obj(cha_element, _ns("SampleRate"), float)
    # Parse the optional sample rate ratio.
    sample_rate_ratio = cha_element.find(_ns("SampleRateRation"))
    if sample_rate_ratio:
        channel.sample_rate_ratio_number_samples = \
            _tag2obj(sample_rate_ratio, _ns("NumberSamples"), int)
        channel.sample_rate_ratio_number_seconds = \
            _tag2obj(sample_rate_ratio, _ns("NumberSeconds"), int)
    channel.storage_format = _tag2obj(cha_element, _ns("StorageFormat"),
                                      unicode)
    # The clock drift is one of the few examples where the attribute name is
    # different from the tag name. This improves clarity.
    channel.clock_drift_in_seconds_per_sample = \
        _tag2obj(cha_element, _ns("ClockDrift"), float)
    # The sensor.
    sensor = cha_element.find(_ns("Sensor"))
    if sensor is not None:
        channel.sensor = _read_equipment(sensor, _ns)
    # The pre-amplifier
    pre_amplifier = cha_element.find(_ns("PreAmplifier"))
    if pre_amplifier is not None:
        channel.pre_amplifier = _read_equipment(pre_amplifier, _ns)
    # The data logger
    data_logger = cha_element.find(_ns("DataLogger"))
    if data_logger is not None:
        channel.data_logger = _read_equipment(data_logger, _ns)
    # Other equipment
    equipment = cha_element.find(_ns("Equipment"))
    if equipment is not None:
        channel.equipment = _read_equipment(equipment, _ns)
    # Finally parse the response.
    response = cha_element.find(_ns("Response"))
    if response is not None:
        channel.response = _read_response(response, _ns)
    return channel


def _read_response(resp_element, _ns):
    response = obspy.station.response.Response()
    instrument_sensitivity = resp_element.find(_ns("InstrumentSensitivity"))
    if instrument_sensitivity is not None:
        response.instrument_sensitivity = \
            _read_instrument_sensitivity(instrument_sensitivity, _ns)
    instrument_polynomial = resp_element.find(_ns("InstrumentPolynomial"))
    if instrument_polynomial is not None:
        response.instrument_polynomial = \
            _read_instrument_polynomial(instrument_polynomial, _ns)
    # Now read all the stages.
    for stage in resp_element.findall(_ns("Stage")):
        response.response_stages.append(_read_response_stage(stage, _ns))
    return response


def _read_response_stage(stage_elem, _ns):
    """
    This parses all ResponseStageTypes. It will return a different object
    depending on the actual response type.
    """
    # The stage sequence number is required!
    stage_sequence_number = int(stage_elem.get("number"))
    # All stages contain a stage gain and potentially a decimation.
    gain_elem = stage_elem.find(_ns("StageGain"))
    stage_gain_value = _tag2obj(gain_elem, _ns("Value"), float)
    stage_gain_frequency = _tag2obj(gain_elem, _ns("Frequency"), float)
    # Parse the decimation.
    decim_elem = stage_elem.find(_ns("Decimation"))
    if decim_elem is not None:
        decimation_input_sample_rate = \
            _tag2obj(decim_elem, _ns("InputSampleRate"), float)
        decimation_factor = _tag2obj(decim_elem, _ns("Factor"), int)
        decimation_offset = _tag2obj(decim_elem, _ns("Offset"), int)
        decimation_delay = _tag2obj(decim_elem, _ns("Delay"), float)
        decimation_correction = _tag2obj(decim_elem, _ns("Correction"), float)
    else:
        decimation_input_sample_rate = None
        decimation_factor = None
        decimation_offset = None
        decimation_delay = None
        decimation_correction = None

    # Now determine which response type it actually is and return the
    # corresponding object.
    poles_zeros_elem = stage_elem.find(_ns("PolesZeros"))
    coefficients_elem = stage_elem.find(_ns("Coefficients"))
    response_list_elem = stage_elem.find(_ns("ResponseList"))
    FIR_elem = stage_elem.find(_ns("FIR"))
    polynomial_elem = stage_elem.find(_ns("Polynomial"))

    type_elems = [poles_zeros_elem, coefficients_elem, response_list_elem,
                  FIR_elem, polynomial_elem]

    # iterate and check for an response element and create alias
    for elem in type_elems:
        if elem is not None:
            break
    else:
        # Raise if none of the previous ones has been found.
        msg = "Could not find a valid Response Stage Type."
        raise ValueError(msg)

    # Now parse all elements the different stages share.
    input_units = elem.find(_ns("InputUnits"))
    input_units_name = _tag2obj(input_units, _ns("Name"), unicode)
    input_units_description = _tag2obj(input_units, _ns("Description"),
                                       unicode)
    output_units = elem.find(_ns("OutputUnits"))
    output_units_name = _tag2obj(output_units, _ns("Name"), unicode)
    output_units_description = _tag2obj(output_units, _ns("Description"),
                                        unicode)
    description = _tag2obj(elem, _ns("Description"), unicode)
    resource_id = _tag2obj(elem, _ns("resourceId"), unicode)
    name = _tag2obj(elem, _ns("name"), unicode)

    # Now collect all shared kwargs to be able to pass them to the different
    # constructors..
    kwargs = {"stage_sequence_number": stage_sequence_number,
              "input_units_name": input_units_name,
              "output_units_name": output_units_name,
              "input_units_description": input_units_description,
              "output_units_description": output_units_description,
              "resource_id": resource_id, "stage_gain_value": stage_gain_value,
              "stage_gain_frequency": stage_gain_frequency, "name": name,
              "description": description,
              "decimation_input_sample_rate": decimation_input_sample_rate,
              "decimation_factor": decimation_factor,
              "decimation_offset": decimation_offset,
              "decimation_delay": decimation_delay,
              "decimation_correction": decimation_correction}

    # Handle Poles and Zeros Response Stage Type.
    if elem is poles_zeros_elem:
        pz_transfer_function_type = \
            _tag2obj(elem, _ns("PzTransferFunctionType"), unicode)
        normalization_factor = \
            _tag2obj(elem, _ns("NormalizationFactor"), float)
        normalization_frequency = \
            _tag2obj(elem, _ns("NormalizationFrequency"), float)
        # Read poles and zeros to list of imaginary numbers.
        zeros = [_tag2obj(i, _ns("Real"), float) +
                 _tag2obj(i, _ns("Imaginary"), float) * 1j
                 for i in elem.findall(_ns("Zero"))]
        poles = [_tag2obj(i, _ns("Real"), float) +
                 _tag2obj(i, _ns("Imaginary"), float) * 1j
                 for i in elem.findall(_ns("Pole"))]
        return obspy.station.PolesZerosResponseStage(
            pz_transfer_function_type=pz_transfer_function_type,
            normalization_frequency=normalization_frequency,
            normalization_factor=normalization_factor, zeros=zeros,
            poles=poles, **kwargs)

    # Handle the coefficients Response Stage Type.
    elif elem is coefficients_elem:
        cf_transfer_function_type = \
            _tag2obj(elem, _ns("CfTransferFunctionType"), unicode)
        numerator = _tags2obj(elem, _ns("Numerator"), float)
        denominator = _tags2obj(elem, _ns("Denominator"), float)
        return obspy.station.CoefficientsTypeResponseStage(
            cf_transfer_function_type=cf_transfer_function_type,
            numerator=numerator, denominator=denominator, **kwargs)

    # Handle the response list response stage type.
    elif elem is response_list_elem:
        rlist_elems = []
        for item in elem.findall(_ns("ResponseListElement")):
            freq = _tag2obj(item, _ns("Frequency"), float)
            amp = _tag2obj(item, _ns("Amplitude"), float)
            phase = _tag2obj(item, _ns("Phase"), float)
            x = obspy.station.response.ResponseListElement(
                frequency=freq, amplitude=amp, phase=phase)
            rlist_elems.append(x)
        return obspy.station.ResponseListResponseStage(
            response_list_elements=rlist_elems, **kwargs)

    # Handle the FIR response stage type.
    elif elem is FIR_elem:
        symmetry = _tag2obj(elem, _ns("Symmetry"), unicode)
        coeffs = _tags2obj(elem, _ns("NumeratorCoefficient"), float)
        return obspy.station.FIRResponseStage(numerator_coefficients=coeffs,
                                              symmetry=symmetry, **kwargs)

    # Handle polynomial instrument responses.
    elif elem is polynomial_elem:
        appr_type = _tag2obj(elem, _ns("ApproximationType"), unicode)
        f_low = _tag2obj(elem, _ns("FrequencyLowerBound"), float)
        f_high = _tag2obj(elem, _ns("FrequencyUpperBound"), float)
        appr_low = _tag2obj(elem, _ns("ApproximationLowerBound"), float)
        appr_high = _tag2obj(elem, _ns("ApproximationUpperBound"), float)
        max_err = _tag2obj(elem, _ns("MaximumError"), float)
        coeffs = _tags2obj(elem, _ns("Coefficient"), float)
        return obspy.station.PolynomialResponseStage(
            approximation_type=appr_type, frequency_lower_bound=f_low,
            frequency_upper_bound=f_high, approximation_lower_bound=appr_low,
            approximation_upper_bound=appr_high, maximum_error=max_err,
            coefficients=coeffs, **kwargs)


def _read_instrument_sensitivity(sensitivity_element, _ns):
    value = _tag2obj(sensitivity_element, _ns("Value"), float)
    frequency = _tag2obj(sensitivity_element, _ns("Frequency"), float)
    input_units = sensitivity_element.find(_ns("InputUnits"))
    output_units = sensitivity_element.find(_ns("OutputUnits"))
    sensitivity = obspy.station.response.InstrumentSensitivity(
        value=value, frequency=frequency,
        input_units_name=_tag2obj(input_units, _ns("Name"), unicode),
        output_units_name=_tag2obj(output_units, _ns("Name"), unicode))
    sensitivity.input_units_description = \
        _tag2obj(input_units, _ns("Description"), unicode)
    sensitivity.output_units_description = \
        _tag2obj(output_units, _ns("Description"), unicode)
    sensitivity.frequency_range_start = \
        _tag2obj(sensitivity_element, _ns("FrequencyStart"), float)
    sensitivity.frequency_range_end = \
        _tag2obj(sensitivity_element, _ns("FrequencyEnd"), float)
    sensitivity.frequency_range_DB_variation = \
        _tag2obj(sensitivity_element, _ns("FrequencyDBVariation"), float)
    return sensitivity


def _read_instrument_polynomial(element, _ns):
    # XXX duplicated code, see reading of PolynomialResponseStage
    input_units = element.find(_ns("InputUnits"))
    input_units_name = _tag2obj(input_units, _ns("Name"), unicode)
    input_units_description = _tag2obj(input_units, _ns("Description"),
                                       unicode)
    output_units = element.find(_ns("OutputUnits"))
    output_units_name = _tag2obj(output_units, _ns("Name"), unicode)
    output_units_description = _tag2obj(output_units, _ns("Description"),
                                        unicode)
    description = _tag2obj(element, _ns("Description"), unicode)
    resource_id = _tag2obj(element, _ns("resourceId"), unicode)
    name = _tag2obj(element, _ns("name"), unicode)
    appr_type = _tag2obj(element, _ns("ApproximationType"), unicode)
    f_low = _tag2obj(element, _ns("FrequencyLowerBound"), float)
    f_high = _tag2obj(element, _ns("FrequencyUpperBound"), float)
    appr_low = _tag2obj(element, _ns("ApproximationLowerBound"), float)
    appr_high = _tag2obj(element, _ns("ApproximationUpperBound"), float)
    max_err = _tag2obj(element, _ns("MaximumError"), float)
    coeffs = _tags2obj(element, _ns("Coefficient"), float)
    return obspy.station.response.InstrumentPolynomial(
        approximation_type=appr_type, frequency_lower_bound=f_low,
        frequency_upper_bound=f_high, approximation_lower_bound=appr_low,
        approximation_upper_bound=appr_high, maximum_error=max_err,
        coefficients=coeffs, input_units_name=input_units_name,
        input_units_description=input_units_description,
        output_units_name=output_units_name,
        output_units_description=output_units_description,
        description=description, resource_id=resource_id, name=name)


def _read_external_reference(ref_element, _ns):
    uri = _tag2obj(ref_element, _ns("URI"), unicode)
    description = _tag2obj(ref_element, _ns("Description"), unicode)
    return obspy.station.ExternalReference(uri=uri, description=description)


def _read_operator(operator_element, _ns):
    agencies = [_i.text for _i in operator_element.findall(_ns("Agency"))]
    contacts = []
    for contact in operator_element.findall(_ns("Contact")):
        contacts.append(_read_person(contact, _ns))
    website = _tag2obj(operator_element, _ns("WebSite"), unicode)
    return obspy.station.Operator(agencies=agencies, contacts=contacts,
                                  website=website)


def _read_equipment(equip_element, _ns):
    resource_id = equip_element.get("resourceId")
    type = _tag2obj(equip_element, _ns("Type"), unicode)
    description = _tag2obj(equip_element, _ns("Description"), unicode)
    manufacturer = _tag2obj(equip_element, _ns("Manufacturer"), unicode)
    vendor = _tag2obj(equip_element, _ns("Vendor"), unicode)
    model = _tag2obj(equip_element, _ns("Model"), unicode)
    serial_number = _tag2obj(equip_element, _ns("SerialNumber"), unicode)
    installation_date = \
        _tag2obj(equip_element, _ns("InstallationDate"), obspy.UTCDateTime)
    removal_date = \
        _tag2obj(equip_element, _ns("RemovalDate"), obspy.UTCDateTime)
    calibration_dates = \
        [obspy.core.UTCDateTime(_i.text)
         for _i in equip_element.findall(_ns("CalibrationDate"))]
    return obspy.station.Equipment(
        resource_id=resource_id, type=type, description=description,
        manufacturer=manufacturer, vendor=vendor, model=model,
        serial_number=serial_number, installation_date=installation_date,
        removal_date=removal_date, calibration_dates=calibration_dates)


def _read_site(site_element, _ns):
    name = _tag2obj(site_element, _ns("Name"), unicode)
    description = _tag2obj(site_element, _ns("Description"), unicode)
    town = _tag2obj(site_element, _ns("Town"), unicode)
    county = _tag2obj(site_element, _ns("County"), unicode)
    region = _tag2obj(site_element, _ns("Region"), unicode)
    country = _tag2obj(site_element, _ns("Country"), unicode)
    return obspy.station.Site(name=name, description=description, town=town,
                              county=county, region=region, country=country)


def _read_comment(comment_element, _ns):
    value = _tag2obj(comment_element, _ns("Value"), unicode)
    begin_effective_time = \
        _tag2obj(comment_element, _ns("BeginEffectiveTime"), obspy.UTCDateTime)
    end_effective_time = \
        _tag2obj(comment_element, _ns("EndEffectiveTime"), obspy.UTCDateTime)
    authors = []
    id = _attr2obj(comment_element, "id", int)
    for author in comment_element.findall(_ns("Author")):
        authors.append(_read_person(author, _ns))
    return obspy.station.Comment(
        value=value, begin_effective_time=begin_effective_time,
        end_effective_time=end_effective_time, authors=authors, id=id)


def _read_person(person_element, _ns):
    names = _tags2obj(person_element, _ns("Name"), unicode)
    agencies = _tags2obj(person_element, _ns("Agency"), unicode)
    emails = _tags2obj(person_element, _ns("Email"), unicode)
    phones = []
    for phone in person_element.findall(_ns("Phone")):
        phones.append(_read_phone(phone, _ns))
    return obspy.station.Person(names=names, agencies=agencies, emails=emails,
                                phones=phones)


def _read_phone(phone_element, _ns):
    country_code = _tag2obj(phone_element, _ns("CountryCode"), int)
    area_code = _tag2obj(phone_element, _ns("AreaCode"), int)
    phone_number = _tag2obj(phone_element, _ns("PhoneNumber"), unicode)
    description = phone_element.get("description")
    return obspy.station.PhoneNumber(
        country_code=country_code, area_code=area_code,
        phone_number=phone_number, description=description)


def write_StationXML(inventory, file_or_file_object, validate=False, **kwargs):
    """
    Writes an inventory object to a buffer.

    :type inventory: :class:`~obspy.station.inventory.Inventory`
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
    if kwargs.get("_suppress_module_tags", False):
        pass
    else:
        etree.SubElement(root, "Module").text = inventory.module
        etree.SubElement(root, "ModuleURI").text = inventory.module_uri

    etree.SubElement(root, "Created").text = _format_time(inventory.created)

    for network in inventory.networks:
        _write_network(root, network)

    tree = root.getroottree()

    # The validation has to be done after parsing once again so that the
    # namespaces are correctly assembled.
    if validate is True:
        buf = BytesIO()
        tree.write(buf)
        buf.seek(0)
        validates, errors = validate_StationXML(buf)
        buf.close()
        if validates is False:
            msg = "The created file fails to validate.\n"
            for err in errors:
                msg += "\t%s\n" % err
            raise Exception(msg)

    tree.write(file_or_file_object, pretty_print=True, xml_declaration=True,
               encoding="UTF-8")


def _get_base_node_attributes(element):
    attributes = {"code": element.code}
    if element.start_date:
        attributes["startDate"] = _format_time(element.start_date)
    if element.end_date:
        attributes["endDate"] = _format_time(element.end_date)
    if element.restricted_status:
        attributes["restrictedStatus"] = element.restricted_status
    if element.alternate_code:
        attributes["alternateCode"] = element.alternate_code
    if element.historical_code:
        attributes["historicalCode"] = element.historical_code
    return attributes


def _write_base_node(element, object_to_read_from):
    if object_to_read_from.description:
        etree.SubElement(element, "Description").text = \
            object_to_read_from.description
    for comment in object_to_read_from.comments:
        _write_comment(element, comment)


def _write_network(parent, network):
    """
    Helper function converting a Network instance to an etree.Element.
    """
    attribs = _get_base_node_attributes(network)
    network_elem = etree.SubElement(parent, "Network", attribs)
    _write_base_node(network_elem, network)

    # Add the two, network specific fields.
    if network.total_number_of_stations is not None:
        etree.SubElement(network_elem, "TotalNumberStations").text = \
            str(network.total_number_of_stations)
    if network.selected_number_of_stations is not None:
        etree.SubElement(network_elem, "SelectedNumberStations").text = \
            str(network.selected_number_of_stations)

    for station in network.stations:
        _write_station(network_elem, station)


def _write_lonlat(parent, obj):
    attribs = {}
    attribs["datum"] = obj.latitude.datum
    attribs["unit"] = obj.latitude.unit
    attribs["plusError"] = obj.latitude.lower_uncertainty
    attribs["minusError"] = obj.latitude.upper_uncertainty
    attribs = dict([(k, v) for k, v in attribs.iteritems() if v is not None])
    etree.SubElement(parent, "Latitude", attribs).text = \
        str(obj.latitude.value)
    attribs = {}
    attribs["datum"] = obj.longitude.datum
    attribs["unit"] = obj.longitude.unit
    attribs["plusError"] = obj.longitude.lower_uncertainty
    attribs["minusError"] = obj.longitude.upper_uncertainty
    attribs = dict([(k, v) for k, v in attribs.iteritems() if v is not None])
    etree.SubElement(parent, "Longitude", attribs).text = \
        str(obj.longitude.value)


def _write_station(parent, station):
    # Write the base node type fields.
    attribs = _get_base_node_attributes(station)
    station_elem = etree.SubElement(parent, "Station", attribs)
    _write_base_node(station_elem, station)

    _write_lonlat(station_elem, station)
    etree.SubElement(station_elem, "Elevation").text = str(station.elevation)

    _write_site(station_elem, station.site)

    # Optional tags.
    _obj2tag(station_elem, "Vault", station.vault)
    _obj2tag(station_elem, "Geology", station.geology)

    for equipment in station.equipments:
        _write_equipment(station_elem, equipment)

    for operator in station.operators:
        operator_elem = etree.SubElement(station_elem, "Operator")
        for agency in operator.agencies:
            etree.SubElement(operator_elem, "Agency").text = agency
        for contact in operator.contacts:
            _write_person(operator_elem, contact, "Contact")
        etree.SubElement(operator_elem, "WebSite").text = operator.website

    etree.SubElement(station_elem, "CreationDate").text = \
        _format_time(station.creation_date)
    if station.termination_date:
        etree.SubElement(station_elem, "TerminationDate").text = \
            _format_time(station.termination_date)
    # The next two tags are optional.
    _obj2tag(station_elem, "TotalNumberChannels",
             station.total_number_of_channels)
    _obj2tag(station_elem, "SelectedNumberChannels",
             station.selected_number_of_channels)

    for ref in station.external_references:
        _write_external_reference(station_elem, ref)


def _write_external_reference(parent, ref):
    ref_elem = etree.SubElement(parent, "ExternalReference")
    etree.SubElement(ref_elem, "URI").text = ref.uri
    etree.SubElement(ref_elem, "Description").text = ref.description


def _write_equipment(parent, equipment):
    if equipment.resource_id is None:
        attr = {}
    else:
        attr = {"resourceId": equipment.resource_id}
    equipment_elem = etree.SubElement(parent, "Equipment", attr)

    # All tags are optional.
    _obj2tag(equipment_elem, "Type", equipment.type)
    _obj2tag(equipment_elem, "Description", equipment.description)
    _obj2tag(equipment_elem, "Manufacturer", equipment.manufacturer)
    _obj2tag(equipment_elem, "Vendor", equipment.vendor)
    _obj2tag(equipment_elem, "Model", equipment.model)
    _obj2tag(equipment_elem, "SerialNumber", equipment.serial_number)
    if equipment.installation_date:
        etree.SubElement(equipment_elem, "InstallationDate").text = \
            _format_time(equipment.installation_date)
    if equipment.removal_date:
        etree.SubElement(equipment_elem, "RemovalDate").text = \
            _format_time(equipment.removal_date)
    for calibration_date in equipment.calibration_dates:
        etree.SubElement(equipment_elem, "CalibrationDate").text = \
            _format_time(calibration_date)


def _write_site(parent, site):
    site_elem = etree.SubElement(parent, "Site")
    etree.SubElement(site_elem, "Name").text = site.name
    # Optional tags
    _obj2tag(site_elem, "Description", site.description)
    _obj2tag(site_elem, "Town", site.town)
    _obj2tag(site_elem, "County", site.county)
    _obj2tag(site_elem, "Region", site.region)
    _obj2tag(site_elem, "Country", site.country)


def _write_comment(parent, comment):
    attribs = {}
    if comment.id is not None:
        attribs["id"] = str(comment.id)
    comment_elem = etree.SubElement(parent, "Comment", attribs)
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


def _tag2obj(element, tag, convert):
    # make sure, only unicode
    if convert is str:
        warnings.warn("overriding 'str' with 'unicode'.")
        convert = unicode
    try:
        return convert(element.find(tag).text)
    except:
        None


def _tags2obj(element, tag, convert):
    values = []
    # make sure, only unicode
    if convert is str:
        warnings.warn("overriding 'str' with 'unicode'.")
        convert = unicode
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


def _obj2tag(parent, tag_name, tag_value):
    """
    If tag_value is not None, append a SubElement to the parent. The text of
    the tag will be tag_value.
    """
    if tag_value is None:
        return
    etree.SubElement(parent, tag_name).text = str(tag_value)


def _format_time(value):
    return value.strftime("%Y-%m-%dT%H:%M:%S+00:00")


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
