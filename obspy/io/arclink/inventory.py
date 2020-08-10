# -*- coding: utf-8 -*-
"""
ObsPy implementation for parsing the arclink inventory format
to an Inventory object.

This is a modified version of obspy.io.stationxml and obspy.io.sc3ml.

:author:
    Mathijs Koymans (koymans@knmi.nl), 29.2016 - [Jollyfant@GitHub]
:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""
import inspect
import math
import os
import re
import warnings

from lxml import etree

import obspy
from obspy.core.util.obspy_types import (ComplexWithUncertainties,
                                         FloatWithUncertaintiesAndUnit)
from obspy.core.inventory import (Azimuth, ClockDrift, Dip,
                                  Distance, Frequency, Latitude,
                                  Longitude, SampleRate)
from obspy.core.inventory import (CoefficientsTypeResponseStage,
                                  FilterCoefficient, FIRResponseStage,
                                  PolesZerosResponseStage,
                                  PolynomialResponseStage)

SOFTWARE_MODULE = "ObsPy %s" % obspy.__version__
SOFTWARE_URI = "http://www.obspy.org"
SCHEMA_VERSION = "1.0"
SCHEMA_NAMESPACE = "http://geofon.gfz-potsdam.de/ns/Inventory/1.0/"


def _is_inventory_xml(path_or_file_object):
    """
    Simple function checking if the passed object contains a valid arclink XML
    1.0 file. Returns True of False.
    The test is not exhaustive - it only checks the root tag but that should
    be good enough for most real world use cases. If the schema is used to
    test for a StationXML file, many real world files are false negatives as
    they don't adhere to the standard.

    :param path_or_file_object: File name or file like object.
    """
    if hasattr(path_or_file_object, "tell") and hasattr(path_or_file_object,
                                                        "seek"):
        current_position = path_or_file_object.tell()

    try:
        if isinstance(path_or_file_object, etree._Element):
            xmldoc = path_or_file_object
        else:
            try:
                xmldoc = etree.parse(path_or_file_object)
            except etree.XMLSyntaxError:
                return False
        root = xmldoc.getroot()
        if re.match(r'{http://geofon.gfz-potsdam.de/ns/Inventory/'
                    r'[0-9]*\.?[0-9]+/}', root.tag) is None:
            return False
        # Match and convert schema number to a float to have positive
        # comparisons between, e.g "1" and "1.0".
        version = float(re.findall(r"\d+\.\d+", root.tag)[0])
        if float(version != float(SCHEMA_VERSION)):
            warnings.warn("The inventory file has version %s, ObsPy can "
                          "deal with version %s. Proceed with caution." % (
                              root.attrib["version"], SCHEMA_VERSION))
        return True
    finally:
        # Make sure to reset file pointer position.
        try:
            path_or_file_object.seek(current_position, 0)
        except Exception:
            pass


def validate_arclink_xml(path_or_object):
    """
    Checks if the given path is a valid arclink_xml file.

    Returns a tuple. The first item is a boolean describing if the validation
    was successful or not. The second item is a list of all found validation
    errors, if existent.

    :param path_or_object: File name or file like object. Can also be an etree
        element.
    """
    # Get the schema location.
    schema_location = os.path.dirname(inspect.getfile(inspect.currentframe()))
    schema_location = os.path.join(schema_location, "data",
                                   "arclink_schema.xsd")

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


def _ns(tagname):
    """
    Hoisted namespace function used to find elements

    :param tagname: name of tag to be extracted
    """
    return "{%s}%s" % (SCHEMA_NAMESPACE, tagname)


def _read_inventory_xml(path_or_file_object):
    """
    Function for reading an Arclink inventory file.

    :param path_or_file_object: File name or file like object.
    """
    root = etree.parse(path_or_file_object).getroot()

    created = None
    sender = "ObsPy Inventory"

    # Set source to this script
    source = "Arclink Inventory Import"
    module = None
    module_uri = None

    # Collect all networks from the arcllink inventory
    networks = []
    for net_element in root.findall(_ns("network")):
        networks.append(_read_network(root, net_element))

    return obspy.core.inventory.Inventory(networks=networks, source=source,
                                          sender=sender, created=created,
                                          module=module, module_uri=module_uri)


def _attr2obj(element, attribute, convert):
    """
    Reads text from attribute in element

    :param element: etree element
    :param attribute: name of attribute to be read
    :param convert: intrinsic function (e.g. int, str, float)
    """
    try:
        if element.get(attribute) is None:
            return None
        elif element.get(attribute) == '':
            return None
        return convert(element.get(attribute))
    except Exception:
        None


def _tag2obj(element, tag, convert):
    """
    Reads text from tag in element

    :param element: etree element
    :param tag: name of tag to be read
    :param convert: intrinsic function (e.g. int, str, float)
    """
    try:
        if element.find(tag).text is None:
            return None
        return convert(element.find(tag).text)
    except Exception:
        None


def _read_network(inventory_root, net_element):
    """
    Reads the network structure

    :param inventory_root: base inventory element of document
    :param net_element: network element to be read
    :param _ns: namespace
    """
    # Get the network code as attribute (e.g. <network code="GB">)
    network = obspy.core.inventory.Network(net_element.get("code"))

    # There is no further information in the attributes of <network>
    # Start and end date are included as tags
    network.start_date = _attr2obj(net_element, 'start', obspy.UTCDateTime)
    network.end_date = _attr2obj(net_element, 'end', obspy.UTCDateTime)
    network.description = _attr2obj(net_element, 'description', str)

    # get the restricted_status (boolean)
    # true is evaluated to 'open'; false to 'closed'
    # to match stationXML format
    network.restricted_status = _get_restricted_status(net_element)

    # Collect the stations
    stations = []
    for sta_element in net_element.findall(_ns("station")):
        stations.append(_read_station(inventory_root, sta_element))
    network.stations = stations

    return network


def _get_restricted_status(element):
    """
    get the restricted_status (boolean)
    true is evaluated to 'open' and false to 'closed'
    to match stationXML formatting

    :param element: xmltree element status is extracted from
    """
    restricted_status = _attr2obj(element, "restricted", str)
    if restricted_status == 'false':
        return 'open'
    else:
        return 'closed'


def _read_station(inventory_root, sta_element):
    """
    Reads the station structure

    :param inventory_root: base inventory element of document
    :param sta_element: station element to be read
    """
    # Read location tags
    longitude = _attr2obj(sta_element, "longitude", Longitude)
    latitude = _attr2obj(sta_element, "latitude", Latitude)
    elevation = _attr2obj(sta_element, "elevation", Distance)

    station = obspy.core.inventory.Station(code=sta_element.get("code"),
                                           latitude=latitude,
                                           longitude=longitude,
                                           elevation=elevation)
    station.site = _read_site(sta_element)

    # There is no relevant info in the base node
    # Read the start and end date (creation, termination) from tags
    station.start_date = _attr2obj(sta_element, "start", obspy.UTCDateTime)
    station.end_date = _attr2obj(sta_element, "end", obspy.UTCDateTime)
    station.creation_date = _attr2obj(sta_element, "start", obspy.UTCDateTime)
    station.termination_date = _attr2obj(sta_element, "end", obspy.UTCDateTime)

    # get the restricted_status (boolean)
    # true is evaluated to 'open'; false to 'closed'
    station.restricted_status = _get_restricted_status(sta_element)

    # Get all the channels, format keeps these in <sensorLocation> tags in the
    # station element. Individual channels are contained within <stream> tags
    channels = []
    for sen_loc_element in sta_element.findall(_ns("sensorLocation")):
        for channel in sen_loc_element.findall(_ns("stream")):
            channels.append(_read_channel(inventory_root, channel))

    station.channels = channels

    return station


def _read_site(sta_element):
    """
    Reads site information from the station element tags
    and region from network element

    In arclinkXML, site information are included as
    tags in the station_element

    :param sta_element: station element
    """
    # The region is defined in the parent network element
    net_element = sta_element.getparent()
    region = _attr2obj(net_element, "region", str)

    # The country, place, description are given in the
    # station element
    country = _attr2obj(sta_element, "country", str)
    place = _attr2obj(sta_element, "place", str)
    description = _attr2obj(sta_element, "description", str)

    # The name is usually the description
    name = description

    return obspy.core.inventory.Site(name=name, description=description,
                                     town=place, county=None, region=region,
                                     country=country)


def _read_datalogger(equip_element):
    """
    Reads equipment information from datalogger
    Some information is not present > to None

    :param equip_element: element to be parsed
    """
    resource_id = equip_element.get("publicID")
    description = _attr2obj(equip_element, "description", str)
    manufacturer = _attr2obj(equip_element, "digitizerManufacturer", str)
    model = _attr2obj(equip_element, "digitizerModel", str)

    # A lot of properties are not specified in the ArclinkXML
    return obspy.core.inventory.Equipment(
        resource_id=resource_id, type=model, description=description,
        manufacturer=manufacturer, vendor=None, model=model,
        serial_number=None, installation_date=None,
        removal_date=None, calibration_dates=None)


def _read_sensor(equip_element):
    """
    Reads equipment information from element
    Some information is not present > to None

    :param equip_element: element to be parsed
    """
    # try to read some element tags, most are missing anyway
    resource_id = equip_element.get("publicID")
    equipment_type = _attr2obj(equip_element, "type", str)
    description = _attr2obj(equip_element, "description", str)
    manufacturer = _attr2obj(equip_element, "manufacturer", str)
    model = _attr2obj(equip_element, "model", str)

    # A lot of properties are not specified in the ArclinkXML
    return obspy.core.inventory.Equipment(
        resource_id=resource_id, type=equipment_type, description=description,
        manufacturer=manufacturer, vendor=None, model=model,
        serial_number=None, installation_date=None,
        removal_date=None, calibration_dates=None)


def _read_channel(inventory_root, cha_element):
    """
    reads channel element from arclinkXML format

    :param inventory_root: root of the XML document
    :param cha_element: channel element to be parsed
    """
    code = cha_element.get("code")

    # Information is also kept within the parent <sensorLocation> element
    sen_loc_element = cha_element.getparent()
    location_code = sen_loc_element.get("code")

    # get site info from the <sensorLocation> element
    longitude = _attr2obj(sen_loc_element, "longitude", Longitude)
    latitude = _attr2obj(sen_loc_element, "latitude", Latitude)
    elevation = _attr2obj(sen_loc_element, "elevation", Distance)
    depth = _attr2obj(cha_element, "depth", Distance)

    channel = obspy.core.inventory.Channel(
        code=code, location_code=location_code, latitude=latitude,
        longitude=longitude, elevation=elevation, depth=depth)

    # obtain the sensorID and link to particular publicID <sensor> element
    # in the inventory base node
    sensor_id = cha_element.get("sensor")
    sensor_element = inventory_root.find(_ns("sensor[@publicID='" +
                                         sensor_id + "']"))
    # obtain the poles and zeros responseID and link to particular
    # <responsePAZ> publicID element in the inventory base node
    if sensor_element is not None:
        response_id = sensor_element.get("response")
        if response_id is not None:
            # Fix #2552
            resp_type = response_id.replace("#", "/").split("/")[0]
            if resp_type == 'ResponsePAZ':
                search = "responsePAZ[@publicID='" + response_id + "']"
                response_element = inventory_root.find(_ns(search))
            elif resp_type == 'ResponsePolynomial':
                search = "responsePolynomial[@publicID='" + response_id + "']"
                response_element = inventory_root.find(_ns(search))
        else:
            response_element = None
    else:
        response_element = None

    # obtain the dataloggerID and link to particular <responsePAZ> publicID
    # element in the inventory base node
    datalogger_id = cha_element.get("datalogger")
    search = "datalogger[@publicID='" + datalogger_id + "']"
    data_log_element = inventory_root.find(_ns(search))

    channel.restricted_status = _get_restricted_status(cha_element)

    # There is no further information in the attributes of <stream>
    # Start and end date are included as tags instead
    channel.start_date = _attr2obj(cha_element, "start", obspy.UTCDateTime)
    channel.end_date = _attr2obj(cha_element, "end", obspy.UTCDateTime)

    # Determine sample rate (given is a numerator, denominator)
    # Assuming numerator is # samples and denominator is # seconds
    numerator = _attr2obj(cha_element, "sampleRateNumerator", int)
    denominator = _attr2obj(cha_element, "sampleRateDenominator", int)

    rate = numerator / denominator

    channel.sample_rate_ratio_number_samples = numerator
    channel.sample_rate_ratio_number_seconds = denominator
    channel.sample_rate = _read_float_var(rate, SampleRate)

    if sensor_element is not None:
        channel.sensor = _read_sensor(sensor_element)
    if data_log_element is not None:
        channel.data_logger = _read_datalogger(data_log_element)
        temp = _attr2obj(data_log_element, "maxClockDrift",
                         ClockDrift)
        if channel.sample_rate != 0.0:
            channel.clock_drift_in_seconds_per_sample = \
                _read_float_var(temp / channel.sample_rate, ClockDrift)
        else:
            msg = "Clock drift division by sample rate of 0: using sec/sample"
            warnings.warn(msg)
            channel.sample_rate = temp

    channel.azimuth = _attr2obj(cha_element, "azimuth", Azimuth)
    channel.dip = _attr2obj(cha_element, "dip", Dip)
    # storage format of channel not supported by StationXML1.1 anymore, keep it
    # as a foreign tag to be nice if anybody needs to access it
    channel.extra = {'format': {
        'value': _tag2obj(cha_element, _ns("format"), str),
        'namespace': SCHEMA_NAMESPACE}}

    if channel.sample_rate == 0.0:
        msg = "Something went hopelessly wrong, found sampling-rate of 0!"
        warnings.warn(msg)

    # Begin to collect digital/analogue filter chains
    # This information is stored as an array in the datalogger element
    response_fir_id = []
    response_paz_id = []
    if data_log_element is not None:
        # Find the decimation element with a particular num/denom
        decim_element = data_log_element.find(_ns(
            "decimation[@sampleRateDenominator='" +
            str(int(denominator)) + "'][@sampleRateNumerator='" +
            str(int(numerator)) + "']"))
        analogue_filter_chain = _tag2obj(decim_element,
                                         _ns("analogueFilterChain"), str)
        if analogue_filter_chain is not None:
            response_paz_id = analogue_filter_chain.split(" ")
        digital_filter_chain = _tag2obj(decim_element,
                                        _ns("digitalFilterChain"), str)
        if digital_filter_chain is not None:
            response_fir_id = digital_filter_chain.split(" ")

    channel.response = _read_response(inventory_root, sensor_element,
                                      response_element, cha_element,
                                      data_log_element,
                                      channel.sample_rate,
                                      response_fir_id, response_paz_id)

    return channel


def _read_instrument_sensitivity(sen_element, cha_element):
    """
    reads the instrument sensitivity (gain) from the sensor and channel element

    :param sen_element: sensor element to be parsed
    :param cha_element: channel element to be parsed
    """
    # Read the gain and gain frequency from attributs
    gain = _attr2obj(cha_element, "gain", float)
    frequency = _attr2obj(cha_element, "gainFrequency", float)

    # Get input units and hardcode output units to counts
    input_units_name = _attr2obj(sen_element, "unit", str)
    output_units_name = 'COUNTS'

    sensitivity = obspy.core.inventory.response.InstrumentSensitivity(
        value=gain, frequency=frequency,
        input_units=input_units_name,
        output_units=output_units_name)

    # assuming these are equal to frequencyStart/frequencyEnd
    sensitivity.frequency_range_start = _attr2obj(sen_element, "lowFrequency",
                                                  float)
    sensitivity.frequency_range_end = _attr2obj(sen_element, "highFrequency",
                                                float)

    return sensitivity


def _read_response(root, sen_element, resp_element, cha_element,
                   data_log_element, samp_rate, fir, analogue):
    """
    reads response from arclinkXML format

    :param root: XML document root element
    :param sen_element: sensor element to be used
    :param resp_element: response element to be parsed
    :param cha_element: channel element to be used
    :param data_log_element: datalogger element to be used
    :param samp_rate: sample rate of stream
    :param fir: list of FIR filter chain identifiers
    :param analogue: list of analogue filter chain identifiers
    """
    response = obspy.core.inventory.response.Response()
    response.instrument_sensitivity = _read_instrument_sensitivity(
        sen_element, cha_element)

    if resp_element is None:
        return response

    # The sampling rate is not given per fir filter as in stationXML
    # We are only given a decimation factor per stage, therefore we are
    # required to reconstruct the sampling rates at a given stage from
    # this chain of factors

    # start with the final sampling_rate after all stages are applied
    # invert the fir stages to reverse engineer (backwards) the sample rate
    # during any fir stage

    samp_rate = float(samp_rate)
    fir_stage_rates = []
    if len(fir):
        fir = fir[::-1]
        for fir_id in fir:
            # get the particular fir stage decimation factor
            # multiply the decimated sample rate by this factor
            search = "responseFIR[@publicID='" + fir_id + "']"
            fir_element = root.find(_ns(search))
            if fir_element is None:
                continue
            dec_fac = _attr2obj(fir_element, "decimationFactor", int)
            if dec_fac is not None and int(dec_fac) != 0:
                samp_rate *= dec_fac
            fir_stage_rates.append(float(samp_rate))

    # Return filter chain to original and also revert the rates
    fir = fir[::-1]
    fir_stage_rates = fir_stage_rates[::-1]

    # Attempt to read stages in the proper order
    # arclinkXML does not group stages by an ID
    # We are required to do stage counting ourselves

    stage = 1
    # Get the sensor units, default to M/S
    sensor_units = _attr2obj(sen_element, "unit", str)
    if sensor_units is None:
        msg = "Sensor unit not set, assuming M/S"
        warnings.warn(msg)
        sensor_units = "M/S"

    # Get the first PAZ stage
    # Input unit: M/S or M/S**2
    # Output unit: V
    if resp_element is not None:
        paz_response = _read_response_stage(resp_element, samp_rate,
                                            stage, sensor_units, 'V')
        if paz_response is not None:
            response.response_stages.append(paz_response)
            stage += 1

    # Apply analogue filter stages (if any)
    # Input unit: V
    # Output unit: V
    if len(analogue):
        for analogue_id in analogue:
            search = "responsePAZ[@publicID='" + analogue_id + "']"
            analogue_element = root.find(_ns(search))
            if analogue_element is None:
                msg = ('Analogue responsePAZ not in inventory:'
                       '%s, stopping before stage %i') % (analogue_id, stage)
                warnings.warn(msg)
                return response
            analogue_response = _read_response_stage(analogue_element,
                                                     samp_rate, stage, 'V',
                                                     'V')
            if analogue_response is not None:
                response.response_stages.append(analogue_response)
                stage += 1

    # Apply datalogger (digitizer)
    # Input unit: V
    # Output unit: COUNTS
    if data_log_element is not None:
        coeff_response = _read_response_stage(data_log_element,
                                              samp_rate, stage, 'V',
                                              'COUNTS')
        if coeff_response is not None:
            response.response_stages.append(coeff_response)
            stage += 1

    # Apply final digital filter stages
    # Input unit: COUNTS
    # Output unit: COUNTS
    for fir_id, rate in zip(fir, fir_stage_rates):
        search = "responseFIR[@publicID='" + fir_id + "']"
        stage_element = root.find(_ns(search))
        if stage_element is None:
            msg = ("fir response not in inventory: %s, stopping correction"
                   "before stage %i") % (fir_id, stage)
            warnings.warn(msg)
            return response
        fir_response = _read_response_stage(stage_element, rate, stage,
                                            'COUNTS', 'COUNTS')
        if fir_response is not None:
            response.response_stages.append(fir_response)
            stage += 1
    return response


def _read_response_stage(stage, rate, stage_number, input_units,
                         output_units):
    """
    Private function to read a response stage

    :param stage: response stage element
    :param rate: stage sample rate
    :param stage_number: response stage number
    :param input_units: input units of stage
    :param output_units output units of stage
    """
    elem_type = stage.tag.split("}")[1]

    stage_sequence_number = stage_number

    # Obtain the stage gain and frequency
    # Default to a gain of 0 and frequency of 0 if missing
    stage_gain = _attr2obj(stage, "gain", float) or 0
    stage_gain_frequency = _attr2obj(stage, "gainFrequency", float) or 0.0

    name = _attr2obj(stage, "name", str)
    resource_id = _attr2obj(stage, "publicID", str)

    # Determine the decimation parameters
    # This is dependent on the type of stage
    # Decimation delay/correction need to be normalized
    if elem_type == "responseFIR":
        decimation_factor = _attr2obj(stage, "decimationFactor", int)
        if rate != 0.0:
            temp = _attr2obj(stage, "delay", float) / rate
            decimation_delay = _read_float_var(temp,
                                               FloatWithUncertaintiesAndUnit,
                                               unit=True)
            temp = _attr2obj(stage, "correction", float) / rate
            decimation_corr = _read_float_var(temp,
                                              FloatWithUncertaintiesAndUnit,
                                              unit=True)
        else:
            decimation_delay = _read_float_var("inf",
                                               FloatWithUncertaintiesAndUnit,
                                               unit=True)
            decimation_corr = _read_float_var("inf",
                                              FloatWithUncertaintiesAndUnit,
                                              unit=True)
        decimation_input_sample_rate = \
            _read_float_var(rate, Frequency)
        decimation_offset = int(0)
    elif elem_type == "datalogger":
        decimation_factor = int(1)
        decimation_delay = _read_float_var(0.00,
                                           FloatWithUncertaintiesAndUnit,
                                           unit=True)
        decimation_corr = _read_float_var(0.00,
                                          FloatWithUncertaintiesAndUnit,
                                          unit=True)
        decimation_input_sample_rate = \
            _read_float_var(rate, Frequency)
        decimation_offset = int(0)
    elif elem_type == "responsePAZ" or elem_type == "responsePolynomial":
        decimation_factor = None
        decimation_delay = None
        decimation_corr = None
        decimation_input_sample_rate = None
        decimation_offset = None
    else:
        raise ValueError("Unknown type of response: " + str(elem_type))

    # set up list of for this stage arguments
    kwargs = {
        "stage_sequence_number": stage_sequence_number,
        "input_units": str(input_units),
        "output_units": str(output_units),
        "input_units_description": None,
        "output_units_description": None,
        "resource_id": None,
        "resource_id2": resource_id,
        "stage_gain": stage_gain,
        "stage_gain_frequency": stage_gain_frequency,
        "name": name,
        "description": None,
        "decimation_input_sample_rate": decimation_input_sample_rate,
        "decimation_factor": decimation_factor,
        "decimation_offset": decimation_offset,
        "decimation_delay": decimation_delay,
        "decimation_correction": decimation_corr
    }

    # Different processing for different types of responses
    # currently supported:
    # PAZ, COEFF, FIR, Polynomial response is not supported;
    # could not find example
    if elem_type == 'responsePAZ':

        # read normalization params
        normalization_freq = _attr2obj(stage, "normalizationFrequency",
                                       Frequency)
        normalization_factor = _attr2obj(stage, "normalizationFactor",
                                         float)

        # Parse the type of the transfer function
        # A: Laplace (rad)
        # B: Laplace (Hz)
        # D: digital (z-transform)
        pz_transfer_function_type = _attr2obj(stage, "type", str)
        if pz_transfer_function_type == 'A':
            pz_transfer_function_type = 'LAPLACE (RADIANS/SECOND)'
        elif pz_transfer_function_type == 'B':
            pz_transfer_function_type = 'LAPLACE (HERTZ)'
        elif pz_transfer_function_type == 'D':
            pz_transfer_function_type = 'DIGITAL (Z-TRANSFORM)'
        else:
            msg = ("Unknown transfer function code %s. Defaulting to Laplace"
                   "(rad)") % pz_transfer_function_type
            warnings.warn(msg)
            pz_transfer_function_type = 'LAPLACE (RADIANS/SECOND)'

        # Parse string of poles and zeros
        # paz are stored as a string in arclinkXML
        # e.g. (-0.01234,0.01234) (-0.01234,-0.01234)
        zeros_array = stage.find(_ns("zeros")).text
        poles_array = stage.find(_ns("poles")).text
        if zeros_array is not None:
            zeros_array = zeros_array.split(" ")
        else:
            zeros_array = []
        if poles_array is not None:
            poles_array = poles_array.split(" ")
        else:
            poles_array = []

        # Keep counter for pole/zero number
        cnt = 0
        poles = []
        zeros = []
        for el in poles_array:
            poles.append(_tag2pole_or_zero(el, cnt))
            cnt += 1
        for el in zeros_array:
            zeros.append(_tag2pole_or_zero(el, cnt))
            cnt += 1

        # Return the paz response
        return PolesZerosResponseStage(
            pz_transfer_function_type=pz_transfer_function_type,
            normalization_frequency=normalization_freq,
            normalization_factor=normalization_factor, zeros=zeros,
            poles=poles, **kwargs)

    elif elem_type == 'datalogger':
        cf_transfer_function_type = "DIGITAL"
        numerator = []
        denominator = []
        return CoefficientsTypeResponseStage(
            cf_transfer_function_type=cf_transfer_function_type,
            numerator=numerator, denominator=denominator, **kwargs)

    elif elem_type == 'responsePolynomial':
        raise NotImplementedError("responsePolynomial not"
                                  "implemented. Contact the ObsPy developers")
        # Polynomial response (UNTESTED)
        # Currently not implemented in ObsPy (20-11-2015)
        f_low = None
        f_high = None
        max_err = None
        appr_type = _attr2obj(stage, "approximationType", str)
        appr_low = _attr2obj(stage, "approximationLowerBound", float)
        appr_high = _attr2obj(stage, "approximationUpperBound", float)
        coeffs_str = _tag2obj(stage, _ns("coefficients"), str)
        if coeffs_str is not None:
            coeffs = coeffs_str.split(" ")
            coeffs_float = []
            i = 0
            # pass additional mapping of coefficient counter
            # so that a proper stationXML can be formatted
            for c in coeffs:
                temp = _read_float_var(c, FilterCoefficient,
                                       additional_mapping={str("number"): i})
                coeffs_float.append(temp)
                i += 1

        return PolynomialResponseStage(
            approximation_type=appr_type, frequency_lower_bound=f_low,
            frequency_upper_bound=f_high, approximation_lower_bound=appr_low,
            approximation_upper_bound=appr_high, maximum_error=max_err,
            coefficients=coeffs, **kwargs)

    elif elem_type == 'responseFIR':
        # For the responseFIR obtain the symmetry and
        # list of coefficients

        coeffs_str = _tag2obj(stage, _ns("coefficients"), str)
        coeffs_float = []
        if coeffs_str is not None and coeffs_str != 'None':
            coeffs = coeffs_str.split(" ")
            i = 0
            # pass additional mapping of coefficient counter
            # so that a proper stationXML can be formatted
            for c in coeffs:
                temp = _read_float_var(c, FilterCoefficient,
                                       additional_mapping={str("number"): i})
                coeffs_float.append(temp)
                i += 1

        # Write the FIR symmetry to what ObsPy expects
        # A = NONE, B = ODD, C = EVEN
        symmetry = _attr2obj(stage, "symmetry", str)
        if symmetry == 'A':
            symmetry = 'NONE'
        elif symmetry == 'B':
            symmetry = 'ODD'
        elif symmetry == 'C':
            symmetry = 'EVEN'
        else:
            raise ValueError('Unknown symmetry metric; expected A, B, or C')

        return FIRResponseStage(
            coefficients=coeffs_float, symmetry=symmetry, **kwargs)

    else:
        raise NotImplementedError


def _tag2pole_or_zero(paz_element, count):
    """
    Parses arclinkXML paz format
    Uncertainties on poles removed, not present in fo
    Always put to None so no internal conflict
    The sanitization removes the first/last parenthesis
    and split by comma, real part is 1st, imaginary 2nd

    :param paz_element: string of poles or zeros e.g. (12320, 23020)
    :param count: sequential numbering of poles/zeros
    """
    paz_element = paz_element[1:-1]
    paz_element = paz_element.split(",")

    real = float(paz_element[0])
    imag = float(paz_element[1])

    if real is not None or imag is not None:
        real = real or 0
        imag = imag or 0
    x = ComplexWithUncertainties(real, imag)
    x.upper_uncertainty = None
    x.upper_uncertainty = None
    x.number = count

    return x


def _read_float_var(elem, cls, unit=False, datum=False, additional_mapping={}):
    """
    function to read floattype to cls object (based on _read_floattype)
    normally ObsPy would read this directly from a tag, but with different
    tag names this is no longer possible; instead we just pass the value
    and not the tag name. We always set the unit/datum/uncertainties to None
    because they are not provided.

    :param elem: float value to be converted
    :param cls: obspy.core.inventory class
    """
    try:
        convert = float(elem)
    except (ValueError, TypeError):
        warnings.warn(
            "Encountered a value '%s' which could not be converted to a "
            "float. Will be skipped. Please contact to report this "
            "issue." % elem,
            UserWarning)
        return None

    if math.isnan(convert):
        warnings.warn("'%s' has a value of NaN. It will be skipped." %
                      elem, UserWarning)
        return None

    obj = cls(convert)
    if unit:
        obj.unit = None
    if datum:
        obj.datum = None
    obj.lower_uncertainty = None
    obj.upper_uncertainty = None
    for key1, key2 in additional_mapping.items():
        setattr(obj, key1, key2)
    return obj
