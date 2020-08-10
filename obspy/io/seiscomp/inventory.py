# -*- coding: utf-8 -*-
"""
ObsPy implementation for parsing the sc3ml format to an Inventory object.

This is a modified version of obspy.io.stationxml.

:author:
    Mathijs Koymans (koymans@knmi.nl), 11.2015 - [Jollyfant@GitHub]
:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""
import math
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
from obspy.io.stationxml.core import _read_floattype


SOFTWARE_MODULE = "ObsPy %s" % obspy.__version__
SOFTWARE_URI = "http://www.obspy.org"
SCHEMA_VERSION = ['0.5', '0.6', '0.7', '0.8', '0.9']
SCHEMA_NAMESPACE_BASE = "http://geofon.gfz-potsdam.de/ns/seiscomp3-schema"


def _get_schema_namespace(version_string):
    """
    >>> print(_get_schema_namespace('0.9'))
    http://geofon.gfz-potsdam.de/ns/seiscomp3-schema/0.9
    >>> print(_get_schema_namespace('0.6'))
    http://geofon.gfz-potsdam.de/ns/seiscomp3-schema/0.6
    """
    namespace = "%s/%s" % (SCHEMA_NAMESPACE_BASE, version_string)
    return namespace


def _count_complex(complex_string):
    """
    Returns number of complex numbers in string (formatted according to
    SeisComp3 XML schema type "ComplexArray"). Raises an Exception if string
    seems invalid.
    """
    counts = set()
    for char in '(,)':
        counts.add(complex_string.count(char))
    if len(counts) != 1:
        msg = ("Invalid string for list of complex numbers:"
               "\n'%s'") % complex_string
        raise ValueError(msg)
    return counts.pop()


def _parse_list_of_complex_string(complex_string):
    """
    Returns a list of complex numbers, parsed from a string (formatted
    according to SeisComp3 XML schema type "ComplexArray").
    """
    count = _count_complex(complex_string)
    numbers = re.findall(r'\(\s*([^,\s]+)\s*,\s*([^)\s]+)\s*\)',
                         complex_string)
    if len(numbers) != count:
        msg = ("Unexpected count of complex numbers parsed from string:"
               "\n  Raw string: '%s'\n  Expected count of complex numbers: %s"
               "\n  Parsed complex numbers: %s") % (complex_string, count,
                                                    numbers)
        raise ValueError(msg)
    return numbers


def _read_sc3ml(path_or_file_object):
    """
    Function for reading a stationXML file.

    :param path_or_file_object: File name or file like object.
    """
    root = etree.parse(path_or_file_object).getroot()

    # Code can be used for version 0.7, 0.8, and 0.9
    for version in SCHEMA_VERSION:
        namespace = _get_schema_namespace(version)
        if root.find("{%s}%s" % (namespace, "Inventory")) is not None:
            break
    else:
        raise ValueError("Schema version not supported.")

    def _ns(tagname):
        return "{%s}%s" % (namespace, tagname)

    # This needs to be tested, did not find an inventory
    # with the journal entry.
    journal = root.find(_ns("Journaling"))
    if journal is not None:
        entry = journal.find(_ns("entry"))
        if entry is not None:
            created = _tag2obj(entry, _ns("created"), obspy.UTCDateTime)
            sender = _tag2obj(entry, _ns("sender"), str)
    else:
        created = None
        sender = "ObsPy Inventory"

    # Set source to this script
    source = "sc3ml import"
    module = None
    module_uri = None

    # Find the inventory root element. (Only finds the first. We expect only
    # one, so any more than that will be ignored.)
    inv_element = root.find(_ns("Inventory"))

    # Pre-generate a dictionary of the sensors, dataloggers and responses to
    # avoid costly linear search when parsing network nodes later.
    # Register sensors
    sensors = {}
    for sensor_element in inv_element.findall(_ns("sensor")):
        public_id = sensor_element.get("publicID")
        if public_id:
            if public_id in sensors:
                msg = ("Found multiple matching sensor tags with the same "
                       "publicID '{}'.".format(public_id))
                raise obspy.ObsPyException(msg)
            else:
                sensors[public_id] = sensor_element
    # Register dataloggers
    dataloggers = {}
    for datalogger_element in inv_element.findall(_ns("datalogger")):
        public_id = datalogger_element.get("publicID")
        if public_id:
            if public_id in dataloggers:
                msg = ("Found multiple matching datalogger tags with the same "
                       "publicID '{}'.".format(public_id))
                raise obspy.ObsPyException(msg)
            else:
                dataloggers[public_id] = datalogger_element
    # Register reponses
    responses = {
        "responsePAZ": {},
        "responsePolynomial": {},
        "responseFIR": {},
        "responseIIR": {}
    }
    for response_type, all_elements in responses.items():
        for response_element in inv_element.findall(_ns(response_type)):
            public_id = response_element.get("publicID")
            if public_id:
                if public_id in all_elements:
                    msg = ("Found multiple matching {} tags with the same "
                           "publicID '{}'.".format(response_type, public_id))
                    raise obspy.ObsPyException(msg)
                else:
                    all_elements[public_id] = response_element
    # Organize all the collection instrument information into a unified
    # intrumentation register.
    instrumentation_register = {
        "sensors": sensors,
        "dataloggers": dataloggers,
        "responses": responses
    }

    # Collect all networks from the sc3ml inventory
    networks = []
    for net_element in inv_element.findall(_ns("network")):
        networks.append(_read_network(instrumentation_register,
                                      net_element, _ns))

    return obspy.core.inventory.Inventory(networks=networks, source=source,
                                          sender=sender, created=created,
                                          module=module, module_uri=module_uri)


def _tag2obj(element, tag, convert):

    """
    Reads text from tag in element

    :param element: etree element
    :param tag: name of tag to be read
    :param convert: intrinsic function (e.g. int, str, float)
    """
    try:
        # Single closing tags e.g. <analogueFilterChain/>.text return None
        # and will be converted to a string 'None' when convert is str
        found_tag_text = element.find(tag).text
        if found_tag_text is None:
            return None
        return convert(found_tag_text)
    except Exception:
        None


def _read_network(instrumentation_register, net_element, _ns):

    """
    Reads the network structure

    :param instrumentation_register: register of instrumentation metadata
    :param net_element: network element to be read
    :param _ns: namespace
    """

    # Get the network code as attribute (e.g. <network code="GB">)
    network = obspy.core.inventory.Network(net_element.get("code"))

    # There is no further information in the attributes of <network>
    # Start and end date are included as tags
    network.start_date = _tag2obj(net_element, _ns("start"), obspy.UTCDateTime)
    network.end_date = _tag2obj(net_element, _ns("end"), obspy.UTCDateTime)
    network.description = _tag2obj(net_element, _ns("description"), str)

    # get the restricted_status (boolean)
    # true is evaluated to 'open'; false to 'closed'
    # to match stationXML format
    network.restricted_status = _get_restricted_status(net_element, _ns)

    # Collect the stations
    stations = []
    for sta_element in net_element.findall(_ns("station")):
        stations.append(_read_station(instrumentation_register,
                                      sta_element, _ns))
    network.stations = stations

    return network


def _get_restricted_status(element, _ns):

    """
    get the restricted_status (boolean)
    true is evaluated to 'open' and false to 'closed'
    to match stationXML formatting
    """

    restricted_status = _tag2obj(element, _ns("restricted"), str)
    if(restricted_status == 'false'):
        return 'open'
    else:
        return 'closed'


def _read_station(instrumentation_register, sta_element, _ns):

    """
    Reads the station structure

    :param instrumentation_register: register of instrumentation metadata
    :param sta_element: station element to be read
    :param _ns: name space
    """

    # Read location tags
    longitude = _read_floattype(sta_element, _ns("longitude"), Longitude,
                                datum=True)
    latitude = _read_floattype(sta_element, _ns("latitude"), Latitude,
                               datum=True)
    elevation = _read_floattype(sta_element, _ns("elevation"), Distance,
                                unit=True)
    station = obspy.core.inventory.Station(code=sta_element.get("code"),
                                           latitude=latitude,
                                           longitude=longitude,
                                           elevation=elevation)
    station.site = _read_site(sta_element, _ns)

    # There is no relevant info in the base node
    # Read the start and end date (creation, termination) from tags
    # "Vault" and "Geology" are not defined in sc3ml ?
    station.start_date = _tag2obj(sta_element, _ns("start"), obspy.UTCDateTime)
    station.end_date = _tag2obj(sta_element, _ns("end"), obspy.UTCDateTime)
    station.creation_date = _tag2obj(sta_element, _ns("start"),
                                     obspy.UTCDateTime)
    station.termination_date = _tag2obj(sta_element, _ns("end"),
                                        obspy.UTCDateTime)

    # get the restricted_status (boolean)
    # true is evaluated to 'open'; false to 'closed'
    station.restricted_status = _get_restricted_status(sta_element, _ns)

    # Get all the channels, sc3ml keeps these in <sensorLocation> tags in the
    # station element. Individual channels are contained within <stream> tags
    channels = []
    for sen_loc_element in sta_element.findall(_ns("sensorLocation")):
        for channel in sen_loc_element.findall(_ns("stream")):
            channels.append(_read_channel(instrumentation_register,
                                          channel, _ns))

    station.channels = channels

    return station


def _read_site(sta_element, _ns):

    """
    Reads site information from the station element tags
    and region from network element

    In sc3ml, site information are included as
    tags in the station_element

    :param sta_element: station element
    :param _ns: namespace
    """

    # The region is defined in the parent network element
    net_element = sta_element.getparent()
    region = _tag2obj(net_element, _ns("region"), str)

    # The country, place, description are given in the
    # station element
    country = _tag2obj(sta_element, _ns("country"), str)
    place = _tag2obj(sta_element, _ns("place"), str)
    description = _tag2obj(sta_element, _ns("description"), str)

    # The name is usually the description
    name = description

    return obspy.core.inventory.Site(name=name, description=None,
                                     town=place, county=None, region=region,
                                     country=country)


def _read_datalogger(equip_element, _ns):

    """
    Reads equipment information from datalogger
    Some information is not present > to None

    :param data_log_element: element to be parsed
    :param _ns: name space
    """

    resource_id = equip_element.get("publicID")
    description = _tag2obj(equip_element, _ns("description"), str)
    manufacturer = _tag2obj(equip_element, _ns("digitizerManufacturer"), str)
    model = _tag2obj(equip_element, _ns("digitizerModel"), str)

    return obspy.core.inventory.Equipment(
        resource_id=resource_id, type=model, description=description,
        manufacturer=manufacturer, vendor=None, model=model,
        serial_number=None, installation_date=None,
        removal_date=None, calibration_dates=None)


def _read_sensor(equip_element, _ns):

    """
    Reads equipment information from element
    Some information is not present > to None

    :param equip_element: element to be parsed
    :param _ns: name space
    """

    # try to read some element tags, most are missing anyway
    resource_id = equip_element.get("publicID")
    equipment_type = _tag2obj(equip_element, _ns("type"), str)
    description = _tag2obj(equip_element, _ns("description"), str)
    manufacturer = _tag2obj(equip_element, _ns("manufacturer"), str)
    model = _tag2obj(equip_element, _ns("model"), str)
    return obspy.core.inventory.Equipment(
        resource_id=resource_id, type=equipment_type, description=description,
        manufacturer=manufacturer, vendor=None, model=model,
        serial_number=None, installation_date=None,
        removal_date=None, calibration_dates=None)


def _read_channel(instrumentation_register, cha_element, _ns):

    """
    reads channel element from sc3ml format

    :param instrumentation_register: register of instrumentation metadata
    :param cha_element: channel element
    :param _ns: namespace
    """

    code = cha_element.get("code")

    # Information is also kept within the parent <sensorLocation> element
    sen_loc_element = cha_element.getparent()
    location_code = sen_loc_element.get("code")

    # get site info from the <sensorLocation> element
    longitude = _read_floattype(sen_loc_element, _ns("longitude"), Longitude,
                                datum=True)
    latitude = _read_floattype(sen_loc_element, _ns("latitude"), Latitude,
                               datum=True)
    elevation = _read_floattype(sen_loc_element, _ns("elevation"), Distance,
                                unit=True)
    depth = _read_floattype(cha_element, _ns("depth"), Distance,
                            unit=True)

    # Set values to 0 if they are is missing (see #1816)
    if longitude is None:
        msg = "Sensor is missing longitude information, using 0.0"
        warnings.warn(msg)
        longitude = 0
    if latitude is None:
        msg = "Sensor is missing latitude information, using 0.0"
        warnings.warn(msg)
        latitude = 0
    if elevation is None:
        msg = "Sensor is missing elevation information, using 0.0"
        warnings.warn(msg)
        elevation = 0
    if depth is None:
        msg = "Channel is missing depth information, using 0.0"
        warnings.warn(msg)
        depth = 0

    channel = obspy.core.inventory.Channel(
        code=code, location_code=location_code, latitude=latitude,
        longitude=longitude, elevation=elevation, depth=depth)

    # obtain the sensorID and link to particular publicID <sensor> element
    # in the inventory base node
    sensor_id = cha_element.get("sensor")
    sensor_element = instrumentation_register["sensors"].get(sensor_id)

    # obtain the poles and zeros responseID and link to particular
    # <responsePAZ> publicID element in the inventory base node
    if (sensor_element is not None and
       sensor_element.get("response") is not None):

        response_id = sensor_element.get("response")
        response_elements = []

        for resp_element in instrumentation_register["responses"].values():
            found_response = resp_element.get(response_id)
            if found_response is not None:
                response_elements.append(found_response)

        if len(response_elements) == 0:
            msg = ("Could not find response tag with public ID "
                   "'{}'.".format(response_id))
            raise obspy.ObsPyException(msg)
        elif len(response_elements) > 1:
            msg = ("Found multiple matching response tags with the same "
                   "public ID '{}'.".format(response_id))
            raise obspy.ObsPyException(msg)
        response_element = response_elements[0]
    else:
        response_element = None

    # obtain the dataloggerID and link to particular <responsePAZ> publicID
    # element in the inventory base node
    datalogger_id = cha_element.get("datalogger")
    data_log_element = \
        instrumentation_register["dataloggers"].get(datalogger_id)

    channel.restricted_status = _get_restricted_status(cha_element, _ns)

    # There is no further information in the attributes of <stream>
    # Start and end date are included as tags instead
    channel.start_date = _tag2obj(cha_element, _ns("start"), obspy.UTCDateTime)
    channel.end_date = _tag2obj(cha_element, _ns("end"), obspy.UTCDateTime)

    # Determine sample rate (given is a numerator, denominator)
    # Assuming numerator is # samples and denominator is # seconds
    numerator = _tag2obj(cha_element, _ns("sampleRateNumerator"), int)
    denominator = _tag2obj(cha_element, _ns("sampleRateDenominator"), int)

    # If numerator is non-zero and denominator zero, will raise
    # ZeroDivisionError.
    rate = numerator / denominator if numerator != 0 else 0

    channel.sample_rate_ratio_number_samples = numerator
    channel.sample_rate_ratio_number_seconds = denominator
    channel.sample_rate = _read_float_var(rate, SampleRate)

    if sensor_element is not None:
        channel.sensor = _read_sensor(sensor_element, _ns)
    if data_log_element is not None:
        channel.data_logger = _read_datalogger(data_log_element, _ns)
        temp = _read_floattype(data_log_element, _ns("maxClockDrift"),
                               ClockDrift)
        if temp is not None:
            if channel.sample_rate != 0.0:
                channel.clock_drift_in_seconds_per_sample = \
                    _read_float_var(temp / channel.sample_rate, ClockDrift)
            else:
                msg = "Clock drift division by sample rate of 0: " \
                      "using sec/sample"
                warnings.warn(msg)
                channel.sample_rate = temp

    channel.azimuth = _read_floattype(cha_element, _ns("azimuth"), Azimuth)
    channel.dip = _read_floattype(cha_element, _ns("dip"), Dip)
    match = re.search(r'{([^}]*)}', cha_element.tag)
    if match:
        namespace = match.group(1)
    else:
        namespace = _get_schema_namespace('0.9')
    channel.extra = {'format': {
        'value': _tag2obj(cha_element, _ns("format"), str),
        # storage format of channel not supported by StationXML1.1 anymore,
        # keep it as a foreign tag to be nice if anybody needs to access it
        'namespace': namespace}}

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

    channel.response = _read_response(instrumentation_register['responses'],
                                      sensor_element, response_element,
                                      cha_element, data_log_element, _ns,
                                      channel.sample_rate,
                                      response_fir_id, response_paz_id)

    return channel


def _read_instrument_sensitivity(sen_element, cha_element, _ns):

    """
    reads the instrument sensitivity (gain) from the sensor and channel element
    """

    gain = _tag2obj(cha_element, _ns("gain"), float)
    frequency = _tag2obj(cha_element, _ns("gainFrequency"), float)

    input_units_name = _tag2obj(sen_element, _ns("unit"), str)
    output_units_name = ''

    sensitivity = obspy.core.inventory.response.InstrumentSensitivity(
        value=gain, frequency=frequency,
        input_units=input_units_name,
        output_units=output_units_name)

    # assuming these are equal to frequencyStart/frequencyEnd
    sensitivity.frequency_range_start = \
        _tag2obj(sen_element, _ns("lowFrequency"), float)
    sensitivity.frequency_range_end = \
        _tag2obj(sen_element, _ns("highFrequency"), float)

    return sensitivity


def _read_response(instrumentation_responses, sen_element, resp_element,
                   cha_element, data_log_element, _ns, samp_rate, fir,
                   analogue):
    """
    reads response from sc3ml format

    :param instrumentation_responses: Dictionary of dictionaries of
        instrumentation response metadata, top level keyed by response type,
        and subdictionaries keyed by response ID.
    :param _ns: namespace
    """
    response = obspy.core.inventory.response.Response()
    response.instrument_sensitivity = _read_instrument_sensitivity(
        sen_element, cha_element, _ns)

    if resp_element is None:
        return response

    """
    uncomment to include resource id for response (not shown in stationXML)

    response.resource_id = resp_element.attrib.get('publicID')
    if response.resource_id is not None:
        response.resource_id = str(response.resource_id)
    """

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
            fir_element = instrumentation_responses["responseFIR"].get(fir_id)
            if fir_element is None:
                continue
            dec_fac = _tag2obj(fir_element, _ns("decimationFactor"), int)
            if dec_fac is not None and int(dec_fac) != 0:
                samp_rate *= dec_fac
            fir_stage_rates.append(float(samp_rate))

    # Return filter chain to original and also revert the rates
    fir = fir[::-1]
    fir_stage_rates = fir_stage_rates[::-1]

    # Attempt to read stages in the proper order
    # sc3ml does not group stages by an ID
    # We are required to do stage counting ourselves

    stage = 1
    # Get the sensor units, default to M/S
    sensor_units = _tag2obj(sen_element, _ns("unit"), str)
    if sensor_units is None:
        msg = "Sensor unit not set, assuming M/S"
        warnings.warn(msg)
        sensor_units = "M/S"

    # Get the first PAZ stage
    # Input unit: M/S or M/S**2
    # Output unit: V
    if resp_element is not None:
        paz_response = _read_response_stage(resp_element, _ns, samp_rate,
                                            stage, sensor_units, 'V')
        if paz_response is not None:
            response.response_stages.append(paz_response)
            stage += 1

    # Apply analogue filter stages (if any)
    # Input unit: V
    # Output unit: V
    if len(analogue):
        for analogue_id in analogue:
            analogue_element = instrumentation_responses["responsePAZ"]\
                .get(analogue_id)
            if analogue_element is None:
                msg = ('Analogue responsePAZ not in inventory:'
                       '%s, stopping before stage %i') % (analogue_id, stage)
                warnings.warn(msg)
                return response
            analogue_response = _read_response_stage(analogue_element, _ns,
                                                     samp_rate, stage, 'V',
                                                     'V')
            if analogue_response is not None:
                response.response_stages.append(analogue_response)
                stage += 1

    # Apply datalogger (digitizer)
    # Input unit: V
    # Output unit: COUNTS
    if data_log_element is not None:
        coeff_response = _read_response_stage(data_log_element, _ns,
                                              samp_rate, stage, 'V',
                                              'COUNTS')
        if coeff_response is not None:
            response.response_stages.append(coeff_response)
            stage += 1

    # Apply final digital filter stages
    # Input unit: COUNTS
    # Output unit: COUNTS
    for fir_id, rate in zip(fir, fir_stage_rates):
        stage_element = instrumentation_responses["responseFIR"].get(fir_id)
        if stage_element is None:
            msg = ("fir response not in inventory: %s, stopping correction"
                   "before stage %i") % (fir_id, stage)
            warnings.warn(msg)
            return response
        fir_response = _read_response_stage(stage_element, _ns, rate, stage,
                                            'COUNTS', 'COUNTS')
        if fir_response is not None:
            response.response_stages.append(fir_response)
            stage += 1
    return response


def _read_response_stage(stage, _ns, rate, stage_number, input_units,
                         output_units):

    elem_type = stage.tag.split("}")[1]

    stage_sequence_number = stage_number

    # Obtain the stage gain and frequency
    # Default to a gain of 0 and frequency of 0 if missing
    stage_gain = _tag2obj(stage, _ns("gain"), float) or 0
    stage_gain_frequency = _tag2obj(stage, _ns("gainFrequency"),
                                    float) or float(0.00)

    name = stage.get("name")
    if name is not None:
        name = str(name)
    resource_id = stage.get("publicID")
    if resource_id is not None:
        resource_id = str(resource_id)

    # Determine the decimation parameters
    # This is dependent on the type of stage
    # Decimation delay/correction need to be normalized
    if(elem_type == "responseFIR"):
        decimation_factor = _tag2obj(stage, _ns("decimationFactor"), int)
        if rate != 0.0:
            temp = _tag2obj(stage, _ns("delay"), float) / rate
            decimation_delay = _read_float_var(temp,
                                               FloatWithUncertaintiesAndUnit,
                                               unit=True)
            temp = _tag2obj(stage, _ns("correction"), float) / rate
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
    elif(elem_type == "datalogger"):
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
    elif(elem_type == "responsePAZ" or elem_type == "responsePolynomial"):
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
    # PAZ
    # COEFF
    # FIR
    # Polynomial response is not supported, could not find example
    if(elem_type == 'responsePAZ'):

        # read normalization params
        normalization_freq = _read_floattype(stage,
                                             _ns("normalizationFrequency"),
                                             Frequency)
        normalization_factor = _tag2obj(stage, _ns("normalizationFactor"),
                                        float)

        # Parse the type of the transfer function
        # A: Laplace (rad)
        # B: Laplace (Hz)
        # D: digital (z-transform)
        pz_transfer_function_type = _tag2obj(stage, _ns("type"), str)
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
        number_of_zeros = _tag2obj(stage, _ns("numberOfZeros"), int)
        number_of_poles = _tag2obj(stage, _ns("numberOfPoles"), int)

        # Parse string of poles and zeros
        # paz are stored as a string in sc3ml
        # e.g. (-0.01234,0.01234) (-0.01234,-0.01234)
        if number_of_zeros > 0:
            zeros_array = stage.find(_ns("zeros")).text
            zeros_array = _parse_list_of_complex_string(zeros_array)
        else:
            zeros_array = []
        if number_of_poles > 0:
            poles_array = stage.find(_ns("poles")).text
            poles_array = _parse_list_of_complex_string(poles_array)
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

    elif(elem_type == 'datalogger'):
        cf_transfer_function_type = "DIGITAL"
        numerator = []
        denominator = []
        return CoefficientsTypeResponseStage(
            cf_transfer_function_type=cf_transfer_function_type,
            numerator=numerator, denominator=denominator, **kwargs)

    elif(elem_type == 'responsePolynomial'):
        # Polynomial response (UNTESTED)
        # Currently not implemented in ObsPy (20-11-2015)
        f_low = None
        f_high = None
        max_err = None
        appr_type = _tag2obj(stage, _ns("approximationType"), str)
        appr_low = _tag2obj(stage, _ns("approximationLowerBound"), float)
        appr_high = _tag2obj(stage, _ns("approximationUpperBound"), float)
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

    elif(elem_type == 'responseFIR'):
        # For the responseFIR obtain the symmetry and
        # list of coefficients

        coeffs_str = _tag2obj(stage, _ns("coefficients"), str)
        coeffs_float = []
        if coeffs_str is not None and coeffs_str != 'None':
            coeffs = coeffs_str.split()
            i = 0
            # pass additional mapping of coefficient counter
            # so that a proper stationXML can be formatted
            for c in coeffs:
                temp = _read_float_var(c, FilterCoefficient,
                                       additional_mapping={str("number"): i})
                coeffs_float.append(temp)
                i += 1

        # Write the FIR symmetry to what ObsPy expects
        # A: NONE,
        # B: ODD,
        # C: EVEN
        symmetry = _tag2obj(stage, _ns("symmetry"), str)
        if(symmetry == 'A'):
            symmetry = 'NONE'
        elif(symmetry == 'B'):
            symmetry = 'ODD'
        elif(symmetry == 'C'):
            symmetry = 'EVEN'
        else:
            raise ValueError('Unknown symmetry metric; expected A, B, or C')

        return FIRResponseStage(
            coefficients=coeffs_float, symmetry=symmetry, **kwargs)


def _tag2pole_or_zero(paz_element, count):

    """
    Parses sc3ml poles and zeros
    Uncertainties on poles removed, not present in sc3ml.xsd?
    Always put to None so no internal conflict
    The sanitization removes the first/last parenthesis
    and split by comma, real part is 1st, imaginary 2nd

    :param paz_element: tuple of poles or zeros e.g. ('12320', '23020')
    """

    real, imag = map(float, paz_element)

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
    because they are not provided by sc3ml ?

    :param elem: float value to be converted
    :param cls: obspy.core.inventory class
    """

    try:
        convert = float(elem)
    except Exception:
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
