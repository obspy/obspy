# -*- coding: utf-8 -*-
"""
Functions dealing with reading and writing StationXML.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2013
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""
import copy
import inspect
import io
import math
import os
import re
import warnings

from lxml import etree

import obspy
from obspy.core import compatibility
from obspy.core.util import AttribDict
from obspy.core.util.deprecation_helpers import ObsPyDeprecationWarning
from obspy.core.util.obspy_types import (ComplexWithUncertainties,
                                         FloatWithUncertaintiesAndUnit)
from obspy.core.inventory import (CoefficientsTypeResponseStage,
                                  CoefficientWithUncertainties,
                                  FilterCoefficient, FIRResponseStage,
                                  PolesZerosResponseStage,
                                  PolynomialResponseStage,
                                  ResponseListResponseStage, ResponseStage)
from obspy.core.inventory import (Angle, Azimuth, ClockDrift, Dip, Distance,
                                  Frequency, Latitude, Longitude, SampleRate)


# Define some constants for writing StationXML files.
SOFTWARE_MODULE = "ObsPy %s" % obspy.__version__
SOFTWARE_URI = "https://www.obspy.org"
SCHEMA_VERSION = "1.1"
READABLE_VERSIONS = ("1.0", "1.1")


def _get_version_from_xmldoc(xmldoc):
    """
    Return StationXML version string or ``None`` if parsing fails.
    """
    root = xmldoc.getroot()
    try:
        match = re.match(
            r'{http://www.fdsn.org/xml/station/[0-9]+}FDSNStationXML',
            root.tag)
        assert match is not None
    except Exception:
        return None
    try:
        version = root.attrib["schemaVersion"]
    except KeyError:
        return None
    return version


def _is_stationxml(path_or_file_object):
    """
    Simple function checking if the passed object contains a valid StationXML
    1.0 or StationXML 1.1 file. Returns True of False.

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
        version = _get_version_from_xmldoc(xmldoc)
        if version is None:
            return False
        if version not in READABLE_VERSIONS:
            warnings.warn("The StationXML file has version %s, ObsPy can "
                          "read versions (%s). Proceed with caution." % (
                              version, ", ".join(READABLE_VERSIONS)))
        return True
    finally:
        # Make sure to reset file pointer position.
        try:
            path_or_file_object.seek(current_position, 0)
        except Exception:
            pass


def validate_stationxml(path_or_object):
    """
    Checks if the given path is a valid StationXML file.

    Returns a tuple. The first item is a boolean describing if the validation
    was successful or not. The second item is a list of all found validation
    errors, if existent.

    :param path_or_object: File name or file like object. Can also be an etree
        element.
    """
    if isinstance(path_or_object, etree._Element):
        xmldoc = path_or_object
    else:
        try:
            xmldoc = etree.parse(path_or_object)
        except etree.XMLSyntaxError:
            return (False, ("Not a XML file.",))
    version = _get_version_from_xmldoc(xmldoc)

    # Get the schema location.
    schema_location = os.path.dirname(inspect.getfile(inspect.currentframe()))
    schema_location = os.path.join(schema_location, "data",
                                   "fdsn-station-%s.xsd" % version)

    if not os.path.exists(schema_location):
        msg = "No schema file found to validate StationXML version '%s'"
        raise ValueError(msg % version)

    xmlschema = etree.XMLSchema(etree.parse(schema_location))

    valid = xmlschema.validate(xmldoc)

    # Pretty error printing if the validation fails.
    if valid is not True:
        return (False, xmlschema.error_log)
    return (True, ())


def _read_stationxml(path_or_file_object):
    """
    Function reading a StationXML file.

    :param path_or_file_object: File name or file like object.
    """
    root = etree.parse(path_or_file_object).getroot()

    # Fix the namespace as its not always the default namespace. Will need
    # to be adjusted if the StationXML format gets another revision!
    namespace = "http://www.fdsn.org/xml/station/1"

    stationxml_version = root.attrib.get('schemaVersion')

    def _ns(tagname):
        return "{%s}%s" % (namespace, tagname)

    # Source and Created field must exist in a StationXML.
    source = root.find(_ns("Source")).text
    created = obspy.UTCDateTime(root.find(_ns("Created")).text)

    # These are optional
    sender = _tag2obj(root, _ns("Sender"), str)
    module = _tag2obj(root, _ns("Module"), str)
    module_uri = _tag2obj(root, _ns("ModuleURI"), str)

    networks = []
    with warnings.catch_warnings():
        if stationxml_version == '1.0':
            warnings.filterwarnings(
                'ignore',
                'Setting Numerator/Denominator with a unit is deprecated.',
                ObsPyDeprecationWarning)
        for network in root.findall(_ns("Network")):
            networks.append(_read_network(network, _ns))

    inv = obspy.core.inventory.Inventory(networks=networks, source=source,
                                         sender=sender, created=created,
                                         module=module, module_uri=module_uri)
    _read_extra(root, inv)  # read extra tags from root element
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
        _attr2obj(element, "restrictedStatus", str)
    object_to_write_to.alternate_code = \
        _attr2obj(element, "alternateCode", str)
    object_to_write_to.historical_code = \
        _attr2obj(element, "historicalCode", str)
    object_to_write_to.source_id = \
        _attr2obj(element, "sourceID", str)
    object_to_write_to.description = \
        _tag2obj(element, _ns("Description"), str)
    identifiers = []
    for identifier in element.findall(_ns("Identifier")):
        scheme = identifier.get('type')
        path = identifier.text
        if scheme is not None:
            identifiers.append(scheme + ':' + path)
        else:
            identifiers.append(path)
    object_to_write_to.identifiers = identifiers
    object_to_write_to.comments = []
    for comment in element.findall(_ns("Comment")):
        object_to_write_to.comments.append(_read_comment(comment, _ns))
    # Availability.
    data_availability = element.find(_ns("DataAvailability"))
    if data_availability is not None:
        object_to_write_to.data_availability = \
            _read_data_availability(data_availability, _ns)
    _read_extra(element, object_to_write_to)


def _read_network(net_element, _ns):
    network = obspy.core.inventory.Network(net_element.get("code"))
    _read_base_node(net_element, network, _ns)
    for operator in net_element.findall(_ns("Operator")):
        network.operators.append(_read_operator(operator, _ns))
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
    longitude = _read_floattype(sta_element, _ns("Longitude"), Longitude,
                                datum=True)
    latitude = _read_floattype(sta_element, _ns("Latitude"), Latitude,
                               datum=True)
    elevation = _read_floattype(sta_element, _ns("Elevation"), Distance,
                                unit=True)
    station = obspy.core.inventory.Station(code=sta_element.get("code"),
                                           latitude=latitude,
                                           longitude=longitude,
                                           elevation=elevation)
    station.site = _read_site(sta_element.find(_ns("Site")), _ns)
    _read_base_node(sta_element, station, _ns)
    # water level only for schemas > 1.0
    station.water_level = _read_floattype(sta_element, _ns("WaterLevel"),
                                          FloatWithUncertaintiesAndUnit,
                                          unit=True)
    station.vault = _tag2obj(sta_element, _ns("Vault"), str)
    station.geology = _tag2obj(sta_element, _ns("Geology"), str)
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
        # Skip empty channels.
        if not channel.items() and not channel.attrib:
            continue
        cha = _read_channel(channel, _ns)
        # Might be None in case the channel could not be parsed.
        if cha is None:
            # This is None if, and only if, one of the coordinates could not
            # be set.
            msg = ("Channel %s.%s of station %s does not have a complete set "
                   "of coordinates and thus it cannot be read. It will not be "
                   "part of the final inventory object." % (
                       channel.get("locationCode"), channel.get("code"),
                       sta_element.get("code")))
            warnings.warn(msg, UserWarning)
        else:
            channels.append(cha)
    station.channels = channels
    return station


def _read_floattype(parent, tag, cls, unit=False, datum=False,
                    additional_mapping={}):
    elem = parent.find(tag)
    if elem is None:
        return None

    # Catch non convertible numbers.
    try:
        convert = float(elem.text)
    except Exception:
        warnings.warn(
            "'%s' could not be converted to a float. Will be skipped. Please "
            "contact to report this issue." % etree.tostring(elem),
            UserWarning)
        return None

    # Catch NaNs.
    if math.isnan(convert):
        warnings.warn("Tag '%s' has a value of NaN. It will be skipped." %
                      tag, UserWarning)
        return None

    obj = cls(convert)
    if unit:
        obj.unit = elem.attrib.get("unit")
    if datum:
        obj.datum = elem.attrib.get("datum")
    lower_uncertainty = elem.attrib.get("minusError")
    if lower_uncertainty is not None:
        lower_uncertainty = float(lower_uncertainty)
    obj.lower_uncertainty = lower_uncertainty
    upper_uncertainty = elem.attrib.get("plusError")
    if upper_uncertainty is not None:
        upper_uncertainty = float(upper_uncertainty)
    obj.upper_uncertainty = upper_uncertainty
    obj.measurement_method = elem.attrib.get("measurementMethod")
    for key1, key2 in additional_mapping.items():
        setattr(obj, key1, elem.attrib.get(key2))
    return obj


def _read_floattype_list(parent, tag, cls, unit=False, datum=False,
                         additional_mapping={}):
    elems = parent.findall(tag)
    objs = []
    for elem in elems:
        obj = cls(float(elem.text))
        if unit:
            obj.unit = elem.attrib.get("unit")
        if datum:
            obj.datum = elem.attrib.get("datum")
        obj.lower_uncertainty = elem.attrib.get("minusError")
        obj.upper_uncertainty = elem.attrib.get("plusError")
        obj.measurement_method = elem.attrib.get("measurementMethod")
        for key1, key2 in additional_mapping.items():
            setattr(obj, key2, elem.attrib.get(key1))
        objs.append(obj)
    return objs


def _read_channel(cha_element, _ns):
    """
    Returns either a :class:`~obspy.core.inventory.channel.Channel` object or
    ``None``.

    It should return ``None`` if and only if it did not manage to
    successfully create a :class:`~obspy.core.inventory.channel.Channel`
    object which can only happen if one of the coordinates is not set. All the
    others are optional. If either the location or channel code is not set it
    raises but that is fine as that would deviate too much from the StationXML
    standard to be worthwhile to recover from.
    """
    code = cha_element.get("code")
    location_code = cha_element.get("locationCode")

    longitude = _read_floattype(cha_element, _ns("Longitude"), Longitude,
                                datum=True)
    latitude = _read_floattype(cha_element, _ns("Latitude"), Latitude,
                               datum=True)
    elevation = _read_floattype(cha_element, _ns("Elevation"), Distance,
                                unit=True)
    depth = _read_floattype(cha_element, _ns("Depth"), Distance, unit=True)

    # All of these must be given, otherwise it is an invalid station.
    if None in [longitude, latitude, elevation, depth]:
        return None

    channel = obspy.core.inventory.Channel(
        code=code, location_code=location_code, latitude=latitude,
        longitude=longitude, elevation=elevation, depth=depth)
    _read_base_node(cha_element, channel, _ns)
    channel.azimuth = _read_floattype(cha_element, _ns("Azimuth"), Azimuth)
    channel.dip = _read_floattype(cha_element, _ns("Dip"), Dip)
    # water level only for schemas > 1.0
    channel.water_level = _read_floattype(cha_element, _ns("WaterLevel"),
                                          FloatWithUncertaintiesAndUnit,
                                          unit=True)
    # Add all types.
    for type_element in cha_element.findall(_ns("Type")):
        channel.types.append(type_element.text)
    # Add all external references.
    channel.external_references = \
        [_read_external_reference(ext_ref, _ns)
         for ext_ref in cha_element.findall(_ns("ExternalReference"))]
    channel.sample_rate = _read_floattype(cha_element, _ns("SampleRate"),
                                          SampleRate)
    # Parse the optional sample rate ratio.
    sample_rate_ratio = cha_element.find(_ns("SampleRateRatio"))
    if sample_rate_ratio is not None:
        channel.sample_rate_ratio_number_samples = \
            _tag2obj(sample_rate_ratio, _ns("NumberSamples"), int)
        channel.sample_rate_ratio_number_seconds = \
            _tag2obj(sample_rate_ratio, _ns("NumberSeconds"), int)
    # The clock drift is one of the few examples where the attribute name is
    # different from the tag name. This improves clarity.
    channel.clock_drift_in_seconds_per_sample = \
        _read_floattype(cha_element, _ns("ClockDrift"), ClockDrift)
    # The sensor.
    calibunits = cha_element.find(_ns("CalibrationUnits"))
    if calibunits is not None:
        channel.calibration_units = _tag2obj(calibunits, _ns("Name"), str)
        channel.calibration_units_description = \
            _tag2obj(calibunits, _ns("Description"), str)
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
    for equipment in cha_element.findall(_ns("Equipment")):
        channel.equipments.append(_read_equipment(equipment, _ns))
    # Finally parse the response.
    response = cha_element.find(_ns("Response"))
    if response is not None:
        channel.response = _read_response(response, _ns)
        channel.response._attempt_to_fix_units()
    return channel


def _read_response(resp_element, _ns):
    response = obspy.core.inventory.response.Response()
    response.resource_id = resp_element.attrib.get('resourceId')
    if response.resource_id is not None:
        response.resource_id = str(response.resource_id)
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
        if not len(stage):
            continue
        response.response_stages.append(_read_response_stage(stage, _ns))
    _read_extra(resp_element, response)
    return response


def _read_response_stage(stage_elem, _ns):
    """
    This parses all ResponseStageTypes. It will return a different object
    depending on the actual response type.
    """
    # The stage sequence number is required!
    stage_sequence_number = int(stage_elem.get("number"))
    resource_id = stage_elem.attrib.get('resourceId')
    if resource_id is not None:
        resource_id = str(resource_id)
    # All stages contain a stage gain and potentially a decimation.
    gain_elem = stage_elem.find(_ns("StageGain"))
    stage_gain = _tag2obj(gain_elem, _ns("Value"), float)
    stage_gain_frequency = _tag2obj(gain_elem, _ns("Frequency"), float)
    # Parse the decimation.
    decim_elem = stage_elem.find(_ns("Decimation"))
    if decim_elem is not None:
        decimation_input_sample_rate = \
            _read_floattype(decim_elem, _ns("InputSampleRate"), Frequency)
        decimation_factor = _tag2obj(decim_elem, _ns("Factor"), int)
        decimation_offset = _tag2obj(decim_elem, _ns("Offset"), int)
        decimation_delay = _read_floattype(decim_elem, _ns("Delay"),
                                           FloatWithUncertaintiesAndUnit,
                                           unit=True)
        decimation_correction = \
            _read_floattype(decim_elem, _ns("Correction"),
                            FloatWithUncertaintiesAndUnit, unit=True)
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
    fir_elem = stage_elem.find(_ns("FIR"))
    polynomial_elem = stage_elem.find(_ns("Polynomial"))
    type_elems = [poles_zeros_elem, coefficients_elem, response_list_elem,
                  fir_elem, polynomial_elem]

    # iterate and check for an response element and create alias
    for elem in type_elems:
        if elem is not None:
            break
    else:
        # Nothing more to parse for gain only blockettes, create minimal
        # ResponseStage and return
        if stage_gain is not None and stage_gain_frequency is not None:
            return obspy.core.inventory.ResponseStage(
                stage_sequence_number=stage_sequence_number,
                stage_gain=stage_gain,
                stage_gain_frequency=stage_gain_frequency,
                resource_id=resource_id, input_units=None, output_units=None)
        # Raise if none of the previous ones has been found.
        msg = "Could not find a valid Response Stage Type."
        raise ValueError(msg)

    # Now parse all elements the different stages share.
    input_units_ = elem.find(_ns("InputUnits"))
    input_units = _tag2obj(input_units_, _ns("Name"), str)
    input_units_description = _tag2obj(input_units_, _ns("Description"),
                                       str)
    output_units_ = elem.find(_ns("OutputUnits"))
    output_units = _tag2obj(output_units_, _ns("Name"), str)
    output_units_description = _tag2obj(output_units_, _ns("Description"),
                                        str)
    description = _tag2obj(elem, _ns("Description"), str)
    name = elem.attrib.get("name")
    if name is not None:
        name = str(name)
    resource_id2 = elem.attrib.get('resourceId')
    if resource_id2 is not None:
        resource_id2 = str(resource_id2)

    # Now collect all shared kwargs to be able to pass them to the different
    # constructors..
    kwargs = {"stage_sequence_number": stage_sequence_number,
              "input_units": input_units,
              "output_units": output_units,
              "input_units_description": input_units_description,
              "output_units_description": output_units_description,
              "resource_id": resource_id, "resource_id2": resource_id2,
              "stage_gain": stage_gain,
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
            _tag2obj(elem, _ns("PzTransferFunctionType"), str)
        normalization_factor = \
            _tag2obj(elem, _ns("NormalizationFactor"), float)
        normalization_frequency = \
            _read_floattype(elem, _ns("NormalizationFrequency"), Frequency)
        # Read poles and zeros to list of imaginary numbers.

        def _tag2pole_or_zero(element):
            real = _tag2obj(element, _ns("Real"), float)
            imag = _tag2obj(element, _ns("Imaginary"), float)
            if real is not None or imag is not None:
                real = real or 0
                imag = imag or 0
            x = ComplexWithUncertainties(real, imag)
            real = _attr2obj(element.find(_ns("Real")), "minusError", float)
            imag = _attr2obj(element.find(_ns("Imaginary")), "minusError",
                             float)
            if any([value is not None for value in (real, imag)]):
                real = real or 0
                imag = imag or 0
                x.lower_uncertainty = complex(real, imag)
            real = _attr2obj(element.find(_ns("Real")), "plusError", float)
            imag = _attr2obj(element.find(_ns("Imaginary")), "plusError",
                             float)
            if any([value is not None for value in (real, imag)]):
                real = real or 0
                imag = imag or 0
                x.upper_uncertainty = complex(real, imag)
            x.number = _attr2obj(element, "number", int)
            x.measurement_method_real = _attr2obj(element.find(_ns("Real")),
                                                  "measurementMethod", str)
            x.measurement_method_imag = _attr2obj(
                element.find(_ns("Imaginary")), "measurementMethod", str)
            return x

        zeros = [_tag2pole_or_zero(el) for el in elem.findall(_ns("Zero"))]
        poles = [_tag2pole_or_zero(el) for el in elem.findall(_ns("Pole"))]
        obj = obspy.core.inventory.PolesZerosResponseStage(
            pz_transfer_function_type=pz_transfer_function_type,
            normalization_frequency=normalization_frequency,
            normalization_factor=normalization_factor, zeros=zeros,
            poles=poles, **kwargs)
        _read_extra(elem, obj)
        return obj

    # Handle the coefficients Response Stage Type.
    elif elem is coefficients_elem:
        cf_transfer_function_type = \
            _tag2obj(elem, _ns("CfTransferFunctionType"), str)
        numerator = \
            _read_floattype_list(elem, _ns("Numerator"),
                                 CoefficientWithUncertainties, unit=False,
                                 additional_mapping={"number": "number"})
        denominator = \
            _read_floattype_list(elem, _ns("Denominator"),
                                 CoefficientWithUncertainties, unit=False,
                                 additional_mapping={"number": "number"})
        obj = obspy.core.inventory.CoefficientsTypeResponseStage(
            cf_transfer_function_type=cf_transfer_function_type,
            numerator=numerator, denominator=denominator, **kwargs)
        _read_extra(elem, obj)
        return obj

    # Handle the response list response stage type.
    elif elem is response_list_elem:
        rlist_elems = []
        for item in elem.findall(_ns("ResponseListElement")):
            freq = _read_floattype(item, _ns("Frequency"), Frequency)
            amp = _read_floattype(item, _ns("Amplitude"),
                                  FloatWithUncertaintiesAndUnit, unit=True)
            phase = _read_floattype(item, _ns("Phase"), Angle)
            rlist_elems.append(
                obspy.core.inventory.response.ResponseListElement(
                    frequency=freq, amplitude=amp, phase=phase))
        obj = obspy.core.inventory.ResponseListResponseStage(
            response_list_elements=rlist_elems, **kwargs)
        _read_extra(elem, obj)
        return obj

    # Handle the FIR response stage type.
    elif elem is fir_elem:
        symmetry = _tag2obj(elem, _ns("Symmetry"), str)
        coeffs = _read_floattype_list(elem, _ns("NumeratorCoefficient"),
                                      FilterCoefficient,
                                      additional_mapping={'i': "number"})
        obj = obspy.core.inventory.FIRResponseStage(
            coefficients=coeffs, symmetry=symmetry, **kwargs)
        _read_extra(elem, obj)
        return obj

    # Handle polynomial instrument responses.
    elif elem is polynomial_elem:
        appr_type = _tag2obj(elem, _ns("ApproximationType"), str)
        f_low = _read_floattype(elem, _ns("FrequencyLowerBound"), Frequency)
        f_high = _read_floattype(elem, _ns("FrequencyUpperBound"), Frequency)
        appr_low = _tag2obj(elem, _ns("ApproximationLowerBound"), float)
        appr_high = _tag2obj(elem, _ns("ApproximationUpperBound"), float)
        max_err = _tag2obj(elem, _ns("MaximumError"), float)
        coeffs = _read_floattype_list(elem, _ns("Coefficient"),
                                      CoefficientWithUncertainties,
                                      additional_mapping={"number": "number"})
        obj = obspy.core.inventory.PolynomialResponseStage(
            approximation_type=appr_type, frequency_lower_bound=f_low,
            frequency_upper_bound=f_high, approximation_lower_bound=appr_low,
            approximation_upper_bound=appr_high, maximum_error=max_err,
            coefficients=coeffs, **kwargs)
        _read_extra(elem, obj)
        return obj


def _read_instrument_sensitivity(sensitivity_element, _ns):
    value = _tag2obj(sensitivity_element, _ns("Value"), float)
    frequency = _tag2obj(sensitivity_element, _ns("Frequency"), float)
    input_units_ = sensitivity_element.find(_ns("InputUnits"))
    output_units_ = sensitivity_element.find(_ns("OutputUnits"))
    sensitivity = obspy.core.inventory.response.InstrumentSensitivity(
        value=value, frequency=frequency,
        input_units=_tag2obj(input_units_, _ns("Name"), str),
        output_units=_tag2obj(output_units_, _ns("Name"), str))
    sensitivity.input_units_description = \
        _tag2obj(input_units_, _ns("Description"), str)
    sensitivity.output_units_description = \
        _tag2obj(output_units_, _ns("Description"), str)
    sensitivity.frequency_range_start = \
        _tag2obj(sensitivity_element, _ns("FrequencyStart"), float)
    sensitivity.frequency_range_end = \
        _tag2obj(sensitivity_element, _ns("FrequencyEnd"), float)
    sensitivity.frequency_range_db_variation = \
        _tag2obj(sensitivity_element, _ns("FrequencyDBVariation"), float)
    _read_extra(sensitivity_element, sensitivity)
    return sensitivity


def _read_instrument_polynomial(element, _ns):
    # XXX duplicated code, see reading of PolynomialResponseStage
    input_units_ = element.find(_ns("InputUnits"))
    input_units = _tag2obj(input_units_, _ns("Name"), str)
    input_units_description = _tag2obj(input_units_, _ns("Description"),
                                       str)
    output_units_ = element.find(_ns("OutputUnits"))
    output_units = _tag2obj(output_units_, _ns("Name"), str)
    output_units_description = _tag2obj(output_units_, _ns("Description"),
                                        str)
    description = _tag2obj(element, _ns("Description"), str)
    resource_id = element.attrib.get("resourceId", None)
    name = element.attrib.get("name", None)
    appr_type = _tag2obj(element, _ns("ApproximationType"), str)
    f_low = _read_floattype(element, _ns("FrequencyLowerBound"), Frequency)
    f_high = _read_floattype(element, _ns("FrequencyUpperBound"), Frequency)
    appr_low = _tag2obj(element, _ns("ApproximationLowerBound"), float)
    appr_high = _tag2obj(element, _ns("ApproximationUpperBound"), float)
    max_err = _tag2obj(element, _ns("MaximumError"), float)
    coeffs = _read_floattype_list(element, _ns("Coefficient"),
                                  CoefficientWithUncertainties,
                                  additional_mapping={"number": "number"})
    obj = obspy.core.inventory.response.InstrumentPolynomial(
        approximation_type=appr_type, frequency_lower_bound=f_low,
        frequency_upper_bound=f_high, approximation_lower_bound=appr_low,
        approximation_upper_bound=appr_high, maximum_error=max_err,
        coefficients=coeffs, input_units=input_units,
        input_units_description=input_units_description,
        output_units=output_units,
        output_units_description=output_units_description,
        description=description, resource_id=resource_id, name=name)
    _read_extra(element, obj)
    return obj


def _read_external_reference(ref_element, _ns):
    uri = _tag2obj(ref_element, _ns("URI"), str)
    description = _tag2obj(ref_element, _ns("Description"), str)
    obj = obspy.core.inventory.ExternalReference(uri=uri,
                                                 description=description)
    _read_extra(ref_element, obj)
    return obj


def _read_operator(operator_element, _ns):
    agency = operator_element.find(_ns("Agency")).text
    contacts = []
    for contact in operator_element.findall(_ns("Contact")):
        contacts.append(_read_person(contact, _ns))
    website = _tag2obj(operator_element, _ns("WebSite"), str)
    obj = obspy.core.inventory.Operator(agency=agency, contacts=contacts,
                                        website=website)
    _read_extra(operator_element, obj)
    return obj


def _read_data_availability(avail_element, _ns):
    extent = avail_element.find(_ns("Extent"))
    spans = avail_element.findall(_ns("Span"))
    # Recovery from empty Extent tag + no spans.
    if extent is None and not spans:
        return None
    start = extent.get("start")
    end = extent.get("end")
    spans = [_read_data_availability_span(span, _ns) for span in spans]
    obj = obspy.core.inventory.util.DataAvailability(start=start, end=end,
                                                     spans=spans)
    _read_extra(avail_element, obj)
    return obj


def _read_data_availability_span(element, _ns):
    start = element.attrib['start']
    if start is not None:
        start = obspy.UTCDateTime(start)
    end = element.attrib['end']
    if end is not None:
        end = obspy.UTCDateTime(end)
    number_of_segments = element.attrib['numberSegments']
    if number_of_segments is not None:
        number_of_segments = int(number_of_segments)
    maximum_time_tear = element.attrib.get('maximumTimeTear')
    if maximum_time_tear is not None:
        maximum_time_tear = float(maximum_time_tear)
    obj = obspy.core.inventory.util.DataAvailabilitySpan(
        start=start, end=end, number_of_segments=number_of_segments,
        maximum_time_tear=maximum_time_tear)
    _read_extra(element, obj)
    return obj


def _read_equipment(equip_element, _ns):
    resource_id = equip_element.get("resourceId")
    type = _tag2obj(equip_element, _ns("Type"), str)
    description = _tag2obj(equip_element, _ns("Description"), str)
    manufacturer = _tag2obj(equip_element, _ns("Manufacturer"), str)
    vendor = _tag2obj(equip_element, _ns("Vendor"), str)
    model = _tag2obj(equip_element, _ns("Model"), str)
    serial_number = _tag2obj(equip_element, _ns("SerialNumber"), str)
    installation_date = \
        _tag2obj(equip_element, _ns("InstallationDate"), obspy.UTCDateTime)
    removal_date = \
        _tag2obj(equip_element, _ns("RemovalDate"), obspy.UTCDateTime)
    calibration_dates = \
        [obspy.core.UTCDateTime(_i.text)
         for _i in equip_element.findall(_ns("CalibrationDate"))]
    obj = obspy.core.inventory.Equipment(
        resource_id=resource_id, type=type, description=description,
        manufacturer=manufacturer, vendor=vendor, model=model,
        serial_number=serial_number, installation_date=installation_date,
        removal_date=removal_date, calibration_dates=calibration_dates)
    _read_extra(equip_element, obj)
    return obj


def _read_site(site_element, _ns):
    name = _tag2obj(site_element, _ns("Name"), str)
    description = _tag2obj(site_element, _ns("Description"), str)
    town = _tag2obj(site_element, _ns("Town"), str)
    county = _tag2obj(site_element, _ns("County"), str)
    region = _tag2obj(site_element, _ns("Region"), str)
    country = _tag2obj(site_element, _ns("Country"), str)
    obj = obspy.core.inventory.Site(name=name, description=description,
                                    town=town, county=county, region=region,
                                    country=country)
    _read_extra(site_element, obj)
    return obj


def _read_comment(comment_element, _ns):
    value = _tag2obj(comment_element, _ns("Value"), str)
    begin_effective_time = \
        _tag2obj(comment_element, _ns("BeginEffectiveTime"), obspy.UTCDateTime)
    end_effective_time = \
        _tag2obj(comment_element, _ns("EndEffectiveTime"), obspy.UTCDateTime)
    authors = []
    id = _attr2obj(comment_element, "id", int)
    subject = _attr2obj(comment_element, "subject", str)
    for author in comment_element.findall(_ns("Author")):
        authors.append(_read_person(author, _ns))
    obj = obspy.core.inventory.Comment(
        value=value, begin_effective_time=begin_effective_time,
        end_effective_time=end_effective_time, authors=authors, id=id,
        subject=subject)
    _read_extra(comment_element, obj)
    return obj


def _read_person(person_element, _ns):
    names = _tags2obj(person_element, _ns("Name"), str)
    agencies = _tags2obj(person_element, _ns("Agency"), str)
    emails = _tags2obj(person_element, _ns("Email"), str)
    phones = []
    for phone in person_element.findall(_ns("Phone")):
        phones.append(_read_phone(phone, _ns))
    obj = obspy.core.inventory.Person(names=names, agencies=agencies,
                                      emails=emails, phones=phones)
    _read_extra(person_element, obj)
    return obj


def _read_phone(phone_element, _ns):
    country_code = _tag2obj(phone_element, _ns("CountryCode"), int)
    area_code = _tag2obj(phone_element, _ns("AreaCode"), int)
    phone_number = _tag2obj(phone_element, _ns("PhoneNumber"), str)
    description = phone_element.get("description")
    obj = obspy.core.inventory.PhoneNumber(
        country_code=country_code, area_code=area_code,
        phone_number=phone_number, description=description)
    _read_extra(phone_element, obj)
    return obj


def _write_stationxml(inventory, file_or_file_object, validate=False,
                      nsmap=None, level="response", **kwargs):
    """
    Writes an inventory object to a buffer.
    :type inventory: :class:`~obspy.core.inventory.Inventory`
    :param inventory: The inventory instance to be written.
    :param file_or_file_object: The file or file-like object to be written to.
    :type validate: bool
    :param validate: If True, the created document will be validated with the
        StationXML schema before being written. Useful for debugging or if you
        don't trust ObsPy. Defaults to False.
    :type nsmap: dict
    :param nsmap: Additional custom namespace abbreviation mappings
        (e.g. `{"edb": "http://erdbeben-in-bayern.de/xmlns/0.1"}`).

    """
    if nsmap is None:
        nsmap = {}
    elif None in nsmap:
        msg = ("Custom namespace mappings do not allow redefinition of "
               "default StationXML namespace (key `None`). "
               "Use other namespace abbreviations for custom namespace tags.")
        raise ValueError(msg)

    nsmap[None] = "http://www.fdsn.org/xml/station/1"
    attrib = {"schemaVersion": SCHEMA_VERSION}

    root = etree.Element("FDSNStationXML", attrib=attrib, nsmap=nsmap)

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

    etree.SubElement(root, "Created").text = str(inventory.created)

    if level not in ["network", "station", "channel", "response"]:
        raise ValueError("Requested stationXML write level is unsupported.")

    for network in inventory.networks:
        _write_network(root, network, level)

    # Add custom namespace tags to root element
    _write_extra(root, inventory)

    tree = root.getroottree()

    # The validation has to be done after parsing once again so that the
    # namespaces are correctly assembled.
    if validate is True:
        buf = io.BytesIO()
        tree.write(buf)
        buf.seek(0)
        validates, errors = validate_stationxml(buf)
        buf.close()
        if validates is False:
            msg = "The created file fails to validate.\n"
            for err in errors:
                msg += "\t%s\n" % err
            raise Exception(msg)

    # Register all namespaces with the tree. This allows for
    # additional namespaces to be added to an inventory that
    # was not created by reading a StationXML file.
    for prefix, ns in nsmap.items():
        if prefix and ns:
            etree.register_namespace(prefix, ns)

    tree.write(file_or_file_object, pretty_print=True, xml_declaration=True,
               encoding="UTF-8")


def _get_base_node_attributes(element):
    attributes = {"code": element.code}
    if element.start_date:
        attributes["startDate"] = str(element.start_date)
    if element.end_date:
        attributes["endDate"] = str(element.end_date)
    if element.restricted_status:
        attributes["restrictedStatus"] = element.restricted_status
    if element.alternate_code:
        attributes["alternateCode"] = element.alternate_code
    if element.historical_code:
        attributes["historicalCode"] = element.historical_code
    if element.source_id is not None:
        attributes["sourceID"] = element.source_id
    return attributes


def _write_base_node(element, object_to_read_from):
    if object_to_read_from.description:
        etree.SubElement(element, "Description").text = \
            object_to_read_from.description
    for identifier in object_to_read_from.identifiers:
        attrib = {}
        if ':' in identifier:
            scheme, path = identifier.split(':', 1)
            attrib['type'] = scheme
        else:
            path = identifier
        etree.SubElement(element, "Identifier", attrib).text = path
    for comment in object_to_read_from.comments:
        _write_comment(element, comment)
    if object_to_read_from.data_availability:
        _write_data_availability(element,
                                 object_to_read_from.data_availability)
    _write_extra(element, object_to_read_from)


def _write_network(parent, network, level):
    """
    Helper function converting a Network instance to an etree.Element.
    """
    attribs = _get_base_node_attributes(network)
    network_elem = etree.SubElement(parent, "Network", attribs)
    _write_base_node(network_elem, network)

    for operator in network.operators:
        operator_elem = etree.SubElement(network_elem, "Operator")
        etree.SubElement(operator_elem, "Agency").text = str(operator.agency)
        for contact in operator.contacts:
            _write_person(operator_elem, contact, "Contact")
        etree.SubElement(operator_elem, "WebSite").text = operator.website
        _write_extra(operator_elem, operator)

    # Add the two, network specific fields.
    if network.total_number_of_stations is not None:
        etree.SubElement(network_elem, "TotalNumberStations").text = \
            str(network.total_number_of_stations)
    if network.selected_number_of_stations is not None:
        etree.SubElement(network_elem, "SelectedNumberStations").text = \
            str(network.selected_number_of_stations)

    if level == "network":
        return

    for station in network.stations:
        _write_station(network_elem, station, level)


def _write_floattype(parent, obj, attr_name, tag, additional_mapping={},
                     cls=None):
    attribs = {}
    obj_ = getattr(obj, attr_name)
    if obj_ is None:
        return
    if cls and not isinstance(obj_, cls):
        obj_ = cls(obj_)
    attribs["datum"] = obj_.__dict__.get("datum")
    if hasattr(obj_, "unit"):
        attribs["unit"] = obj_.unit
    attribs["minusError"] = obj_.lower_uncertainty
    attribs["plusError"] = obj_.upper_uncertainty
    attribs["measurementMethod"] = obj_.measurement_method
    for key1, key2 in additional_mapping.items():
        attribs[key1] = getattr(obj_, key2)
    attribs = {k: str(v) for k, v in attribs.items() if v is not None}
    etree.SubElement(parent, tag, attribs).text = _float_to_str(obj_)


def _write_floattype_list(parent, obj, attr_list_name, tag,
                          additional_mapping={}, unit=True):
    for obj_ in getattr(obj, attr_list_name):
        attribs = {}
        attribs["datum"] = obj_.__dict__.get("datum")
        if hasattr(obj_, "unit") and unit:
            attribs["unit"] = obj_.unit
        attribs["minusError"] = obj_.lower_uncertainty
        attribs["plusError"] = obj_.upper_uncertainty
        attribs["measurementMethod"] = obj_.measurement_method
        for key1, key2 in additional_mapping.items():
            attribs[key2] = getattr(obj_, key1)
        attribs = {k: str(v) for k, v in attribs.items() if v is not None}
        etree.SubElement(parent, tag, attribs).text = _float_to_str(obj_)


def _float_to_str(x):
    """
    Converts a float to str making. For most numbers this results in a
    decimal representation (for xs:decimal) while for very large or very
    small numbers this results in an exponential representation suitable for
    xs:float and xs:double.
    """
    return "%s" % x


def _write_polezero_list(parent, obj):
    def _polezero2tag(parent, tag, obj_):
        attribs = {}
        attribs["number"] = str(obj_.number)
        sub = etree.SubElement(parent, tag, attribs)
        attribs_real = {}
        attribs_imag = {}
        if obj_.lower_uncertainty is not None:
            attribs_real['minusError'] = \
                _float_to_str(obj_.lower_uncertainty.real)
            attribs_imag['minusError'] = \
                _float_to_str(obj_.lower_uncertainty.imag)
        if obj_.upper_uncertainty is not None:
            attribs_real['plusError'] = \
                _float_to_str(obj_.upper_uncertainty.real)
            attribs_imag['plusError'] = \
                _float_to_str(obj_.upper_uncertainty.imag)
        if obj_.measurement_method_real is not None:
            attribs_real['measurement_method'] = obj_.measurement_method_real
        if obj_.measurement_method_imag is not None:
            attribs_imag['measurement_method'] = obj_.measurement_method_imag
        etree.SubElement(sub, "Real", attribs_real).text = \
            _float_to_str(obj_.real)
        etree.SubElement(sub, "Imaginary", attribs_imag).text = \
            _float_to_str(obj_.imag)

    # Assign numbers if not given - stationxml requires them. Create copies
    # to not modify the original objects.
    zeros = copy.deepcopy(obj.zeros)
    poles = copy.deepcopy(obj.poles)

    # All or nothing - either all have numbers or we manually assign them
    # starting with the zeros.
    #
    # This is done here to catch all cases - otherwise it would have to be
    # done at poles + zeros assignment time and that would be pretty nasty
    # as it would have to work across poles and zeros.
    if not all([hasattr(i, "number") and i.number is not None
                for i in zeros + poles]):
        # Start numbers for both at zero - other variants are out there but
        # this is what IRIS does.
        for i, v in enumerate(zeros):
            v.number = i
        for i, v in enumerate(poles):
            v.number = i

    for obj_ in zeros:
        _polezero2tag(parent, "Zero", obj_)
    for obj_ in poles:
        _polezero2tag(parent, "Pole", obj_)
    _write_extra(parent, obj)


def _write_station(parent, station, level):
    # Write the base node type fields.
    attribs = _get_base_node_attributes(station)
    station_elem = etree.SubElement(parent, "Station", attribs)
    _write_base_node(station_elem, station)

    _write_floattype(station_elem, station, "latitude", "Latitude")
    _write_floattype(station_elem, station, "longitude", "Longitude")
    _write_floattype(station_elem, station, "elevation", "Elevation")

    _write_site(station_elem, station.site)

    if station.water_level is not None:
        _write_floattype(station_elem, station, "water_level", "WaterLevel")

    # Optional tags.
    _obj2tag(station_elem, "Vault", station.vault)
    _obj2tag(station_elem, "Geology", station.geology)

    for equipment in station.equipments:
        _write_equipment(station_elem, equipment)

    for operator in station.operators:
        operator_elem = etree.SubElement(station_elem, "Operator")
        etree.SubElement(operator_elem, "Agency").text = str(operator.agency)
        for contact in operator.contacts:
            _write_person(operator_elem, contact, "Contact")
        etree.SubElement(operator_elem, "WebSite").text = operator.website
        _write_extra(operator_elem, operator)

    if station.creation_date is not None:
        # CreationDate is optional in schemas > 1.0
        etree.SubElement(station_elem, "CreationDate").text = \
            str(station.creation_date)

    if station.termination_date:
        etree.SubElement(station_elem, "TerminationDate").text = \
            str(station.termination_date)
    # The next two tags are optional.
    _obj2tag(station_elem, "TotalNumberChannels",
             station.total_number_of_channels)
    _obj2tag(station_elem, "SelectedNumberChannels",
             station.selected_number_of_channels)

    for ref in station.external_references:
        _write_external_reference(station_elem, ref)

    if level == "station":
        return

    for channel in station.channels:
        _write_channel(station_elem, channel, level)


def _write_channel(parent, channel, level):
    # Write the base node type fields.
    attribs = _get_base_node_attributes(channel)
    attribs['locationCode'] = channel.location_code
    channel_elem = etree.SubElement(parent, "Channel", attribs)
    _write_base_node(channel_elem, channel)

    for ref in channel.external_references:
        _write_external_reference(channel_elem, ref)

    _write_floattype(channel_elem, channel, "latitude", "Latitude")
    _write_floattype(channel_elem, channel, "longitude", "Longitude")
    _write_floattype(channel_elem, channel, "elevation", "Elevation")
    _write_floattype(channel_elem, channel, "depth", "Depth")

    # Optional tags.
    _write_floattype(channel_elem, channel, "azimuth", "Azimuth")
    _write_floattype(channel_elem, channel, "dip", "Dip")

    if channel.water_level is not None:
        _write_floattype(channel_elem, channel, "water_level", "WaterLevel")

    for type_ in channel.types:
        etree.SubElement(channel_elem, "Type").text = type_

    _write_floattype(channel_elem, channel, "sample_rate", "SampleRate")
    if channel.sample_rate_ratio_number_samples and \
            channel.sample_rate_ratio_number_seconds:
        srr = etree.SubElement(channel_elem, "SampleRateRatio")
        etree.SubElement(srr, "NumberSamples").text = \
            str(channel.sample_rate_ratio_number_samples)
        etree.SubElement(srr, "NumberSeconds").text = \
            str(channel.sample_rate_ratio_number_seconds)

    _write_floattype(channel_elem, channel,
                     "clock_drift_in_seconds_per_sample", "ClockDrift")

    if channel.calibration_units:
        cu = etree.SubElement(channel_elem, "CalibrationUnits")
        etree.SubElement(cu, "Name").text = \
            str(channel.calibration_units)
        if channel.calibration_units_description:
            etree.SubElement(cu, "Description").text = \
                str(channel.calibration_units_description)
    _write_equipment(channel_elem, channel.sensor, "Sensor")
    _write_equipment(channel_elem, channel.pre_amplifier, "PreAmplifier")
    _write_equipment(channel_elem, channel.data_logger, "DataLogger")
    for equipment in channel.equipments:
        _write_equipment(channel_elem, equipment, "Equipment")

    if level == "channel":
        return

    if channel.response is not None:
        _write_response(channel_elem, channel.response)


def _write_io_units(parent, obj):
    sub = etree.SubElement(parent, "InputUnits")
    etree.SubElement(sub, "Name").text = \
        str(obj.input_units)
    etree.SubElement(sub, "Description").text = \
        str(obj.input_units_description)
    sub = etree.SubElement(parent, "OutputUnits")
    etree.SubElement(sub, "Name").text = \
        str(obj.output_units)
    etree.SubElement(sub, "Description").text = \
        str(obj.output_units_description)
    _write_extra(parent, obj)


def _write_polynomial_common_fields(element, polynomial):
    etree.SubElement(element, "ApproximationType").text = \
        str(polynomial.approximation_type)
    _write_floattype(element, polynomial,
                     "frequency_lower_bound", "FrequencyLowerBound",
                     cls=Frequency)
    _write_floattype(element, polynomial,
                     "frequency_upper_bound", "FrequencyUpperBound",
                     cls=Frequency)
    etree.SubElement(element, "ApproximationLowerBound").text = \
        _float_to_str(polynomial.approximation_lower_bound)
    etree.SubElement(element, "ApproximationUpperBound").text = \
        _float_to_str(polynomial.approximation_upper_bound)
    etree.SubElement(element, "MaximumError").text = \
        _float_to_str(polynomial.maximum_error)

    # Patch the polynomial to make sure the coefficients have the correct type.
    p = copy.deepcopy(polynomial)
    coeffs = []
    for _i, c in enumerate(polynomial.coefficients):
        if not isinstance(c, CoefficientWithUncertainties):
            c = CoefficientWithUncertainties(c)
        if "number" not in c.__dict__:
            c.__dict__["number"] = _i + 1
        coeffs.append(c)
    p.coefficients = coeffs

    _write_floattype_list(element, p,
                          "coefficients", "Coefficient",
                          additional_mapping={"number": "number"})
    _write_extra(element, polynomial)


def _write_response(parent, resp):
    attr = {}
    if resp.resource_id is not None:
        attr["resourceId"] = resp.resource_id
    parent = etree.SubElement(parent, "Response", attr)
    # write instrument sensitivity
    if resp.instrument_sensitivity is not None and \
            any(resp.instrument_sensitivity.__dict__.values()):
        ins_sens = resp.instrument_sensitivity
        sub = etree.SubElement(parent, "InstrumentSensitivity")
        etree.SubElement(sub, "Value").text = \
            _float_to_str(ins_sens.value)
        etree.SubElement(sub, "Frequency").text = \
            _float_to_str(ins_sens.frequency)
        _write_extra(sub, resp.instrument_sensitivity)
        _write_io_units(sub, ins_sens)
        freq_range_group = [True if getattr(ins_sens, key, None) is not None
                            else False
                            for key in ['frequency_range_start',
                                        'frequency_range_end',
                                        'frequency_range_db_variation']]
        # frequency range group properly described
        if all(freq_range_group):
            etree.SubElement(sub, "FrequencyStart").text = \
                _float_to_str(ins_sens.frequency_range_start)
            etree.SubElement(sub, "FrequencyEnd").text = \
                _float_to_str(ins_sens.frequency_range_end)
            etree.SubElement(sub, "FrequencyDBVariation").text = \
                _float_to_str(ins_sens.frequency_range_db_variation)
        # frequency range group not present
        elif not any(freq_range_group):
            pass
        # frequency range group only partly present
        else:
            msg = ("Frequency range group of instrument sensitivity "
                   "specification invalid")
            raise Exception(msg)
    # write instrument polynomial
    if resp.instrument_polynomial is not None:
        attribs = {}
        if resp.instrument_polynomial.name is not None:
            attribs['name'] = resp.instrument_polynomial.name
        if resp.instrument_polynomial.resource_id is not None:
            attribs['resourceId'] = resp.instrument_polynomial.resource_id
        sub = etree.SubElement(parent, "InstrumentPolynomial", attribs)
        etree.SubElement(sub, "Description").text = \
            str(resp.instrument_polynomial.description)
        _write_io_units(sub, resp.instrument_polynomial)
        _write_polynomial_common_fields(sub, resp.instrument_polynomial)
        _write_extra(sub, resp.instrument_polynomial)
    # write response stages
    for stage in resp.response_stages:
        _write_response_stage(parent, stage)
    _write_extra(parent, resp)


def _write_response_stage(parent, stage):
    attr = {'number': str(stage.stage_sequence_number)}
    if stage.resource_id is not None:
        attr["resourceId"] = stage.resource_id
    sub = etree.SubElement(parent, "Stage", attr)
    # do nothing for gain only response stages
    if type(stage) == ResponseStage:
        pass
    else:
        # create tag for stage type
        tagname_map = {PolesZerosResponseStage: "PolesZeros",
                       CoefficientsTypeResponseStage: "Coefficients",
                       ResponseListResponseStage: "ResponseList",
                       FIRResponseStage: "FIR",
                       PolynomialResponseStage: "Polynomial"}
        subel_attrs = {}
        if stage.name is not None:
            subel_attrs["name"] = str(stage.name)
        if stage.resource_id2 is not None:
            subel_attrs["resourceId"] = stage.resource_id2
        sub_ = etree.SubElement(sub, tagname_map[type(stage)], subel_attrs)
        # write operations common to all stage types
        _obj2tag(sub_, "Description", stage.description)
        sub__ = etree.SubElement(sub_, "InputUnits")
        _obj2tag(sub__, "Name", stage.input_units)
        _obj2tag(sub__, "Description", stage.input_units_description)
        sub__ = etree.SubElement(sub_, "OutputUnits")
        _obj2tag(sub__, "Name", stage.output_units)
        _obj2tag(sub__, "Description", stage.output_units_description)

        # write custom fields of respective stage type
        if type(stage) == ResponseStage:
            pass
        elif isinstance(stage, PolesZerosResponseStage):
            _obj2tag(sub_, "PzTransferFunctionType",
                     stage.pz_transfer_function_type)
            _obj2tag(sub_, "NormalizationFactor",
                     stage.normalization_factor)
            _write_floattype(sub_, stage, "normalization_frequency",
                             "NormalizationFrequency")
            _write_polezero_list(sub_, stage)
        elif isinstance(stage, CoefficientsTypeResponseStage):
            _obj2tag(sub_, "CfTransferFunctionType",
                     stage.cf_transfer_function_type)
            _write_floattype_list(sub_, stage,
                                  "numerator", "Numerator",
                                  additional_mapping={'number': 'number'},
                                  unit=False)
            _write_floattype_list(sub_, stage,
                                  "denominator", "Denominator",
                                  additional_mapping={'number': 'number'},
                                  unit=False)
        elif isinstance(stage, ResponseListResponseStage):
            for rlelem in stage.response_list_elements:
                sub__ = etree.SubElement(sub_, "ResponseListElement")
                _write_floattype(sub__, rlelem, "frequency", "Frequency")
                _write_floattype(sub__, rlelem, "amplitude", "Amplitude")
                _write_floattype(sub__, rlelem, "phase", "Phase")
        elif isinstance(stage, FIRResponseStage):
            _obj2tag(sub_, "Symmetry", stage.symmetry)
            _write_floattype_list(sub_, stage, "coefficients",
                                  "NumeratorCoefficient",
                                  additional_mapping={'number': 'i'})
        elif isinstance(stage, PolynomialResponseStage):
            _write_polynomial_common_fields(sub_, stage)

    if isinstance(stage, PolynomialResponseStage):
        # <StageGain> and <Decimation> were removed for <Polynomial> response
        # stages in StationXML 1.1
        if stage.decimation_input_sample_rate is not None:
            msg = ("Decimation is not allowed for Polynomial response "
                   "stages in StationXML schema > 1.0. Ignoring.")
            warnings.warn(msg)
        if stage.stage_gain not in (None, 1.0):
            msg = ("Stage gain is not allowed for Polynomial response "
                   "stages in StationXML schema > 1.0. Ignoring.")
            warnings.warn(msg)
    else:
        # write decimation
        if stage.decimation_input_sample_rate is not None:
            sub_ = etree.SubElement(sub, "Decimation")
            _write_floattype(sub_, stage, "decimation_input_sample_rate",
                             "InputSampleRate")
            _obj2tag(sub_, "Factor", stage.decimation_factor)
            _obj2tag(sub_, "Offset", stage.decimation_offset)
            _write_floattype(sub_, stage, "decimation_delay", "Delay")
            _write_floattype(sub_, stage, "decimation_correction",
                             "Correction")
        # write gain
        sub_ = etree.SubElement(sub, "StageGain")
        _obj2tag(sub_, "Value", stage.stage_gain)
        _obj2tag(sub_, "Frequency", stage.stage_gain_frequency)
    _write_extra(parent, stage)


def _write_external_reference(parent, ref):
    ref_elem = etree.SubElement(parent, "ExternalReference")
    etree.SubElement(ref_elem, "URI").text = ref.uri
    etree.SubElement(ref_elem, "Description").text = ref.description
    _write_extra(parent, ref)


def _write_equipment(parent, equipment, tag="Equipment"):
    if equipment is None:
        return
    attr = {}
    if equipment.resource_id is not None:
        attr["resourceId"] = equipment.resource_id
    equipment_elem = etree.SubElement(parent, tag, attr)

    # All tags are optional.
    _obj2tag(equipment_elem, "Type", equipment.type)
    _obj2tag(equipment_elem, "Description", equipment.description)
    _obj2tag(equipment_elem, "Manufacturer", equipment.manufacturer)
    _obj2tag(equipment_elem, "Vendor", equipment.vendor)
    _obj2tag(equipment_elem, "Model", equipment.model)
    _obj2tag(equipment_elem, "SerialNumber", equipment.serial_number)
    if equipment.installation_date:
        etree.SubElement(equipment_elem, "InstallationDate").text = \
            str(equipment.installation_date)
    if equipment.removal_date:
        etree.SubElement(equipment_elem, "RemovalDate").text = \
            str(equipment.removal_date)
    for calibration_date in equipment.calibration_dates:
        etree.SubElement(equipment_elem, "CalibrationDate").text = \
            str(calibration_date)
    _write_extra(parent, equipment)


def _write_site(parent, site):
    site_elem = etree.SubElement(parent, "Site")
    etree.SubElement(site_elem, "Name").text = site.name
    # Optional tags
    _obj2tag(site_elem, "Description", site.description)
    _obj2tag(site_elem, "Town", site.town)
    _obj2tag(site_elem, "County", site.county)
    _obj2tag(site_elem, "Region", site.region)
    _obj2tag(site_elem, "Country", site.country)
    _write_extra(parent, site)


def _write_comment(parent, comment):
    attribs = {}
    if comment.id is not None:
        attribs["id"] = str(comment.id)
    if comment.subject is not None:
        attribs["subject"] = str(comment.subject)
    comment_elem = etree.SubElement(parent, "Comment", attribs)
    etree.SubElement(comment_elem, "Value").text = comment.value
    if comment.begin_effective_time:
        etree.SubElement(comment_elem, "BeginEffectiveTime").text = \
            str(comment.begin_effective_time)
    if comment.end_effective_time:
        etree.SubElement(comment_elem, "EndEffectiveTime").text = \
            str(comment.end_effective_time)
    for author in comment.authors:
        _write_person(comment_elem, author, "Author")
    _write_extra(parent, comment)


def _write_data_availability(parent, data_availability):
    data_availability_elem = etree.SubElement(parent, "DataAvailability")
    if any(value is not None for value in (data_availability.start,
                                           data_availability.end)):
        # we need both start and end to write a valid Extent tag
        if any(value is None for value in (data_availability.start,
                                           data_availability.end)):
            # play safe and raise an exception
            msg = ("Both start/end need to be set to write a valid "
                   "DataAvailability Extent tag ('%s').") % data_availability
            raise ValueError(msg)
        # XXX foreign namespace custom attributes not yet implemented
        attribs = {'start': str(data_availability.start),
                   'end': str(data_availability.end)}
        etree.SubElement(data_availability_elem, "Extent", attribs)
    for span in data_availability.spans:
        _write_data_availability_span(data_availability_elem, span)
    _write_extra(data_availability_elem, data_availability)


def _write_data_availability_span(parent, span):
    if any(value is None for value in (span.start, span.end,
                                       span.number_of_segments)):
        msg = ("All of start/end/number_of_segments need to be set to write "
               " a valid DataAvailability Span tag ('%s').") % span
        raise ValueError(msg)
    attribs = {'start': str(span.start), 'end': str(span.end),
               'numberSegments': str(span.number_of_segments)}
    if span.maximum_time_tear is not None:
        attribs['maximumTimeTear'] = str(span.maximum_time_tear)
    # XXX foreign namespace custom attributes not yet implemented
    etree.SubElement(parent, "Span", attribs)


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
    _write_extra(parent, person)


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
    _write_extra(parent, phone)


def _write_element(parent, element, name):
    """
    Recursively write custom namespace elements.
    """
    custom_name = "{%s}%s" % (
        element['namespace'], name)  # name of the attribute/tag
    attrib = element.get("attrib", {})
    if hasattr(element, "type") and \
            element['type'].lower() in ("attribute", "attrib"):
        parent.set(custom_name, element['value'])
    else:  # if not a attribute, then create a tag
        sub = etree.SubElement(parent, custom_name, attrib=attrib)
        if isinstance(
                element['value'],
                compatibility.collections_abc.Mapping):  # nested extra tags
            for tagname, tag_element in element['value'].items():
                _write_element(sub, tag_element, tagname)
        else:
            sub.text = _float_to_str(element['value'])


def _write_extra(parent, obj):
    """
    Write information stored in custom tags/attributes in obj.extra.
    """
    if hasattr(obj, "extra"):
        for tagname, element in obj.extra.items():
            _write_element(parent, element, tagname)


def _convert(text, convert):
    if convert is str:
        return _convert_str(text)
    try:
        return convert(text)
    except Exception:
        None


def _convert_str(text):
    # we use future.builtins.str and are sure we have unicode here
    # lxml gives ``None`` for tags with empty text, best way to handle this is
    # probably to give back an empty string
    if not text:
        # Returning an empty string
        return ''
    return text


def _tag2obj(element, tag, convert):
    try:
        text = element.find(tag).text
    except Exception:
        return None
    return _convert(text, convert)


def _tags2obj(element, tag, convert):
    values = []
    for elem in element.findall(tag):
        values.append(_convert(elem.text, convert))
    return values


def _attr2obj(element, attr, convert):
    attribute = element.get(attr)
    if attribute is None:
        return None
    try:
        return convert(attribute)
    except Exception:
        None


def _obj2tag(parent, tag_name, tag_value):
    """
    If tag_value is not None, append a SubElement to the parent. The text of
    the tag will be tag_value.
    """
    if tag_value is None:
        return
    if isinstance(tag_value, float):
        text = _float_to_str(tag_value)
    else:
        text = str(tag_value)
    etree.SubElement(parent, tag_name).text = text


def _read_element(prefix, ns, element, extra):
    """
    Recursively read custom namespace elements and add them to extra.
    """
    # remove namespace from tag name
    _, name = element.tag.split("}")
    etree.register_namespace(prefix, ns)
    extra[name] = AttribDict()
    extra[name].namespace = ns
    if(len(element) > 0):  # element contains nested elements
        extra[name].value = AttribDict()
        for nested_el in element:
            _read_element(prefix, ns, nested_el, extra[name].value)
    else:  # element contains values
        extra[name].value = element.text
        if element.attrib:  # adds custom attributes dictionary to tag
            extra[name].attrib = element.attrib
    return extra


def _read_extra(element, obj):
    """
    Add information stored in custom tags/attributes in obj.extra.
    """
    # search all namespaces in current scope
    for prefix, ns in element.nsmap.items():
        # skip any fdsn namespaces,
        # we're not interested in StationXML defined tags here
        if re.match("(http|https)://www.(fdsn|w3).org/", ns):
            continue
        # process all elements of this custom namespace, if any
        if hasattr(obj, "extra"):
            extra = obj.extra
        else:
            extra = AttribDict()
            obj.extra = extra
        for el in element.iterfind("{%s}*" % ns):
            extra = _read_element(prefix, ns, el, extra)
    # process all attributes of custom namespaces, if any
    for key, value in element.attrib.items():
        # no custom namespace
        if "}" not in key:
            continue
        # separate namespace from tag name
        _t = etree.QName(key)
        ns, name = _t.namespace, _t.localname
        if re.match("(http|https)://www.(fdsn|w3).org/", ns):
            continue
        if hasattr(obj, "extra"):
            extra = obj.extra
        else:
            extra = AttribDict()
            obj.extra = extra
        extra[name] = {'value': str(value),
                       'namespace': '%s' % ns,
                       'type': 'attribute'}
