# -*- coding: utf-8 -*-
"""
Functions dealing with reading SiteXML.
Metadata is stored in a SERASite object.

:copyright:
    ORFEUS, 2025
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""

import inspect
from pathlib import Path
import re
import warnings

from lxml import etree

import obspy
from obspy.io.stationxml.core import _tag2obj, _attr2obj, _tags2obj
from obspy.core.inventory.util import ExternalReference
from obspy.io.sitexml.core import (SERASite, SiteDescription, SiteCharacterizationParameters, SERASiteOwner, 
                                   EC8, H800, BedrockDepth, GeologicalUnit, ResonanceFrequency, VelocityS30, 
                                   VelocityProfile, VelocityProfileData, ValueWithUncertainty,
                                   LiteratureSource)

# Define some constants for writing SiteXML files.
SCHEMA_VERSION = "1.2"
NAMESPACE = "http://www.orfeus-eu.org/xml/site/1"
#READABLE_VERSIONS = ("1.0", "1.1", "1.2")

def _ns(tagname):
        return "{%s}%s" % (NAMESPACE, tagname)

def _get_version_from_xmldoc(xmldoc):
    """
    Return SiteXML version string or ``None`` if parsing fails.
    """
    root = xmldoc.getroot()
    try:
        match = re.match(
            r'{http://www.orfeus-eu.org/xml/site/[0-9]+}SERA_quakeml',
            root.tag)
        assert match is not None
    except Exception:
        return None
    try:
        version = xmldoc.find(_ns("schemaVersion")).text
        #root.attrib["schemaVersion"]
    except KeyError:
        return None
    return version

def validate_sitexml(path_or_object):
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
    print(version)

    # Get the schema location.
    schema_location = Path(inspect.getfile(inspect.currentframe())).parent

    schema_location = schema_location / "data"
    schema_location = str(schema_location / ("QuakeML-SERA-%s.xsd" % version))

    if not Path(schema_location).exists():
        msg = "No schema file found to validate SiteXML version '%s'"
        raise ValueError(msg % version)

    xmlschema = etree.XMLSchema(etree.parse(schema_location))

    valid = xmlschema.validate(xmldoc)

    # Pretty error printing if the validation fails.
    if valid is not True:
        return (False, xmlschema.error_log)
    return (True, ())

def _read_sitexml(path_or_file_object):
    """
    Function reading a SiteXML file.

    :param path_or_file_object: File name or file like object.
    """
    #root = etree.parse(path_or_file_object).getroot()
    xmldoc = etree.parse(path_or_file_object)

    #namespace = "http://www.orfeus-eu.org/xml/site/1"

    created = obspy.UTCDateTime(xmldoc.find(_ns("created")).text)

    site_owner_element = xmldoc.find(_ns("siteOwner"))
    if site_owner_element is not None:
        site_owner = _read_site_owner(site_owner_element)

    site_description_element = xmldoc.find(_ns(
        "siteDescription"))
    if site_description_element is not None:
        site_description = _read_site_description(
            site_description_element)
    
    site_characterization_element = xmldoc.find(
        _ns("siteCharacterizationParameters"))
    if site_characterization_element is not None:
        site_characterization = _read_site_characterization(
            site_characterization_element)
     
    sera_site = SERASite(station_code="ARG1",
                         site_owner=site_owner, 
                         site_description=site_description, 
                         site_characterization=site_characterization,
                         created=created)

    return sera_site

def _read_site_owner(owner_element):
    """
    <siteOwner> element structure:

    - publicID, codeName, fullName
    - contact
        - person
            - personID, firstname, lastname, mbox, homepage
        - affiliation
            - department, function
            - institution
                - name, mbox, phone, homepage
                - postalAddress
                    - streetAddress, locality, postalCode
                    - country
                        - country, code
    """

    ownerID = _attr2obj(owner_element, "publicID", str)
    codeName = _tag2obj(owner_element, _ns("codeName"), str)
    fullName = _tag2obj(owner_element, _ns("fullName"), str)

    # To create the site_owner object we need codeName AND fullName to be present.
    # Otherwise, skip reading the rest of the siteOwner element and return None.
    if codeName is None or fullName is None:
        warnings.warn("Missing owner_codename and owner_fullname value. " \
                    "Processing of site owner element will be skipped.", UserWarning)
        return None
    
    site_owner = SERASiteOwner(owner_codename = codeName, 
                                owner_fullname = fullName,
                                ownerID = ownerID)
    
    # Read person element
    person_element = owner_element.find(_ns("contact")).find(_ns("person"))
    if person_element is not None:
        site_owner.personID = _attr2obj(person_element, "personID", str)
        site_owner.person_firstname = _tag2obj(person_element, _ns("firstname"), str)
        site_owner.person_lastname = _tag2obj(person_element, _ns("lastname"), str)
        site_owner.person_mbox = _tag2obj(person_element, _ns("mbox"), str)
        site_owner.person_homepage = _tag2obj(person_element, _ns("homepage"), str)
    
    # Read affiliation element
    affiliation_element = owner_element.find(_ns("contact")).find(_ns("affiliation"))
    if affiliation_element is None:
        return site_owner
    
    site_owner.affiliation_department = _tag2obj(affiliation_element, _ns("department"), str)
    site_owner.affiliation_function = _tag2obj(affiliation_element, _ns("function"), str)

    # Read institution element
    institution_element = affiliation_element.find(_ns("institution"))
    if institution_element is None:
        return site_owner
    
    site_owner.institution_name = _tag2obj(institution_element, _ns("name"), str) 
    site_owner.institution_mbox = _tag2obj(institution_element, _ns("mbox"), str) 
    site_owner.institution_phone = _tag2obj(institution_element, _ns("phone"), str)
    site_owner.institution_homepage = _tag2obj(institution_element, _ns("homepage"), str)

    identifier_element = institution_element.find(_ns("identifier"))
    site_owner.institution_ID = _tag2obj(identifier_element, _ns("resourceID"), str) 
    #resourceID_element = identifier_element.find(_ns("identifier"))
    #site_owner.institution_ID = institution_ID,

    # Read postalAddress element
    postal_address_element = institution_element.find(_ns("postalAddress"))
    if postal_address_element is None:
        return site_owner
    
    site_owner.address_street = _tag2obj(postal_address_element, _ns("streetAddress"), str) 
    site_owner.address_locality = _tag2obj(postal_address_element, _ns("locality"), str)
    site_owner.address_postal_code = _tag2obj(postal_address_element, _ns("postalCode"), str)
    
    # Read country element
    country_element = postal_address_element.find(_ns("country"))
    if country_element is None:
        return site_owner
    
    site_owner.address_country = _tag2obj(country_element, _ns("country"), str)
    site_owner.address_country_code = _tag2obj(country_element, _ns("code"), str) 

    return site_owner

def _read_site_description(site_description_element):
    """

    <siteDescription> element structure:

    - latitude, longitude, altitude, minDistanceFromStation, maxDistanceFromStation
    - OverallQindex
    - siteTopology
        - schemeA, schemeB
    - siteMorphology
        - morphology
        - siteClassEC8, siteClassEC8Qindex1, siteClassEC8Reference
        - bedrockDepth, bedrockDepthQindex1, bedrockDepthReference
        - h800, h800Qindex1, h800Reference
        - geologicalUnit, geologicalUnitQindex1, geologicalUnitReference, 
        - geologicalMapScale, geologicalUnitOGE
    """

    # SERA SiteXML allows latitude and longitude to be missing.
    #
    latitude = _read_value(site_description_element, "latitude", float)
    longitude = _read_value(site_description_element, "longitude", float)
    if latitude is None or longitude is None:
        warnings.warn("Missing latitude or longitude value. " \
                    "Processing of site description element " \
                    "will be skipped.", UserWarning)
        return None
    
    altitude = _read_value(site_description_element, "altitude", float)
    min_distance_from_station = _read_value(site_description_element, 
                                            "minDistanceFromStation", float)
    max_distance_from_station = _read_value(site_description_element, 
                                            "maxDistanceFromStation", float)

    site_description = SiteDescription(latitude=latitude, 
                                       longitude=longitude, 
                                       altitude=altitude,
                                       min_distance_from_station=min_distance_from_station,
                                       max_distance_from_station=max_distance_from_station)
    
    # Topology
    topology_element = site_description_element.find(_ns("siteTopology"))
    if topology_element is not None:
        site_description.topologyA = _tag2obj(topology_element, _ns("schemeA"), str)
        site_description.topologyB = _tag2obj(topology_element, _ns("schemeB"), str)

    # Qindex
    #site_description.overall_qindex = _read_value(site_description_element, "OverallQindex", float)
    
    # Morphology
    #
    # Everything else in siteDescription is under the siteMorphology element.
    # if this element is missing, return the site_description object created so far.
    #
    morphology_element = site_description_element.find(_ns("siteMorphology"))
    if morphology_element is None: 
        return site_description
    
    site_description.morphology = _tag2obj(morphology_element, _ns("morphology"), str)

    # EC8 Class
    ec8_value = _tag2obj(morphology_element, _ns("siteClassEC8"), str)
    if ec8_value is not None: 
        ec8_qindex = _read_value(morphology_element, "siteClassEC8Qindex1", float)
        [ec8_literature_source, ec8_file_resource] = _read_reference(
            morphology_element, "siteClassEC8Reference")
        site_description.ec8 = EC8(
                value = ec8_value,
                quality_index = ec8_qindex,
                literature_source = ec8_literature_source, 
                file_resource = ec8_file_resource)

    # H800
    [h800_value, h800_uncertainty] = _read_value_with_uncertainty(
        morphology_element, "h800", int)
    if h800_value is not None: 
        h800_qindex = _read_value(morphology_element, "h800Qindex1", float)
        [h800_literature_source, h800_file_resource] = _read_reference(
            morphology_element, "h800Reference")
        site_description.h800 = H800(
                value = ValueWithUncertainty(h800_value, h800_uncertainty, int), 
                quality_index = h800_qindex,
                literature_source = h800_literature_source, 
                file_resource = h800_file_resource)

    # Bedrock Depth
    [bdepth_value, bdepth_uncertainty] = _read_value_with_uncertainty(
        morphology_element, "bedrockDepth", int)
    if bdepth_value is not None: 
        bdepth_qindex = _read_value(morphology_element, "bedrockDepthQindex1", float)
        [bdepth_literature_source, bdepth_file_resource] = _read_reference(
            morphology_element, "bedrockDepthReference")
        site_description.bedrock_depth = BedrockDepth(
                value = ValueWithUncertainty(bdepth_value, bdepth_uncertainty, int),
                quality_index = bdepth_qindex,
                literature_source = bdepth_literature_source,
                file_resource = bdepth_file_resource)
    
    # Geological Unit
    gunit_value = _tag2obj(morphology_element, _ns("geologicalUnit"), str)
    if gunit_value is not None:
        gunit_qindex = _read_value(morphology_element, "geologicalUnitQindex1", float)
        gunit_map_scale = _tag2obj(morphology_element, _ns("geologicalMapScale"), str)
        gunit_oge = _tag2obj(morphology_element, _ns("geologicalUnitOGE"), str)
        [gunit_literature_source, gunit_file_resource] = _read_reference(
            morphology_element, "geologicalUnitReference")
        site_description.geological_unit = GeologicalUnit(
                value = gunit_value, 
                quality_index = gunit_qindex,
                geological_map_scale = gunit_map_scale,
                geological_unit_OGE = gunit_oge,
                literature_source = gunit_literature_source,
                file_resource = gunit_file_resource)
    
    return site_description

def _read_site_characterization(site_char_element):
    """
    <siteCharacterizationParameters> element structure:

    - PublicID (attr)
    - Analysis
        - PublicID (attr)
        - resonanceFrequency, resonanceFrequencyQIndex1, 
        - resonanceFrequencyReference, resonanceFrequencyMethod
        - velocityS30, velocityS30Qindex1, velocityS30Reference, 
        - velocityS30Method, velocityS30ManualIndex, velocityS30MethodCombIndex
        - velocityProfileCount
        - sptLogsCount
        - cptLogsCount
        - boreholeLogsCount
    - VelocityProfile
        - layerCount
        - velocityProfileData
            - density
            - velocityP
            - velocityS
            - layerThickness
                - layerTopDepth
                - layerBottomDepth
    - velocityProfileQindex1
    - velocityProfileReference
    """
    
    # Create an empty site_characterization object
    #
    publicID = _attr2obj(site_char_element, "publicID", str)
    site_char_obj = SiteCharacterizationParameters(publicID = publicID)
    
    ### Read Analysis element. Store values in site_char_obj.
    #
    analysis_element = site_char_element.find(_ns("Analysis"))
    if analysis_element is not None:
        _read_analysis(analysis_element, site_char_obj)
    
    _read_velocity_profile(site_char_element, site_char_obj)

    """
    # If both analysis and velocityProfile elements are missing 
    # return with an empty site_characterization object
    if analysis_element is None and velocity_profile_element is None:
        return None
    
    ### Read Velocity Profile. Store values in site_char_obj.
    #
    if velocity_profile_element is not None:
        _read_velocity_profile(site_char_element, site_char_obj)
    """
    # This could be empty if no Analysis or VP data is present in SiteXML
    # Maybe return None in this case
    return site_char_obj

def _read_analysis(analysis_element, site_char_obj):
    """
    Read the <Analysis> element

    :type analysis_element: :class:`~lxml.etree._Element`
    :param analysis_element: 
    :type site_characterization_obj: :class:`~obspy.core.io.sitexml.SiteCharacterizationParameters`
    :param site_characterization_obj: The SiteCharacterizationParameters object to store the values 
        read from the <Analysis> element. It should be pre-initialized by the calling function.
    """
    site_char_obj.analysis_publicID = _attr2obj(analysis_element, "publicID", str)

    # Resonance Frequency 
    [rfreq_value, rfreq_uncertainty] = \
        _read_value_with_uncertainty(analysis_element, "resonanceFrequency", float)
    if rfreq_value is not None: 
        rfreq_qindex = _read_value(analysis_element, "resonanceFrequencyQindex1", float)
        rfreq_methods = _tags2obj(analysis_element, _ns("resonanceFrequencyMethod"), str)
        [rfreq_literature_source, rfreq_file_resource] = \
            _read_reference(analysis_element, "resonanceFrequencyReference")

        site_char_obj.resonance_frequency = ResonanceFrequency(
                value = ValueWithUncertainty(rfreq_value, rfreq_uncertainty, float),
                quality_index = rfreq_qindex,
                methods = rfreq_methods,
                literature_source = rfreq_literature_source,
                file_resource = rfreq_file_resource)

    # Velocity S30
    [vs30_value, vs30_uncertainty] = \
        _read_value_with_uncertainty(analysis_element, "velocityS30", float)
    if vs30_value is not None: 
        vs30_qindex = _read_value(analysis_element, "velocityS30Qindex1", float)
        vs30_methods = _tags2obj(analysis_element, _ns("velocityS30Method"), str)
        vs30_methods_index = _tag2obj(analysis_element, _ns("velocityS30MethodCombIndex"), str)
        vs30_manual_index = _tag2obj(analysis_element, _ns("velocityS30ManualIndex"), str)
        [vs30_literature_source, vs30_file_resource] = \
            _read_reference(analysis_element, "velocityS30Reference")

        site_char_obj.velocity_s30 = VelocityS30(
                value = ValueWithUncertainty(vs30_value, vs30_uncertainty, float),
                quality_index = vs30_qindex,
                methods = vs30_methods,
                method_combined_quality_index = vs30_methods_index,
                manual_quality_index = vs30_manual_index,
                literature_source = vs30_literature_source,
                file_resource = vs30_file_resource)

    site_char_obj.velocity_profile_count = \
        _read_value(analysis_element, "velocityProfileCount", int)
    site_char_obj.spt_logs_count = \
        _read_value(analysis_element, "sptLogsCount", int)
    site_char_obj.cpt_logs_count = \
        _read_value(analysis_element, "cptLogsCount", int)
    site_char_obj.borehole_logs_count = \
        _read_value(analysis_element, "boreholeLogsCount", int)

def _read_velocity_profile(site_char_element, site_char_obj):
    """
    Read the <VelocityProfile> element

    :type velocity_profile_element: :class:`~lxml.etree._Element`
    :param velocity_profile_element: 
    :type site_characterization_obj: :class:`~obspy.core.io.sitexml.SiteCharacterizationParameters`
    :param site_characterization_obj: The SiteCharacterizationParameters object to store the values read from the <VelocityProfile> element. 
                                      It should be pre-initialized by the calling function.
    """

    vp_element_list=site_char_element.findall(_ns("VelocityProfile"))
    vp_qindex = _read_value(site_char_element, "velocityProfileQindex1", float)
    [vp_literature_source, vp_file_resource] = \
            _read_reference(site_char_element, "velocityProfileReference")

    # At least one VelocityProfile or a velocityProfileReference 
    # should be present in SiteXML in order to create the VelocityProfile object
    if len(vp_element_list) == 0 \
            and vp_literature_source is None \
            and vp_file_resource is None:
        return None

    site_char_obj.velocity_profile = \
            VelocityProfile(velocity_profile_data = [],    # We will fill this later
                            quality_index = vp_qindex,
                            literature_source = vp_literature_source,
                            file_resource = vp_file_resource)

    if site_char_obj.velocity_profile_count is None \
        or site_char_obj.velocity_profile_count != len(vp_element_list):
            warnings.warn("Number of Velocity Profiles in SiteXML " \
                    "doesn't much the <velocityProfileCount> value", UserWarning)
        
    for vp_element in vp_element_list:
        layer_count = _read_value(vp_element, "layerCount", int)
        vp_data_element = vp_element.find(_ns("velocityProfileData"))
        if vp_data_element is not None:
            vp_data = VelocityProfileData(layer_count=layer_count,
                                          density=[], velocityP=[], velocityS=[],
                                          top_depth=[], bottom_depth=[])
        _read_velocity_profile_data(vp_data_element, 
                                    vp_data, vp_element_list.index(vp_element))
        site_char_obj.velocity_profile.velocity_profile_data.append(vp_data)
    
    return None

def _read_velocity_profile_data(vp_data_element, vp_data, vp_no):

    layer_count = vp_data.layer_count
    density_list = vp_data_element.findall(_ns("density"))
    velocityP_list = vp_data_element.findall(_ns("velocityP"))
    velocityS_list = vp_data_element.findall(_ns("velocityS"))
    layerThickness_list = vp_data_element.findall(_ns("layerThickness"))
    
    if not all([x == layer_count for x in (len(density_list), len(velocityP_list), 
                                           len(velocityS_list), len(layerThickness_list))]):
         warnings.warn("layerCount value '%s' of Velocity Profile '%s' doesn't much " 
                    "the number of child elements: " 
                    "density: '%s', velocityP: '%s', velocityS: '%s', layerThickness: '%s'" 
                    % (layer_count, vp_no, len(density_list), len(velocityP_list), 
                    len(velocityS_list), len(layerThickness_list)), UserWarning)
         # Set layer_count to match the max among all length values
         return

    for layer in range(0, layer_count):
        
        density_value = _tag2obj(density_list[layer], _ns("value"), float)
        density_uncertainty = _tag2obj(density_list[layer], _ns("uncertainty"), float)
        if density_value:
            vp_data.density.append(ValueWithUncertainty(density_value, density_uncertainty, float))

        velocityP_value = _tag2obj(velocityP_list[layer], _ns("value"), float)
        velocityP_uncertainty = _tag2obj(velocityP_list[layer], _ns("uncertainty"), float)
        if velocityP_value:
            vp_data.velocityP.append(ValueWithUncertainty(velocityP_value, velocityP_uncertainty, float))

        velocityS_value = _tag2obj(velocityS_list[layer], _ns("value"), float)
        velocityS_uncertainty = _tag2obj(velocityS_list[layer], _ns("uncertainty"), float)
        if velocityS_value:
            vp_data.velocityS.append(ValueWithUncertainty(velocityS_value, velocityS_uncertainty, float))

        [top_depth_value, top_depth_uncer] = \
            _read_value_with_uncertainty(layerThickness_list[layer], 
                                         "layerTopDepth", float)
        if top_depth_value:
            vp_data.top_depth.append(ValueWithUncertainty(top_depth_value, top_depth_uncer, float))

        [bottom_depth_value, bottom_depth_uncer] = \
            _read_value_with_uncertainty(layerThickness_list[layer],
                                         "layerBottomDepth", float)
        #print(top_depth_value, bottom_depth_value)
        if bottom_depth_value:
            vp_data.bottom_depth.append(ValueWithUncertainty(bottom_depth_value, bottom_depth_uncer, float))
        #print(top_depth_value, bottom_depth_value)
        #print(vp_data.top_depth[layer].value, vp_data.bottom_depth[layer].value)


def _read_reference(parent, tag):
    reference_element = parent.find(_ns(tag))
    if reference_element is None:
        return None, None

    literature_source_element = reference_element.find(_ns("literatureSource"))
    literature_source = (
        _read_literature_source(literature_source_element)
        if literature_source_element is not None
        else None
    )

    file_resource_element = reference_element.find(_ns("FileResource"))
    file_resource = (
        _read_file_resource(file_resource_element)
        if file_resource_element is not None
        else None
    )

    return literature_source, file_resource

def _read_literature_source(literature_source_element):
    """
    Read a literatureSource element.
    Return an object only if title or doi is provided
    """
    title = _tag2obj(literature_source_element, _ns("title"), str)
    first_author = _tag2obj(literature_source_element, _ns("firstAuthor"), str)
    secondary_authors = _tag2obj(literature_source_element, _ns("secondaryAuthors"), str)
    year = _tag2obj(literature_source_element, _ns("year"), str)
    booktitle = _tag2obj(literature_source_element, _ns("booktitle"), str)
    doi = _tag2obj(literature_source_element, _ns("DOI"), str)
    
    language_element = literature_source_element.find(_ns("language"))
    if language_element is not None:
        language = _tag2obj(language_element, _ns("code"), str)

    if title or doi:
        return LiteratureSource(title=title, 
                                first_author=first_author, 
                                secondary_authors=secondary_authors,
                                year=year,
                                booktitle=booktitle,
                                language=language,
                                doi=doi)
    else:
        return None

def _read_file_resource(file_resource_element):
    """
    Read a fileResource element.
    Return an object only if uri is provided
    """
    uri = _tag2obj(file_resource_element, _ns("url"), str)
    description = _tag2obj(file_resource_element, _ns("description"), str)

    if uri:
        return ExternalReference(uri=uri, description=description)
    else:
        return None

def _read_value(parent, tag, type):
    """
    Method used to read a value 
    from an element of the following structure
    
    <xs:element name="parent">
        <xs:element name="tag">
		    <xs:element name="value" type="type"/>
        </xs:element>
	</xs:element>
    """
    element = parent.find(_ns(tag))
    if element is None:
        return None
    return _tag2obj(element, _ns("value"), type)

def _read_value_with_uncertainty(parent, tag, type):
    """
    Method used to read a value / uncertainty pair 
    from an element of the following structure
    
    <xs:element name="parent">
        <xs:element name="tag">
            <xs:element name="value" type="type"/>
            <xs:element name="uncertainty" type="type"/>
        </xs:element>
    </xs:element>
    """
    element = parent.find(_ns(tag))
    if element is None:
        print(parent)
        return None, None

    value = _tag2obj(element, _ns("value"), type)
    uncertainty = _tag2obj(element, _ns("uncertainty"), type)

    return value, uncertainty
