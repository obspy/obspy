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
import io
from pathlib import Path
import re
import warnings

from lxml import etree

import obspy
from obspy.io.stationxml.core import _tag2obj, _attr2obj, _tags2obj
from obspy.core.inventory.util import ExternalReference
from .core import (SERASite, SERASiteOwner, SiteDescription, Analysis,
                   EC8, H800, BedrockDepth, GeologicalUnit, ResonanceFrequency,
                   VelocityProfileSurvey, VelocityProfile, VelocityProfileData,
                   VelocityS30, ValueWithUncertainty, LiteratureSource)
from .util import SiteXMLValidationError

# Define some constants for writing SiteXML files.
SCHEMA_VERSION = "1.3"
NAMESPACE = "http://www.orfeus-eu.org/xml/site/1"

def _ns(tagname):
    """
    Return a namespaced SiteXML tag name for lxml lookups.

    :rtype: str
    """
    return "{%s}%s" % (NAMESPACE, tagname)

def _get_version_from_xmldoc(xmldoc):
    """
    Return SiteXML version string or ``None`` if parsing fails.

    :rtype: str or None
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
        version = root.attrib["schemaVersion"]
    except KeyError:
        return None
    return version

def _is_sitexml(path_or_file_object):
    """
    Simple function checking if the passed object contains a valid SiteXML
    file. Returns True of False.

    The test is not exhaustive - 
    it only checks the root tag and the schema version. 

    :param path_or_file_object: File name or file like object.
    :rtype: bool
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
        if version != SCHEMA_VERSION:
            warnings.warn("The SiteXML file has version %s, ObsPy can "
                          "read version (%s)." % (
                              version, ", ".join(SCHEMA_VERSION)))
        return True
    finally:
        # Make sure to reset file pointer position.
        try:
            path_or_file_object.seek(current_position, 0)
        except Exception:
            pass

def validate_sitexml(path_or_object):
    """
    Checks if the given path is a valid SiteXML file.

    Returns a tuple. The first item is a boolean describing if the validation
    was successful or not. The second item is a list of all found validation
    errors, if existent.

    :param path_or_object: File name or file like object. Can also be an etree
        element.
    :rtype: tuple
    """
    if hasattr(path_or_object, "tell") and hasattr(path_or_object, "seek"):
        current_position = path_or_object.tell()
    else:
        current_position = None

    try:
        if isinstance(path_or_object, etree._Element):
            xmldoc = path_or_object
        else:
            try:
                xmldoc = etree.parse(path_or_object)
            except etree.XMLSyntaxError:
                return (False, ("Not a XML file.",))
        version = _get_version_from_xmldoc(xmldoc)

        # Get the schema location.
        schema_location = Path(inspect.getfile(inspect.currentframe())).parent
        schema_location = schema_location / "data"
        schema_location = str(schema_location / ("QuakeML-SERA-%s.xsd" % version))
        
        if not Path(schema_location).exists():
            msg = "No schema file found to validate SiteXML version '%s'"
            raise SiteXMLValidationError(msg % version)

        xmlschema = etree.XMLSchema(etree.parse(schema_location))

        valid = xmlschema.validate(xmldoc)

        # Pretty error printing if the validation fails.
        if valid is not True:
            return (False, xmlschema.error_log)
        return (True, ())
    finally:
        if current_position is not None:
            try:
                path_or_object.seek(current_position, 0)
            except Exception:
                pass

def read_sitexml(path_or_file_object):
    """
    Function reading a SiteXML file.

    :param file_or_file_object: The file name or file-like object to read from.
    :rtype: :class:`~obspy.io.sitexml.core.SERASite`

    Returns a SERASite object with metadata read from the provided SiteXML file.
    At least site owner and site description metadata should be present in XMl file 
    in order to create the SERASite object.

    Example

    >>> from obspy.io.sitexml.sitexml import read_sitexml
    >>> site = read_sitexml("site.xml")

    """
    validates, errors = validate_sitexml(path_or_file_object)
    if validates is False:
        msg = "The provided SiteXML file fails to validate against the schema.\n"
        for err in errors:
            msg += "\t%s\n" % err
        raise SiteXMLValidationError(msg)
        
    root = etree.parse(path_or_file_object).getroot()
    
    siteID = _attr2obj(root, "publicID", str)
    created = obspy.UTCDateTime(root.find(_ns("creationTime")).text)

    site_owner_element = root.find(_ns("siteOwner"))
    if site_owner_element is not None:
        site_owner = _read_site_owner(site_owner_element)

    site_description_element = root.find(_ns(
        "siteDescription"))
    if site_description_element is not None:
        site_description = _read_site_description(
            site_description_element)
    
    # Create the SERA_Site object only if both 
    # site_owner and site_description exists
    #
    if site_owner and site_description:
        sera_site = SERASite(site_owner = site_owner, 
                         site_description = site_description, 
                         resource_id = siteID,
                         created = created)
    else:
        raise SiteXMLValidationError(
            "Missing site owner and/or site description in provided SiteXML file."
        )
    
    # Analysis element is optional
    #
    analysis_element_list = root.findall(_ns("analysis"))
    if len(analysis_element_list) != 0:
        analysis = []
        for analysis_element in analysis_element_list:
            analysis.append(_read_analysis(analysis_element))
        sera_site.analysis = analysis
        
    # Read External References
    #
    ref_element_list = root.findall(_ns("externalReference"))
    if len(ref_element_list) != 0:
        references = []
        for reference_element in ref_element_list:
            references.append(_read_external_reference(reference_element))
        sera_site.external_references = references

    return sera_site

def _read_site_owner(owner_element):
    """
    <siteOwner> element structure:

    - publicID (attribute)
    - codeName, fullName
    - contact
        - person
            - publicID (attribute) 
            - firstname, lastname, mbox, homepage
        - affiliation
            - department, function
            - institution
                - publicID (attribute)
                - name, mbox, phone, homepage
                - postalAddress
                    - streetAddress, locality, postalCode
                    - country
                        - country, code
    :rtype: :class:`~obspy.io.sitexml.core.SERASiteOwner`
    """

    ownerID = _attr2obj(owner_element, "publicID", str)
    codeName = _tag2obj(owner_element, _ns("codeName"), str)
    fullName = _tag2obj(owner_element, _ns("fullName"), str)

    # Read person element
    contact_element = owner_element.find(_ns("contact"))
    person_element = (
        contact_element.find(_ns("person"))
        if contact_element is not None
        else None
    )
    if person_element is None:
        personID = None
        person_firstname = None
        person_lastname = None
        person_mbox = None
        person_homepage = None
    else:
        personID = _attr2obj(person_element, "publicID", str)
        person_firstname = _tag2obj(
            person_element, _ns("firstname"), str)
        person_lastname = _tag2obj(person_element, _ns("lastname"), str)
        person_mbox = _tag2obj(person_element, _ns("mbox"), str)
        person_homepage = _tag2obj(person_element, _ns("homepage"), str)

    site_owner = SERASiteOwner(
        owner_codename=codeName,
        owner_fullname=fullName,
        ownerID=ownerID,
        personID=personID,
        person_firstname=person_firstname,
        person_lastname=person_lastname,
        person_mbox=person_mbox,
        person_homepage=person_homepage)
    
    # Read affiliation element
    affiliation_element = contact_element.find(_ns("affiliation"))
    if affiliation_element is None:
        return site_owner
    
    site_owner.affiliation_department = _tag2obj(affiliation_element, _ns("department"), str)
    site_owner.affiliation_function = _tag2obj(affiliation_element, _ns("function"), str)

    # Read institution element
    institution_element = affiliation_element.find(_ns("institution"))
    if institution_element is None:
        return site_owner
    
    site_owner.institutionID = _attr2obj(institution_element, "publicID", str) 
    site_owner.institution_name = _tag2obj(institution_element, _ns("name"), str) 
    site_owner.institution_mbox = _tag2obj(institution_element, _ns("mbox"), str) 
    site_owner.institution_phone = _tag2obj(institution_element, _ns("phone"), str)
    site_owner.institution_homepage = _tag2obj(institution_element, _ns("homepage"), str)

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

    - publicID (attribute)
    - station_code, latitude, longitude, altitude, minDistanceFromStation, maxDistanceFromStation
    - OverallQindex
    - siteTopography
        - schemaA, schemaB
    - siteMorphology
        - morphology
        - siteClassEC8
            - value, qualityIndex, reference
        - bedrockDepth
            - value, qualityIndex, reference
        - h800
            - value, qualityIndex, reference
        - geologicalUnit
            - value, geologicalMapScale, geologicalUnitOGE,
              qualityIndex, reference
    - preferredSiteAnalysisID
    - preferredVelocityProfileID
    - comment (0-unbounded)
    :rtype: :class:`~obspy.io.sitexml.core.SiteDescription`
    """
    resource_id = _attr2obj(site_description_element, "publicID", str) 
    station_code = _tag2obj(site_description_element, _ns("station"), str)
    
    latitude = _tag2obj(site_description_element, _ns("latitude"), float)
    longitude = _tag2obj(site_description_element, _ns("longitude"), float)
    if resource_id is None or latitude is None or longitude is None:
        raise SiteXMLValidationError(
            "Missing required site description publicID, latitude or "
            "longitude value."
        )
    
    site_description = SiteDescription(resource_id=resource_id,
                                       station_code=station_code, 
                                       latitude=latitude, 
                                       longitude=longitude)
    
    site_description.altitude = _tag2obj(site_description_element, _ns("altitude"), float)
    site_description.min_distance_from_station = _tag2obj(
        site_description_element, _ns("minDistanceFromStation"), float)
    site_description.max_distance_from_station = _tag2obj(
        site_description_element, _ns("maxDistanceFromStation"), float)
    
    # Topography
    topography_element = site_description_element.find(_ns("siteTopography"))
    if topography_element is not None:
        site_description.topographyA = _tag2obj(topography_element, _ns("schemaA"), str)
        site_description.topographyB = _tag2obj(topography_element, _ns("schemaB"), str)

    # Morphology
    #
    morphology_element = site_description_element.find(_ns("siteMorphology"))
    if morphology_element is not None: 
        _read_morphology(morphology_element, site_description)
    
    site_description.preferred_site_analysisID = \
        _tag2obj(site_description_element, _ns("preferredSiteAnalysisID"), str)
    site_description.preferred_velocity_profileID = \
        _tag2obj(site_description_element, _ns("preferredVelocityProfileID"), str)
 
    # Overall Quality Index
    site_description.overall_quality_index = \
        _tag2obj(site_description_element, _ns("overallQindex"), float)
    # Comments
    #
    return site_description

def _read_morphology(morphology_element, site_description_obj):
    """
     <siteMorphology> element structure:

    - siteMorphology
        - morphology
        - siteClassEC8
            - value, qualityIndex, reference
        - bedrockDepth
            - value, qualityIndex, reference
        - h800
            - value, qualityIndex, reference
        - geologicalUnit
            - value, geologicalMapScale, geologicalUnitOGE,
              qualityIndex, reference
    :rtype: None
    """
    site_description_obj.morphology = _tag2obj(morphology_element, _ns("morphology"), str)

    site_description_obj.ec8 = _read_site_indicator(
        morphology_element, "siteClassEC8", EC8)
    site_description_obj.bedrock_depth = _read_site_indicator(
        morphology_element, "bedrockDepth", BedrockDepth,
        value_with_uncertainty=True)
    site_description_obj.h800 = _read_site_indicator(
        morphology_element, "h800", H800,
        value_with_uncertainty=True)
    site_description_obj.geological_unit = _read_site_indicator(
        morphology_element, "geologicalUnit", GeologicalUnit)
    
def _read_analysis(analysis_element):
    """
    Read the <Analysis> element

    :type analysis_element: :class:`~lxml.etree._Element`
    :param analysis_element: 

    Returns:
    :type analysis_obj: :class:`~obspy.core.io.sitexml.core.Analysis`
    :param analysis_obj: The Analysis object to store the values 
        read from the <Analysis> element.
    :rtype: :class:`~obspy.io.sitexml.core.Analysis`

    <Analysis> element structure:

    - Analysis [List]
        - PublicID (attr)
        - creationTime
        - resonanceFrequency
            - value, method, qualityIndex, reference
        - velocityS30
            - value, method, manualIndex, methodCombIndex,
              qualityIndex, reference
        - sptLogsCount
        - cptLogsCount
        - boreholeLogsCount
        - velocityProfile
            - profile [List]
                - PublicID (attr)
                - layerCount
                - velocityProfileData [List]
                    - density
                    - velocityP
                    - velocityS
                    - layerThickness
                        - layerTopDepth
                        - layerBottomDepth
            - qualityIndex
            - reference
    """
    
    resource_id = _attr2obj(analysis_element, "publicID", str)
    site_descriptionID = _tag2obj(
        analysis_element, _ns("siteDescriptionID"), str)

    analysis_obj = Analysis(resource_id=resource_id,
                            site_descriptionID=site_descriptionID)
    
    creation_time = _tag2obj(analysis_element, _ns("creationTime"), str)
    if creation_time is not None:
        analysis_obj.creation_date = obspy.UTCDateTime(creation_time)
    analysis_obj.resonance_frequency = _read_site_indicator(
        analysis_element, "resonanceFrequency", ResonanceFrequency,
        value_with_uncertainty=True)
    analysis_obj.velocity_s30 = _read_site_indicator(
        analysis_element, "velocityS30", VelocityS30,
        value_with_uncertainty=True)

    analysis_obj.spt_logs_count = \
        _tag2obj(analysis_element, _ns("sptLogsCount"), int)
    analysis_obj.cpt_logs_count = \
        _tag2obj(analysis_element, _ns("cptLogsCount"), int)
    analysis_obj.borehole_logs_count = \
        _tag2obj(analysis_element, _ns("boreholeLogsCount"), int)
    
    _read_velocity_profile(analysis_element, analysis_obj)

    return analysis_obj

def _read_velocity_profile(analysis_element, analysis_obj):
    """
    Read the <velocityProfile> element

    :type analysis_element: :class:`~lxml.etree._Element`
    :param analysis_element: 
    :type analysis_obj: :class:`~obspy.core.io.sitexml.core.Analysis`
    :param analysis_obj: The Analysis object to store the values read from the <velocityProfile> element. 
                        It should be pre-initialized by the calling function.
    :rtype: :class:`~obspy.io.sitexml.core.VelocityProfileSurvey`
    """

    velocity_profile_element = analysis_element.find(_ns("velocityProfile"))
    if velocity_profile_element is None:
        return None

    vp_element_list = velocity_profile_element.findall(_ns("profile"))
    value = _tag2obj(velocity_profile_element, _ns("qualityIndex"), float)
    vp_qindex = value if value is not None else 0
    [vp_literature_source, vp_external_reference] = \
            _read_reference(velocity_profile_element)

    # At least one profile or a reference
    # should be present in SiteXML in order to create the VelocityProfileSurvey object
    if len(vp_element_list) == 0 \
            and vp_literature_source is None \
            and vp_external_reference is None:
        return None

    analysis_obj.velocity_profile_survey = \
            VelocityProfileSurvey(velocity_profiles = [],    # We will fill this later
                            quality_index = vp_qindex,
                            literature_source = vp_literature_source,
                            external_reference = vp_external_reference)

    # Go through the list of Velocity Profiles. 
    # For each velocityProfile tree element create a VelocityProfile object.
    #
    for vp_element in vp_element_list:
        resource_id = _attr2obj(vp_element, "publicID", str)
        layer_count = _tag2obj(vp_element, _ns("layerCount"), int)
        vp_data_element_list = vp_element.findall(_ns("velocityProfileData"))
        vp_data_list = []
        
        # Go through the velocityProfileData elements. 
        # For each velocityProfileData tree element create a VelocityProfileData object
        # and add it to the VelocityProfile object
        #      
        if vp_data_element_list is not None:
            for vp_data_element in vp_data_element_list:
                vp_data = _read_velocity_profile_data(vp_data_element)
                vp_data_list.append(vp_data)

        vp = VelocityProfile(resource_id=resource_id,
                             velocity_profile_data=vp_data_list,
                             layer_count=layer_count)
    
        analysis_obj.velocity_profile_survey.velocity_profiles.append(vp)

def _read_velocity_profile_data(vp_data_element):
    """
    Read one velocityProfileData element into a layer object.

    :rtype: :class:`~obspy.io.sitexml.core.VelocityProfileData`
    """

    density = _read_value_with_uncertainty(vp_data_element, "density", float)
    velocityS = _read_value_with_uncertainty(vp_data_element, "velocityS", float)
    velocityP = _read_value_with_uncertainty(vp_data_element, "velocityP", float)
    
    geometry_element = vp_data_element.find(_ns("layerThickness"))
    if geometry_element is None:
        raise SiteXMLValidationError(
            "velocityProfileData requires a layerThickness element."
        )
    top_depth = _read_value_with_uncertainty(geometry_element, "layerTopDepth", float)
    bottom_depth = _read_value_with_uncertainty(geometry_element, "layerBottomDepth", float)

    if top_depth is None:
        raise SiteXMLValidationError(
            "velocityProfileData requires layerTopDepth."
        )
    
    vp_data = VelocityProfileData(top_depth = top_depth,
                                bottom_depth = bottom_depth,
                                density = density,
                                velocityS = velocityS,
                                velocityP = velocityP
                                )
    return vp_data

def _read_site_indicator(parent, site_indicator_name, site_indicator_cls,
                         value_with_uncertainty=False):
    """
    Read one nested SiteXML site indicator into its API object.

    :rtype: :class:`~obspy.io.sitexml.core.SiteIndicator` or None
    """
    indicator_element = parent.find(_ns(site_indicator_name))
    if indicator_element is None:
        return None

    if value_with_uncertainty:
        indicator_value = _read_value_with_uncertainty(
            indicator_element, "value", float)
    else:
        indicator_value = _tag2obj(indicator_element, _ns("value"), str)

    value = _tag2obj(indicator_element, _ns("qualityIndex"), float)
    quality_index = value if value is not None else 0
    literature_source, external_reference = _read_reference(indicator_element)

    kwargs = {
        "value": indicator_value,
        "quality_index": quality_index,
        "literature_source": literature_source,
        "external_reference": external_reference,
    }

    if site_indicator_name in ("resonanceFrequency", "velocityS30"):
        kwargs["methods"] = _tags2obj(indicator_element, _ns("method"), str)

    if site_indicator_name == "geologicalUnit":
        kwargs["geological_map_scale"] = _tag2obj(
            indicator_element, _ns("geologicalMapScale"), str)
        kwargs["geological_unit_OGE"] = _tag2obj(
            indicator_element, _ns("geologicalUnitOGE"), str)

    if site_indicator_name == "velocityS30":
        kwargs["method_combined_qindex"] = _tag2obj(
            indicator_element, _ns("methodCombIndex"), str)
        kwargs["manual_qindex"] = _tag2obj(
            indicator_element, _ns("manualIndex"), str)

    return site_indicator_cls(**kwargs)

def _read_reference(parent):
    """
    Read literature and external references from a site indicator element.

    :rtype: tuple
    """
    literature_source_element = parent.find(_ns("literatureSource"))
    literature_source = (
        _read_literature_source(literature_source_element)
        if literature_source_element is not None
        else None
    )

    external_reference_element = parent.find(_ns("externalReference"))
    external_reference = (
        _read_external_reference(external_reference_element)
        if external_reference_element is not None
        else None
    )

    return literature_source, external_reference

def _read_literature_source(literature_source_element):
    """
    Read a literatureSource element.

    :rtype: :class:`~obspy.io.sitexml.core.LiteratureSource`
    """
    title = _tag2obj(literature_source_element, _ns("title"), str)
    first_author = _tag2obj(literature_source_element, _ns("firstAuthor"), str)
    secondary_authors = _tag2obj(literature_source_element, _ns("secondaryAuthors"), str)
    year = _tag2obj(literature_source_element, _ns("year"), str)
    booktitle = _tag2obj(literature_source_element, _ns("booktitle"), str)
    doi = _tag2obj(literature_source_element, _ns("doi"), str)
    language = _tag2obj(literature_source_element, _ns("languageCode"), str)

    return LiteratureSource(title=title,
                            first_author=first_author,
                            secondary_authors=secondary_authors,
                            year=year,
                            booktitle=booktitle,
                            language=language,
                            doi=doi)

def _read_value(parent, tag, type):
    """
    Method used to read a value 
    from an element of the following structure
    
    <xs:element name="parent">
        <xs:element name="tag">
		    <xs:element name="value" type="type"/>
        </xs:element>
	</xs:element>

    :rtype: object or None
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

    :rtype: :class:`~obspy.io.sitexml.core.ValueWithUncertainty` or None
    """
    element = parent.find(_ns(tag))
    if element is None:
        return None

    value = _tag2obj(element, _ns("value"), type)
    uncertainty = _tag2obj(element, _ns("uncertainty"), type)

    return ValueWithUncertainty(value, uncertainty)

def _read_external_reference(ref_element):
    """
    Read an ExternalReference object.

    :rtype: :class:`~obspy.core.inventory.util.ExternalReference`
    """
    uri = _tag2obj(ref_element, _ns("uri"), str)
    description = _tag2obj(ref_element, _ns("description"), str)
    return ExternalReference(uri=uri, description=description)

def quality_index1(method=None, evaluation=None, reliability=None, completeness=None):
    """
    This function calculates the Quality Index #1 according to SERA Deliverable 7.2. 
    It varies from 0 to 1 and refers to a single mandatory indicator. 
    
    Four criteria are used for the calculation:
        A. Method of acquisition and analysis
        B. Estimation of indicator
        C. Reliability of the value
        D. Completeness of the report
    Each criterion is assinged a value between 0 and 1.
    
    The Quality Index #1 is then calculated using the following formula
        Q_Index1 = [ (A + B + C) * D ] / (Amax + Bmax + Cmax)

    :type method: float       
    :param method: It defines the reliability of the method of acquisition and 
        analysis to infer the value of the target indicator, on the basis of 
        peer-reviewed papers
    :param evaluation: It defines the way of evaluating the target indicator: direct or
        proxy. The evaluation is direct if derived from in-situ field experiments; 
        whereas it is inferred if derived from proxies or empirical relationships.
    :param reliability: It indicates the confidence on the single indicator (the 
        reliability of its value) and it is based on the available information 
        summarized within the intermediate report.
    :param completeness: it defines whether there exists a report describing step by step 
        the field survey and the data processing to evaluate the target indicator. 
        Please note that the presence of a detailed report is very important; 
        in case of the absence of any report documenting the value of a given indicator, 
        the corresponding quality_index1 is assigned a zero value.
    :rtype: float
    """

    # A. Method of acquisition and analysis quality index. Takes two values:
    #       1 - Documented method through several papers: The method of acquisition and analysis 
    #           to estimate the target indicator is well documented through several peer-reviewed 
    #           papers.
    #       0 - Undocumented method: The method of acquisition and analysis is not published

    if method == "documented" or method == 1:
        A = 1
    else:
        A = 0

    # B. Estimation of indicator. Takes two values:
    #       2 - Direct evaluation: The evaluation is based on specific field experiments
    #       0 - Inferred evaluation: The evaluation is based on inferred values from proxies, 
    #           empirical relationships or modeling

    if evaluation == "direct" or evaluation == 2:
        B = 2
    else:
        B = 0

    # C. Reliability of the value. Takes three values:
    #       1   - Yes: The indicator (its value or description) is very reliable
    #       0.5 - Partial: In case of partial/moderate confidence
    #       0   - No: The indicator, although described in the report, is not reliable
    
    if reliability == "yes" or reliability == 1:
        C = 1
    elif reliability == "partial" or reliability == 0.5:
        C = 0.5
    else:
        C = 0

    # D. Completeness of the report. Takes three values:
    #       1   - Yes: A well-documented report for the specific indicator is present
    #       0.5 - Partial: A report associated to a site is present, but the information 
    #               is partial and not very detailed
    #       0   - No: The value is provided without any documentation
    
    if completeness == "yes" or completeness == 1:
        D = 1
    elif completeness == "partial" or completeness == 0.5:
        D = 0.5
    else:
        D = 0

    # Sum of maximum values for criteria A, B and C: max(A) + max(B) + max(C)
    max = 4 
    quality_index1 = (A + B + C) * D / max

    return quality_index1

def quality_index2(sera_site):
    """
    This function calculates the Quality Index #2 for a site, according to SERA Deliverable 7.2. 

    Quality Index #2 is a weighted sum computed on the quality index #1 of all site 
    indicators evaluated at the target site and varies from 0 to 1.

    The formula used for the calculation is :
    Q_Index2 = (w1*Q_Index1_si1 + w2*Q_Index1_si2 + ... + w7*Q_Index1_si7) / (w1 + w2 + ... + w7)

    The weights used for this calculation for each site indicator, as proposed by SERA, are:
    - Resonance Frequency   : 1
    - Velocity Profile      : 1
    - Velocity S30          : 0.5
    - Bedrock Depth         : 0.5
    - H800                  : 0.5
    - Geological Unit       : 0.5
    - Soil Class EC8        : 0.25
    
    :type sera_site: :class:`~obspy.io.sitexml.core.SERASite
    :param sera_site: The site for which to calculate quality index #2
    :rtype: float or None
    """

    if not sera_site:
        return None
    
    weights = {}
    weights["resonanceFrequency"] = 1
    weights["velocityProfile"] = 1
    weights["velocityS30"] = 0.5
    weights["bedrockDepth"] = 0.5
    weights["h800"] = 0.5
    weights["geologicalUnit"] = 0.5
    weights["siteClassEC8"] = 0.25
    
    weights_sum = 0
    for value in weights.values():
        weights_sum = weights_sum + value
    #print("Qindex2 weights sum : ", weights_sum)

    Qindex1 = {}
    if sera_site.site_description:
        if sera_site.site_description.ec8:
            Qindex1["siteClassEC8"] = sera_site.site_description.ec8.quality_index
        if sera_site.site_description.h800:
            Qindex1["h800"] = sera_site.site_description.h800.quality_index
        if sera_site.site_description.bedrock_depth:
            Qindex1["bedrockDepth"] = sera_site.site_description.bedrock_depth.quality_index
        if sera_site.site_description.geological_unit:
            Qindex1["geologicalUnit"] = sera_site.site_description.geological_unit.quality_index
    
    # TODOs: We must select the prefered analysis and prefered VP for the calculation of QI2
    #
    if sera_site.analysis:
        if sera_site.analysis[0].resonance_frequency:
            Qindex1["resonanceFrequency"] = sera_site.analysis[0].resonance_frequency.quality_index
        if sera_site.analysis[0].velocity_profile_survey:
            Qindex1["velocityProfile"] = sera_site.analysis[0].velocity_profile_survey.quality_index
        if sera_site.analysis[0].velocity_s30:
            Qindex1["velocityS30"] = sera_site.analysis[0].velocity_s30.quality_index
    
    quality_index2_sum = 0
    for key in Qindex1:
        quality_index2_sum = quality_index2_sum + (weights[key] * Qindex1[key])
    #print(quality_index2_sum)
    quality_index2 = quality_index2_sum / weights_sum

    return quality_index2

def quality_index3(f0_vs30 = 0, f0_bedrock_depth = 0, f0_h800 = 0, vs30_h800 = 0, vs30_geology = 0):
    """
    This function calculates the Quality Index #3 for a site, according to SERA Deliverable 7.2. 

    Quality Index #3 refers to the overall consistency between the various 
    indicators and varies from 0 to 1.
    
    Specifically, Q_Index3 evaluates consistency of various couples of indicators according to the 
    current state of knowledge of the community. If estimates for a given couple of indicators 
    (e.g f0 and Vs30, geology and Vs30, etc.) are not within the range of reported values, then 
    these two estimates are considered as not consistent with one another.

    The consistency among various couple of indicators should be performed between the following 
    mandatory indicators: f0, Vs(z), Vs30, H800 (engineering bedrock), seismic bedrock depth and 
    surface geology.

    The computation of Q_Index3 is given by the sum of consistency values among the following 
    five couples of indicators, for which published references are available. 
        1. f0 and Vs30
        2. f0 and seismic_bedrock_depth
        3. f0 and engineering_bedrock_depth
        4. Vs30 and H800 
        5. Vs30 and geology

    The consistency at a specific site is computed only for the available indicators 
    (e.g. if only Vs30 and geological information are reported for a site, then the consistency 
    (cons) should be checked only for the couple Vs30-surface geology).

    Q_Index3 = [cons(f0, Vs30) + cons(f0, seismic_bedrock_depth) + 
                cons(f0, engineering_bedrock_depth) + cons(H800, Vs30) + 
                cons(Vs30, geology)] / n
    :rtype: float
    """
    n = 5       # Number of couples used for the calculation of quality_index3
    quality_index3 = (f0_vs30 + f0_bedrock_depth + f0_h800 + vs30_h800 + vs30_geology) / n

    return quality_index3

def overall_quality_index(quality_index2 = 0, quality_index3 = 0):
    """
    This function calculates the Quality Index #3 for a site, according to SERA Deliverable 7.2. 

    The overall quality index is computed as the arithmetic mean between Q_Index2 and Q_Index3. 
    
    Overall_Quality_Index = (Q_Index2 + Q_Index3) / 2 

    The range of values of Overall_Quality_Index is spanning from 0 to 1. 
    A value of 1 is for a site with a very thorough and reliable seismic characterization, 
    0 is assigned to a site badly or not characterized.
    :rtype: float
    """
    overall_quality_index = (quality_index2 + quality_index3) / 2

    return overall_quality_index


def write_sitexml(sera_site, file_or_file_object, validate=True):
    """
    Writes a sera_site object to a buffer.

    :type sera_site: :class:`~obspy.io.sitexml.core.SERASite`
    :param sera_site: The sitexml instance to be written.
    :param file_or_file_object: The file or file-like object to be written to.
    :type validate: bool, optional
    :param validate: If True, the created document will be validated with the
        SiteXML schema before being written. Defaults to True which is the
        recommended usage.
    :rtype: None
    """
    # Validate cross-references in the in-memory SiteXML object graph before
    # emitting XML, so broken internal IDs fail early with API-level errors.
    sera_site.validate_references()

    attribs = {"schemaVersion": SCHEMA_VERSION}
    if sera_site.resource_id:
        attribs["publicID"] = sera_site.resource_id

    root = etree.Element("SERA_quakeml", attribs, nsmap={None: NAMESPACE})

    # Root-level creationTime is document serialization metadata. Always
    # stamp it with the current write time, even when rewriting an unchanged
    # SERASite object that was read from an existing XML document.
    creation_time = obspy.UTCDateTime()
    sera_site.created = creation_time
    etree.SubElement(root, "creationTime").text = str(creation_time)

    if sera_site.external_references:
        for ref in sera_site.external_references:
            _write_external_reference(root, ref)

    if sera_site.site_owner:
        _write_site_owner(root, sera_site.site_owner)
    if sera_site.site_description:
        _write_site_description(root, sera_site.site_description)
    if sera_site.analysis:
        _write_analysis(root, sera_site.analysis)

    tree = root.getroottree()

    if validate is True:
        buf = io.BytesIO()
        tree.write(buf)
        buf.seek(0)
        validates, errors = validate_sitexml(buf)
        buf.close()
        if validates is False:
            msg = "The created file fails to validate.\n"
            for err in errors:
                msg += "\t%s\n" % err
            raise SiteXMLValidationError(msg)

    etree.indent(tree, "    ")
    tree.write(file_or_file_object, pretty_print=True, xml_declaration=True,
               encoding="UTF-8")


def _write_site_owner(parent, site_owner):
    """
    Append a siteOwner element to ``parent``.

    :rtype: None
    """
    if site_owner.owner_codename and site_owner.owner_fullname:
        attribs = {"publicID": site_owner.ownerID} if site_owner.ownerID else None
        site_owner_elem = etree.SubElement(parent, "siteOwner", attribs)
        _obj2tag(site_owner_elem, "codeName", site_owner.owner_codename)
        _obj2tag(site_owner_elem, "fullName", site_owner.owner_fullname)
    else:
        raise SiteXMLValidationError(
            "Site owner requires owner_codename and owner_fullname."
        )

    if site_owner.person_firstname and site_owner.person_lastname and site_owner.person_mbox:
        contact_elem = etree.SubElement(site_owner_elem, "contact")
        attribs = {"publicID": site_owner.personID} if site_owner.personID else None
        person_elem = etree.SubElement(contact_elem, "person", attribs)
        _obj2tag(person_elem, "firstname", site_owner.person_firstname)
        _obj2tag(person_elem, "lastname", site_owner.person_lastname)
        _obj2tag(person_elem, "mbox", site_owner.person_mbox)
        _obj2tag(person_elem, "homepage", site_owner.person_homepage)
    else:
        raise SiteXMLValidationError(
            "Site owner contact person requires firstname, lastname and mbox."
        )

    if site_owner.institution_name and site_owner.institution_mbox:
        affiliation_elem = etree.SubElement(contact_elem, "affiliation")
        attribs = {"publicID": site_owner.institutionID} if site_owner.institutionID else None
        institution_elem = etree.SubElement(affiliation_elem, "institution", attribs)
        _obj2tag(institution_elem, "name", site_owner.institution_name)
        _obj2tag(institution_elem, "mbox", site_owner.institution_mbox)
        _obj2tag(institution_elem, "phone", site_owner.institution_phone)
        _obj2tag(institution_elem, "homepage", site_owner.institution_homepage)
        _obj2tag(affiliation_elem, "department", site_owner.affiliation_department)
        _obj2tag(affiliation_elem, "function", site_owner.affiliation_function)

        if site_owner.address_street:
            postal_address_elem = etree.SubElement(institution_elem, "postalAddress")
            _obj2tag(postal_address_elem, "streetAddress", site_owner.address_street)
            _obj2tag(postal_address_elem, "locality", site_owner.address_locality)
            _obj2tag(postal_address_elem, "postalCode", site_owner.address_postal_code)

            country_elem = etree.SubElement(postal_address_elem, "country")
            _obj2tag(country_elem, "code", site_owner.address_country_code)
            _obj2tag(country_elem, "country", site_owner.address_country)


def _write_site_description(parent, site_description):
    """
    Append a siteDescription element to ``parent``.

    :rtype: None
    """
    attribs = {"publicID": site_description.resource_id} if site_description.resource_id else None
    site_description_elem = etree.SubElement(parent, "siteDescription", attribs)

    _obj2tag(site_description_elem, "station", site_description.station_code)
    _obj2tag(site_description_elem, "latitude", site_description.latitude)
    _obj2tag(site_description_elem, "longitude", site_description.longitude)
    _obj2tag(site_description_elem, "altitude", site_description.altitude)
    _obj2tag(site_description_elem, "minDistanceFromStation",
             site_description.min_distance_from_station)
    _obj2tag(site_description_elem, "maxDistanceFromStation",
             site_description.max_distance_from_station)

    if site_description.topographyA or site_description.topographyB:
        site_topography_elem = etree.SubElement(site_description_elem, "siteTopography")
        _obj2tag(site_topography_elem, "schemaA", site_description.topographyA)
        _obj2tag(site_topography_elem, "schemaB", site_description.topographyB)

    if (site_description.morphology or site_description.ec8 or
            site_description.bedrock_depth or site_description.h800 or
            site_description.geological_unit):
        site_morphology_elem = etree.SubElement(site_description_elem, "siteMorphology")
        _obj2tag(site_morphology_elem, "morphology", site_description.morphology)

        _write_site_indicator(site_morphology_elem, "siteClassEC8", site_description.ec8)
        _write_site_indicator(site_morphology_elem, "bedrockDepth",
                              site_description.bedrock_depth)
        _write_site_indicator(site_morphology_elem, "h800", site_description.h800)
        _write_site_indicator(site_morphology_elem, "geologicalUnit",
                              site_description.geological_unit)

    _obj2tag(site_description_elem, "preferredSiteAnalysisID",
             site_description.preferred_site_analysisID)
    _obj2tag(site_description_elem, "preferredVelocityProfileID",
             site_description.preferred_velocity_profileID)
    _obj2tag(site_description_elem, "overallQindex",
             site_description.overall_quality_index)


def _write_analysis(parent, analysis_list):
    """
    Append all analysis elements to ``parent``.

    :rtype: None
    """
    for analysis in analysis_list:
        attribs = {"publicID": analysis.resource_id} if analysis.resource_id else None
        analysis_elem = etree.SubElement(parent, "analysis", attribs)

        _obj2tag(analysis_elem, "siteDescriptionID", analysis.site_descriptionID)
        _obj2tag(analysis_elem, "creationTime", analysis.creation_date)
        _write_site_indicator(analysis_elem, "resonanceFrequency",
                              analysis.resonance_frequency)
        _write_site_indicator(analysis_elem, "velocityS30", analysis.velocity_s30)
        _obj2tag(analysis_elem, "sptLogsCount", analysis.spt_logs_count)
        _obj2tag(analysis_elem, "cptLogsCount", analysis.cpt_logs_count)
        _obj2tag(analysis_elem, "boreholeLogsCount", analysis.borehole_logs_count)
        _write_velocity_profile(analysis_elem, analysis.velocity_profile_survey)


def _write_velocity_profile(parent, velocity_profile_survey):
    """
    Append velocity-profile elements and survey metadata to ``parent``.

    :rtype: None
    """
    if velocity_profile_survey:
        velocity_profile_elem = etree.SubElement(parent, "velocityProfile")
        if velocity_profile_survey.velocity_profiles:
            for vp in velocity_profile_survey.velocity_profiles:
                index = velocity_profile_survey.velocity_profiles.index(vp)
                comment = etree.Comment(f" Velocity profile # {index + 1} ")
                velocity_profile_elem.append(comment)

                attribs = {"publicID": vp.resource_id}
                vp_elem = etree.SubElement(velocity_profile_elem, "profile", attribs)
                if vp.layer_count != len(vp.velocity_profile_data):
                    raise SiteXMLValidationError(
                        "Number of velocity profile data layers does not "
                        "match the layer_count value."
                    )
                _obj2tag(vp_elem, "layerCount", vp.layer_count)

                for vp_data in vp.velocity_profile_data:
                    vp_data_elem = etree.SubElement(vp_elem, "velocityProfileData")
                    _write_value_with_uncertainty(vp_data_elem, "velocityP",
                                                  vp_data.velocityP)
                    _write_value_with_uncertainty(vp_data_elem, "velocityS",
                                                  vp_data.velocityS)
                    _write_value_with_uncertainty(vp_data_elem, "density",
                                                  vp_data.density)

                    geometry_elem = etree.SubElement(vp_data_elem, "layerThickness")
                    _write_value_with_uncertainty(geometry_elem, "layerTopDepth",
                                                  vp_data.top_depth)
                    _write_value_with_uncertainty(geometry_elem, "layerBottomDepth",
                                                  vp_data.bottom_depth)

        _obj2tag(velocity_profile_elem, "qualityIndex",
                 velocity_profile_survey.quality_index)
        _write_reference(velocity_profile_elem, velocity_profile_survey)


def _write_site_indicator(parent, site_indicator_name, site_indicator_obj):
    """
    Append a site indicator value, methods, quality index, and reference.

    :rtype: None
    """
    if site_indicator_obj:
        site_indicator_elem = etree.SubElement(parent, site_indicator_name)
        if isinstance(site_indicator_obj.value, ValueWithUncertainty):
            _write_value_with_uncertainty(site_indicator_elem, "value",
                                          site_indicator_obj.value)
        else:
            etree.SubElement(site_indicator_elem, "value").text = (
                str(site_indicator_obj.value)
            )

        _write_methods(site_indicator_elem, site_indicator_obj)

        if site_indicator_name == "velocityS30":
            _obj2tag(site_indicator_elem, "methodCombIndex",
                     site_indicator_obj.method_combined_qindex)
            _obj2tag(site_indicator_elem, "manualIndex",
                     site_indicator_obj.manual_qindex)

        if site_indicator_name == "geologicalUnit":
            _obj2tag(site_indicator_elem, "geologicalMapScale",
                     site_indicator_obj.geological_map_scale)
            _obj2tag(site_indicator_elem, "geologicalUnitOGE",
                     site_indicator_obj.geological_unit_OGE)

        _obj2tag(site_indicator_elem, "qualityIndex",
                 site_indicator_obj.quality_index)

        _write_reference(site_indicator_elem, site_indicator_obj)


def _write_reference(parent, site_indicator_obj):
    """
    Append site-indicator reference metadata when present.

    :rtype: None
    """
    literature_obj = site_indicator_obj.literature_source
    external_reference_obj = site_indicator_obj.external_reference

    if literature_obj:
        literature_elem = etree.SubElement(parent, "literatureSource")
        _obj2tag(literature_elem, "title", literature_obj.title)
        _obj2tag(literature_elem, "firstAuthor", literature_obj.first_author)
        _obj2tag(literature_elem, "secondaryAuthors",
                 literature_obj.secondary_authors)
        _obj2tag(literature_elem, "year", literature_obj.year)
        _obj2tag(literature_elem, "booktitle", literature_obj.booktitle)
        _obj2tag(literature_elem, "doi", literature_obj.doi)
        _obj2tag(literature_elem, "languageCode", literature_obj.language)

    if external_reference_obj:
        _write_external_reference(parent, external_reference_obj)


def _write_methods(parent, site_indicator_obj):
    """
    Append method elements for a site indicator.

    :rtype: None
    """
    if site_indicator_obj.methods:
        for method in site_indicator_obj.methods:
            _obj2tag(parent, "method", method)


def _write_value(parent, tag, value):
    """
    Append an element containing a nested value child.

    :rtype: None
    """
    if value is not None:
        element = etree.SubElement(parent, tag)
        etree.SubElement(element, "value").text = str(value)


def _write_value_with_uncertainty(parent, tag, value):
    """
    Append a value/uncertainty quantity element.

    :rtype: None
    """
    if isinstance(value, ValueWithUncertainty):
        element = etree.SubElement(parent, tag)
        etree.SubElement(element, "value").text = str(value.value)
        if value.uncertainty is not None:
            etree.SubElement(element, "uncertainty").text = str(value.uncertainty)


def _write_external_reference(parent, ref):
    """
    Append an externalReference element.

    :rtype: None
    """
    ref_elem = etree.SubElement(parent, "externalReference")
    etree.SubElement(ref_elem, "uri").text = ref.uri
    etree.SubElement(ref_elem, "description").text = ref.description


def _obj2tag(parent, tag_name, tag_value):
    """
    Append a simple text element when ``tag_value`` is present.

    :rtype: None
    """
    if tag_value is not None:
        etree.SubElement(parent, tag_name).text = str(tag_value)
