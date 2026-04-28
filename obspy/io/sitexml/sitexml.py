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
from .util import SiteXMLIOError, SiteXMLValidationError

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

    Example

    >>> from obspy.io.sitexml.sitexml import validate_sitexml
    >>> validates, errors = validate_sitexml(path_or_file_object)
    >>> if validates:
    ...     print("This is valid SiteXML file")
    ... else:
    ...     print("The provided SiteXML file fails to validate "
    ...           "against the schema.")
    
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
        schema_location = Path(
            inspect.getfile(inspect.currentframe())).parent
        schema_location = schema_location / "data"
        schema_location = str(
            schema_location / ("QuakeML-SERA-%s.xsd" % version))
        
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

###### READ SiteXML functionality
#
def sitexml_to_sitedict(path_or_file_object, pattern="*.xml"):
    """
    Read one SiteXML file or all matching files in a directory.

    The returned dictionary is keyed by each site's resource ID.

    :type path_or_file_object: str, pathlib.Path, or file-like object
    :param path_or_file_object: SiteXML file, file-like object, or directory
        containing SiteXML files.
    :type pattern: str, optional
    :param pattern: Glob pattern used when ``path_or_file_object`` is a
        directory. Defaults to ``"*.xml"``.
    :rtype: dict
    :return: Dictionary of :class:`~obspy.io.sitexml.core.SERASite` objects.
    """
    def _add_site(sera_site_dict, sera_site):
        if sera_site.resource_id in sera_site_dict:
            raise SiteXMLValidationError(
                f"Duplicate SiteXML site resource_id: {sera_site.resource_id}"
            )
        sera_site_dict[sera_site.resource_id] = sera_site

    sera_site_dict = {}

    if hasattr(path_or_file_object, "read"):
        _add_site(sera_site_dict, read_sitexml(path_or_file_object))
        return sera_site_dict

    path = Path(path_or_file_object)
    if path.is_file():
        _add_site(sera_site_dict, read_sitexml(path))
        return sera_site_dict

    if path.is_dir():
        for filename in sorted(path.glob(pattern)):
            if filename.is_file():
                _add_site(sera_site_dict, read_sitexml(filename))
        return sera_site_dict

    raise SiteXMLIOError(
        f"Could not access SiteXML file or directory: {path_or_file_object}"
    )

def read_sitexml(path_or_file_object):
    """
    Function reading a SiteXML file.

    :param file_or_file_object: The file name or file-like object to read from.
    :rtype: :class:`~obspy.io.sitexml.core.SERASite`

    Returns a SERASite object with metadata read from the provided SiteXML
    file. At least site owner and site description metadata should be present
    in XML file in order to create the SERASite object.

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
    Read the <siteOwner> element

    :type owner_element: :class:`~lxml.etree._Element`, required

    :rtype: :class:`~obspy.io.sitexml.core.SERASiteOwner`
    :return: A `SERASiteOwner` object populated with the values read from 
            the <siteOwner> element.

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
    Read the <siteDescription> element

    :type site_description_element: :class:`~lxml.etree._Element`, required

    :rtype: :class:`~obspy.io.sitexml.core.SiteDescription`
    :return: A `SiteDescription` object populated with the values read from 
            the <siteDescription> element.

    <siteDescription> element structure:

    - publicID (attribute)
    - station_code, latitude, longitude, altitude, minDistanceFromStation,
      maxDistanceFromStation
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
    Read the <siteMorphology> element

    :rtype: None

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

    :type analysis_element: :class:`~lxml.etree._Element`, required

    :rtype: :class:`~obspy.io.sitexml.core.Analysis`
    :return: An Analysis object populated with the values read from 
            the <Analysis> element.

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

    :type analysis_element: :class:`~lxml.etree._Element`, required
    :param analysis_element: 
    :type analysis_obj:
        :class:`~obspy.core.io.sitexml.core.Analysis`, required
    :param analysis_obj: Analysis object to store values read from the
        <velocityProfile> element. It should be pre-initialized by the calling
        function.
    :rtype: :class:`~obspy.io.sitexml.core.VelocityProfileSurvey`
    """

    velocity_profile_element = analysis_element.find(_ns("velocityProfile"))
    if velocity_profile_element is None:
        return None

    vp_element_list = velocity_profile_element.findall(_ns("profile"))
    vp_qindex = _tag2obj(velocity_profile_element, _ns("qualityIndex"), float)
    vp_literature_source = _read_literature_source(velocity_profile_element)
    vp_external_references = _read_external_references(velocity_profile_element)

    # At least one profile or a reference
    # should be present in SiteXML in order to create the VelocityProfileSurvey object
    if len(vp_element_list) == 0 \
            and vp_literature_source is None \
            and not vp_external_references:
        return None

    analysis_obj.velocity_profile_survey = \
            VelocityProfileSurvey(velocity_profiles = [],    # We will fill this later
                            quality_index = vp_qindex,
                            literature_source = vp_literature_source,
                            external_references = vp_external_references)

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

    quality_index = _tag2obj(indicator_element, _ns("qualityIndex"), float)
    literature_source = _read_literature_source(indicator_element)
    external_references = _read_external_references(indicator_element)

    kwargs = {
        "value": indicator_value,
        "quality_index": quality_index,
        "literature_source": literature_source,
        "external_references": external_references,
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

def _read_external_references(parent):
    """
    Read all externalReference elements from ``parent``.

    :rtype: list or None
    """
    return [
        _read_external_reference(external_reference_element)
        for external_reference_element in parent.findall(_ns("externalReference"))
    ] or None

def _read_literature_source(parent):
    """
    Read a literatureSource element from ``parent``.

    :rtype: :class:`~obspy.io.sitexml.core.LiteratureSource` or None
    """
    literature_source_element = parent.find(_ns("literatureSource"))
    if literature_source_element is None:
        return None

    title = _tag2obj(
        literature_source_element, _ns("title"), str)
    first_author = _tag2obj(
        literature_source_element, _ns("firstAuthor"), str)
    secondary_authors = _tag2obj(
        literature_source_element, _ns("secondaryAuthors"), str)
    year = _tag2obj(
        literature_source_element, _ns("year"), str)
    booktitle = _tag2obj(
        literature_source_element, _ns("booktitle"), str)
    doi = _tag2obj(
        literature_source_element, _ns("doi"), str)
    language = _tag2obj(
        literature_source_element, _ns("languageCode"), str)

    return LiteratureSource(title=title,
                            first_author=first_author,
                            secondary_authors=secondary_authors,
                            year=year,
                            booktitle=booktitle,
                            language=language,
                            doi=doi)

### NOT USED anymore
#
def _read_value(parent, tag, type):
    """
    Read a nested value from an element.

    :rtype: object or None

    The element should have the following structure

        <parent>
            <tag>
                <value>...</value>
            </tag>
        </parent>
    """
    element = parent.find(_ns(tag))
    if element is None:
        return None
    return _tag2obj(element, _ns("value"), type)

def _read_value_with_uncertainty(parent, tag, type):
    """
    Read a nested value/uncertainty pair from an element.

    :rtype: :class:`~obspy.io.sitexml.core.ValueWithUncertainty` or None

    The element should have the following structure

        <parent>
            <tag>
                <value>...</value>
                <uncertainty>...</uncertainty>
            </tag>
        </parent>
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

###### WRITE SiteXML functionality
#
def sitedict_to_sitexml(sera_site_dict, output_folder="."):
    """
    Exports a dictionary of SERASite objects to SiteXML files.

    The files are written to a folder given with argument ``output_folder``.
    The name of each SiteXML file is either:

    * The station code in ``network.station`` notation if the metadata belong
      to a station site
    * The siteID otherwise

    :type sera_site_dict: dict of
        :class:`~obspy.io.sitexml.core.SERASite`, required
    :param sera_site_dict: Dictionary of SERASite objects.
    :type output_folder: str or pathlib.Path, optional
    :param output_folder: Output folder to write the SiteXML files. If not
        provided writes to the current folder.
    :rtype: None
    """
    output_folder = Path(output_folder)
    for sera_site in sera_site_dict.values():
        output_file = output_folder / sera_site.get_sitexml_filename()
        write_sitexml(sera_site, output_file, validate=True)

def write_sitexml(sera_site, file_or_file_object, validate=True):
    """
    Writes a sera_site object to a buffer.

    :type sera_site: :class:`~obspy.io.sitexml.core.SERASite`, required
    :param sera_site: The sitexml instance to be written.
    :type file_or_file_object: str or file-like object, required
    :param file_or_file_object: The file or file-like object to be written to.
    :type validate: bool, optional
    :param validate: If True, the created document will be validated with the
        SiteXML schema before being written. Defaults to True which is the
        recommended usage.
    :rtype: None

    Example

    >>> from obspy.io.sitexml.sitexml import write_sitexml
    >>> write_sitexml(
    ...     sera_site, sera_site.get_sitexml_filename(), validate=True)

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
        if velocity_profile_survey.literature_source:
            _write_literature_source(
                velocity_profile_elem, velocity_profile_survey.literature_source)
        if velocity_profile_survey.external_references:
            for external_reference in velocity_profile_survey.external_references:
                _write_external_reference(velocity_profile_elem, external_reference)


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

        if site_indicator_obj.literature_source:
            _write_literature_source(
                site_indicator_elem, site_indicator_obj.literature_source)
        if site_indicator_obj.external_references:
            for external_reference in site_indicator_obj.external_references:
                _write_external_reference(site_indicator_elem, external_reference)


def _write_literature_source(parent, literature_obj):
    """
    Append a literatureSource element.

    :rtype: None
    """
    literature_elem = etree.SubElement(parent, "literatureSource")
    _obj2tag(literature_elem, "title", literature_obj.title)
    _obj2tag(literature_elem, "firstAuthor", literature_obj.first_author)
    _obj2tag(literature_elem, "secondaryAuthors",
             literature_obj.secondary_authors)
    _obj2tag(literature_elem, "year", literature_obj.year)
    _obj2tag(literature_elem, "booktitle", literature_obj.booktitle)
    _obj2tag(literature_elem, "doi", literature_obj.doi)
    _obj2tag(literature_elem, "languageCode", literature_obj.language)


def _write_methods(parent, site_indicator_obj):
    """
    Append method elements for a site indicator.

    :rtype: None
    """
    if site_indicator_obj.methods:
        for method in site_indicator_obj.methods:
            _obj2tag(parent, "method", method)

### NOT USED anymore
#
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
