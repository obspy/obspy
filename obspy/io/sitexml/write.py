# -*- coding: utf-8 -*-
"""
Functions dealing with reading and writing SiteXML.

:copyright:
	ORFEUS, 2025
:license:
	GNU Lesser General Public License, Version 3
	(https://www.gnu.org/copyleft/lesser.html)
"""

from pathlib import Path
import re
import warnings
import io

from lxml import etree

from obspy.core.inventory.util import ExternalReference
from obspy.io.sitexml.core import ValueWithUncertainty
from obspy.io.sitexml.sitexml import validate_sitexml

# Define some constants for writing SiteXML files.
SCHEMA_VERSION = "1.3"
NAMESPACE = "http://www.orfeus-eu.org/xml/site/1"
#READABLE_VERSIONS = ("1.0", "1.1", "1.2")

def _write_sitexml(sera_site, file_or_file_object, validate=False,
					  nsmap=None):
	"""
	Writes a sera_site object to a buffer.

	:type sera_site: :class:`~obspy.io.sitexml.core.SERASite`
	:param sera_site: The sitexml instance to be written.
	:param file_or_file_object: The file or file-like object to be written to.
	:type validate: bool
	:param validate: If True, the created document will be validated with the
		SiteXML schema before being written. Useful for debugging or if you
		don't trust ObsPy. Defaults to False.
	:type nsmap: dict
	:param nsmap: Additional custom namespace abbreviation
		mappings (e.g. `{"edb": "http://erdbeben-in-bayern.de/xmlns/0.1"}`).
	"""
	if nsmap is None:
		nsmap = {}
	elif None in nsmap:
		msg = ("Custom namespace mappings do not allow redefinition of "
			   "default SiteXML namespace (key `None`). "
			   "Use other namespace abbreviations for custom namespace tags.")
		raise ValueError(msg)

	nsmap[None] = NAMESPACE
	#attrib = {"schemaVersion": SCHEMA_VERSION}

	root = etree.Element("SERA_quakeml", nsmap=nsmap)
	
	etree.SubElement(root, "schemaVersion").text = SCHEMA_VERSION
	etree.SubElement(root, "creationTime").text = str(sera_site.created)

	if sera_site.site_owner:
		_write_site_owner(root, sera_site.site_owner)
	if sera_site.site_description:
		_write_site_description(root, sera_site.site_description)
	if sera_site.site_characterization:
		_write_site_characterization(root, sera_site.site_characterization)

	#_write_site_characterization(root, sera_site.site_characterization)

	tree = root.getroottree()

	# The validation has to be done after parsing once again so that the
	# namespaces are correctly assembled.
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
			raise Exception(msg)
	
	# Register all namespaces with the tree. This allows for
	# additional namespaces to be added to an inventory that
	# was not created by reading a SiteXML file.
	for prefix, ns in nsmap.items():
		if prefix and ns:
			etree.register_namespace(prefix, ns)
	
	etree.indent(tree, "    ")
	tree.write(file_or_file_object, pretty_print=True, xml_declaration=True,
			   encoding="UTF-8")

def _get_base_node_attributes(element):
	attributes = {}
	if element.ownerID:
		attributes["publicID"] = str(element.ownerID)

def _write_site_owner(parent, site_owner):

	#attribs = _get_base_node_attributes(site_owner)
	attribs = {"publicID": site_owner.ownerID} if site_owner.ownerID else None
	site_owner_elem = etree.SubElement(parent, "siteOwner", attribs)
	_obj2tag(site_owner_elem, "codeName", site_owner.owner_codename)
	_obj2tag(site_owner_elem, "fullName", site_owner.owner_fullname)
	
	contact_elem = etree.SubElement(site_owner_elem, "contact")

	attribs = {"publicID": site_owner.personID} if site_owner.personID else None
	person_elem = etree.SubElement(contact_elem, "person", attribs)
	_obj2tag(person_elem, "firstname", site_owner.person_firstname)
	_obj2tag(person_elem, "lastname", site_owner.person_lastname)
	_obj2tag(person_elem, "mbox", site_owner.person_mbox)
	_obj2tag(person_elem, "homepage", site_owner.person_homepage)
	
	affiliation_elem = etree.SubElement(contact_elem, "affiliation")
	
	attribs = {"publicID": site_owner.institutionID} if site_owner.institutionID else None
	institution_elem = etree.SubElement(affiliation_elem, "institution", attribs)	
	_obj2tag(institution_elem, "name", site_owner.institution_name)
	_obj2tag(institution_elem, "mbox", site_owner.institution_mbox)
	_obj2tag(institution_elem, "phone", site_owner.institution_phone)
	_obj2tag(institution_elem, "homepage", site_owner.institution_homepage)
	_obj2tag(affiliation_elem, "department", site_owner.affiliation_department)
	_obj2tag(affiliation_elem, "function", site_owner.affiliation_function)

	
	postal_address_elem = etree.SubElement(institution_elem, "postalAddress")
	_obj2tag(postal_address_elem, "streetAddress", site_owner.address_street)
	_obj2tag(postal_address_elem, "locality", site_owner.address_locality)
	_obj2tag(postal_address_elem, "postalCode", site_owner.address_postal_code)
	
	country_elem = etree.SubElement(postal_address_elem, "country")
	_obj2tag(country_elem, "code", site_owner.address_country_code)
	_obj2tag(country_elem, "country", site_owner.address_country)
	
def _write_site_description(parent, site_description):

	attribs = {"publicID": site_description.publicID} if site_description.publicID else None
	site_description_elem = etree.SubElement(parent, "siteDescription", attribs)

	_obj2tag(site_description_elem, "station", site_description.station_code)
	_write_value(site_description_elem, "latitude", 
				site_description.latitude)
	
	_write_value(site_description_elem, "longitude", 
				site_description.longitude)

	_write_value(site_description_elem, "altitude", 
				site_description.altitude)

	_write_value(site_description_elem, "minDistanceFromStation", 
				site_description.min_distance_from_station)
	
	_write_value(site_description_elem, "maxDistanceFromStation", 
				site_description.max_distance_from_station)
	
	if site_description.topographyA or site_description.topographyB:
		site_topography_elem = etree.SubElement(site_description_elem, "siteTopography")
		_obj2tag(site_topography_elem, "schemeA", site_description.topographyA)
		_obj2tag(site_topography_elem, "schemeB", site_description.topographyB)

	site_morphology_elem = etree.SubElement(site_description_elem, "siteMorphology")
	_obj2tag(site_morphology_elem, "morphology", site_description.morphology)
	
	_write_site_indicator(site_morphology_elem, "siteClassEC8", 
					   site_description.ec8)
	_write_site_indicator(site_morphology_elem, "bedrockDepth", 
					   site_description.bedrock_depth)
	_write_site_indicator(site_morphology_elem, "h800", 
					   site_description.h800)
	_write_site_indicator(site_morphology_elem, "geologicalUnit", 
					   site_description.geological_unit)

	_obj2tag(site_description_elem, "preferredSiteAnalysisID", 
		  site_description.preferred_site_analysisID)
	_obj2tag(site_description_elem, "preferredVelocityProfileID", 
		  site_description.preferred_velocity_profileID)

def _write_site_characterization(parent, site_characterization):

	attribs = {"publicID": site_characterization.publicID} if site_characterization.publicID else None
	site_characterization_elem = etree.SubElement(parent, 
						"siteCharacterizationParameters", attribs)
	
	if site_characterization.analysis:
		for analysis in site_characterization.analysis:
			
			attribs = {"publicID": analysis.publicID} if analysis.publicID else None
			analysis_elem = etree.SubElement(site_characterization_elem, "analysis", attribs)

			_obj2tag(analysis_elem, "siteDescriptionID", analysis.site_descriptionID)

			# TODOs
			# Write CreationInfo
			# Write Comments

			_write_site_indicator(analysis_elem, "resonanceFrequency", 
					   analysis.resonance_frequency)
			_write_site_indicator(analysis_elem, "velocityS30", 
					   analysis.velocity_s30)

			_obj2tag(analysis_elem, "velocityProfileCount", analysis.velocity_profile_count)
			_obj2tag(analysis_elem, "sptLogsCount", analysis.spt_logs_count)
			_obj2tag(analysis_elem, "cptLogsCount", analysis.cpt_logs_count)
			_obj2tag(analysis_elem, "boreholeLogsCount", analysis.borehole_logs_count)

			_write_velocity_profile(analysis_elem, 
								analysis.velocity_profile_survey)	

def _write_velocity_profile(parent, velocity_profile_survey):

	if velocity_profile_survey:
		if velocity_profile_survey.velocity_profiles:
			for vp in velocity_profile_survey.velocity_profiles:
				index = velocity_profile_survey.velocity_profiles.index(vp)
				comment = etree.Comment(f" Velocity profile # {index+1} ")
				parent.append(comment)

				attribs = {"publicID": vp.publicID} if vp.publicID else None
				vp_elem = etree.SubElement(parent, "velocityProfile", attribs)
				_obj2tag(vp_elem, "layerCount", vp.layer_count)

				for vp_data in vp.velocity_profile_data:
					vp_data_elem = etree.SubElement(vp_elem, "velocityProfileData")
					_write_value_with_uncertainty(vp_data_elem, 
									"velocityP", 
									vp_data.velocityP)
					_write_value_with_uncertainty(vp_data_elem, 
									"velocityS", 
									vp_data.velocityS)
					_write_value_with_uncertainty(vp_data_elem, 
									"density", 
									vp_data.density)
					
					geometry_elem = etree.SubElement(vp_data_elem, "layerThickness")
					_write_value_with_uncertainty(geometry_elem, 
									"layerTopDepth", 
									vp_data.top_depth)
					_write_value_with_uncertainty(geometry_elem, 
									"layerBottomDepth", 
									vp_data.bottom_depth)
		# 
		#  
		_write_value(parent, "velocityProfileQindex1", velocity_profile_survey.quality_index)
		_write_reference(parent, velocity_profile_survey)

		# Σε συνδυασμό  με το read_Site_xml να γίνει κατάλληλος έλεγχος 
		# και διαχείριση της περίπτωσης που
		# το layer_count δεν είναι ίσο με το μέγεθος της λίστας των μεγεθών.

def _write_site_indicator(parent, site_indicator_name, site_indicator_obj):

	if site_indicator_obj:

		# Write site indicator value
		# ec8 / geological_unit don't have a value sub-element !!
		if isinstance(site_indicator_obj.value, ValueWithUncertainty):
			_write_value_with_uncertainty(parent, site_indicator_name, site_indicator_obj.value)
		else:
			etree.SubElement(parent, site_indicator_name).text = \
				str(site_indicator_obj.value)
		
		# Write site indicator quality index
		_write_value_with_uncertainty(parent, 
			site_indicator_name + "Qindex1", 
			site_indicator_obj.quality_index)

		# Write site indicator methods (valid for resonanceFrequency and velocityS30)
		_write_methods(parent, site_indicator_name, site_indicator_obj)

		if site_indicator_name == "geologicalUnit":
			_obj2tag(parent, "geologicalMapScale", 
			site_indicator_obj.geological_map_scale)
			_obj2tag(parent, "geologicalUnitOGE", 
			site_indicator_obj.geological_unit_OGE)

		if site_indicator_name == "velocityS30":
			_obj2tag(parent, "velocityS30MethodCombIndex", 
			site_indicator_obj.method_combined_quality_index)
			_obj2tag(parent, "velocityS30ManualIndex", 
			site_indicator_obj.manual_quality_index)

		# Write site indicator reference
		_write_reference(parent, site_indicator_obj)
	
def _write_reference(parent, site_indicator_obj):

	literature_obj = site_indicator_obj.literature_source
	file_obj = site_indicator_obj.file_resource
	 
	if literature_obj or file_obj:
		reference_elem = etree.SubElement(parent, site_indicator_obj.name + "Reference")

	if literature_obj:
		literature_elem = etree.SubElement(reference_elem, "literatureSource")
		_obj2tag(literature_elem, "title", literature_obj.title)
		_obj2tag(literature_elem, "firstAuthor", literature_obj.first_author)
		_obj2tag(literature_elem, "secondaryAuthors", literature_obj.secondary_authors)
		_obj2tag(literature_elem, "year", literature_obj.year)
		_obj2tag(literature_elem, "booktitle", literature_obj.booktitle)
		_obj2tag(literature_elem, "doi", literature_obj.doi)
		
		if literature_obj.language:
			language_elem = etree.SubElement(literature_elem, "language")
			_obj2tag(language_elem, "code", literature_obj.language)
	 
	if file_obj:    
		file_resource_elem = etree.SubElement(reference_elem, "fileResource")
		_obj2tag(file_resource_elem, "description", file_obj.description)
		_obj2tag(file_resource_elem, "url", file_obj.uri)
		
def _write_methods(parent, site_indicator_name, site_indicator_obj):

	if site_indicator_obj.methods:
		for method in site_indicator_obj.methods:
			_obj2tag(parent, site_indicator_name + "Method", method)

def _write_value_with_uncertainty_v1(parent, value, uncertainty=None):

	etree.SubElement(parent, "value").text = str(value)
	if uncertainty:
		etree.SubElement(parent, "uncertainty").text = str(uncertainty)

def _write_value(parent, tag, value):
	"""
	Method used to write a value 
	to an element of the following structure
	
	<xs:element name="parent">
		<xs:element name="tag">
			<xs:element name="value" type="type"/>
		</xs:element>
	</xs:element>
	"""
	if value:
		element = etree.SubElement(parent, tag)
		etree.SubElement(element, "value").text = str(value)
	
def _write_value_with_uncertainty(parent, tag, value):
	"""
	Method used to write a value / uncertainty pair 
	to an element of the following structure
	
	<xs:element name="parent">
		<xs:element name="tag">
			<xs:element name="value" type="type"/>
			<xs:element name="uncertainty" type="type"/>
		</xs:element>
	</xs:element>

	:type parent: str
	:param parent: Name of parent element
	:type tag: str
	:param tag: Name of element to be created
	:type value: :class:`~obspy.io.sitexml.core.ValueWithUncertainty`
	:param value: Object with value / uncertainty values.
	"""
	if isinstance(value, ValueWithUncertainty):
		element = etree.SubElement(parent, tag)
		etree.SubElement(element, "value").text = str(value.value)
		if value.uncertainty:
			etree.SubElement(element, "uncertainty").text = str(value.uncertainty)
			
def _obj2tag(parent, tag_name, tag_value):
	"""
	If tag_value is not None, append a SubElement to the parent. The text of
	the tag will be tag_value.
	"""
	if tag_value is not None:
		if isinstance(tag_value, float):
			text = str(tag_value)
		else:
			text = str(tag_value)
		etree.SubElement(parent, tag_name).text = text