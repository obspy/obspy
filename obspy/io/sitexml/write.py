# -*- coding: utf-8 -*-
"""
Functions dealing with reading and writing SiteXML.

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
import io

from lxml import etree

import obspy
from obspy.io.stationxml.core import _tag2obj, _attr2obj, _tags2obj
from obspy.core.inventory.util import ExternalReference
from obspy.io.sitexml.core import (SERASite, SiteDescription, SiteCharacterizationParameters, SERASiteOwner, 
								   EC8, H800, BedrockDepth, GeologicalUnit, ResonanceFrequency, VelocityS30, 
								   VelocityProfile, VelocityProfileData, ValueWithUncertainty,
								   LiteratureSource)
from obspy.io.sitexml.sitexml import validate_sitexml

# Define some constants for writing SiteXML files.
SCHEMA_VERSION = "1.2"
NAMESPACE = "http://www.orfeus-eu.org/xml/site/1"
#READABLE_VERSIONS = ("1.0", "1.1", "1.2")

def _write_sitexml(sera_site, file_or_file_object, validate=False,
					  nsmap=None, level="response", **kwargs):
	"""
	Writes an inventory object to a buffer.

	:type sitexml: :class:`~obspy.io.sitexml.core.SERASite`
	:param sitexml: The sitexml instance to be written.
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
	etree.SubElement(root, "created").text = str(sera_site.created)

	"""
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
	

	if level not in ["network", "station", "channel", "response"]:
		raise ValueError("Requested stationXML write level is unsupported.")

	for network in inventory.networks:
		_write_network(root, network, level)

	# Add custom namespace tags to root element
	_write_extra(root, inventory)
	"""

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
	attribs = {"publicID": site_owner.ownerID}
	site_owner_elem = etree.SubElement(parent, "siteOwner", attribs)
	_obj2tag(site_owner_elem, "codeName", site_owner.owner_codename)
	_obj2tag(site_owner_elem, "fullName", site_owner.owner_fullname)
	
	contact_elem = etree.SubElement(site_owner_elem, "contact")

	attribs = {"personID": site_owner.personID}
	person_elem = etree.SubElement(contact_elem, "person", attribs)
	_obj2tag(person_elem, "firstname", site_owner.person_firstname)
	_obj2tag(person_elem, "lastname", site_owner.person_lastname)
	_obj2tag(person_elem, "mbox", site_owner.person_mbox)
	_obj2tag(person_elem, "homepage", site_owner.person_homepage)
	
	affiliation_elem = etree.SubElement(contact_elem, "affiliation")
	_obj2tag(affiliation_elem, "department", site_owner.affiliation_department)
	_obj2tag(affiliation_elem, "function", site_owner.affiliation_function)
	
	institution_elem = etree.SubElement(affiliation_elem, "institution")
	identifier_elem = etree.SubElement(institution_elem, "identifier")
	_obj2tag(identifier_elem, "resourceID", site_owner.institution_ID)
	_obj2tag(institution_elem, "name", site_owner.institution_name)
	_obj2tag(institution_elem, "mbox", site_owner.institution_mbox)
	_obj2tag(institution_elem, "phone", site_owner.institution_phone)
	_obj2tag(institution_elem, "homepage", site_owner.institution_homepage)
	
	postal_address_elem = etree.SubElement(institution_elem, "postalAddress")
	_obj2tag(postal_address_elem, "streetAddress", site_owner.address_street)
	_obj2tag(postal_address_elem, "locality", site_owner.address_locality)
	_obj2tag(postal_address_elem, "postalCode", site_owner.address_postal_code)
	
	country_elem = etree.SubElement(postal_address_elem, "country")
	_obj2tag(country_elem, "code", site_owner.address_country_code)
	_obj2tag(country_elem, "country", site_owner.address_country)
	
def _write_site_description(parent, site_description):

	site_description_elem = etree.SubElement(parent, "siteDescription")

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
	
	site_morphology_elem = etree.SubElement(site_description_elem, "siteMorphology")
	_obj2tag(site_morphology_elem, "morphology", site_description.morphology)
	
	if site_description.topologyA or site_description.topologyB:
		site_topology_elem = etree.SubElement(site_morphology_elem, "siteTopology")
		_obj2tag(site_topology_elem, "schemeA", site_description.topologyA)
		_obj2tag(site_topology_elem, "schemeB", site_description.topologyB)

	_write_site_indicator(site_morphology_elem, "siteClassEC8", site_description.ec8)
	_write_site_indicator(site_morphology_elem, "bedrockDepth", site_description.bedrock_depth)
	_write_site_indicator(site_morphology_elem, "h800", site_description.h800)
	# Bellow also write mapscale and geomap if available
	_write_site_indicator(site_morphology_elem, "geologicalUnit", site_description.geological_unit)

def _write_site_characterization(parent, site_characterization):

	attribs = {"publicID": site_characterization.publicID}
	site_characterization_elem = etree.SubElement(parent, 
						"siteCharacterizationParameters", attribs)
	
	attribs = {"publicID": site_characterization.analysis_publicID}
	analysis_elem = etree.SubElement(site_characterization_elem, "Analysis", attribs)

	_write_site_indicator(analysis_elem, 
					   "resonanceFrequency", 
					   site_characterization.resonance_frequency)
	_write_site_indicator(analysis_elem, 
					   "velocityS30", 
					   site_characterization.velocity_s30)
	
	_obj2tag(analysis_elem, 
		  "velocityProfileCount", 
		  site_characterization.velocity_profile_count)
	_obj2tag(analysis_elem, 
		  "sptLogsCount", 
		  site_characterization.spt_logs_count)
	_obj2tag(analysis_elem, 
		  "cptLogsCount", 
		  site_characterization.cpt_logs_count)
	_obj2tag(analysis_elem, 
		  "boreholeLogsCount", 
		  site_characterization.borehole_logs_count)

	if site_characterization.velocity_profile:
		_write_velocity_profile(site_characterization_elem, 
						  site_characterization.velocity_profile)
	
def _write_site_indicator(parent, site_indicator_name, site_indicator_obj):

	if site_indicator_obj:

		# ec8 / geological_unit don't have a value sub-element !!
		if isinstance(site_indicator_obj.value, ValueWithUncertainty):
			_write_value_with_uncertainty(parent, site_indicator_name, site_indicator_obj.value)
		else:
			etree.SubElement(parent, site_indicator_name).text = \
				str(site_indicator_obj.value)
		
		_write_value(parent, site_indicator_name + "Qindex1", site_indicator_obj.quality_index)

		_write_methods(parent, site_indicator_name, site_indicator_obj)

		# Write velocityS30MethodCombIndex / velocityS30ManualIndex
		_write_reference(parent, site_indicator_obj)

	return

def _write_velocity_profile(parent, vp_obj):
	
	_write_value(parent, "velocityProfileQindex1", vp_obj.quality_index)
	_write_reference(parent, vp_obj)

	for vp_data in vp_obj.velocity_profile_data:
		index = vp_obj.velocity_profile_data.index(vp_data)
		comment = etree.Comment(f" Velocity profile # {index+1} ")
		parent.append(comment)

		vp_elem = etree.SubElement(parent, "VelocityProfile")
		
		_write_value(vp_elem, "layerCount", vp_data.layer_count)
		vp_data_elem = etree.SubElement(vp_elem, "velocityProfileData")
		
		# Σε συνδυασμό  με το read_Site_xml να γίνει κατάλληλος έλεγχος 
		# και διαχείριση της περίπτωσης που
		# το layer_count δεν είναι ίσο με το μέγεθος της λίστας των μεγεθών.
		# for i in range(vp_data.layer_count):
		# Important: We need to have only one loope!!!
		
		for i in range(len(vp_data.density)):
			_write_value_with_uncertainty(vp_data_elem, 
								  "density", 
								  vp_data.density[i])
		for i in range(len(vp_data.velocityP)):	
			_write_value_with_uncertainty(vp_data_elem, 
								  "velocityP", 
								  vp_data.velocityP[i])
		for i in range(len(vp_data.velocityS)):	
			_write_value_with_uncertainty(vp_data_elem, 
								  "velocityS", 
								  vp_data.velocityS[i])
		for i in range(len(vp_data.top_depth)):	
			_write_value_with_uncertainty(vp_data_elem, 
								  "layerTopDepth", 
								  vp_data.top_depth[i])
		for i in range(len(vp_data.bottom_depth)):	
			_write_value_with_uncertainty(vp_data_elem, 
								  "layerBottomDepth", 
								  vp_data.bottom_depth[i])
			#_obj2tag(vp_data_elem, "density", vp_data.density[i])
	

def _write_reference(parent, site_indicator_obj):

	literature_obj = site_indicator_obj.literature_source
	file_obj = site_indicator_obj.file_resource
	 
	if literature_obj or file_obj:
		reference_elem = etree.SubElement(parent, site_indicator_obj.name + "Reference")

	if literature_obj:
		literature_elem = etree.SubElement(reference_elem, "literatureSource")
		_obj2tag(literature_elem, "title", literature_obj.title)
		_obj2tag(literature_elem, "first_author", literature_obj.first_author)
		_obj2tag(literature_elem, "secondary_authors", literature_obj.secondary_authors)
		_obj2tag(literature_elem, "year", literature_obj.year)
		_obj2tag(literature_elem, "booktitle", literature_obj.booktitle)
		_obj2tag(literature_elem, "DOI", literature_obj.doi)
		
		if literature_obj.language:
			language_elem = etree.SubElement(literature_elem, "language")
			_obj2tag(language_elem, "code", literature_obj.language)
	 
	if file_obj:    
		file_resource_elem = etree.SubElement(reference_elem, "FileResource")
		_obj2tag(file_resource_elem, "url", file_obj.uri)
		_obj2tag(file_resource_elem, "description", file_obj.description)

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
	if tag_value:
		if isinstance(tag_value, float):
			text = _float_to_str(tag_value)
		else:
			text = str(tag_value)
		etree.SubElement(parent, tag_name).text = text