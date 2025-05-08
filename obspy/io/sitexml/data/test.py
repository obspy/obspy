import obspy
from lxml import etree
from obspy.io.stationxml.core import _tag2obj, _attr2obj, _tags2obj
import obspy.io.sitexml.core 
import obspy.io.sitexml.util
from obspy.io.sitexml.core import (SERASite, SiteDescription, SiteCharacterizationParameters, EC8, LiteratureSource, SERASiteOwner)
from obspy.io.sitexml.util import (TopographySchemaA, TopographySchemaB, EC8Class, _sitexml_check_enum)
from obspy.io.sitexml.sitexml import (validate_stationxml, _read_sitexml, _read_literature_source, _read_reference,
                                       _read_site_description, _read_file_resource, _read_value, _read_value_with_uncertainty,
                                       _read_site_characterization)

xml_file = "./test_site.xml"
xml_file = "./SiteOGPC_SERA_v1.2.xml"

#validate_stationxml(xml_file)
xmldoc = etree.parse(xml_file)

sera_site = _read_sitexml(xml_file)
so = sera_site.site_owner
print(so)
sd = sera_site.site_description
print(sd)
sc = sera_site.site_characterization_parameters
print(sc)
print(sd.ec8)
print(sc.velocity_s30)


"""
#print(sera_site)
#print(sera_site.station_code)
sd = sera_site.site_description
print(sd)
print(sd.ec8)
print(sd.h800)
print(sd.bedrock_depth)
print(sd.geological_unit)

site=SERASite("ARG")
site.site_description=SiteDescription(23.44, 45.33)

site.site_description.topologyA = "T1"

site.site_description.ec8="A"
site.site_description.ec8.file_resource="http://some/uri"
site.site_description.ec8.literature_source="Some title"
e=EC8("B")
"""
