from lxml import etree
from obspy.io.sitexml.sitexml import (validate_sitexml, _read_sitexml, quality_index2,
                                      quality_index3, overall_quality_index)
from obspy.io.sitexml.write import _write_sitexml

NAMESPACE = "http://www.orfeus-eu.org/xml/site/1"

def _ns(tagname):
        return "{%s}%s" % (NAMESPACE, tagname)

xml_file = "./SiteOGPC_SERA_v1.3.4.xml"

validate_sitexml(xml_file)

#xmldoc = etree.parse(xml_file)
#site_owner_element = xmldoc.find(_ns("siteOwner"))
#etree.dump(site_owner_element)

sera_site = _read_sitexml(xml_file)

so = sera_site.site_owner
print(so)
print(so.personID)
print(so.institutionID)
sd = sera_site.site_description
print(sd)
#sc = sera_site.site_characterization
#print(sc)
#print(sd.ec8)
#print(sd.h800)
#print(sd.bedrock_depth)
#print(sd.geological_unit)
print(sera_site.analysis[0])
#print(sera_site.external_references[0].uri)
#print(sera_site.analysis[0].velocity_profile_survey)

print(sera_site.analysis[0].velocity_s30)
#print(sc.analysis[0].resonance_frequency)
print(sera_site)

qi2 = quality_index2(sera_site)
print("quality_index2 :" , qi2)
qi3 = quality_index3(1, 1, 0, 1, 1)
print("quality_index3 :" , qi3)
oqi = overall_quality_index(qi2, qi3)
print("overall_quality_index : " , oqi)

_write_sitexml(sera_site, "output_v1.3.4.xml", validate=True)
