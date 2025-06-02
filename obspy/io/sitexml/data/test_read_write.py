from lxml import etree
from obspy.io.sitexml.sitexml import (validate_sitexml, _read_sitexml)
from obspy.io.sitexml.write import _write_sitexml

xml_file = "./SiteOGPC_SERA_v1.2.xml"

validate_sitexml(xml_file)
xmldoc = etree.parse(xml_file)
sera_site = _read_sitexml(xml_file)

so = sera_site.site_owner
print(so)
sd = sera_site.site_description
print(sd)
sc = sera_site.site_characterization
print(sc)
print(sd.ec8)
print(sd.h800)
print(sd.bedrock_depth)
print(sd.geological_unit)
print(sc.velocity_s30)
print(sc.velocity_profile)

_write_sitexml(sera_site, "output.xml",validate=False)
