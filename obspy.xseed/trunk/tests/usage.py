# -*- coding: utf-8 -*-

from obspy.xseed import SEEDParser
from lxml import etree
import StringIO

sp = SEEDParser(verify=True)
sp.parseSEEDFile('data/dataless.seed.BW_ZUGS')
fp = open('output/dataless.seed.BW_ZUGS.xml','w')
fp.write(sp.getXML())
fp.close()

#sp = SEEDParser()
#sp.parseSEEDFile('data/dataless.seed')
#fp = open('output/dataless.seed.xml','w')
#fp.write(sp.getXML())
#fp.close()


# read schema

xmlschema_doc = etree.parse('xml-seed.modified.xsd')
xmlschema = etree.XMLSchema(xmlschema_doc)

doc = etree.parse(StringIO.StringIO(sp.getXML()))
xmlschema.assertValid(doc)