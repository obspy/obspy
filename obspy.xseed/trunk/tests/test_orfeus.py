# -*- coding: utf-8 -*-

from lxml import etree
from obspy.xseed import SEEDParser
import StringIO
import os


xmlschema = etree.parse('xml-seed.modified.xsd')
xmlschema = etree.XMLSchema(xmlschema)

for root,dir,files in os.walk(os.path.join("data", "orfeus")):
    filelist = [ os.path.join(root,fi) for fi in files]
    for filename in filelist:
        if 'svn' in filename:
            continue
        print filename
        sp = SEEDParser(strict=True)
        try:
            sp.parseSEEDFile(filename)
            doc = etree.parse(StringIO.StringIO(sp.getXML()))
            xmlschema.assertValid(doc)
        except:
            sp.debug = True
            sp.parseSEEDFile(filename)
            raise
