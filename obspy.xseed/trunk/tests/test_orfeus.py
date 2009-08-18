# -*- coding: utf-8 -*-
"""
Checks all dataless SEED files archived by ORFEUS (12/2008).
"""

from lxml import etree
from obspy.xseed import SEEDParser
import StringIO
import os


xmlschema = etree.parse('xml-seed.xsd')
xmlschema = etree.XMLSchema(xmlschema)

input_base = os.path.join("data", "orfeus")
output_base = os.path.join("output")

# generate output directory
if not os.path.isdir(output_base + os.sep + "data"):
    os.mkdir(output_base + os.sep + "data")
if not os.path.isdir(output_base + os.sep + input_base):
    os.mkdir(output_base + os.sep + input_base)

# Count all wrong validations.
validation_count = 0

for root, dirs, files in os.walk(input_base):
    # skip empty or SVN directories
    if not files or '.svn' in root:
        continue
    # generate output directory
    if not os.path.isdir(output_base + os.sep + root):
        os.mkdir(output_base + os.sep + root)
    # generate file list
    filelist = [os.path.join(root, fi) for fi in files if '.svn' not in fi]
    for filename in filelist:
        print filename
        # Cheap hack to leave the Spanish networks out as they do not work
        # currently.
        try:
            sp = SEEDParser(strict=True)
            # try to parse
            sp.parseSEEDFile(filename)
            # generate a XML file and validate it with a given schema
            xml = sp.getXSEED()
            doc = etree.parse(StringIO.StringIO(xml))
            if not xmlschema.validate(doc):
                validation_count += 1
                print 'Wrong XSEED for:', filename
            fp = open('output' + os.sep + filename + '.xml', 'w')
            fp.write(xml)
            fp.close()
        except:
            sp = SEEDParser(strict=True, debug=True)
            sp.parseSEEDFile(filename)
            fp = open('output' + os.sep + 'error.xml', 'w')
            fp.write(sp.getXSEED())
            fp.close()
            raise

print 'Number of not correctly validated files:', validation_count
