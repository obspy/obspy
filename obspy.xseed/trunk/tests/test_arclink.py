# -*- coding: utf-8 -*-
"""
Checks dataless SEED fetched via ArcLink.
"""

from lxml import etree
from obspy.xseed import Parser
from obspy.xseed.utils import compareSEED
from StringIO import StringIO
import os


xmlschema = etree.parse('xml-seed-1.1.xsd')
xmlschema = etree.XMLSchema(xmlschema)

input_base = os.path.join("data", "arclink")
output_base = os.path.join("output")

# generate output directory
if not os.path.isdir(output_base + os.sep + "data"):
    os.mkdir(output_base + os.sep + "data")
if not os.path.isdir(output_base + os.sep + input_base):
    os.mkdir(output_base + os.sep + input_base)

for root, dirs, files in os.walk(input_base):
    # skip empty or SVN directories
    if not files or '.svn' in root:
        continue
    # generate output directory
    if not os.path.isdir(output_base + os.sep + root):
        os.mkdir(output_base + os.sep + root)
    # generate file list
    filelist = [os.path.join(root, fi) for fi in files if '.svn' not in fi]
    for seedfile in filelist:
        print seedfile
        x1seedfile = 'output' + os.sep + seedfile + '.1.xml'
        x2seedfile = 'output' + os.sep + seedfile + '.2.xml'
        oseedfile = 'output' + os.sep + seedfile
        # fetch original SEED file
        fp = open(seedfile, 'r')
        seed1 = fp.read()
        fp.close()
        # parse SEED
        sp = Parser()
        sp.parseSEED(StringIO(seed1))
        # generate XSEED
        xml1 = sp.getXSEED()
        fp = open(x1seedfile, 'w')
        fp.write(xml1)
        fp.close()
        # test against schema
        #doc = etree.parse(x1seedfile)
        #xmlschema.assertValid(doc)
        # parse XSEED
        sp = Parser(strict=True)
        sp.parseXSEEDFile(x1seedfile)
        # generate SEED
        seed2 = sp.getSEED()
        fp = open(oseedfile, 'wb')
        fp.write(seed2)
        fp.close()
        # now parse this generate SEED 
        sp = Parser(strict=True)
        sp.parseSEED(StringIO(seed2))
        # generate XSEED
        xml2 = sp.getXSEED()
        fp = open(x2seedfile, 'w')
        fp.write(xml2)
        fp.close()
        # test against schema
        #doc = etree.parse(x2seedfile)
        #xmlschema.assertValid(doc)
        # parse XSEED
        sp = Parser(strict=True)
        sp.parseXSEED(StringIO(xml2))
        seed3 = sp.getSEED()
        # compare XSEED and SEED files
        assert xml1 == xml2
        assert seed2 == seed3
        # comparing original with generated SEED is more complicated
        compareSEED(seed1, seed2)

