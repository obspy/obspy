# -*- coding: utf-8 -*-
"""
Checks all dataless SEED files archived by ORFEUS (12/2008).
"""

from StringIO import StringIO
from lxml import etree
from obspy.xseed import Parser
from obspy.xseed.utils import compareSEED
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
        # The originals of those four files contain compact date strings
        if seedfile.endswith('FR_CALF') or seedfile.endswith('FR_SAOF') or \
           seedfile.endswith('IU_SFJD') or seedfile.endswith('NO_JMIC'):
            compact = True
        else:
            compact = False
        x1seedfile = 'output' + os.sep + seedfile + '.1.xml'
        x2seedfile = 'output' + os.sep + seedfile + '.2.xml'
        oseedfile = 'output' + os.sep + seedfile
        # fetch original SEED file
        fp = open(seedfile, 'r')
        seed1 = fp.read()
        fp.close()
        # parse SEED
        sp = Parser(strict=True)
        sp.parseSEED(StringIO(seed1))
        # generate XSEED
        xml1 = sp.getXSEED()
        fp = open(x1seedfile, 'w')
        fp.write(xml1)
        fp.close()
        # test against schema
        doc = etree.parse(x1seedfile)
        xmlschema.assertValid(doc)
        # parse XSEED
        sp = Parser(strict=True, compact=compact)
        sp.parseXSEEDFile(x1seedfile)
        # generate SEED
        seed2 = sp.getSEED()
        fp = open(oseedfile, 'wb')
        fp.write(seed2)
        fp.close()
        # now parse this generate SEED 
        sp = Parser(strict=True)
        sp.parseSEEDFile(oseedfile)
        # generate XSEED
        xml2 = sp.getXSEED()
        fp = open(x2seedfile, 'w')
        fp.write(xml2)
        fp.close()
        # test against schema
        doc = etree.parse(x2seedfile)
        xmlschema.assertValid(doc)
        # parse XSEED
        sp = Parser(strict=True, compact=compact)
        sp.parseXSEED(StringIO(xml2))
        seed3 = sp.getSEED()
        # compare XSEED and SEED files
        try:
            assert xml1 == xml2
        except:
            import pdb;pdb.set_trace()
        try:
            assert seed2 == seed3
        except:
            import pdb;pdb.set_trace()
        # comparing original with generated SEED is more complicated
        compareSEED(seed1, seed2)
