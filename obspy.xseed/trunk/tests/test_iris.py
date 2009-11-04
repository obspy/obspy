# -*- coding: utf-8 -*-
"""
Checks all BB dataless SEED files archived by IRIS.
"""

from lxml import etree
from obspy.xseed import Parser
from obspy.xseed.utils import compareSEED
import os
from StringIO import StringIO


xmlschema = etree.parse('xml-seed-1.0-modified.xsd')
xmlschema = etree.XMLSchema(xmlschema)

input_base = os.path.join("data", "iris")
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
    for filename in filelist:
        print filename
        sp = Parser(strict=True)
        # try to parse
        print 'Parsing SEED...'
        sp.parseSEEDFile(filename)
        # generate a XML file and validate it with a given schema
        print 'Generating XSEED...'
        xml = sp.getXSEED()
        out = 'output' + os.sep + filename + '.xml'
        fp = open(out, 'w')
        fp.write(xml)
        fp.close()
        print 'Validating XSEED...'
        doc = etree.parse(out)
        xmlschema.assertValid(doc)
        print 'Parsing XSEED...'
        sp2 = Parser(strict=True)
        sp2.parseXSEED(StringIO(xml))
        print 'Generating SEED again...'
        seed2 = sp2.getSEED()
        # Open file.
        file = open(filename, 'r')
        seed1 = file.read()
        file.close()
        print 'Comparing SEED strings...'
        compareSEED(seed1, seed2)
        print 'DONE :-)'