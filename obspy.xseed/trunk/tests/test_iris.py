# -*- coding: utf-8 -*-
"""
Checks all BB dataless SEED files archived by IRIS.
"""

from lxml import etree
from obspy.xseed import Parser
import os


xmlschema = etree.parse('xml-seed.xsd')
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
        sp.parseSEEDFile(filename)
        # generate a XML file and validate it with a given schema
        xml = sp.getXSEED()
        out = 'output' + os.sep + filename + '.xml'
        fp = open(out, 'w')
        fp.write(xml)
        fp.close()
        doc = etree.parse(out)
        xmlschema.assertValid(doc)
