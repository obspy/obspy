# -*- coding: utf-8 -*-
"""
Checks all dataless SEED files from the Bavarian network (12/2008).
"""

from lxml import etree
from obspy.xseed import Parser, utils
import glob
import os

# paths
input_base = os.path.join("data", "dataless")
output_base = os.path.join("output", "dataless")

# validation schemas
# original 1.0
xmlschema = etree.parse('xml-seed.xsd')
xmlschema = etree.XMLSchema(xmlschema)
# modified 1.0
xmlschema2 = etree.parse('xml-seed-1.0-modified.xsd')
xmlschema2 = etree.XMLSchema(xmlschema2)


files = []
files += glob.glob(os.path.join(input_base, '*', '*'))
files += glob.glob(os.path.join(input_base, '*', '*', '*'))
for file in files:
    # check and eventually generate output directory
    path = os.path.dirname(file)
    relpath = os.path.relpath(path, input_base)
    path = os.path.join(output_base, relpath)
    if not os.path.isdir(path):
        os.mkdir(path)
    # skip directories
    if not os.path.isfile(file):
        continue
    seedfile = os.path.basename(file)
    # create filenames for output directory
    x1seedfile = path + os.sep + seedfile + '.1.xml'
    x2seedfile = path + os.sep + seedfile + '.2.xml'
    oseedfile = path + os.sep + seedfile
    # skip existing files
    if os.path.isfile(x2seedfile):
        print "Skipping " + seedfile
        continue
    else:
        print "Creating " + seedfile
    # fetch original SEED file
    fp = open(file, 'r')
    seed1 = fp.read()
    fp.close()
    # parse SEED
    sp = Parser()
    sp.read(seed1)
    # generate XSEED
    xml1 = sp.getXSEED()
    fp = open(x1seedfile, 'w')
    fp.write(xml1)
    fp.close()
    # test against schemas
    doc = etree.parse(x1seedfile)
    #xmlschema.assertValid(doc)
    #xmlschema2.assertValid(doc)
    # parse XSEED
    sp = Parser(strict=True)
    sp.read(x1seedfile)
    # generate SEED
    seed2 = sp.getSEED()
    fp = open(oseedfile, 'wb')
    fp.write(seed2)
    fp.close()
    # now parse this generate SEED 
    sp = Parser(strict=True)
    sp.read(seed2)
    # generate XSEED
    xml2 = sp.getXSEED()
    fp = open(x2seedfile, 'w')
    fp.write(xml2)
    fp.close()
    # test against schema
    doc = etree.parse(x2seedfile)
    #xmlschema.assertValid(doc)
    #xmlschema2.assertValid(doc)
    # parse XSEED
    sp = Parser(strict=True)
    sp.read(xml2)
    seed3 = sp.getSEED()
    # compare XSEED and SEED files
    assert xml1 == xml2
    assert seed2 == seed3
    # comparing original with generated SEED is more complicated
    utils.compareSEED(seed1, seed2)

