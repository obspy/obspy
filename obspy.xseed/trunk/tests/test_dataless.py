# -*- coding: utf-8 -*-
"""
Conversion test suite for Dataless SEED into XML-SEED and vice versa.

Runs tests against all Dataless SEED files within the data/dataless directory. 
Output is created within the output/dataless folder. Once generated files will
be skipped. Clear the output/dataless folder in order to rerun all tests.
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
xml_doc = etree.parse('xml-seed-1.0.xsd')
xmlschema100 = etree.XMLSchema(xml_doc)
# modified 1.0
xml_doc = etree.parse('xml-seed-1.0.1.xsd')
xmlschema101 = etree.XMLSchema(xml_doc)

# Exceptions
# The originals of those files contain compact date strings
compact_date_files = ['dataless-odc.FR_SAOF', 'dataless-odc.FR_CALF',
                      'dataless-odc.IU_SFJD', 'dataless-odc.NO_JMIC']
# not 100% XSEED 1.0 compatible, due to Blockette 060
xseed_incompatible = ['arclink.dataless.seed', '_US-BB.dataless']

# generate output directory 
if not os.path.isdir(output_base):
    os.mkdir(output_base)

# build up file list and loop over all files
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
        print "Skipping", os.path.join(relpath, seedfile)
        continue
    else:
        msg = "Parsing %s\t\t" % os.path.join(relpath, seedfile)
        print msg,
    # fetch original SEED file
    fp = open(file, 'r')
    seed1 = fp.read()
    fp.close()
    # set compact date flag
    compact = False
    if seedfile in compact_date_files:
        compact = True
    # start parsing
    try:
        print "rS",
        # parse SEED
        sp = Parser()
        sp.read(seed1)
        print "wX",
        # generate XSEED
        xml1 = sp.getXSEED()
        fp = open(x1seedfile, 'w')
        fp.write(xml1)
        fp.close()
        print "vX",
        # test against schemas
        doc = etree.parse(x1seedfile)
        if seedfile not in xseed_incompatible:
            xmlschema100.assertValid(doc)
        xmlschema101.assertValid(doc)
        print "rX",
        # parse XSEED
        sp = Parser(strict=True, compact=compact)
        sp.read(x1seedfile)
        print "wS",
        # generate SEED
        seed2 = sp.getSEED()
        sp.writeSEED(oseedfile)
        print "rS",
        # now parse this generate SEED 
        sp = Parser(strict=True)
        sp.read(seed2)
        print "wX",
        # generate XSEED
        xml2 = sp.getXSEED()
        fp = open(x2seedfile, 'w')
        fp.write(xml2)
        fp.close()
        print "vX",
        # test against schema
        doc = etree.parse(x2seedfile)
        if seedfile not in xseed_incompatible:
            xmlschema100.assertValid(doc)
        xmlschema101.assertValid(doc)
        print "rX",
        # parse XSEED
        sp = Parser(strict=True, compact=compact)
        sp.read(xml2)
        seed3 = sp.getSEED()
        print "c",
        # compare XSEED and SEED files
        if xml1 != xml2:
            raise Exception("XML-SEED strings differ")
        if seed2 != seed3:
            raise Exception("SEED strings differ")
        # comparing original with generated SEED is more complicated
        utils.compareSEED(seed1, seed2)
        print "."
    except Exception, e:
        # remove all related files
        if os.path.isfile(x1seedfile):
            os.remove(x1seedfile)
        if os.path.isfile(x2seedfile):
            os.remove(x2seedfile)
        if os.path.isfile(oseedfile):
            os.remove(oseedfile)
        # raise actual exception
        raise

