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
dataless_path = os.path.join("data", "dataless")
output_path = os.path.join("output", "dataless")

# validation schemas
schema_10 = os.path.join(os.pardir, 'obspy', 'xseed' ,'tests', 'data',
                                                        'xml-seed-1.0.xsd')
schema_11 = os.path.join(os.pardir, 'obspy', 'xseed' ,'tests', 'data',
                                                        'xml-seed-1.1.xsd')
xml_doc_10 = etree.parse(schema_10)
xml_doc_11 = etree.parse(schema_11)
xmlschema10 = etree.XMLSchema(xml_doc_10)
xmlschema11 = etree.XMLSchema(xml_doc_11)

# Exceptions
# The originals of those files contain compact date strings
compact_date_files = ['dataless-odc.FR_SAOF', 'dataless-odc.FR_CALF',
                      'dataless-odc.IU_SFJD', 'dataless-odc.NO_JMIC']

# generate output directory 
if not os.path.isdir(output_path):
    os.mkdir(output_path)

# build up file list and loop over all files
files = []
files += glob.glob(os.path.join(dataless_path, '*', '*'))
files += glob.glob(os.path.join(dataless_path, '*', '*', '*'))
for file in files:
    # Check for arclink file. This file contains blockette 60 and cannot
    # produce a valid XSEED-1.0. Iris file contain blockette 62 which also
    # cannot be validated.
    skip = False
    if 'arclink' in file or 'iris' in file:
        skip = True
    # check and eventually generate output directory
    path = os.path.dirname(file)
    relpath = os.path.relpath(path, dataless_path)
    path = os.path.join(output_path, relpath)
    if not os.path.isdir(path):
        os.mkdir(path)
    # skip directories
    if not os.path.isfile(file):
        continue
    seedfile = os.path.basename(file)
    # create filenames for output directory
    x1seedfile_10 = path + os.sep + seedfile + '.1_0.1.xml'
    x2seedfile_10 = path + os.sep + seedfile + '.1_0.2.xml'
    x1seedfile_11 = path + os.sep + seedfile + '.1_1.1.xml'
    x2seedfile_11 = path + os.sep + seedfile + '.1_1.2.xml'
    oseedfile10 = path + os.sep + seedfile + '.10'
    oseedfile11 = path + os.sep + seedfile + '.11'
    # skip existing files
    if os.path.isfile(x2seedfile_10):
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
        sp = Parser(seed1)
        print "wX",
        # generate XSEED versions 1.0 and 1.1
        f1 = open(x1seedfile_10, 'w')
        f2 = open(x1seedfile_11, 'w')
        xml1_10 = sp.getXSEED(version = '1.0')
        xml1_11 = sp.getXSEED()
        f1.write(xml1_10)
        f1.close()
        f2.write(xml1_11)
        f2.close()
        print "vX",
        # test against schemas
        if not skip:
            doc1 = etree.parse(x1seedfile_10)
            xmlschema10.assertValid(doc1)
        doc2 = etree.parse(x1seedfile_11)
        xmlschema11.assertValid(doc2)
        print "rX",
        # parse XSEED in both versions.
        sp_10 = Parser(x1seedfile_10, strict=True, compact=compact)
        sp_11 = Parser(x1seedfile_11, strict=True, compact=compact)
        print "wS",
        # generate SEED again from both versions.
        f1 = open(oseedfile10, 'w')
        f2 = open(oseedfile11, 'w')
        seed1_10 = sp_10.getSEED()
        seed1_11 = sp_11.getSEED()
        f1.write(seed1_10)
        f2.write(seed1_11)
        f1.close()
        f2.close()
        print "rS",
        # now parse this generate SEED 
        sp_10 = Parser(oseedfile10)
        sp_11 = Parser(oseedfile11)
        print "wX",
        # generate XSEED again.
        f1 = open(x1seedfile_10, 'w')
        f2 = open(x1seedfile_11, 'w')
        f3 = open(x2seedfile_10, 'w')
        f4 = open(x2seedfile_11, 'w')
        xml2_10 = sp_10.getXSEED(version = '1.0')
        xml2_11 = sp_10.getXSEED()
        xml3_10 = sp_11.getXSEED(version = '1.0')
        xml3_11 = sp_11.getXSEED()
        f1.write(xml2_10)
        f1.close()
        f2.write(xml2_11)
        f2.close()
        f3.write(xml3_10)
        f3.close()
        f4.write(xml3_11)
        f4.close()
        print "vX",
        # test against schema
        if not skip:
            doc1 = etree.parse(x1seedfile_10)
            xmlschema10.assertValid(doc1)
        doc2 = etree.parse(x1seedfile_11)
        xmlschema11.assertValid(doc2)
        if not skip:
            doc3 = etree.parse(x2seedfile_10)
            xmlschema10.assertValid(doc3)
        doc4 = etree.parse(x2seedfile_11)
        xmlschema11.assertValid(doc4)
        print "rX",
        # parse XSEED
        sp_10 = Parser(xml2_10, strict=True, compact=compact)
        sp_11 = Parser(xml2_11, strict=True, compact=compact)
        sp_20 = Parser(xml3_10, strict=True, compact=compact)
        sp_21 = Parser(xml3_11, strict=True, compact=compact)
        seed2_10 = sp_10.getSEED()
        seed2_11 = sp_11.getSEED()
        seed3_10 = sp_20.getSEED()
        seed3_11 = sp_21.getSEED()
        print "c",
        # compare XSEED and SEED files
        if xml1_10 != xml2_10 or xml1_10 != xml3_10:
            raise Exception("XML-SEED 1.0 strings differ")
        if xml1_11 != xml2_11 or xml1_11 != xml3_11:
            raise Exception("XML-SEED 1.1 strings differ")
        if seed1_10 != seed1_11 or seed1_10 != seed2_10 or \
           seed1_10 != seed2_11 or seed1_10 != seed3_10 or \
           seed1_10 != seed3_11:
            raise Exception("SEED strings differ")
        # comparing original with generated SEED is more complicated
        utils.compareSEED(seed1, seed1_10)
        print "."
    except Exception, e:
        import pdb;pdb.set_trace()
        # remove all related files
        if os.path.isfile(x1seedfile_10):
            os.remove(x1seedfile_10)
        if os.path.isfile(x2seedfile_10):
            os.remove(x2seedfile_10)
        if os.path.isfile(x1seedfile_11):
            os.remove(x1seedfile_11)
        if os.path.isfile(x2seedfile_11):
            os.remove(x2seedfile_11)
        if os.path.isfile(oseedfile_10):
            os.remove(oseedfile_10)
        if os.path.isfile(oseedfile_11):
            os.remove(oseedfile_11)
        # raise actual exception
        raise

