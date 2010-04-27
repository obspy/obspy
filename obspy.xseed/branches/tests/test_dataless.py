# -*- coding: utf-8 -*-
"""
Conversion test suite for Dataless SEED into XML-SEED and vice versa.

Runs tests against all Dataless SEED files within the data/dataless directory. 
Output is created within the output/dataless folder. Once generated files will
be skipped. Clear the output/dataless folder in order to rerun all tests.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""

from lxml import etree
from obspy.xseed import Parser, utils
import glob
import os
import sys


# paths
dataless_path = os.path.join("data", "dataless")
output_path = os.path.join("output", "dataless")

# validation schemas
schema_10 = 'xml-seed-1.0.xsd'
schema_11 = 'xml-seed-1.1.xsd'
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
    xseedfile_10 = path + os.sep + seedfile + '.1_0.xml'
    xseedfile_11 = path + os.sep + seedfile + '.1_1.xml'
    seedfile = path + os.sep + seedfile
    # skip existing files
    if os.path.isfile(xseedfile_10):
        print "Skipping", os.path.join(relpath, os.path.basename(seedfile))
        continue
    else:
        msg = "Parsing %s\t\t" % os.path.join(relpath,
                                              os.path.basename(seedfile))
        print msg,
    # fetch original SEED file
    fp = open(file, 'r')
    org_seed = fp.read()
    fp.close()
    # set compact date flag
    compact = False
    if os.path.basename(file) in compact_date_files:
        compact = True
    # start parsing
    try:
        print "rS",
        sys.stdout.flush()
        # parse SEED
        sp = Parser(org_seed)
        print "wS",
        sys.stdout.flush()
        # write SEED to compare to original SEED.
        f1 = open(seedfile, 'w')
        seed = sp.getSEED(compact=compact)
        f1.write(seed)
        f1.close()
        print "cS",
        sys.stdout.flush()
        # Compare to original SEED.
        utils.compareSEED(org_seed, seed)
        print "wX",
        sys.stdout.flush()
        # generate XSEED versions 1.0 and 1.1
        f1 = open(xseedfile_10, 'w')
        f2 = open(xseedfile_11, 'w')
        xml_10 = sp.getXSEED(version='1.0')
        xml_11 = sp.getXSEED()
        f1.write(xml_10)
        f1.close()
        f2.write(xml_11)
        f2.close()
        print "vX",
        sys.stdout.flush()
        # test against schemas
        if not skip:
            doc1 = etree.parse(xseedfile_10)
            xmlschema10.assertValid(doc1)
        doc2 = etree.parse(xseedfile_11)
        xmlschema11.assertValid(doc2)
        print "rS",
        sys.stdout.flush()
        # parse the created SEED file.
        sp1 = Parser(seedfile)
        print "rX",
        sys.stdout.flush()
        # parse XSEED in both versions.
        sp2 = Parser(xseedfile_10, strict=True)
        sp3 = Parser(xseedfile_11, strict=True)
        print "wScS",
        sys.stdout.flush()
        parsers = [sp1, sp2, sp3]
        # generate SEED again from all three versions and compare to seed.
        for parser in parsers:
            if seed != parser.getSEED(compact=compact):
                raise Exception("SEED strings differ")
        print "wX1.0cX",
        sys.stdout.flush()
        # generate XSEED 1.0 again from all three versions and compare.
        for parser in parsers:
            if xml_10 != parser.getXSEED(version='1.0'):
                raise Exception("XML-SEED 1.0 strings differ")
        print "wX1.1cX",
        sys.stdout.flush()
        # generate XSEED 1.0 again from all three versions and compare.
        for parser in parsers:
            if xml_11 != parser.getXSEED():
                raise Exception("XML-SEED 1.1 strings differ")
        print "."
        sys.stdout.flush()
    except Exception, e:
        # remove all related files
        if os.path.isfile(xseedfile_10):
            os.remove(xseedfile_10)
        if os.path.isfile(xseedfile_11):
            os.remove(xseedfile_11)
        if os.path.isfile(seedfile):
            os.remove(seedfile)
        # raise actual exception
        raise
