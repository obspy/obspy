# -*- coding: utf-8 -*-

from glob import iglob
from lxml import etree
from obspy.xseed.blockette import Blockette054, Blockette060
from obspy.xseed.blockette.blockette import BlocketteLengthException
import os
import sys
import unittest


class BlocketteTestCase(unittest.TestCase):
    """
    Test cases for all blockettes.
    """
    def setUp(self):
        # directory where the test files are located
        self.path = os.path.dirname(__file__)

    def test_invalidBlocketteLength(self):
        """
        A wrong blockette length should raise an exception.
        """
        # create a blockette 054 which is way to long
        b054 = "0540240A0400300300000020" + ("+1.58748E-03" * 40)
        blockette = Blockette054(strict=True)
        self.assertRaises(BlocketteLengthException, blockette.parseSEED, b054)

    def parseFile(self, blkt_file):
        """
        Parses the test definition file and creates a list of it.
        """
        # Create a new empty list to store all information of the test in it.
        test_examples = []
        # Now read the corresponding file and parse it.
        file = open(blkt_file, 'r')
        # Helper variable to parse the file. Might be a little bit slow but its
        # just for tests.
        cur_stat = None
        for line in file:
            # Skip unnecessary content.
            if not cur_stat:
                if line[0:2] != '--':
                    continue
                else:
                    # If new example number append new list.
                    if len(test_examples) != int(line[2:4]):
                        test_examples.append([])
                    # Append new list to the current example of the list. The
                    # list contains the type of the example.
                    test_examples[-1].append([line[5:].replace('\n', '')])
                    cur_stat = 1
                    continue
            # Filter out any empty/commentary lines still remaining and also
            # set cur_stat to None.
            if line.strip() == '':
                cur_stat = None
                continue
            elif line.strip()[0] == '#':
                cur_stat = None
                continue
            test_examples[-1][-1].append(line)
        file.close()
        # Simplify and Validate the list.
        self.simplifyAndValidateAndCreateDictionary(test_examples)
        return test_examples

    def simplifyAndValidateAndCreateDictionary(self, examples):
        """
        Takes an examples list and combines the XSEED strings and validates the
        list.
        Afterwards in creates a list containing a dictionary for each example
        set.
        """
        # Loop over each example:
        for example in examples:
            if len(example) < 2:
                msg = 'At least one SEED and XSEED string for each examples.'
                raise Exception(msg)
            # Exactly one SEED string needed.
            if [item[0] for item in example].count('SEED') != 1:
                msg = 'Only one SEED string per example!'
                raise Exception(msg)
            # Some other stuff is not tested! Please be careful and adhere to
            # the format rules for writing blockette tests.
            for _i in xrange(len(example)):
                ex_type = example[_i]
                # Loop over each type in the example and differentiate between
                # SEED and not SEED types.
                if ex_type[0] == 'SEED':
                    # Nothing to do here except to remove the line ending.
                    ex_type[1] = ex_type[1][:-1]
                    continue
                # Remove spaces and line endings.
                for _j in range(len(ex_type)):
                    temp_string = ex_type[_j].strip()
                    if temp_string.endswith('\n'):
                        temp_string = temp_string[:-2]
                    ex_type[_j] = temp_string
                # Combine all XSEED strings to one large string. The weird
                # PLACEHOLDER syntax assures that flat fields are parsed
                # correctly because you need to have a space between them.
                ex_type.insert(1, 'PLACEHOLDER'.join(ex_type[1:]))
                ex_type[1] = ex_type[1].replace('>PLACEHOLDER<', '><')
                ex_type[1] = ex_type[1].replace('>PLACEHOLDER', '>')
                ex_type[1] = ex_type[1].replace('PLACEHOLDER<', '<')
                ex_type[1] = ex_type[1].replace('PLACEHOLDER', ' ')
                example[_i] = ex_type[0:2]
        # Now create a dictionary for each example.
        for _i in xrange(len(examples)):
            ex_dict = {}
            for part in examples[_i]:
                ex_dict[part[0]] = part[1]
            examples[_i] = ex_dict

    def SEEDAndXSEEDConversion(self, test_examples, blkt_number):
        """
        Takes everything in the prepared list and tests the SEED/XSEED
        conversion for all given formats.
        """
        # Another loop over all examples.
        for example in test_examples:
            # Create several blockette instances
            # One to read from SEED and one for each XSEED version.
            blkt_module = 'obspy.xseed.blockette.blockette' + blkt_number
            blkt_class_name = 'Blockette' + blkt_number
            blkt = sys.modules[blkt_module].__dict__[blkt_class_name]

            versions = {}
            # prepare SEED
            versions['SEED'] = {}
            versions['SEED']['blkt'] = blkt()
            versions['SEED']['data'] = example['SEED']
            versions['SEED']['blkt'].parseSEED(example['SEED'])

            # prepare XSEED
            for key, data in example.iteritems():
                if not 'XSEED' in key:
                    continue
                if key == 'XSEED':
                    key = ''
                versions[key] = {}
                versions[key]['version'] = key[6:]
                versions[key]['blkt'] = blkt(xseed_version=key[6:])
                versions[key]['blkt'].parseXML(etree.fromstring(data))
                versions[key]['data'] = data
            # loop over all combinations
            errmsg = 'Blockette %s - Getting %s from %s\n%s\n!=\n%s'
            for key1, blkt1 in versions.iteritems():
                # conversion to SEED
                seed = blkt1['blkt'].getSEED()
                self.assertEqual(seed, versions['SEED']['data'],
                                 errmsg % (blkt_number, 'SEED', key1,
                                           seed, versions['SEED']['data']))
                for key2, blkt2 in versions.iteritems():
                    if key2 == 'SEED':
                        continue
                    xseed = etree.tostring(blkt1['blkt'].getXML(\
                        xseed_version=blkt2['version']))
                    self.assertEqual(xseed, versions[key2]['data'],
                                     errmsg % (blkt_number, 'XSEED', key2,
                                               xseed, blkt2['data']))

    def test_allBlockettes(self):
        """
        Tests all Blockettes.
        """
        # Loop over all files in the blockette-tests directory.
        path = os.path.join(self.path, 'blockette-tests', 'blockette*.txt')
        for blkt_file in iglob(path):
            # Get blockette number.
            blkt_number = blkt_file[-7:-4]
            # Check whether the blockette class can be loaded.
            try:
                __import__('obspy.xseed.blockette.blockette' + blkt_number)
            except:
                msg = 'Failed to import blockette', blkt_number
                raise ImportError(msg)
            # Parse the file.
            test_examples = self.parseFile(blkt_file)
            # The last step is to actually test the conversions to and from
            # SEED/XSEED for every example in every direction.
            self.SEEDAndXSEEDConversion(test_examples, blkt_number)

    def test_blockette60_has_blockette_id(self):
        """
        Blockette 60 overwrites the init method. Check that the parent class is
        called.
        """
        blkt = Blockette060()
        self.assertEqual(blkt.blockette_id, "060")
        self.assertEqual(blkt.id, 60)


def suite():
    return unittest.makeSuite(BlocketteTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
