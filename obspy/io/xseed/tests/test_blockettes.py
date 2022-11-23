# -*- coding: utf-8 -*-
import importlib
import io
import os
import sys
import unittest
import warnings
from glob import iglob

from lxml import etree

from obspy.io.xseed.blockette import (
    Blockette041, Blockette050, Blockette054, Blockette060)
from obspy.io.xseed.blockette.blockette import BlocketteLengthException
from obspy.io.xseed.fields import SEEDTypeException


class BlocketteTestCase(unittest.TestCase):
    """
    Test cases for all blockettes.
    """
    def setUp(self):
        # directory where the test files are located
        self.path = os.path.dirname(__file__)

    def test_invalid_blockette_length(self):
        """
        A wrong blockette length should raise an exception.
        """
        # create a blockette 054 which is way to long
        b054 = b"0540240A0400300300000020" + (b"+1.58748E-03" * 40)
        blockette = Blockette054(strict=True)
        self.assertRaises(BlocketteLengthException, blockette.parse_seed, b054)

    def parse_file(self, blkt_file):
        """
        Parses the test definition file and creates a list of it.
        """
        # Create a new empty list to store all information of the test in it.
        test_examples = []
        # Now read the corresponding file and parse it.
        with io.open(blkt_file, 'rt') as fh:
            # Helper variable to parse the file. Might be a little bit slow but
            # its just for tests.
            cur_stat = None
            for line in fh:
                # Skip unnecessary content.
                if not cur_stat:
                    if line[0:2] != '--':
                        continue
                    else:
                        # If new example number append new list.
                        if len(test_examples) != int(line[2:4]):
                            test_examples.append([])
                        # Append new list to the current example of the list.
                        # The list contains the type of the example.
                        test_examples[-1].append([line[5:].replace('\n', '')])
                        cur_stat = 1
                        continue
                # Filter out any empty/commentary lines still remaining and
                # also set cur_stat to None.
                if line.strip() == '':
                    cur_stat = None
                    continue
                elif line.strip()[0] == '#':
                    cur_stat = None
                    continue
                test_examples[-1][-1].append(line)
        # Simplify and Validate the list.
        self.simplify_and_validate_and_create_dictionary(test_examples)
        # Convert everything to bytes - a lot of the logic in this method
        # here assumes strings to its easier to work with strings and
        # convert to bytes after everything is done.
        for _i in test_examples:
            for key, value in _i.items():
                _i[key] = value.encode()
        return test_examples

    def simplify_and_validate_and_create_dictionary(self, examples):
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
            for _i in range(len(example)):
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
        for _i in range(len(examples)):
            ex_dict = {}
            for part in examples[_i]:
                ex_dict[part[0]] = part[1]
            examples[_i] = ex_dict

    def seed_and_xseed_conversion(self, test_examples, blkt_number):
        """
        Takes everything in the prepared list and tests the SEED/XSEED
        conversion for all given formats.
        """
        # Another loop over all examples.
        for example in test_examples:
            # Create several blockette instances
            # One to read from SEED and one for each XSEED version.
            blkt_module = 'obspy.io.xseed.blockette.blockette' + blkt_number
            blkt_class_name = 'Blockette' + blkt_number
            blkt = sys.modules[blkt_module].__dict__[blkt_class_name]

            versions = {}
            # prepare SEED
            versions['SEED'] = {}
            versions['SEED']['Blkt'] = blkt()
            versions['SEED']['data'] = example['SEED']
            versions['SEED']['Blkt'].parse_seed(example['SEED'])

            # prepare XSEED
            for key, data in example.items():
                if 'XSEED' not in key:
                    continue
                if key == 'XSEED':
                    key = ''
                versions[key] = {}
                versions[key]['version'] = key[6:]
                versions[key]['Blkt'] = blkt(xseed_version=key[6:])
                versions[key]['Blkt'].parse_xml(etree.fromstring(data))
                versions[key]['data'] = data
            # loop over all combinations
            errmsg = 'Blockette %s - Getting %s from %s\n%s\n!=\n%s'
            for key1, blkt1 in versions.items():
                # conversion to SEED
                seed = blkt1['Blkt'].get_seed()
                self.assertEqual(seed, versions['SEED']['data'],
                                 errmsg % (blkt_number, 'SEED', key1,
                                           seed, versions['SEED']['data']))
                for key2, blkt2 in versions.items():
                    if key2 == 'SEED':
                        continue
                    xseed = etree.tostring(blkt1['Blkt'].get_xml(
                        xseed_version=blkt2['version']))
                    self.assertEqual(xseed, versions[key2]['data'],
                                     errmsg % (blkt_number, 'XSEED', key2,
                                               xseed, blkt2['data']))

    def test_all_blockettes(self):
        """
        Tests all Blockettes.
        """
        # Loop over all files in the blockette-tests directory.
        path = os.path.join(self.path, 'blockette-tests', 'blockette*.txt')
        test_example_count = 0
        for blkt_file in iglob(path):
            # Get blockette number.
            blkt_number = blkt_file[-7:-4]
            # Check whether the blockette class can be loaded.
            try:
                __import__('obspy.io.xseed.blockette.blockette' + blkt_number)
            except Exception:
                msg = 'Failed to import blockette', blkt_number
                raise ImportError(msg)
            # Parse the file.
            test_examples = self.parse_file(blkt_file)
            # Check how many examples were actually tested.
            test_example_count += len(test_examples)
            # The last step is to actually test the conversions to and from
            # SEED/XSEED for every example in every direction.
            self.seed_and_xseed_conversion(test_examples, blkt_number)
        self.assertGreater(test_example_count, 0)

    def test_all_blockettes_get_resp(self):
        """
        Tests get_resp() for all blockettes that have that method.
        """
        tested_blockettes = 0
        # Loop over all files in the blockette-tests directory.
        path = os.path.join(self.path, 'blockette-tests', 'blockette*.txt')
        for blkt_file in iglob(path):

            # Import the corresponding blockette.
            blkt_number = blkt_file[-7:-4]
            blkt = importlib.import_module(
                'obspy.io.xseed.blockette.blockette%s' % blkt_number)
            blkt = getattr(blkt, "Blockette%s" % blkt_number)

            # Skip if it has no get_resp() method.
            if not hasattr(blkt, "get_resp"):
                continue
            tested_blockettes += 1

            # Some of the blockettes requires access to other blockettes.
            # They don't have it for this simplistic test and thus raise
            # warnings which therefore have to be caught.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                test_examples = self.parse_file(blkt_file)
                for ex in test_examples:
                    b = blkt()
                    b.parse_seed(ex["SEED"])
                    if blkt_number != "060":
                        r = b.get_resp("AA", "BB", [])
                    else:
                        # Blockette 60 is special as it also needs access to
                        # other blockettes. Create some dummy data here.
                        b.stages = [[0]]
                        blkt41 = Blockette041()
                        blkt41.response_lookup_key = 0
                        blkt41.symmetry_code = 0
                        blkt41.signal_in_units = "m/s"
                        blkt41.signal_out_units = "m/s"
                        blkt41.number_of_factors = 0
                        r = b.get_resp("AA", "BB", [blkt41])
                    # For now only check that it contains something and that it
                    # returns bytes.
                    self.assertGreater(len(r), 0)
                    self.assertTrue(isinstance(r, bytes))

        self.assertGreater(tested_blockettes, 0)

    def test_blockette60_has_blockette_id(self):
        """
        Blockette 60 overwrites the init method. Check that the parent class is
        called.
        """
        blkt = Blockette060()
        self.assertEqual(blkt.blockette_id, "060")
        self.assertEqual(blkt.id, 60)

    def test_issue701(self):
        """
        Testing an oversized site name.
        """

        b050_orig = "0500168ANTF +43.564000  +7.123000  +54.0   6  0" + \
            "Antibes - 06004 - Alpes-Maritimes - Provence-Alpes-Côte d'" + \
            "Azur - France~ 363210102003,211,11:18:00~2004,146,08:52:00~NFR"
        b050_cut = "0500166ANTF +43.564000  +7.123000  +54.00006000" + \
            "Antibes - 06004 - Alpes-Maritimes - Provence-Alpes-Côte d'A~" + \
            "0363210102003,211,11:18:00.0000~2004,146,08:52:00.0000~NFR"
        # reading should work but without issues
        blockette = Blockette050()
        blockette.parse_seed(b050_orig.encode('utf-8'))
        self.assertEqual(len(blockette.site_name.encode('utf-8')), 72)
        with warnings.catch_warnings(record=True):
            warnings.simplefilter('error', UserWarning)
            self.assertRaises(UserWarning, blockette.get_seed)
            # Now ignore the warnings and test the default values.
            warnings.simplefilter('ignore', UserWarning)
            # writing should cut to 60 chars
            out = blockette.get_seed()
            # utf-8 only needed for PY2
            self.assertEqual(out.decode('utf-8'), b050_cut)
            # reading it again should have cut length
            blockette = Blockette050()
            blockette.parse_seed(out)
            self.assertEqual(len(blockette.site_name.encode('utf-8')), 60)
        # writing with strict=True will raise
        blockette = Blockette050(strict=True)
        blockette.parse_seed(b050_orig.encode('utf-8'))
        self.assertRaises(SEEDTypeException, blockette.get_seed)

    def test_equality_and_unequality(self):
        """
        Tests the rich (un)equality dunder methods.
        """
        a = Blockette050()
        b = Blockette050()
        c = Blockette050()
        d = Blockette060()

        a.test = 1
        b.test = 1
        c.test = 2
        d.test = 1

        self.assertTrue(a == a)
        self.assertTrue(a == b)
        self.assertTrue(a != c)
        self.assertTrue(a != d)
        # Revert order
        self.assertTrue(a == a)
        self.assertTrue(b == a)
        self.assertTrue(c != a)
        self.assertTrue(d != a)
        # Negations
        self.assertFalse(a != a)
        self.assertFalse(a != b)
        self.assertFalse(a == c)
        self.assertFalse(a == d)
