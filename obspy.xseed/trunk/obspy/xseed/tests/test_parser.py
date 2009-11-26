# -*- coding: utf-8 -*-

from StringIO import StringIO
from glob import glob
from lxml import etree
from obspy.core.util import NamedTemporaryFile
from obspy.xseed.blockette.blockette010 import Blockette010
from obspy.xseed.blockette.blockette051 import Blockette051
from obspy.xseed.blockette.blockette053 import Blockette053
from obspy.xseed.blockette.blockette054 import Blockette054
from obspy.xseed.parser import Parser, SEEDParserException
from obspy.xseed.utils import compareSEED
import inspect
import os
import unittest


class ParserTestCase(unittest.TestCase):

    def setUp(self):
        # directory where the test files are located
        self.dir = os.path.dirname(inspect.getsourcefile(self.__class__))
        self.path = os.path.join(self.dir, 'data')

    def tearDown(self):
        pass

    def test_invalidStartHeader(self):
        """
        A SEED Volume must start with a Volume Index Control Header.
        """
        data = "000001S 0510019~~0001000000"
        sp = Parser(strict=True)
        self.assertRaises(SEEDParserException, sp.read, data)

    def test_invalidStartBlockette(self):
        """
        A SEED Volume must start with Blockette 010.
        """
        data = "000001V 0510019~~0001000000"
        sp = Parser(strict=True)
        self.assertRaises(SEEDParserException, sp.read, data)

    def test_string(self):
        """
        Tests string representation of L{obspy.xseed.Parser} object.
        """
        filename = os.path.join(self.path, 'dataless.seed.BW_MANZ')
        p = Parser(filename)
        sp = str(p).split()
        self.assertEquals(sp, ["BW.MANZ..EHZ", "BW.MANZ..EHN", "BW.MANZ..EHE"])

    def test_nonExistingFilename(self):
        """
        Test reading non existing file.
        """
        self.assertRaises(IOError, Parser, "XYZ")

    def test_blocketteStartsAfterRecord(self):
        """
        '... 058003504 1.00000E+00 0.00000E+0000 000006S*0543864 ... '
        ' 0543864' -> results in Blockette 005
        """
        # create a valid blockette 010 with record length 256
        b010 = "0100026 2.408~2038,001~~~~"
        blockette = Blockette010(strict=True, compact=True)
        blockette.parseSEED(b010)
        self.assertEquals(b010, blockette.getSEED())
        # create a valid blockette 054
        b054 = "0540240A0400300300000009" + ("+1.58748E-03" * 18)
        blockette = Blockette054(strict=True, compact=True)
        blockette.parseSEED(b054)
        self.assertEquals(b054, blockette.getSEED())
        # combine data
        data = "000001V " + b010 + (' ' * 222)
        data += "000002S " + b054 + (' ' * 8)
        data += "000003S*" + b054 + (' ' * 8)
        # read records
        parser = Parser(strict=True)
        parser.read(data)

    def test_multipleContinuedStationControlHeader(self):
        """
        """
        # create a valid blockette 010 with record length 256
        b010 = "0100026 2.408~2038,001~~~~"
        blockette = Blockette010(strict=True, compact=True)
        blockette.parseSEED(b010)
        self.assertEquals(b010, blockette.getSEED())
        # create a valid blockette 054
        b054 = "0540960A0400300300000039"
        nr = ""
        for i in range(0, 78):
            nr = nr + "+1.000%02dE-03" % i # 960 chars
        blockette = Blockette054(strict=True, compact=True)
        blockette.parseSEED(b054 + nr)
        self.assertEquals(b054 + nr, blockette.getSEED())
        # create a blockette 052
        b051 = '05100271999,123~~0001000000'
        blockette = Blockette051(strict=True)
        blockette.parseSEED(b051)
        # combine data (each line equals 256 chars)
        data = "000001V " + b010 + (' ' * 222)
        data += "000002S " + b054 + nr[0:224] # 256-8-24 = 224
        data += "000003S*" + nr[224:472] # 256-8 = 248
        data += "000004S*" + nr[472:720]
        data += "000005S*" + nr[720:] + b051 + ' ' * 5 # 5 spaces left
        self.assertEqual(len(data), 256 * 5)
        data += "000006S " + b054 + nr[0:224] # 256-8-24 = 224
        data += "000007S*" + nr[224:472] # 256-8 = 248
        data += "000008S*" + nr[472:720]
        data += "000009S*" + nr[720:] + ' ' * 32 # 32 spaces left
        self.assertEqual(len(data), 256 * 9)
        # read records
        parser = Parser(strict=True)
        parser.read(data)
        # check results
        self.assertEquals(sorted(parser.blockettes.keys()), [10, 51, 54])
        self.assertEquals(len(parser.blockettes[10]), 1)
        self.assertEquals(len(parser.blockettes[51]), 1)
        self.assertEquals(len(parser.blockettes[54]), 2)

    def test_blocketteLongerThanRecordLength(self):
        """
        If a blockette is longer than the record length it should result in
        more than one record.
        """
        parser = Parser(strict=True)
        # Set record length to 100.
        parser.record_length = 100
        # Use a blockette 53 string.
        SEED_string = '0530382A01002003+6.00770E+07+2.00000E-02002+0.00000E' \
            '+00+0.00000E+00+0.00000E+00+0.00000E+00+0.00000E+00+0.00000E+0' \
            '0+0.00000E+00+0.00000E+00005-3.70040E-02-3.70160E-02+0.00000E+' \
            '00+0.00000E+00-3.70040E-02+3.70160E-02+0.00000E+00+0.00000E+00' \
            '-2.51330E+02+0.00000E+00+0.00000E+00+0.00000E+00-1.31040E+02-4' \
            '.67290E+02+0.00000E+00+0.00000E+00-1.31040E+02+4.67290E+02+0.0' \
            '0000E+00+0.00000E+00'
        blkt_53 = Blockette053()
        blkt_53.parseSEED(SEED_string)
        # This just tests an internal SEED method.
        records = parser._createCutAndFlushRecord([blkt_53], 'S')
        # This should result in five records.
        self.assertEqual(len(records), 5)
        # Each records should be 100 - 6 = 94 long.
        for record in records:
            self.assertEqual(len(record), 94)
        # Reassemble the String.
        new_string = ''
        for record in records:
            new_string += record[2:]
        # Compare the new and the old string.
        self.assertEqual(new_string.strip(), SEED_string)

    def test_readAndWriteSEED(self):
        """
        Reads all SEED records from the Bavarian network and writes them
        again.
        
        This should not change them.
        
        There are some differences which will be edited before comparison:
        - The written SEED file will always have the version 2.4. BW uses
          version 2.3.
        
        The different formating of numbers in the stations blockettes will not
        be changed but 'evened'. Both are valid ways to do it - see SEED-Manual
        chapter 3 for more informations.
        """
        # Get all filenames.
        BW_SEED_files = glob(os.path.join(self.path, u'dataless.seed.BW*'))
        # Loop over all files.
        for file in BW_SEED_files:
            parser = Parser()
            f = open(file, 'r')
            # Original SEED file.
            original_seed = f.read()
            f.seek(0)
            # Parse and write the data.
            parser.read(f)
            f.close()
            new_seed = parser.getSEED()
            # compare both SEED strings
            compareSEED(original_seed, new_seed)
            del parser
            parser1 = Parser(original_seed)
            parser2 = Parser(new_seed)
            self.assertEqual(parser1.getSEED(), parser2.getSEED())
            del parser1, parser2

    def test_createReadAssertAndWriteXSEED(self):
        """
        This test takes some SEED files, reads them to a Parser object
        and converts them back to SEED once. This is done to avoid any
        formating issues as seen in test_readAndWriteSEED.
        
        Therefore the reading and writing of SEED files is considered to be
        correct.
        
        Finally the resulting SEED gets converted to XSEED and back to SEED
        and the two SEED strings are then evaluated to be identical.
        
        This tests also checks for XML validity using a XML schema.
        """
        # Get all filenames.
        BW_SEED_files = glob(os.path.join(self.path, u'dataless.seed.BW*'))
        # Loop over all files and versions.
        for version in ['1.0', '1.1']:
            # Path to xsd-file.
            xsd_path = os.path.join(self.path, 'xml-seed-%s.xsd' % version)
            # Prepare validator.
            f = open(xsd_path, 'r')
            xmlschema_doc = etree.parse(f)
            f.close()
            xmlschema = etree.XMLSchema(xmlschema_doc)
            for file in BW_SEED_files:
                # Parse the file.
                parser1 = Parser(file)
                # Convert to SEED once to avoid any issues seen in
                # test_readAndWriteSEED.
                original_seed = parser1.getSEED()
                del parser1
                # Now read the file, parse it, write XSEED, read XSEED and 
                # write SEED again. The output should be totally identical.
                parser2 = Parser(original_seed)
                xseed_string = parser2.getXSEED(version=version)
                del parser2
                # Validate XSEED.
                doc = etree.parse(StringIO(xseed_string))
                self.assertTrue(xmlschema.validate(doc))
                del doc
                parser3 = Parser(xseed_string)
                new_seed = parser3.getSEED()
                self.assertEqual(original_seed, new_seed)
                del parser3, original_seed, new_seed

    def test_readFullSEED(self):
        """
        Test the reading of a full-SEED file. The data portion will be omitted.
        """
        filename = os.path.join(self.path, 'arclink_full.seed')
        sp = Parser(filename)
        # Just checks whether certain blockettes are written.
        self.assertEqual(len(sp.stations), 1)
        self.assertEqual([_i.id for _i in sp.volume], [10])
        self.assertEqual([_i.id for _i in sp.abbreviations],
                [30, 33, 33, 34, 34, 34, 34, 41, 43, 44, 47, 47, 48, 48, 48])
        self.assertEqual([_i.id for _i in sp.stations[0]], [50, 52, 60, 58])
        self.assertEqual(sp.stations[0][0].network_code, 'GR')
        self.assertEqual(sp.stations[0][0].station_call_letters, 'FUR')
#
#    def test_createRESPFromXSEED(self):
#        """
#        Tests RESP file creation from XML-SEED.
#        """
#        ### example 1
#        # parse Dataless SEED
#        filename = os.path.join(self.path, 'dataless.seed.BW_FURT')
#        sp1 = Parser(filename)
#        # write XML-SEED
#        tempfile = NamedTemporaryFile().name
#        sp1.writeXSEED(tempfile)
#        # parse XML-SEED
#        sp2 = Parser(tempfile)
#        # create RESP files
#        _resp_list = sp2.getRESP()
#        os.remove(tempfile)
#        ### example 2
#        # parse Dataless SEED
#        filename = os.path.join(self.path, 'arclink_full.seed')
#        sp1 = Parser(filename)
#        # write XML-SEED
#        tempfile = NamedTemporaryFile().name
#        sp1.writeXSEED(tempfile)
#        # parse XML-SEED
#        sp2 = Parser(tempfile)
#        # create RESP files
#        _resp_list = sp2.getRESP()
#        os.remove(tempfile)


def suite():
    return unittest.makeSuite(ParserTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
