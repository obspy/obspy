# -*- coding: utf-8 -*-

# With statement for Python 2.5. Necessary although not used in Python 2.5.
from __future__ import with_statement

from StringIO import StringIO
from lxml import etree
from obspy.core.util import NamedTemporaryFile
from obspy.core import UTCDateTime
from obspy.xseed.blockette.blockette010 import Blockette010
from obspy.xseed.blockette.blockette051 import Blockette051
from obspy.xseed.blockette.blockette053 import Blockette053
from obspy.xseed.blockette.blockette054 import Blockette054
from obspy.xseed.parser import Parser
from obspy.xseed.utils import compareSEED, SEEDParserException
import os
import unittest
import warnings
import gzip


class ParserTestCase(unittest.TestCase):
    """
    """
    def setUp(self):
        # directory where the test files are located
        self.path = os.path.join(os.path.dirname(__file__), 'data')
        self.BW_SEED_files = [os.path.join(self.path, file) for file in
                ['dataless.seed.BW_FURT', 'dataless.seed.BW_MANZ',
                 'dataless.seed.BW_ROTZ', 'dataless.seed.BW_ZUGS']]

    def test_bug165(self):
        """
        Test cases related to #165:
         - number of poles or zeros can be 0
         - an unsupported response information somewhere in the metadata should
           not automatically raise an Error, if the desired information can
           still be retrieved

        This test also tests if a warning is raised if no startime is given.
        """
        parser = Parser()
        file = os.path.join(self.path, "bug165.dataless")
        t = UTCDateTime("2010-01-01T00:00:00")
        if hasattr(warnings, 'catch_warnings'):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                # Trigger a warning.
                parser.read(file)
                self.assertEqual(len(w), 1)
                self.assertTrue(issubclass(w[-1].category, UserWarning))
                self.assertTrue('date' and 'required' in \
                                        w[-1].message.message.lower())
        else:
            # Just raise the warning using Python 2.5.
            parser.read(file)
        paz = parser.getPAZ("NZ.DCZ.20.HNZ", t)
        result = {'digitizer_gain': 419430.0, 'gain': 24595700000000.0,
                  'poles': [(-981 + 1009j), (-981 - 1009j), (-3290 + 1263j),
                            (-3290 - 1263j)],
                  'seismometer_gain': 1.01885, 'sensitivity': 427336.0,
                  'zeros': []}
        self.assertEqual(paz, result)
        self.assertRaises(SEEDParserException, parser.getPAZ,
                          "NZ.DCZ.10.HHZ", t)

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
        sp = str(p).split(os.linesep)
        self.assertEquals(sp, ["BW.MANZ..EHZ | 2005-12-06T00:00:00.000000Z - ",
                               "BW.MANZ..EHN | 2005-12-06T00:00:00.000000Z - ",
                               "BW.MANZ..EHE | 2005-12-06T00:00:00.000000Z -"])

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
        b010 = "0100042 2.4082008,001~2038,001~2009,001~~~"
        blockette = Blockette010(strict=True, compact=True)
        blockette.parseSEED(b010)
        self.assertEquals(b010, blockette.getSEED())
        # create a valid blockette 054
        b054 = "0540240A0400300300000009" + ("+1.58748E-03" * 18)
        blockette = Blockette054(strict=True, compact=True)
        blockette.parseSEED(b054)
        self.assertEquals(b054, blockette.getSEED())
        # combine data
        data = "000001V " + b010 + (' ' * 206)
        data += "000002S " + b054 + (' ' * 8)
        data += "000003S*" + b054 + (' ' * 8)
        # read records
        parser = Parser(strict=True)
        parser.read(data)

    def test_multipleContinuedStationControlHeader(self):
        """
        """
        # create a valid blockette 010 with record length 256
        b010 = "0100042 2.4082008,001~2038,001~2009,001~~~"
        blockette = Blockette010(strict=True, compact=True)
        blockette.parseSEED(b010)
        self.assertEquals(b010, blockette.getSEED())
        # create a valid blockette 054
        b054 = "0540960A0400300300000039"
        nr = ""
        for i in range(0, 78):
            nr = nr + "+1.000%02dE-03" % i  # 960 chars
        blockette = Blockette054(strict=True, compact=True)
        blockette.parseSEED(b054 + nr)
        self.assertEquals(b054 + nr, blockette.getSEED())
        # create a blockette 051
        b051 = '05100271999,123~~0001000000'
        blockette = Blockette051(strict=False)
        # Only suppress warnings starting with Python 2.6. This is necessary
        # because there is no suitable context manager for Python 2.5 that
        # can suppress warnings.
        if hasattr(warnings, 'catch_warnings'):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                blockette.parseSEED(b051)
        else:
            # Just raise the warning using Python 2.5.
            blockette.parseSEED(b051)
        # combine data (each line equals 256 chars)
        data = "000001V " + b010 + (' ' * 206)
        data += "000002S " + b054 + nr[0:224]  # 256-8-24 = 224
        data += "000003S*" + nr[224:472]  # 256-8 = 248
        data += "000004S*" + nr[472:720]
        data += "000005S*" + nr[720:] + b051 + ' ' * 5  # 5 spaces left
        self.assertEqual(len(data), 256 * 5)
        data += "000006S " + b054 + nr[0:224]  # 256-8-24 = 224
        data += "000007S*" + nr[224:472]  # 256-8 = 248
        data += "000008S*" + nr[472:720]
        data += "000009S*" + nr[720:] + ' ' * 32  # 32 spaces left
        self.assertEqual(len(data), 256 * 9)
        # read records
        parser = Parser(strict=False)
        if hasattr(warnings, 'catch_warnings'):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                parser.read(data)
        else:
            # Just raise the warning using Python 2.5.
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
        # Loop over all files.
        for file in self.BW_SEED_files:
            f = open(file, 'r')
            # Original SEED file.
            original_seed = f.read()
            f.seek(0)
            # Parse and write the data.
            parser = Parser(f)
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
        # Loop over all files and versions.
        for version in ['1.0', '1.1']:
            # Path to XML schema file.
            xsd_path = os.path.join(self.path, 'xml-seed-%s.xsd' % version)
            # Prepare validator.
            f = open(xsd_path, 'r')
            xmlschema_doc = etree.parse(f)
            f.close()
            xmlschema = etree.XMLSchema(xmlschema_doc)
            for file in self.BW_SEED_files:
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

    def test_getPAZ(self):
        """
        Test extracting poles and zeros information
        """
        filename = os.path.join(self.path, 'arclink_full.seed')
        sp = Parser(filename)
        paz = sp.getPAZ('BHE')
        self.assertEqual(paz['gain'], +6.00770e+07)
        self.assertEqual(paz['zeros'], [0j, 0j])
        self.assertEqual(paz['poles'], [(-3.70040e-02 + 3.70160e-02j),
            (-3.70040e-02 - 3.70160e-02j), (-2.51330e+02 + 0.00000e+00j),
            (-1.31040e+02 - 4.67290e+02j), (-1.31040e+02 + 4.67290e+02j)])
        self.assertEqual(paz['sensitivity'], +7.86576e+08)
        self.assertEqual(paz['seismometer_gain'], +1.50000E+03)
        # Raise exception for undefined channels
        self.assertRaises(SEEDParserException, sp.getPAZ, 'EHE')
        #
        # Do the same for another dataless file
        #
        filename = os.path.join(self.path, 'dataless.seed.BW_FURT')
        sp = Parser(filename)
        paz = sp.getPAZ('EHE')
        self.assertEqual(paz['gain'], +1.00000e+00)
        self.assertEqual(paz['zeros'], [0j, 0j, 0j])
        self.assertEqual(paz['poles'], [(-4.44400e+00 + 4.44400e+00j),
                                        (-4.44400e+00 - 4.44400e+00j),
                                        (-1.08300e+00 + 0.00000e+00j)])
        self.assertEqual(paz['sensitivity'], +6.71140E+08)
        self.assertEqual(paz['seismometer_gain'], 4.00000E+02)
        # Raise exception for undefined channels
        self.assertRaises(SEEDParserException, sp.getPAZ, 'BHE')
        # Raise exception if not a Laplacian transfer function ('A').
        # Modify transfer_fuction_type on the fly
        for blk in sp.blockettes[53]:
            blk.transfer_function_types = 'X'
        self.assertRaises(SEEDParserException, sp.getPAZ, 'EHE')
        #
        # And the same for yet another dataless file
        #
        filename = os.path.join(self.path, 'nied.dataless.gz')
        f = StringIO(gzip.open(filename).read())
        sp = Parser(f)
        gain = [+3.94857E+03, +4.87393E+04, +3.94857E+03]
        zeros = [[+0.00000E+00 + 0.00000E+00j, +0.00000E+00 + 0.00000E+00j],
                 [+0.00000E+00 + 0.00000E+00j, +0.00000E+00 + 0.00000E+00j,
                  -6.32511E+02 + 0.00000E+00j],
                 [+0.00000E+00 + 0.00000E+00j, +0.00000E+00 + 0.00000E+00j]]
        poles = [[-1.23413E-02 + 1.23413E-02j, -1.23413E-02 - 1.23413E-02j,
                  -3.91757E+01 + 4.91234E+01j, -3.91757E+01 - 4.91234E+01j],
                 [-3.58123E-02 - 4.44766E-02j, -3.58123E-02 + 4.44766E-02j,
                  -5.13245E+02 + 0.00000E+00j, -6.14791E+04 + 0.00000E+00j],
                 [-1.23413E-02 + 1.23413E-02j, -1.23413E-02 - 1.23413E-02j,
                  -3.91757E+01 + 4.91234E+01j, -3.91757E+01 - 4.91234E+01j]]
        sensitivity = [+4.92360E+08, +2.20419E+06, +9.84720E+08]
        seismometer_gain = [+2.29145E+03, +1.02583E+01, +2.29145E+03]
        for i, channel in enumerate(['BHZ', 'BLZ', 'LHZ']):
            paz = sp.getPAZ(channel)
            self.assertEqual(paz['gain'], gain[i])
            self.assertEqual(paz['zeros'], zeros[i])
            self.assertEqual(paz['poles'], poles[i])
            self.assertEqual(paz['sensitivity'], sensitivity[i])
            self.assertEqual(paz['seismometer_gain'], seismometer_gain[i])
        sp = Parser(os.path.join(self.path, 'dataless.seed.BW_RJOB'))
        paz = sp.getPAZ("BW.RJOB..EHZ", UTCDateTime("2007-01-01"))
        result = {'gain': 1.0,
                  'poles': [(-4.444 + 4.444j), (-4.444 - 4.444j),
                            (-1.083 + 0j)],
                  'seismometer_gain': 400.0,
                  'sensitivity': 671140000.0,
                  'zeros': [0j, 0j, 0j],
                  'digitizer_gain': 1677850.0}
        self.assertEqual(paz, result)
        paz = sp.getPAZ("BW.RJOB..EHZ", UTCDateTime("2010-01-01"))
        result = {'gain': 60077000.0,
                  'poles': [(-0.037004000000000002 + 0.037016j),
                            (-0.037004000000000002 - 0.037016j),
                            (-251.33000000000001 + 0j),
                            (-131.03999999999999 - 467.29000000000002j),
                            (-131.03999999999999 + 467.29000000000002j)],
                  'seismometer_gain': 1500.0,
                  'sensitivity': 2516800000.0,
                  'zeros': [0j, 0j],
                  'digitizer_gain':  1677850.0}
        self.assertEqual(sorted(paz.items()), sorted(result.items()))

    def test_getPAZFromXSEED(self):
        """
        Get PAZ from XSEED file, testcase for #146
        """
        filename = os.path.join(self.path, 'dataless.seed.BW_FURT')
        sp1 = Parser(filename)
        sp2 = Parser(sp1.getXSEED())
        paz = sp2.getPAZ('EHE')
        result = {'gain': 1.00000e+00,
                  'zeros': [0j, 0j, 0j],
                  'poles': [(-4.44400e+00 + 4.44400e+00j),
                            (-4.44400e+00 - 4.44400e+00j),
                            (-1.08300e+00 + 0.00000e+00j)],
                  'sensitivity': 6.71140E+08,
                  'seismometer_gain': 4.00000E+02,
                  'digitizer_gain': 1677850.0}
        self.assertEqual(sorted(paz.items()), sorted(result.items()))

    def test_getCoordinates(self):
        """
        Test extracting coordinates for seed and xseed (including #146)
        """
        # seed
        sp = Parser(os.path.join(self.path, 'dataless.seed.BW_RJOB'))
        result = {'elevation': 860.0, 'latitude': 47.737166999999999,
                  'longitude': 12.795714}
        paz = sp.getCoordinates("BW.RJOB..EHZ", UTCDateTime("2007-01-01"))
        self.assertEqual(sorted(paz.items()), sorted(result.items()))
        paz = sp.getCoordinates("BW.RJOB..EHZ", UTCDateTime("2010-01-01"))
        self.assertEqual(sorted(paz.items()), sorted(result.items()))
        # xseed
        sp2 = Parser(sp.getXSEED())
        paz = sp2.getCoordinates("BW.RJOB..EHZ", UTCDateTime("2007-01-01"))
        self.assertEqual(sorted(paz.items()), sorted(result.items()))
        paz = sp2.getCoordinates("BW.RJOB..EHZ", UTCDateTime("2010-01-01"))
        self.assertEqual(sorted(paz.items()), sorted(result.items()))

    def test_createRESPFromXSEED(self):
        """
        Tests RESP file creation from XML-SEED.
        """
        ### example 1
        # parse Dataless SEED
        filename = os.path.join(self.path, 'dataless.seed.BW_FURT')
        sp1 = Parser(filename)
        # write XML-SEED
        tempfile = NamedTemporaryFile().name
        sp1.writeXSEED(tempfile)
        # parse XML-SEED
        sp2 = Parser(tempfile)
        # create RESP files
        _resp_list = sp2.getRESP()
        os.remove(tempfile)
        ### example 2
        # parse Dataless SEED
        filename = os.path.join(self.path, 'arclink_full.seed')
        sp1 = Parser(filename)
        # write XML-SEED
        tempfile = NamedTemporaryFile().name
        sp1.writeXSEED(tempfile)
        # parse XML-SEED
        sp2 = Parser(tempfile)
        # create RESP files
        _resp_list = sp2.getRESP()
        os.remove(tempfile)

    def test_compareBlockettes(self):
        """
        Tests the comparison of two blockettes.
        """
        p = Parser()
        b010_1 = "0100042 2.4082008,001~2038,001~2009,001~~~"
        blockette1 = Blockette010(strict=True, compact=True,
                                  xseed_version='1.0')
        blockette1.parseSEED(b010_1)
        blockette2 = Blockette010()
        blockette2.parseSEED(b010_1)
        b010_3 = "0100042 2.4082009,001~2038,001~2009,001~~~"
        blockette3 = Blockette010(strict=True, compact=True)
        blockette3.parseSEED(b010_3)
        blockette4 = Blockette010(xseed_version='1.0')
        blockette4.parseSEED(b010_3)
        self.assertTrue(p._compareBlockettes(blockette1, blockette2))
        self.assertFalse(p._compareBlockettes(blockette1, blockette3))
        self.assertFalse(p._compareBlockettes(blockette2, blockette3))
        self.assertTrue(p._compareBlockettes(blockette3, blockette4))

    def test_missingRequiredDateTimes(self):
        """
        A warning should be raised if a blockette misses a required date.
        """
        # blockette 10 - missing start time
        b010 = "0100034 2.408~2038,001~2009,001~~~"
        # strict raises an exception
        blockette = Blockette010(strict=True)
        self.assertRaises(SEEDParserException, blockette.parseSEED, b010)
        # If strict is false, a warning is raised. This is tested in
        # test_bug165 due to some issues with the warning being raised only
        # once.
        blockette = Blockette010()
        blockette.parseSEED(b010)
        self.assertEquals(b010, blockette.getSEED())

        # blockette 10 - missing volume time
        b010 = "0100034 2.4082008,001~2038,001~~~~"
        # strict raises an exception
        blockette = Blockette010(strict=True)
        self.assertRaises(SEEDParserException, blockette.parseSEED, b010)
        # non-strict warns
        blockette = Blockette010()
        # The warning cannot be tested due to being issued only once.
        # A similar case is tested in test_bug165.
        blockette.parseSEED(b010)
        self.assertEquals(b010, blockette.getSEED())

    def test_issue298a(self):
        """
        Test case for issue #298.
        """
        file = os.path.join(self.path, "AI.ESPZ._.BHE.dataless")
        parser = Parser(file)
        parser.getRESP()

    def test_issue298b(self):
        """
        Second test case for issue #298.
        """
        file = os.path.join(self.path, "AI.ESPZ._.BH_.dataless")
        parser = Parser(file)
        parser.getRESP()


def suite():
    return unittest.makeSuite(ParserTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
