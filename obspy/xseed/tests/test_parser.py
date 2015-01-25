# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA @UnusedWildImport

import gzip
import io
import os
import unittest
import warnings

import numpy as np
from lxml import etree

from obspy import read, UTCDateTime
from obspy.core.util import NamedTemporaryFile
from obspy.xseed.blockette.blockette010 import Blockette010
from obspy.xseed.blockette.blockette051 import Blockette051
from obspy.xseed.blockette.blockette053 import Blockette053
from obspy.xseed.blockette.blockette054 import Blockette054
from obspy.xseed.parser import Parser
from obspy.xseed.utils import compareSEED, SEEDParserException


class ParserTestCase(unittest.TestCase):
    """
    Parser test suite.
    """
    def setUp(self):
        # directory where the test files are located
        self.path = os.path.join(os.path.dirname(__file__), 'data')
        self.BW_SEED_files = [
            os.path.join(self.path, file) for file in
            ['dataless.seed.BW_FURT', 'dataless.seed.BW_MANZ',
             'dataless.seed.BW_ROTZ', 'dataless.seed.BW_ZUGS']]

    def test_issue165(self):
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
        # raises UserWarning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            # Trigger a warning.
            parser.read(file)
            self.assertEqual(len(w), 1)
            self.assertTrue(issubclass(w[-1].category, UserWarning))
            self.assertTrue('date' and 'required' in
                            str(w[-1].message).lower())
            # Triggers a warning.
            paz = parser.getPAZ("NZ.DCZ.20.HNZ", t)
            result = {'digitizer_gain': 419430.0, 'gain': 24595700000000.0,
                      'poles': [(-981 + 1009j), (-981 - 1009j),
                                (-3290 + 1263j), (-3290 - 1263j)],
                      'seismometer_gain': 1.01885, 'sensitivity': 427336.0,
                      'zeros': []}
            self.assertEqual(paz, result)

    def test_invalidStartHeader(self):
        """
        A SEED Volume must start with a Volume Index Control Header.
        """
        data = b"000001S 0510019~~0001000000"
        sp = Parser(strict=True)
        self.assertRaises(SEEDParserException, sp.read, data)

    def test_invalidStartBlockette(self):
        """
        A SEED Volume must start with Blockette 010.
        """
        data = b"000001V 0510019~~0001000000"
        sp = Parser(strict=True)
        self.assertRaises(SEEDParserException, sp.read, data)

    def test_string(self):
        """
        Tests string representation of L{obspy.xseed.Parser} object.
        """
        filename = os.path.join(self.path, 'dataless.seed.BW_MANZ')
        p = Parser(filename)
        sp = str(p).splitlines()
        sp = [_i.strip() for _i in sp]
        self.assertEqual(sp, [
            "Networks:",
            "BW (BayernNetz)",
            "Stations:",
            "BW.MANZ (Manzenberg,Bavaria, BW-Net)",
            "Channels:",
            ("BW.MANZ..EHE | 200.00 Hz | Streckeisen STS-2/N seismometer | "
                "2005-12-06 -  | Lat: 50.0, Lng: 12.1"),
            ("BW.MANZ..EHN | 200.00 Hz | Streckeisen STS-2/N seismometer | "
                "2005-12-06 -  | Lat: 50.0, Lng: 12.1"),
            ("BW.MANZ..EHZ | 200.00 Hz | Streckeisen STS-2/N seismometer | "
                "2005-12-06 -  | Lat: 50.0, Lng: 12.1")])

    def test_get_inventory(self):
        """
        Tests the parser's getInventory() method.
        """
        filename = os.path.join(self.path, 'dataless.seed.BW_FURT')
        p = Parser(filename)
        self.assertEqual(
            p.getInventory(),
            {'networks': [{'network_code': 'BW',
             'network_name': 'BayernNetz'}],
             'stations': [{'station_name': 'Furstenfeldbruck, Bavaria, BW-Net',
                          'station_id': 'BW.FURT'}],
             'channels': [
                 {'channel_id': 'BW.FURT..EHZ',
                  'start_date': UTCDateTime(2001, 1, 1, 0, 0),
                  'instrument': 'Lennartz LE-3D/1 seismometer',
                  'elevation_in_m': 565.0,
                  'latitude': 48.162899,
                  'local_depth_in_m': 0.0,
                  'longitude': 11.2752,
                  'end_date': '', 'sampling_rate': 200.0},
                 {'channel_id': 'BW.FURT..EHN',
                  'start_date': UTCDateTime(2001, 1, 1, 0, 0),
                  'instrument': 'Lennartz LE-3D/1 seismometer',
                  'elevation_in_m': 565.0,
                  'latitude': 48.162899,
                  'local_depth_in_m': 0.0,
                  'longitude': 11.2752,
                  'end_date': '',
                  'sampling_rate': 200.0},
                 {'channel_id': 'BW.FURT..EHE',
                  'start_date': UTCDateTime(2001, 1, 1, 0, 0),
                  'instrument': 'Lennartz LE-3D/1 seismometer',
                  'elevation_in_m': 565.0,
                  'latitude': 48.162899,
                  'local_depth_in_m': 0.0,
                  'longitude': 11.2752,
                  'end_date': '',
                  'sampling_rate': 200.0}]})

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
        b010 = b"0100042 2.4082008,001~2038,001~2009,001~~~"
        blockette = Blockette010(strict=True, compact=True)
        blockette.parseSEED(b010)
        self.assertEqual(b010, blockette.getSEED())
        # create a valid blockette 054
        b054 = b"0540240A0400300300000009" + (b"+1.58748E-03" * 18)
        blockette = Blockette054(strict=True, compact=True)
        blockette.parseSEED(b054)
        self.assertEqual(b054, blockette.getSEED())
        # combine data
        data = b"000001V " + b010 + (b' ' * 206)
        data += b"000002S " + b054 + (b' ' * 8)
        data += b"000003S*" + b054 + (b' ' * 8)
        # read records
        parser = Parser(strict=True)
        parser.read(data)

    def test_multipleContinuedStationControlHeader(self):
        """
        """
        # create a valid blockette 010 with record length 256
        b010 = b"0100042 2.4082008,001~2038,001~2009,001~~~"
        blockette = Blockette010(strict=True, compact=True)
        blockette.parseSEED(b010)
        self.assertEqual(b010, blockette.getSEED())
        # create a valid blockette 054
        b054 = b"0540960A0400300300000039"
        nr = b""
        for i in range(0, 78):
            # 960 chars
            nr = nr + ("+1.000%02dE-03" % i).encode('ascii', 'strict')
        blockette = Blockette054(strict=True, compact=True)
        blockette.parseSEED(b054 + nr)
        self.assertEqual(b054 + nr, blockette.getSEED())
        # create a blockette 051
        b051 = b'05100271999,123~~0001000000'
        blockette = Blockette051(strict=False)
        # ignore user warning
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("ignore")
            blockette.parseSEED(b051)
        # combine data (each line equals 256 chars)
        data = b"000001V " + b010 + (b' ' * 206)
        data += b"000002S " + b054 + nr[0:224]  # 256-8-24 = 224
        data += b"000003S*" + nr[224:472]  # 256-8 = 248
        data += b"000004S*" + nr[472:720]
        data += b"000005S*" + nr[720:] + b051 + b' ' * 5  # 5 spaces left
        self.assertEqual(len(data), 256 * 5)
        data += b"000006S " + b054 + nr[0:224]  # 256-8-24 = 224
        data += b"000007S*" + nr[224:472]  # 256-8 = 248
        data += b"000008S*" + nr[472:720]
        data += b"000009S*" + nr[720:] + b' ' * 32  # 32 spaces left
        self.assertEqual(len(data), 256 * 9)
        # read records
        parser = Parser(strict=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            parser.read(data)
        # check results
        self.assertEqual(sorted(parser.blockettes.keys()), [10, 51, 54])
        self.assertEqual(len(parser.blockettes[10]), 1)
        self.assertEqual(len(parser.blockettes[51]), 1)
        self.assertEqual(len(parser.blockettes[54]), 2)

    def test_blocketteLongerThanRecordLength(self):
        """
        If a blockette is longer than the record length it should result in
        more than one record.
        """
        parser = Parser(strict=True)
        # Set record length to 100.
        parser.record_length = 100
        # Use a blockette 53 string.
        SEED_string = b'0530382A01002003+6.00770E+07+2.00000E-02002+0.00000E' \
            b'+00+0.00000E+00+0.00000E+00+0.00000E+00+0.00000E+00+0.00000E+0' \
            b'0+0.00000E+00+0.00000E+00005-3.70040E-02-3.70160E-02+0.00000E+' \
            b'00+0.00000E+00-3.70040E-02+3.70160E-02+0.00000E+00+0.00000E+00' \
            b'-2.51330E+02+0.00000E+00+0.00000E+00+0.00000E+00-1.31040E+02-4' \
            b'.67290E+02+0.00000E+00+0.00000E+00-1.31040E+02+4.67290E+02+0.0' \
            b'0000E+00+0.00000E+00'
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
        new_string = b''
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
        for file in (self.BW_SEED_files[-1],):
            f = open(file, 'rb')
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
            f = open(xsd_path, 'rb')
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
                doc = etree.parse(io.BytesIO(xseed_string))
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
        self.assertEqual(
            [_i.id for _i in sp.abbreviations],
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
        self.assertEqual(
            paz['poles'],
            [(-3.70040e-02 + 3.70160e-02j),
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
        # Raise UserWarning if not a Laplacian transfer function ('A').
        # Modify transfer_fuction_type on the fly
        for blk in sp.blockettes[53]:
            blk.transfer_function_types = 'X'
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("error", UserWarning)
            self.assertRaises(UserWarning, sp.getPAZ, 'EHE')
        #
        # And the same for yet another dataless file
        #
        filename = os.path.join(self.path, 'nied.dataless.gz')
        f = io.BytesIO(gzip.open(filename).read())
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
                  'digitizer_gain': 1677850.0}
        self.assertEqual(sorted(paz.items()), sorted(result.items()))
        # last test again, check arg name changed in [3722]
        result = {'gain': 60077000.0,
                  'poles': [(-0.037004000000000002 + 0.037016j),
                            (-0.037004000000000002 - 0.037016j),
                            (-251.33000000000001 + 0j),
                            (-131.03999999999999 - 467.29000000000002j),
                            (-131.03999999999999 + 467.29000000000002j)],
                  'seismometer_gain': 1500.0,
                  'sensitivity': 2516800000.0,
                  'zeros': [0j, 0j],
                  'digitizer_gain': 1677850.0}
        paz = sp.getPAZ(seed_id="BW.RJOB..EHZ",
                        datetime=UTCDateTime("2010-01-01"))
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
        Test extracting coordinates for SEED and XSEED (including #146)
        """
        # SEED
        sp = Parser(os.path.join(self.path, 'dataless.seed.BW_RJOB'))
        result = {'elevation': 860.0, 'latitude': 47.737166999999999,
                  'longitude': 12.795714, 'local_depth': 0}
        paz = sp.getCoordinates("BW.RJOB..EHZ", UTCDateTime("2007-01-01"))
        self.assertEqual(sorted(paz.items()), sorted(result.items()))
        paz = sp.getCoordinates("BW.RJOB..EHZ", UTCDateTime("2010-01-01"))
        self.assertEqual(sorted(paz.items()), sorted(result.items()))
        # XSEED
        sp2 = Parser(sp.getXSEED())
        paz = sp2.getCoordinates("BW.RJOB..EHZ", UTCDateTime("2007-01-01"))
        self.assertEqual(sorted(paz.items()), sorted(result.items()))
        paz = sp2.getCoordinates("BW.RJOB..EHZ", UTCDateTime("2010-01-01"))
        self.assertEqual(sorted(paz.items()), sorted(result.items()))

    def test_selectDoesNotChangeTheParserFormat(self):
        """
        Test that using the _select() method of the Parser object does
        not change the _format attribute.
        """
        p = Parser(os.path.join(self.path, "dataless.seed.BW_FURT.xml"))
        self.assertEqual(p._format, "XSEED")
        p._select(p.getInventory()["channels"][0]["channel_id"])
        self.assertEqual(p._format, "XSEED")

    def test_createRESPFromXSEED(self):
        """
        Tests RESP file creation from XML-SEED.
        """
        # 1
        # parse Dataless SEED
        filename = os.path.join(self.path, 'dataless.seed.BW_FURT')
        sp1 = Parser(filename)
        # write XML-SEED
        with NamedTemporaryFile() as fh:
            tempfile = fh.name
            sp1.writeXSEED(tempfile)
            # parse XML-SEED
            sp2 = Parser(tempfile)
            # create RESP files
            sp2.getRESP()
        # 2
        # parse Dataless SEED
        filename = os.path.join(self.path, 'arclink_full.seed')
        sp1 = Parser(filename)
        # write XML-SEED
        with NamedTemporaryFile() as fh:
            tempfile = fh.name
            sp1.writeXSEED(tempfile)
            # parse XML-SEED
            sp2 = Parser(tempfile)
            # create RESP files
            sp2.getRESP()

    def test_compareBlockettes(self):
        """
        Tests the comparison of two blockettes.
        """
        p = Parser()
        b010_1 = b"0100042 2.4082008,001~2038,001~2009,001~~~"
        blockette1 = Blockette010(strict=True, compact=True,
                                  xseed_version='1.0')
        blockette1.parseSEED(b010_1)
        blockette2 = Blockette010()
        blockette2.parseSEED(b010_1)
        b010_3 = b"0100042 2.4082009,001~2038,001~2009,001~~~"
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
        b010 = b"0100034 2.408~2038,001~2009,001~~~"
        # strict raises an exception
        blockette = Blockette010(strict=True)
        self.assertRaises(SEEDParserException, blockette.parseSEED, b010)
        # If strict is false, a warning is raised. This is tested in
        # test_bug165.
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("ignore", UserWarning)
            blockette = Blockette010()
            blockette.parseSEED(b010)
            self.assertEqual(b010, blockette.getSEED())
        # blockette 10 - missing volume time
        b010 = b"0100034 2.4082008,001~2038,001~~~~"
        # strict raises an exception
        blockette = Blockette010(strict=True)
        self.assertRaises(SEEDParserException, blockette.parseSEED, b010)
        # non-strict
        blockette = Blockette010()
        # The warning cannot be tested due to being issued only once.
        # A similar case is tested in test_bug165.
        blockette.parseSEED(b010)
        self.assertEqual(b010, blockette.getSEED())

    def test_issue298a(self):
        """
        Test case for issue #298: blockette size exceeds 9999 bytes.
        """
        file = os.path.join(self.path, "AI.ESPZ._.BHE.dataless")
        parser = Parser(file)
        parser.getRESP()

    def test_issue298b(self):
        """
        Second test case for issue #298: blockette size exceeds 9999 bytes.
        """
        file = os.path.join(self.path, "AI.ESPZ._.BH_.dataless")
        parser = Parser(file)
        parser.getRESP()

    def test_issue319(self):
        """
        Test case for issue #319: multiple abbreviation dictionaries.
        """
        filename = os.path.join(self.path, 'BN.LPW._.BHE.dataless')
        # raises a UserWarning: More than one Abbreviation Dictionary Control
        # Headers found!
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("error", UserWarning)
            self.assertRaises(UserWarning, Parser, filename)
            warnings.simplefilter("ignore", UserWarning)
            parser = Parser(filename)
            self.assertEqual(parser.version, 2.3)

    def test_issue157(self):
        """
        Test case for issue #157: re-using parser object.
        """
        expected = {'latitude': 48.162899, 'elevation': 565.0,
                    'longitude': 11.2752, 'local_depth': 0.0}
        filename1 = os.path.join(self.path, 'dataless.seed.BW_FURT')
        filename2 = os.path.join(self.path, 'dataless.seed.BW_MANZ')
        t = UTCDateTime("2010-07-01")
        parser = Parser()
        parser.read(filename2)
        # parsing a second time will raise a UserWarning: Clearing parser
        # before every subsequent read()
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("error", UserWarning)
            self.assertRaises(UserWarning, parser.read, filename1)
            warnings.simplefilter("ignore", UserWarning)
            parser.read(filename1)
            result = parser.getCoordinates("BW.FURT..EHZ", t)
            self.assertEqual(expected, result)

    def test_issue358(self):
        """
        Test case for issue #358.
        """
        filename = os.path.join(self.path, 'CL.AIO.dataless')
        parser = Parser()
        parser.read(filename)
        dt = UTCDateTime('2012-01-01')
        parser.getPAZ('CL.AIO.00.EHZ', dt)

    def test_issue361(self):
        """
        Test case for issue #361.
        """
        filename = os.path.join(self.path, 'G.SPB.dataless')
        parser = Parser()
        parser.read(filename)
        # 1 - G.SPB..BHZ - no Laplace transform - works
        parser.getPAZ('G.SPB..BHZ')
        # 2 - G.SPB.00.BHZ - raises exception because of multiple results
        self.assertRaises(SEEDParserException, parser.getPAZ, 'G.SPB.00.BHZ')
        # 3 - G.SPB.00.BHZ with datetime - no Laplace transform - works
        dt = UTCDateTime('2007-01-01')
        parser.getPAZ('G.SPB.00.BHZ', dt)
        # 4 - G.SPB.00.BHZ with later datetime works
        dt = UTCDateTime('2012-01-01')
        parser.getPAZ('G.SPB.00.BHZ', dt)

    def test_splitStationsDataless2XSEED(self):
        """
        Test case for writing dataless to XSEED with multiple entries.
        """
        filename = os.path.join(self.path, 'dataless.seed.BW_DHFO')
        parser = Parser()
        parser.read(filename)
        with NamedTemporaryFile() as fh:
            tempfile = fh.name
            # this will create two files due to two entries in dataless
            parser.writeXSEED(tempfile, split_stations=True)
            # the second file name is appended with the timestamp of start
            # period
            os.remove(tempfile + '.1301529600.0.xml')

    def test_rotationToZNE(self):
        """
        Weak test for rotation of arbitrarily rotated components to ZNE.
        """
        st = read(os.path.join(self.path,
                               "II_COCO_three_channel_borehole.mseed"))
        # Read the SEED file and rotate the Traces with the information stored
        # in the SEED file.
        p = Parser(os.path.join(self.path, "dataless.seed.II_COCO"))
        st_r = p.rotateToZNE(st)

        # Still three channels left.
        self.assertEqual(len(st_r), 3)

        # Extract the components for easier assertions. This also asserts that
        # the channel renaming worked.
        tr_z = st.select(channel="BHZ")[0]
        tr_1 = st.select(channel="BH1")[0]
        tr_2 = st.select(channel="BH2")[0]
        tr_r_z = st_r.select(channel="BHZ")[0]
        tr_r_n = st_r.select(channel="BHN")[0]
        tr_r_e = st_r.select(channel="BHE")[0]

        # Convert all components to float for easier assertions.
        tr_z.data = np.require(tr_z.data, dtype=np.float64)
        tr_1.data = np.require(tr_1.data, dtype=np.float64)
        tr_2.data = np.require(tr_2.data, dtype=np.float64)

        # The total energy should not be different.
        energy_before = np.sum((tr_z.data ** 2) + (tr_1.data ** 2) +
                               (tr_2.data ** 2))
        energy_after = np.sum((tr_r_z.data ** 2) + (tr_r_n.data ** 2) +
                              (tr_r_e.data ** 2))
        self.assertTrue(np.allclose(energy_before, energy_after))

        # The vertical channel should not have changed at all.
        np.testing.assert_array_equal(tr_z.data, tr_r_z.data)
        # The other two are only rotated by 2 degree so should also not have
        # changed much but at least a little bit. And the components should be
        # renamed.
        self.assertTrue(np.allclose(tr_1, tr_r_n, rtol=10E-3))
        # The east channel carries very little energy for this particular
        # example. Thus it changes quite a lot even for this very subtle
        # rotation. The energy comparison should still ensure a sensible
        # result.
        self.assertTrue(np.allclose(tr_2, tr_r_e, atol=tr_r_e.max() / 4.0))


def suite():
    return unittest.makeSuite(ParserTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
