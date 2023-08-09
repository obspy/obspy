# -*- coding: utf-8 -*-
import gzip
import io
import os
import unittest
import warnings

import numpy as np
from lxml import etree

import obspy
from obspy import UTCDateTime, read
from obspy.core.util import NamedTemporaryFile
from obspy.io.xseed.blockette.blockette010 import Blockette010
from obspy.io.xseed.blockette.blockette051 import Blockette051
from obspy.io.xseed.blockette.blockette053 import Blockette053
from obspy.io.xseed.blockette.blockette054 import Blockette054
import obspy.io.xseed.parser
from obspy.io.xseed.parser import Parser
from obspy.io.xseed.utils import SEEDParserException, compare_seed
from obspy.signal.invsim import evalresp_for_frequencies


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
        """
        parser = Parser(strict=True)
        file = os.path.join(self.path, "bug165.dataless")
        t = UTCDateTime("2010-01-01T00:00:00")
        parser.read(file)
        paz = parser.get_paz("NZ.DCZ.20.HNZ", t)
        result = {'digitizer_gain': 419430.0, 'gain': 24595700000000.0,
                  'poles': [(-981 + 1009j), (-981 - 1009j),
                            (-3290 + 1263j), (-3290 - 1263j)],
                  'seismometer_gain': 1.01885, 'sensitivity': 427336.0,
                  'zeros': []}
        self.assertEqual(paz, result)

    def test_invalid_start_header(self):
        """
        A SEED Volume must start with a Volume Index Control Header.
        """
        data = b"000001S 0510019~~0001000000"
        sp = Parser(strict=True)
        self.assertRaises(SEEDParserException, sp.read, data)

    def test_invalid_start_blockette(self):
        """
        A SEED Volume must start with Blockette 010.
        """
        data = b"000001V 0510019~~0001000000"
        sp = Parser(strict=True)
        self.assertRaises(SEEDParserException, sp.read, data)

    def test_newline_between_blockettes(self):
        """
        A very rare case.
        """
        # Handcrafted files.
        filename = os.path.join(self.path,
                                'dataless.seed.newline_between_blockettes')
        p = Parser(filename)
        self.assertEqual(sorted(list(p.blockettes.keys())),
                         [10, 11, 30, 33, 34])

    def test_string(self):
        """
        Tests string representation of L{obspy.io.xseed.Parser} object.
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
        Tests the parser's get_inventory() method.
        """
        filename = os.path.join(self.path, 'dataless.seed.BW_FURT')
        p = Parser(filename)
        self.assertEqual(
            p.get_inventory(),
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

    def test_non_existing_file_name(self):
        """
        Test reading non existing file.
        """
        self.assertRaises(IOError, Parser, "XYZ")

    def test_blockette_starts_after_record(self):
        """
        '... 058003504 1.00000E+00 0.00000E+0000 000006S*0543864 ... '
        ' 0543864' -> results in Blockette 005
        """
        # create a valid blockette 010 with record length 256
        b010 = b"0100042 2.4082008,001~2038,001~2009,001~~~"
        blockette = Blockette010(strict=True, compact=True)
        blockette.parse_seed(b010)
        self.assertEqual(b010, blockette.get_seed())
        # create a valid blockette 054
        b054 = b"0540240A0400300300000009" + (b"+1.58748E-03" * 18)
        blockette = Blockette054(strict=True, compact=True)
        blockette.parse_seed(b054)
        self.assertEqual(b054, blockette.get_seed())
        # combine data
        data = b"000001V " + b010 + (b' ' * 206)
        data += b"000002S " + b054 + (b' ' * 8)
        data += b"000003S*" + b054 + (b' ' * 8)
        # read records
        parser = Parser(strict=True)
        parser.read(data)

    def test_multiple_continued_station_control_header(self):
        """
        """
        # create a valid blockette 010 with record length 256
        b010 = b"0100042 2.4082008,001~2038,001~2009,001~~~"
        blockette = Blockette010(strict=True, compact=True)
        blockette.parse_seed(b010)
        self.assertEqual(b010, blockette.get_seed())
        # create a valid blockette 054
        b054 = b"0540960A0400300300000039"
        nr = b""
        for i in range(0, 78):
            # 960 chars
            nr = nr + ("+1.000%02dE-03" % i).encode('ascii', 'strict')
        blockette = Blockette054(strict=True, compact=True)
        blockette.parse_seed(b054 + nr)
        self.assertEqual(b054 + nr, blockette.get_seed())
        # create a blockette 051
        b051 = b'05100271999,123~~0001000000'
        blockette = Blockette051(strict=False)
        # ignore user warning
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("ignore")
            blockette.parse_seed(b051)
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

    def test_blockette_longer_than_record_length(self):
        """
        If a blockette is longer than the record length it should result in
        more than one record.
        """
        parser = Parser(strict=True)
        # Set record length to 100.
        parser.record_length = 100
        # Use a blockette 53 string.
        seed_string = b'0530382A01002003+6.00770E+07+2.00000E-02002+0.00000E' \
            b'+00+0.00000E+00+0.00000E+00+0.00000E+00+0.00000E+00+0.00000E+0' \
            b'0+0.00000E+00+0.00000E+00005-3.70040E-02-3.70160E-02+0.00000E+' \
            b'00+0.00000E+00-3.70040E-02+3.70160E-02+0.00000E+00+0.00000E+00' \
            b'-2.51330E+02+0.00000E+00+0.00000E+00+0.00000E+00-1.31040E+02-4' \
            b'.67290E+02+0.00000E+00+0.00000E+00-1.31040E+02+4.67290E+02+0.0' \
            b'0000E+00+0.00000E+00'
        blkt_53 = Blockette053()
        blkt_53.parse_seed(seed_string)
        # This just tests an internal SEED method.
        records = parser._create_cut_and_flush_record([blkt_53], 'S')
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
        self.assertEqual(new_string.strip(), seed_string)

    def test_read_and_write_seed(self):
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
            new_seed = parser.get_seed()
            # compare both SEED strings
            compare_seed(original_seed, new_seed)
            del parser
            parser1 = Parser(original_seed)
            parser2 = Parser(new_seed)
            self.assertEqual(parser1.get_seed(), parser2.get_seed())
            del parser1, parser2

    def test_create_read_assert_and_write_xseed(self):
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
                original_seed = parser1.get_seed()
                del parser1
                # Now read the file, parse it, write XSEED, read XSEED and
                # write SEED again. The output should be totally identical.
                parser2 = Parser(original_seed)
                xseed_string = parser2.get_xseed(version=version)
                del parser2
                # Validate XSEED.
                doc = etree.parse(io.BytesIO(xseed_string))
                self.assertTrue(xmlschema.validate(doc))
                del doc
                parser3 = Parser(xseed_string)
                new_seed = parser3.get_seed()
                self.assertEqual(original_seed, new_seed)
                del parser3, original_seed, new_seed

    def test_read_full_seed(self):
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

    def test_get_paz(self):
        """
        Test extracting poles and zeros information
        """
        filename = os.path.join(self.path, 'arclink_full.seed')
        sp = Parser(filename)
        paz = sp.get_paz('BHE')
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
        self.assertRaises(SEEDParserException, sp.get_paz, 'EHE')
        #
        # Do the same for another dataless file
        #
        filename = os.path.join(self.path, 'dataless.seed.BW_FURT')
        sp = Parser(filename)
        paz = sp.get_paz('EHE')
        self.assertEqual(paz['gain'], +1.00000e+00)
        self.assertEqual(paz['zeros'], [0j, 0j, 0j])
        self.assertEqual(paz['poles'], [(-4.44400e+00 + 4.44400e+00j),
                                        (-4.44400e+00 - 4.44400e+00j),
                                        (-1.08300e+00 + 0.00000e+00j)])
        self.assertEqual(paz['sensitivity'], +6.71140E+08)
        self.assertEqual(paz['seismometer_gain'], 4.00000E+02)
        # Raise exception for undefined channels
        self.assertRaises(SEEDParserException, sp.get_paz, 'BHE')
        # Raise UserWarning if not a Laplacian transfer function ('A').
        # Modify transfer_fuction_type on the fly
        for blk in sp.blockettes[53]:
            blk.transfer_function_types = 'X'
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("error", UserWarning)
            self.assertRaises(UserWarning, sp.get_paz, 'EHE')
        #
        # And the same for yet another dataless file
        #
        filename = os.path.join(self.path, 'nied.dataless.gz')
        with gzip.open(filename) as g:
            f = io.BytesIO(g.read())
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
            paz = sp.get_paz(channel)
            self.assertEqual(paz['gain'], gain[i])
            self.assertEqual(paz['zeros'], zeros[i])
            self.assertEqual(paz['poles'], poles[i])
            self.assertEqual(paz['sensitivity'], sensitivity[i])
            self.assertEqual(paz['seismometer_gain'], seismometer_gain[i])
        sp = Parser(os.path.join(self.path, 'dataless.seed.BW_RJOB'))
        paz = sp.get_paz("BW.RJOB..EHZ", UTCDateTime("2007-01-01"))
        result = {'gain': 1.0,
                  'poles': [(-4.444 + 4.444j), (-4.444 - 4.444j),
                            (-1.083 + 0j)],
                  'seismometer_gain': 400.0,
                  'sensitivity': 671140000.0,
                  'zeros': [0j, 0j, 0j],
                  'digitizer_gain': 1677850.0}
        self.assertEqual(paz, result)
        paz = sp.get_paz("BW.RJOB..EHZ", UTCDateTime("2010-01-01"))
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
        # check arg name changed in [3722]
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
        paz = sp.get_paz(seed_id="BW.RJOB..EHZ",
                         datetime=UTCDateTime("2010-01-01"))
        self.assertEqual(sorted(paz.items()), sorted(result.items()))
        # test for multiple blockette 53s using II dataless
        sp = Parser(os.path.join(self.path, 'dataless.seed.II_COCO'))
        paz = sp.get_paz("II.COCO.00.BHZ", UTCDateTime("2013-01-01"))
        result = {'gain': 1057.5083723679224,
                  'poles': [(-0.004799989149937387 + 0j),
                            (-0.07341022385496342 + 0j),
                            (-21.852101684280665 + 23.497667916231002j),
                            (-21.852101684280665 - 23.497667916231002j)],
                  'seismometer_gain': 2164.8,
                  'sensitivity': 3598470000.0,
                  'zeros': [0j, 0j],
                  'digitizer_gain': 1662150.0}
        self.assertEqual(sorted(paz.items()), sorted(result.items()))

    def test_get_paz_from_xseed(self):
        """
        Get PAZ from XSEED file, testcase for #146
        """
        filename = os.path.join(self.path, 'dataless.seed.BW_FURT')
        sp1 = Parser(filename)
        sp2 = Parser(sp1.get_xseed())
        paz = sp2.get_paz('EHE')
        result = {'gain': 1.00000e+00,
                  'zeros': [0j, 0j, 0j],
                  'poles': [(-4.44400e+00 + 4.44400e+00j),
                            (-4.44400e+00 - 4.44400e+00j),
                            (-1.08300e+00 + 0.00000e+00j)],
                  'sensitivity': 6.71140E+08,
                  'seismometer_gain': 4.00000E+02,
                  'digitizer_gain': 1677850.0}
        self.assertEqual(sorted(paz.items()), sorted(result.items()))

    def test_get_coordinates(self):
        """
        Test extracting coordinates for SEED and XSEED (including #146)
        """
        # SEED
        sp = Parser(os.path.join(self.path, 'dataless.seed.BW_RJOB'))
        result = {'elevation': 860.0, 'latitude': 47.737166999999999,
                  'longitude': 12.795714, 'local_depth': 0,
                  'azimuth': 0.0, 'local_depth': 0, 'dip': -90.0}
        paz = sp.get_coordinates("BW.RJOB..EHZ", UTCDateTime("2007-01-01"))
        self.assertEqual(sorted(paz.items()), sorted(result.items()))
        paz = sp.get_coordinates("BW.RJOB..EHZ", UTCDateTime("2010-01-01"))
        self.assertEqual(sorted(paz.items()), sorted(result.items()))
        # XSEED
        sp2 = Parser(sp.get_xseed())
        paz = sp2.get_coordinates("BW.RJOB..EHZ", UTCDateTime("2007-01-01"))
        self.assertEqual(sorted(paz.items()), sorted(result.items()))
        paz = sp2.get_coordinates("BW.RJOB..EHZ", UTCDateTime("2010-01-01"))
        self.assertEqual(sorted(paz.items()), sorted(result.items()))
        # Additional test with non-trivial azimuth
        sp = Parser(os.path.join(self.path, 'dataless.seed.II_COCO'))
        result = {'elevation': 1.0, 'latitude': -12.1901,
                  'longitude': 96.8349, 'local_depth': 1.3,
                  'azimuth': 92.0, 'local_depth': 1.3, 'dip': 0.0}
        paz = sp.get_coordinates("II.COCO.10.BH2", UTCDateTime("2010-11-11"))
        self.assertEqual(sorted(paz.items()), sorted(result.items()))

    def test_select_does_not_change_the_parser_format(self):
        """
        Test that using the _select() method of the Parser object does
        not change the _format attribute.
        """
        p = Parser(os.path.join(self.path, "dataless.seed.BW_FURT.xml"))
        self.assertEqual(p._format, "XSEED")
        p._select(p.get_inventory()["channels"][0]["channel_id"])
        self.assertEqual(p._format, "XSEED")

    def test_create_resp_from_xseed(self):
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
            sp1.write_xseed(tempfile)
            # parse XML-SEED
            sp2 = Parser(tempfile)
            # create RESP files
            sp2.get_resp()
        # 2
        # parse Dataless SEED
        filename = os.path.join(self.path, 'arclink_full.seed')
        sp1 = Parser(filename)
        # write XML-SEED
        with NamedTemporaryFile() as fh:
            tempfile = fh.name
            sp1.write_xseed(tempfile)
            # parse XML-SEED
            sp2 = Parser(tempfile)
            # create RESP files
            sp2.get_resp()

    def test_read_resp(self):
        """
        Tests reading a respfile by calling Parser(filename)
        """
        sts2_resp_file = os.path.join(self.path,
                                      'RESP.XX.NS085..BHZ.STS2_gen3.120.1500')
        p = Parser(sts2_resp_file)
        # Weak but at least tests that something has been read.
        assert set(p.blockettes.keys()) == {34, 50, 52, 53, 54, 57, 58}

        rt130_resp_file = os.path.join(self.path,
                                       'RESP.XX.NR008..HHZ.130.1.100')
        p = Parser(rt130_resp_file)
        # Weak but at least tests that something has been read.
        assert set(p.blockettes.keys()) == {34, 50, 52, 53, 54, 57, 58}

    def test_read_resp_data(self):
        """
        Tests reading a resp string by calling Parser(string)
        """
        sts2_resp_file = os.path.join(self.path,
                                      'RESP.XX.NS085..BHZ.STS2_gen3.120.1500')
        with open(sts2_resp_file, "rt") as fh:
            p = Parser(fh.read())
        # Weak but at least tests that something has been read.
        assert set(p.blockettes.keys()) == {34, 50, 52, 53, 54, 57, 58}

        rt130_resp_file = os.path.join(self.path,
                                       'RESP.XX.NR008..HHZ.130.1.100')
        with open(rt130_resp_file, "rt") as fh:
            p = Parser(fh.read())
        # Weak but at least tests that something has been read.
        assert set(p.blockettes.keys()) == {34, 50, 52, 53, 54, 57, 58}

    def clean_unit_string(self, string):
        """
        Returns a list of cleaned strings
        """
        # Strip out string constants that differ.
        # Unit descriptions, and case
        dirty_fields = ['B054F05', 'B054F06', 'B053F05', 'B053F06']
        ret = list()

        for line in string.split(b'\n'):
            line = line.decode('ascii')
            if line[:7] not in dirty_fields:
                line = line.upper()
            else:
                line = line.split('-')[0].upper()
            ret.append(line)
        return ret

    def test_resp_round_trip(self):
        single_seed = os.path.join(
            self.path,
            '../../../../',
            'core/tests/data/IRIS_single_channel_with_response.seed')
        # Make parser and get resp from SEED
        seed_p = Parser(single_seed)
        resp_from_seed = seed_p.get_resp()[0][1]
        resp_from_seed.seek(0)
        resp_from_seed = resp_from_seed.read()
        seed_list = self.clean_unit_string(resp_from_seed)

        # make parser from resp made above and make a resp from it
        resp_p = Parser(resp_from_seed.decode('ascii'))
        resp_from_resp = resp_p.get_resp()[0][1]
        resp_from_resp.seek(0)
        resp_from_resp = resp_from_resp.read()
        resp_list = self.clean_unit_string(resp_from_resp)

        # compare
        self.assertEqual(seed_list, resp_list)

    def test_compare_blockettes(self):
        """
        Tests the comparison of two blockettes.
        """
        p = Parser()
        b010_1 = b"0100042 2.4082008,001~2038,001~2009,001~~~"
        blockette1 = Blockette010(strict=True, compact=True,
                                  xseed_version='1.0')
        blockette1.parse_seed(b010_1)
        blockette2 = Blockette010()
        blockette2.parse_seed(b010_1)
        b010_3 = b"0100042 2.4082009,001~2038,001~2009,001~~~"
        blockette3 = Blockette010(strict=True, compact=True)
        blockette3.parse_seed(b010_3)
        blockette4 = Blockette010(xseed_version='1.0')
        blockette4.parse_seed(b010_3)
        self.assertTrue(p._compare_blockettes(blockette1, blockette2))
        self.assertFalse(p._compare_blockettes(blockette1, blockette3))
        self.assertFalse(p._compare_blockettes(blockette2, blockette3))
        self.assertTrue(p._compare_blockettes(blockette3, blockette4))

    def test_missing_required_date_times(self):
        """
        A warning should be raised if a blockette misses a required date.
        """
        # blockette 10 - missing start time
        b010 = b"0100034 2.408~2038,001~2009,001~~~"
        # strict raises an exception
        blockette = Blockette010(strict=True)
        self.assertRaises(SEEDParserException, blockette.parse_seed, b010)
        # If strict is false, a warning is raised. This is tested in
        # test_bug165.
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("ignore", UserWarning)
            blockette = Blockette010()
            blockette.parse_seed(b010)
            self.assertEqual(b010, blockette.get_seed())
        # blockette 10 - missing volume time
        b010 = b"0100034 2.4082008,001~2038,001~~~~"
        # strict raises an exception
        blockette = Blockette010(strict=True)
        self.assertRaises(SEEDParserException, blockette.parse_seed, b010)
        # non-strict
        blockette = Blockette010()
        # The warning cannot be tested due to being issued only once, but will
        # be ignored - a similar case is tested in test_bug165.
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("ignore", UserWarning)
            blockette.parse_seed(b010)
        self.assertEqual(b010, blockette.get_seed())

    def test_issue_298a(self):
        """
        Test case for issue #298: blockette size exceeds 9999 bytes.
        """
        file = os.path.join(self.path, "AI.ESPZ._.BHE.dataless")
        parser = Parser(file)
        parser.get_resp()

    def test_issue_298b(self):
        """
        Second test case for issue #298: blockette size exceeds 9999 bytes.
        """
        file = os.path.join(self.path, "AI.ESPZ._.BH_.dataless")
        parser = Parser(file)
        parser.get_resp()

    def test_issue_319(self):
        """
        Test case for issue #319: multiple abbreviation dictionaries.
        """
        # We have to clear the warnings registry here as some other tests
        # also trigger the warning.
        if hasattr(obspy.io.xseed.parser, "__warningregistry__"):
            obspy.io.xseed.parser.__warningregistry__.clear()

        filename = os.path.join(self.path, 'BN.LPW._.BHE.dataless')
        # raises a UserWarning: More than one Abbreviation Dictionary Control
        # Headers found!
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            parser = Parser(filename)
        self.assertEqual(
            w[0].message.args[0],
            "More than one Abbreviation Dictionary Control Headers found!")
        self.assertEqual(parser.version, 2.3)

    def test_issue_157(self):
        """
        Test case for issue #157: re-using parser object.
        """
        expected = {'latitude': 48.162899, 'elevation': 565.0,
                    'longitude': 11.2752, 'local_depth': 0.0,
                    'azimuth': 0.0, 'dip': -90.0}
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
            result = parser.get_coordinates("BW.FURT..EHZ", t)
            self.assertEqual(expected, result)

    def test_issue_358(self):
        """
        Test case for issue #358.
        """
        filename = os.path.join(self.path, 'CL.AIO.dataless')
        parser = Parser()
        parser.read(filename)
        dt = UTCDateTime('2012-01-01')
        parser.get_paz('CL.AIO.00.EHZ', dt)

    def test_issue_361(self):
        """
        Test case for issue #361.
        """
        filename = os.path.join(self.path, 'G.SPB.dataless')
        parser = Parser()
        parser.read(filename)
        # 1 - G.SPB..BHZ - no Laplace transform - works
        parser.get_paz('G.SPB..BHZ')
        # 2 - G.SPB.00.BHZ - raises exception because of multiple results
        self.assertRaises(SEEDParserException, parser.get_paz, 'G.SPB.00.BHZ')
        # 3 - G.SPB.00.BHZ with datetime - no Laplace transform - works
        dt = UTCDateTime('2007-01-01')
        parser.get_paz('G.SPB.00.BHZ', dt)
        # 4 - G.SPB.00.BHZ with later datetime works
        dt = UTCDateTime('2012-01-01')
        parser.get_paz('G.SPB.00.BHZ', dt)

    def test_split_stations_dataless_to_xseed(self):
        """
        Test case for writing dataless to XSEED with multiple entries.
        """
        filename = os.path.join(self.path, 'dataless.seed.BW_DHFO')
        parser = Parser()
        parser.read(filename)
        with NamedTemporaryFile() as fh:
            tempfile = fh.name
            # this will create two files due to two entries in dataless
            parser.write_xseed(tempfile, split_stations=True)
            # the second file name is appended with the timestamp of start
            # period
            os.remove(tempfile + '.1301529600.0.xml')

    def test_rotation_to_zne(self):
        """
        Weak test for rotation of arbitrarily rotated components to ZNE.
        """
        st = read(os.path.join(self.path,
                               "II_COCO_three_channel_borehole.mseed"))
        # Read the SEED file and rotate the Traces with the information stored
        # in the SEED file.
        p = Parser(os.path.join(self.path, "dataless.seed.II_COCO"))
        st_r = p.rotate_to_zne(st)

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
        np.testing.assert_allclose(tr_z.data, tr_r_z.data, rtol=1e-10)
        # The other two are only rotated by 2 degree so should also not have
        # changed much but at least a little bit. And the components should be
        # renamed.
        self.assertTrue(np.allclose(tr_1, tr_r_n, rtol=10E-3))
        # The east channel carries very little energy for this particular
        # example. Thus it changes quite a lot even for this very subtle
        # rotation. The energy comparison should still ensure a sensible
        # result.
        self.assertTrue(np.allclose(tr_2, tr_r_e, atol=tr_r_e.max() / 4.0))

    def test_underline_in_site_name(self):
        """
        Test case for issue #1893.
        """
        filename = os.path.join(self.path, 'UP_BACU_HH.dataless')
        parser = Parser()
        parser.read(filename)
        # value given by pdccgg
        self.assertEqual(parser.blockettes[50][0].site_name,
                         'T3930_b A6689 3930')

    def test_parsing_resp_file_without_clear_blkt_separation(self):
        """
        This is a slightly malformed RESP file that has two blockettes 58 at
        the end. Most RESP files separate blockettes with comments of which
        at least one contains a plus sign. This one does not so additional
        heuristics are needed.
        """
        filename = os.path.join(self.path, '6D6-Trillium-250sps.resp')
        parser = Parser()
        parser.read(filename)
        b = parser.blockettes[58][-1]
        self.assertEqual(b.stage_sequence_number, 0)
        self.assertEqual(b.number_of_history_values, 0)
        np.testing.assert_allclose(b.sensitivity_gain, 8.043400E+10)
        np.testing.assert_allclose(b.frequency, 1.0)

        # Also compare directly against what evalresp would do.
        obs_r = obspy.read_inventory(filename)[0][0][0].response\
            .get_evalresp_response_for_frequencies([0.0, 1.0, 10.0])
        evresp = evalresp_for_frequencies(0.01, [0.0, 1.0, 10.0], filename,
                                          obspy.UTCDateTime(2015, 1, 2))
        np.testing.assert_allclose(obs_r, evresp)
