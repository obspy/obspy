# -*- coding: utf-8 -*-
import gzip
import io
import os
import warnings

import numpy as np
from lxml import etree

import obspy
from obspy import UTCDateTime, read, read_inventory
from obspy.core.util import NamedTemporaryFile
from obspy.io.xseed.blockette.blockette010 import Blockette010
from obspy.io.xseed.blockette.blockette051 import Blockette051
from obspy.io.xseed.blockette.blockette053 import Blockette053
from obspy.io.xseed.blockette.blockette054 import Blockette054
import obspy.io.xseed.parser
from obspy.io.xseed.parser import Parser
from obspy.io.xseed.utils import SEEDParserException, compare_seed
from obspy.signal.invsim import evalresp_for_frequencies
import pytest


class TestParser():
    """
    Parser test suite.
    """
    @pytest.fixture(autouse=True, scope="function")
    def setup(self, testdata):
        self.BW_SEED_files = [
            testdata[file] for file in
            ['dataless.seed.BW_FURT', 'dataless.seed.BW_MANZ',
             'dataless.seed.BW_ROTZ', 'dataless.seed.BW_ZUGS']]

    def test_issue165(self, testdata):
        """
        Test cases related to #165:
         - number of poles or zeros can be 0
         - an unsupported response information somewhere in the metadata should
           not automatically raise an Error, if the desired information can
           still be retrieved
        """
        parser = Parser(strict=True)
        file = testdata["bug165.dataless"]
        t = UTCDateTime("2010-01-01T00:00:00")
        parser.read(file)
        paz = parser.get_paz("NZ.DCZ.20.HNZ", t)
        result = {'digitizer_gain': 419430.0, 'gain': 24595700000000.0,
                  'poles': [(-981 + 1009j), (-981 - 1009j),
                            (-3290 + 1263j), (-3290 - 1263j)],
                  'seismometer_gain': 1.01885, 'sensitivity': 427336.0,
                  'zeros': []}
        assert paz == result

    def test_invalid_start_header(self):
        """
        A SEED Volume must start with a Volume Index Control Header.
        """
        data = b"000001S 0510019~~0001000000"
        sp = Parser(strict=True)
        with pytest.raises(SEEDParserException):
            sp.read(data)

    def test_invalid_start_blockette(self):
        """
        A SEED Volume must start with Blockette 010.
        """
        data = b"000001V 0510019~~0001000000"
        sp = Parser(strict=True)
        with pytest.raises(SEEDParserException):
            sp.read(data)

    def test_newline_between_blockettes(self, testdata):
        """
        A very rare case.
        """
        # Handcrafted files.
        filename = testdata['dataless.seed.newline_between_blockettes']
        p = Parser(filename)
        assert sorted(list(p.blockettes.keys())) == [10, 11, 30, 33, 34]

    def test_string(self, testdata):
        """
        Tests string representation of L{obspy.io.xseed.Parser} object.
        """
        filename = testdata['dataless.seed.BW_MANZ']
        p = Parser(filename)
        sp = str(p).splitlines()
        sp = [_i.strip() for _i in sp]
        assert sp == [
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
                "2005-12-06 -  | Lat: 50.0, Lng: 12.1")]

    def test_get_inventory(self, testdata):
        """
        Tests the parser's get_inventory() method.
        """
        filename = testdata['dataless.seed.BW_FURT']
        p = Parser(filename)
        assert p.get_inventory() == \
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
                  'sampling_rate': 200.0}]}

    def test_non_existing_file_name(self):
        """
        Test reading non existing file.
        """
        with pytest.raises(IOError):
            Parser("XYZ")

    def test_blockette_starts_after_record(self):
        """
        '... 058003504 1.00000E+00 0.00000E+0000 000006S*0543864 ... '
        ' 0543864' -> results in Blockette 005
        """
        # create a valid blockette 010 with record length 256
        b010 = b"0100042 2.4082008,001~2038,001~2009,001~~~"
        blockette = Blockette010(strict=True, compact=True)
        blockette.parse_seed(b010)
        assert b010 == blockette.get_seed()
        # create a valid blockette 054
        b054 = b"0540240A0400300300000009" + (b"+1.58748E-03" * 18)
        blockette = Blockette054(strict=True, compact=True)
        blockette.parse_seed(b054)
        assert b054 == blockette.get_seed()
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
        assert b010 == blockette.get_seed()
        # create a valid blockette 054
        b054 = b"0540960A0400300300000039"
        nr = b""
        for i in range(0, 78):
            # 960 chars
            nr = nr + ("+1.000%02dE-03" % i).encode('ascii', 'strict')
        blockette = Blockette054(strict=True, compact=True)
        blockette.parse_seed(b054 + nr)
        assert b054 + nr == blockette.get_seed()
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
        assert len(data) == 256 * 5
        data += b"000006S " + b054 + nr[0:224]  # 256-8-24 = 224
        data += b"000007S*" + nr[224:472]  # 256-8 = 248
        data += b"000008S*" + nr[472:720]
        data += b"000009S*" + nr[720:] + b' ' * 32  # 32 spaces left
        assert len(data) == 256 * 9
        # read records
        parser = Parser(strict=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            parser.read(data)
        # check results
        assert sorted(parser.blockettes.keys()) == [10, 51, 54]
        assert len(parser.blockettes[10]) == 1
        assert len(parser.blockettes[51]) == 1
        assert len(parser.blockettes[54]) == 2

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
        assert len(records) == 5
        # Each records should be 100 - 6 = 94 long.
        for record in records:
            assert len(record) == 94
        # Reassemble the String.
        new_string = b''
        for record in records:
            new_string += record[2:]
        # Compare the new and the old string.
        assert new_string.strip() == seed_string

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
            assert parser1.get_seed() == parser2.get_seed()
            del parser1, parser2

    def test_create_read_assert_and_write_xseed(self, testdata):
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
            xsd_path = testdata['xml-seed-%s.xsd' % version]
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
                assert xmlschema.validate(doc)
                del doc
                parser3 = Parser(xseed_string)
                new_seed = parser3.get_seed()
                assert original_seed == new_seed
                del parser3, original_seed, new_seed

    def test_read_full_seed(self, testdata):
        """
        Test the reading of a full-SEED file. The data portion will be omitted.
        """
        filename = testdata['arclink_full.seed']
        sp = Parser(filename)
        # Just checks whether certain blockettes are written.
        assert len(sp.stations) == 1
        assert [_i.id for _i in sp.volume] == [10]
        assert [_i.id for _i in sp.abbreviations] == \
            [30, 33, 33, 34, 34, 34, 34, 41, 43, 44, 47, 47, 48, 48, 48]
        assert [_i.id for _i in sp.stations[0]] == [50, 52, 60, 58]
        assert sp.stations[0][0].network_code == 'GR'
        assert sp.stations[0][0].station_call_letters == 'FUR'

    def test_get_paz(self, testdata):
        """
        Test extracting poles and zeros information
        """
        filename = testdata['arclink_full.seed']
        sp = Parser(filename)
        paz = sp.get_paz('BHE')
        assert paz['gain'] == +6.00770e+07
        assert paz['zeros'] == [0j, 0j]
        assert paz['poles'] == \
            [(-3.70040e-02 + 3.70160e-02j),
             (-3.70040e-02 - 3.70160e-02j), (-2.51330e+02 + 0.00000e+00j),
             (-1.31040e+02 - 4.67290e+02j), (-1.31040e+02 + 4.67290e+02j)]
        assert paz['sensitivity'] == +7.86576e+08
        assert paz['seismometer_gain'] == +1.50000E+03
        # Raise exception for undefined channels
        with pytest.raises(SEEDParserException):
            sp.get_paz('EHE')
        #
        # Do the same for another dataless file
        #
        filename = testdata['dataless.seed.BW_FURT']
        sp = Parser(filename)
        paz = sp.get_paz('EHE')
        assert paz['gain'] == +1.00000e+00
        assert paz['zeros'] == [0j, 0j, 0j]
        assert paz['poles'] == [(-4.44400e+00 + 4.44400e+00j),
                                (-4.44400e+00 - 4.44400e+00j),
                                (-1.08300e+00 + 0.00000e+00j)]
        assert paz['sensitivity'] == +6.71140E+08
        assert paz['seismometer_gain'] == 4.00000E+02
        # Raise exception for undefined channels
        with pytest.raises(SEEDParserException):
            sp.get_paz('BHE')
        # Raise UserWarning if not a Laplacian transfer function ('A').
        # Modify transfer_fuction_type on the fly
        for blk in sp.blockettes[53]:
            blk.transfer_function_types = 'X'
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("error", UserWarning)
            with pytest.raises(UserWarning):
                sp.get_paz('EHE')
        #
        # And the same for yet another dataless file
        #
        filename = testdata['nied.dataless.gz']
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
            assert paz['gain'] == gain[i]
            assert paz['zeros'] == zeros[i]
            assert paz['poles'] == poles[i]
            assert paz['sensitivity'] == sensitivity[i]
            assert paz['seismometer_gain'] == seismometer_gain[i]
        sp = Parser(testdata['dataless.seed.BW_RJOB'])
        paz = sp.get_paz("BW.RJOB..EHZ", UTCDateTime("2007-01-01"))
        result = {'gain': 1.0,
                  'poles': [(-4.444 + 4.444j), (-4.444 - 4.444j),
                            (-1.083 + 0j)],
                  'seismometer_gain': 400.0,
                  'sensitivity': 671140000.0,
                  'zeros': [0j, 0j, 0j],
                  'digitizer_gain': 1677850.0}
        assert paz == result
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
        assert sorted(paz.items()) == sorted(result.items())
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
        assert sorted(paz.items()) == sorted(result.items())
        # test for multiple blockette 53s using II dataless
        sp = Parser(testdata['dataless.seed.II_COCO'])
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
        assert sorted(paz.items()) == sorted(result.items())

    def test_get_paz_from_xseed(self, testdata):
        """
        Get PAZ from XSEED file, testcase for #146
        """
        filename = testdata['dataless.seed.BW_FURT']
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
        assert sorted(paz.items()) == sorted(result.items())

    def test_get_coordinates(self, testdata):
        """
        Test extracting coordinates for SEED and XSEED (including #146)
        """
        # SEED
        sp = Parser(testdata['dataless.seed.BW_RJOB'])
        result = {'elevation': 860.0, 'latitude': 47.737166999999999,
                  'longitude': 12.795714, 'local_depth': 0,
                  'azimuth': 0.0, 'local_depth': 0, 'dip': -90.0}
        paz = sp.get_coordinates("BW.RJOB..EHZ", UTCDateTime("2007-01-01"))
        assert sorted(paz.items()) == sorted(result.items())
        paz = sp.get_coordinates("BW.RJOB..EHZ", UTCDateTime("2010-01-01"))
        assert sorted(paz.items()) == sorted(result.items())
        # XSEED
        sp2 = Parser(sp.get_xseed())
        paz = sp2.get_coordinates("BW.RJOB..EHZ", UTCDateTime("2007-01-01"))
        assert sorted(paz.items()) == sorted(result.items())
        paz = sp2.get_coordinates("BW.RJOB..EHZ", UTCDateTime("2010-01-01"))
        assert sorted(paz.items()) == sorted(result.items())
        # Additional test with non-trivial azimuth
        sp = Parser(testdata['dataless.seed.II_COCO'])
        result = {'elevation': 1.0, 'latitude': -12.1901,
                  'longitude': 96.8349, 'local_depth': 1.3,
                  'azimuth': 92.0, 'local_depth': 1.3, 'dip': 0.0}
        paz = sp.get_coordinates("II.COCO.10.BH2", UTCDateTime("2010-11-11"))
        assert sorted(paz.items()) == sorted(result.items())

    def test_select_does_not_change_the_parser_format(self, testdata):
        """
        Test that using the _select() method of the Parser object does
        not change the _format attribute.
        """
        p = Parser(testdata["dataless.seed.BW_FURT.xml"])
        assert p._format == "XSEED"
        p._select(p.get_inventory()["channels"][0]["channel_id"])
        assert p._format == "XSEED"

    def test_create_resp_from_xseed(self, testdata):
        """
        Tests RESP file creation from XML-SEED.
        """
        # 1
        # parse Dataless SEED
        filename = testdata['dataless.seed.BW_FURT']
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
        filename = testdata['arclink_full.seed']
        sp1 = Parser(filename)
        # write XML-SEED
        with NamedTemporaryFile() as fh:
            tempfile = fh.name
            sp1.write_xseed(tempfile)
            # parse XML-SEED
            sp2 = Parser(tempfile)
            # create RESP files
            sp2.get_resp()

    def test_read_resp(self, testdata):
        """
        Tests reading a respfile by calling Parser(filename)
        """
        sts2_resp_file = testdata['RESP.XX.NS085..BHZ.STS2_gen3.120.1500']
        p = Parser(sts2_resp_file)
        # Weak but at least tests that something has been read.
        assert set(p.blockettes.keys()) == {34, 50, 52, 53, 54, 57, 58}

        rt130_resp_file = testdata['RESP.XX.NR008..HHZ.130.1.100']
        p = Parser(rt130_resp_file)
        # Weak but at least tests that something has been read.
        assert set(p.blockettes.keys()) == {34, 50, 52, 53, 54, 57, 58}

    def test_read_resp_data(self, testdata):
        """
        Tests reading a resp string by calling Parser(string)
        """
        sts2_resp_file = testdata['RESP.XX.NS085..BHZ.STS2_gen3.120.1500']
        with open(sts2_resp_file, "rt") as fh:
            p = Parser(fh.read())
        # Weak but at least tests that something has been read.
        assert set(p.blockettes.keys()) == {34, 50, 52, 53, 54, 57, 58}

        rt130_resp_file = testdata['RESP.XX.NR008..HHZ.130.1.100']
        with open(rt130_resp_file, "rt") as fh:
            p = Parser(fh.read())
        # Weak but at least tests that something has been read.
        assert set(p.blockettes.keys()) == {34, 50, 52, 53, 54, 57, 58}

    def test_read_resp_paz_uncertainties(self, testdata):
        """
        Regression test for correctly reading uncertainties in PAZ complex
        numbers
        """
        inv = read_inventory(testdata['RESP.HRV.IU.00.BHZ_cropped'], "RESP")
        resp = inv[0][0][0].response
        pole = resp.response_stages[0].poles[0]
        assert round(pole.imag, 4) == -0.0123
        assert round(pole.imag.upper_uncertainty, 6) == 0.000735
        assert round(pole.imag.lower_uncertainty, 6) == 0.000735
        zero = resp.response_stages[0].zeros[2]
        assert round(zero.real, 4) == -0.0261
        assert round(zero.real.lower_uncertainty, 4) == 0.0238
        assert round(zero.real.upper_uncertainty, 4) == 0.0238

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

    def test_resp_round_trip(self, root):
        single_seed = (root / 'core' / 'tests' / 'data' /
                       'IRIS_single_channel_with_response.seed')
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
        assert seed_list == resp_list

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
        assert p._compare_blockettes(blockette1, blockette2)
        assert not p._compare_blockettes(blockette1, blockette3)
        assert not p._compare_blockettes(blockette2, blockette3)
        assert p._compare_blockettes(blockette3, blockette4)

    def test_missing_required_date_times(self):
        """
        A warning should be raised if a blockette misses a required date.
        """
        # blockette 10 - missing start time
        b010 = b"0100034 2.408~2038,001~2009,001~~~"
        # strict raises an exception
        blockette = Blockette010(strict=True)
        with pytest.raises(SEEDParserException):
            blockette.parse_seed(b010)
        # If strict is false, a warning is raised. This is tested in
        # test_bug165.
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("ignore", UserWarning)
            blockette = Blockette010()
            blockette.parse_seed(b010)
            assert b010 == blockette.get_seed()
        # blockette 10 - missing volume time
        b010 = b"0100034 2.4082008,001~2038,001~~~~"
        # strict raises an exception
        blockette = Blockette010(strict=True)
        with pytest.raises(SEEDParserException):
            blockette.parse_seed(b010)
        # non-strict
        blockette = Blockette010()
        # The warning cannot be tested due to being issued only once, but will
        # be ignored - a similar case is tested in test_bug165.
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("ignore", UserWarning)
            blockette.parse_seed(b010)
        assert b010 == blockette.get_seed()

    def test_issue_298a(self, testdata):
        """
        Test case for issue #298: blockette size exceeds 9999 bytes.
        """
        file = testdata["AI.ESPZ._.BHE.dataless"]
        parser = Parser(file)
        parser.get_resp()

    def test_issue_298b(self, testdata):
        """
        Second test case for issue #298: blockette size exceeds 9999 bytes.
        """
        file = testdata["AI.ESPZ._.BH_.dataless"]
        parser = Parser(file)
        parser.get_resp()

    def test_issue_319(self, testdata):
        """
        Test case for issue #319: multiple abbreviation dictionaries.
        """
        # We have to clear the warnings registry here as some other tests
        # also trigger the warning.
        if hasattr(obspy.io.xseed.parser, "__warningregistry__"):
            obspy.io.xseed.parser.__warningregistry__.clear()

        filename = testdata['BN.LPW._.BHE.dataless']
        # raises a UserWarning: More than one Abbreviation Dictionary Control
        # Headers found!
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            parser = Parser(filename)
        assert w[0].message.args[0] == \
            "More than one Abbreviation Dictionary Control Headers found!"
        assert parser.version == 2.3

    def test_issue_157(self, testdata):
        """
        Test case for issue #157: re-using parser object.
        """
        expected = {'latitude': 48.162899, 'elevation': 565.0,
                    'longitude': 11.2752, 'local_depth': 0.0,
                    'azimuth': 0.0, 'dip': -90.0}
        filename1 = testdata['dataless.seed.BW_FURT']
        filename2 = testdata['dataless.seed.BW_MANZ']
        t = UTCDateTime("2010-07-01")
        parser = Parser()
        parser.read(filename2)
        # parsing a second time will raise a UserWarning: Clearing parser
        # before every subsequent read()
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("error", UserWarning)
            with pytest.raises(UserWarning):
                parser.read(filename1)
            warnings.simplefilter("ignore", UserWarning)
            parser.read(filename1)
            result = parser.get_coordinates("BW.FURT..EHZ", t)
            assert expected == result

    def test_issue_358(self, testdata):
        """
        Test case for issue #358.
        """
        filename = testdata['CL.AIO.dataless']
        parser = Parser()
        parser.read(filename)
        dt = UTCDateTime('2012-01-01')
        parser.get_paz('CL.AIO.00.EHZ', dt)

    def test_issue_361(self, testdata):
        """
        Test case for issue #361.
        """
        filename = testdata['G.SPB.dataless']
        parser = Parser()
        parser.read(filename)
        # 1 - G.SPB..BHZ - no Laplace transform - works
        parser.get_paz('G.SPB..BHZ')
        # 2 - G.SPB.00.BHZ - raises exception because of multiple results
        with pytest.raises(SEEDParserException):
            parser.get_paz('G.SPB.00.BHZ')
        # 3 - G.SPB.00.BHZ with datetime - no Laplace transform - works
        dt = UTCDateTime('2007-01-01')
        parser.get_paz('G.SPB.00.BHZ', dt)
        # 4 - G.SPB.00.BHZ with later datetime works
        dt = UTCDateTime('2012-01-01')
        parser.get_paz('G.SPB.00.BHZ', dt)

    def test_split_stations_dataless_to_xseed(self, testdata):
        """
        Test case for writing dataless to XSEED with multiple entries.
        """
        filename = testdata['dataless.seed.BW_DHFO']
        parser = Parser()
        parser.read(filename)
        with NamedTemporaryFile() as fh:
            tempfile = fh.name
            # this will create two files due to two entries in dataless
            parser.write_xseed(tempfile, split_stations=True)
            # the second file name is appended with the timestamp of start
            # period
            os.remove(tempfile + '.1301529600.0.xml')

    def test_rotation_to_zne(self, testdata):
        """
        Weak test for rotation of arbitrarily rotated components to ZNE.
        """
        st = read(testdata["II_COCO_three_channel_borehole.mseed"])
        # Read the SEED file and rotate the Traces with the information stored
        # in the SEED file.
        p = Parser(testdata["dataless.seed.II_COCO"])
        st_r = p.rotate_to_zne(st)

        # Still three channels left.
        assert len(st_r) == 3

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
        assert np.allclose(energy_before, energy_after)

        # The vertical channel should not have changed at all.
        np.testing.assert_allclose(tr_z.data, tr_r_z.data, rtol=1e-10)
        # The other two are only rotated by 2 degree so should also not have
        # changed much but at least a little bit. And the components should be
        # renamed.
        assert np.allclose(tr_1, tr_r_n, rtol=10E-3)
        # The east channel carries very little energy for this particular
        # example. Thus it changes quite a lot even for this very subtle
        # rotation. The energy comparison should still ensure a sensible
        # result.
        assert np.allclose(tr_2, tr_r_e, atol=tr_r_e.max() / 4.0)

    def test_underline_in_site_name(self, testdata):
        """
        Test case for issue #1893.
        """
        filename = testdata['UP_BACU_HH.dataless']
        parser = Parser()
        parser.read(filename)
        # value given by pdccgg
        assert parser.blockettes[50][0].site_name == 'T3930_b A6689 3930'

    def test_parsing_resp_file_without_clear_blkt_separation(self, testdata):
        """
        This is a slightly malformed RESP file that has two blockettes 58 at
        the end. Most RESP files separate blockettes with comments of which
        at least one contains a plus sign. This one does not so additional
        heuristics are needed.
        """
        filename = testdata['6D6-Trillium-250sps.resp']
        parser = Parser()
        parser.read(filename)
        b = parser.blockettes[58][-1]
        assert b.stage_sequence_number == 0
        assert b.number_of_history_values == 0
        np.testing.assert_allclose(b.sensitivity_gain, 8.043400E+10)
        np.testing.assert_allclose(b.frequency, 1.0)

        # Also compare directly against what evalresp would do.
        obs_r = obspy.read_inventory(filename)[0][0][0].response\
            .get_evalresp_response_for_frequencies([0.0, 1.0, 10.0])
        evresp = evalresp_for_frequencies(0.01, [0.0, 1.0, 10.0], filename,
                                          obspy.UTCDateTime(2015, 1, 2))
        np.testing.assert_allclose(obs_r, evresp)

    def test_parsing_resp_file_with_mutiple_blockette55(self, testdata):
        """
        Test case for issue #3275. Allow reading mutiple blockette 55.
        """
        inv_xml = obspy.read_inventory(testdata['issue3275.xml'])
        with warnings.catch_warnings(record=True):
            warnings.filterwarnings("ignore", "Date is required")
            inv_seed = obspy.read_inventory(testdata['issue3275.seed'])
        freq_xml = [f.frequency for f in inv_xml[0][0][0].response
                     .response_stages[0].response_list_elements]
        # test file with two blockettes 55
        freq_seed = [f.frequency for f in inv_seed[0][0][0].response
                     .response_stages[0].response_list_elements]
        assert len(freq_xml) == len(freq_seed)
        assert freq_xml == freq_seed
