#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test suite for the station text parser.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2015
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import inspect
import io
import os
import unittest

import obspy
import obspy.station
from obspy.station import fdsn_text


class StationTextTestCase(unittest.TestCase):
    """
    """
    def setUp(self):
        # Most generic way to get the actual data directory.
        self.data_dir = os.path.join(os.path.dirname(os.path.abspath(
            inspect.getfile(inspect.currentframe()))), "data")

    def test_is_txt_file(self):
        """
        Tests the is_txt() file routine.
        """
        txt_files = [os.path.join(self.data_dir, _i) for _i in [
            "network_level_fdsn.txt", "station_level_fdsn.txt",
            "channel_level_fdsn.txt"]]
        non_txt_files = [os.path.join(self.data_dir, _i) for _i in
                         ["BW_RJOB.xml", "XM.05.seed"]]

        # Test with filenames.
        for filename in txt_files:
            self.assertTrue(fdsn_text.is_FDSN_station_text_file(filename))
        for filename in non_txt_files:
            self.assertFalse(fdsn_text.is_FDSN_station_text_file(filename))

        # Text with open files in binary mode.
        for filename in txt_files:
            with open(filename, "rb") as fh:
                self.assertTrue(fdsn_text.is_FDSN_station_text_file(fh))
                self.assertEqual(fh.tell(), 0)
        for filename in non_txt_files:
            with open(filename, "rb") as fh:
                self.assertFalse(fdsn_text.is_FDSN_station_text_file(fh))
                self.assertEqual(fh.tell(), 0)

        # Text with open files in text mode.
        for filename in txt_files:
            with open(filename, "rt") as fh:
                self.assertTrue(fdsn_text.is_FDSN_station_text_file(fh))
                self.assertEqual(fh.tell(), 0)
        for filename in non_txt_files:
            with open(filename, "rt") as fh:
                self.assertFalse(fdsn_text.is_FDSN_station_text_file(fh))
                self.assertEqual(fh.tell(), 0)

        # Text with BytesIO.
        for filename in txt_files:
            with open(filename, "rb") as fh:
                with io.BytesIO(fh.read()) as buf:
                    self.assertTrue(fdsn_text.is_FDSN_station_text_file(buf))
                    self.assertEqual(buf.tell(), 0)
        for filename in non_txt_files:
            with open(filename, "rb") as fh:
                with io.BytesIO(fh.read()) as buf:
                    self.assertFalse(fdsn_text.is_FDSN_station_text_file(buf))
                    self.assertEqual(buf.tell(), 0)

        # Text with StringIO.
        for filename in txt_files:
            with open(filename, "rt") as fh:
                with io.StringIO(fh.read()) as buf:
                    self.assertTrue(fdsn_text.is_FDSN_station_text_file(buf))
                    self.assertEqual(buf.tell(), 0)
        for filename in non_txt_files:
            with open(filename, "rt") as fh:
                with io.StringIO(fh.read()) as buf:
                    self.assertFalse(fdsn_text.is_FDSN_station_text_file(buf))
                    self.assertEqual(buf.tell(), 0)

    def test_reading_network_file(self):
        """
        Test reading a file at the network level.
        """
        # Manually create an expected Inventory object.
        expected_inv = obspy.station.Inventory(source=None, networks=[
            obspy.station.Network(
                code="TA", total_number_of_stations=1700,
                start_date=obspy.UTCDateTime("2003-01-01T00:00:00"),
                end_date=obspy.UTCDateTime("2500-12-31T23:59:59"),
                description="USArray Transportable Array (NSF EarthScope "
                            "Project)"),

            obspy.station.Network(
                code="TC", total_number_of_stations=0,
                start_date=obspy.UTCDateTime("2011-01-01T00:00:00"),
                end_date=obspy.UTCDateTime("2500-12-31T23:59:59"),
                description="Red Sismologica Nacional")
        ])

        # Read from a filename.
        filename = os.path.join(self.data_dir, "network_level_fdsn.txt")
        inv = fdsn_text.read_FDSN_station_text_file(filename)
        inv_obs = obspy.read_inventory(filename)

        # Copy creation date as it will be slightly different otherwise.
        inv.created = expected_inv.created
        inv_obs.created = expected_inv.created
        self.assertEqual(inv, expected_inv)
        self.assertEqual(inv_obs, expected_inv)

        # Read from open file in text mode.
        with open(filename, "rt") as fh:
            inv = fdsn_text.read_FDSN_station_text_file(fh)
            fh.seek(0, 0)
            inv_obs = obspy.read_inventory(fh)
        inv.created = expected_inv.created
        inv_obs.created = expected_inv.created
        self.assertEqual(inv, expected_inv)
        self.assertEqual(inv_obs, expected_inv)

        # Read from open file in binary mode.
        with open(filename, "rb") as fh:
            inv = fdsn_text.read_FDSN_station_text_file(fh)
            fh.seek(0, 0)
            inv_obs = obspy.read_inventory(fh)
        inv.created = expected_inv.created
        inv_obs.created = expected_inv.created
        self.assertEqual(inv, expected_inv)
        self.assertEqual(inv_obs, expected_inv)

        # Read from StringIO.
        with open(filename, "rt") as fh:
            with io.StringIO(fh.read()) as buf:
                buf.seek(0, 0)
                inv = fdsn_text.read_FDSN_station_text_file(buf)
                buf.seek(0, 0)
                inv_obs = obspy.read_inventory(buf)
        inv.created = expected_inv.created
        inv_obs.created = expected_inv.created
        self.assertEqual(inv, expected_inv)
        self.assertEqual(inv_obs, expected_inv)

        # Read from BytesIO.
        with open(filename, "rb") as fh:
            with io.BytesIO(fh.read()) as buf:
                buf.seek(0, 0)
                inv = fdsn_text.read_FDSN_station_text_file(buf)
                buf.seek(0, 0)
                inv_obs = obspy.read_inventory(buf)
        inv.created = expected_inv.created
        inv_obs.created = expected_inv.created
        self.assertEqual(inv, expected_inv)
        self.assertEqual(inv_obs, expected_inv)

    def test_reading_station_file(self):
        """
        Test reading a file at the station level.
        """
        # Manually create an expected Inventory object.
        expected_inv = obspy.station.Inventory(source=None, networks=[
            obspy.station.Network(
                code="TA", stations=[
                    obspy.station.Station(
                        code="A04A", latitude=48.7197, longitude=-122.707,
                        elevation=23.0, site=obspy.station.Site(
                            name="Legoe Bay, Lummi Island, WA, USA"),
                        start_date=obspy.UTCDateTime("2004-09-19T00:00:00"),
                        end_date=obspy.UTCDateTime("2008-02-19T23:59:59")),
                    obspy.station.Station(
                        code="A04D", latitude=48.7201, longitude=-122.7063,
                        elevation=13.0, site=obspy.station.Site(
                            name="Lummi Island, WA, USA"),
                        start_date=obspy.UTCDateTime("2010-08-18T00:00:00"),
                        end_date=obspy.UTCDateTime("2599-12-31T23:59:59"))
                ]),
            obspy.station.Network(
                code="TR", stations=[
                    obspy.station.Station(
                        code="ALNG", latitude=10.1814, longitude=-61.6883,
                        elevation=10.0, site=obspy.station.Site(
                            name="Trinidad, Point Fortin"),
                        start_date=obspy.UTCDateTime("2000-01-01T00:00:00"),
                        end_date=obspy.UTCDateTime("2599-12-31T23:59:59"))])])

        # Read from a filename.
        filename = os.path.join(self.data_dir, "station_level_fdsn.txt")
        inv = fdsn_text.read_FDSN_station_text_file(filename)
        inv_obs = obspy.read_inventory(filename)

        # Copy creation date as it will be slightly different otherwise.
        inv.created = expected_inv.created
        inv_obs.created = expected_inv.created
        self.assertEqual(inv, expected_inv)
        self.assertEqual(inv_obs, expected_inv)

        # Read from open file in text mode.
        with open(filename, "rt") as fh:
            inv = fdsn_text.read_FDSN_station_text_file(fh)
            fh.seek(0, 0)
            inv_obs = obspy.read_inventory(fh)
        inv.created = expected_inv.created
        inv_obs.created = expected_inv.created
        self.assertEqual(inv, expected_inv)
        self.assertEqual(inv_obs, expected_inv)

        # Read from open file in binary mode.
        with open(filename, "rb") as fh:
            inv = fdsn_text.read_FDSN_station_text_file(fh)
            fh.seek(0, 0)
            inv_obs = obspy.read_inventory(fh)
        inv.created = expected_inv.created
        inv_obs.created = expected_inv.created
        self.assertEqual(inv, expected_inv)
        self.assertEqual(inv_obs, expected_inv)

        # Read from StringIO.
        with open(filename, "rt") as fh:
            with io.StringIO(fh.read()) as buf:
                buf.seek(0, 0)
                inv = fdsn_text.read_FDSN_station_text_file(buf)
                buf.seek(0, 0)
                inv_obs = obspy.read_inventory(buf)
        inv.created = expected_inv.created
        inv_obs.created = expected_inv.created
        self.assertEqual(inv, expected_inv)
        self.assertEqual(inv_obs, expected_inv)

        # Read from BytesIO.
        with open(filename, "rb") as fh:
            with io.BytesIO(fh.read()) as buf:
                buf.seek(0, 0)
                inv = fdsn_text.read_FDSN_station_text_file(buf)
                buf.seek(0, 0)
                inv_obs = obspy.read_inventory(buf)
        inv.created = expected_inv.created
        inv_obs.created = expected_inv.created
        self.assertEqual(inv, expected_inv)
        self.assertEqual(inv_obs, expected_inv)

    def test_reading_channel_file(self):
        """
        Test reading a file at the channel level.
        """
        # Manually create an expected Inventory object.
        expected_inv = obspy.station.Inventory(source=None, networks=[
            obspy.station.Network(
                code="AK", stations=[
                    obspy.station.Station(
                        code="BAGL",
                        latitude=60.4896,
                        longitude=-142.0915,
                        elevation=1470,
                        channels=[
                            obspy.station.Channel(
                                code="LHZ", location_code="",
                                latitude=60.4896,
                                longitude=-142.0915,
                                elevation=1470,
                                depth=0.0,
                                azimuth=0.0,
                                dip=-90.0,
                                sample_rate=1.0,
                                sensor=obspy.station.Equipment(
                                    type="Nanometrics Trillium 240 Sec "
                                         "Response sn 400 and a"),
                                start_date=obspy.UTCDateTime(
                                    "2013-01-01T00:00:00"),
                                end_date=obspy.UTCDateTime(
                                    "2599-12-31T23:59:59"),
                                response=obspy.station.Response(
                                    instrument_sensitivity=obspy.station
                                    .InstrumentSensitivity(
                                        frequency=0.02, input_units="M/S",
                                        output_units=None, value=4.88233E8)))
                        ]),
                    obspy.station.Station(
                        code="BWN",
                        latitude=64.1732,
                        longitude=-149.2991,
                        elevation=356.0,
                        channels=[
                            obspy.station.Channel(
                                code="LHZ", location_code="",
                                latitude=64.1732,
                                longitude=-149.2991,
                                elevation=356.0,
                                depth=0.0,
                                azimuth=0.0,
                                dip=-90.0,
                                sample_rate=1.0,
                                sensor=obspy.station.Equipment(
                                    type="Nanometrics Trillium 240 Sec "
                                         "Response sn 400 and a"),
                                start_date=obspy.UTCDateTime(
                                    "2010-07-23T00:00:00"),
                                end_date=obspy.UTCDateTime(
                                    "2014-05-28T23:59:59"),
                                response=obspy.station.Response(
                                    instrument_sensitivity=obspy.station
                                    .InstrumentSensitivity(
                                        frequency=0.02, input_units="M/S",
                                        output_units=None, value=4.88233E8))),
                            obspy.station.Channel(
                                code="LHZ", location_code="",
                                latitude=64.1732,
                                longitude=-149.2991,
                                elevation=356.0,
                                depth=1.5,
                                azimuth=0.0,
                                dip=-90.0,
                                sample_rate=1.0,
                                sensor=obspy.station.Equipment(
                                    type="Nanometrics Trillium 120 Sec "
                                         "Response/Quanterra 33"),
                                start_date=obspy.UTCDateTime(
                                    "2014-08-01T00:00:00"),
                                end_date=obspy.UTCDateTime(
                                    "2599-12-31T23:59:59"),
                                response=obspy.station.Response(
                                    instrument_sensitivity=obspy.station
                                    .InstrumentSensitivity(
                                        frequency=0.03, input_units="M/S",
                                        output_units=None, value=4.98112E8)))
                        ])
                ]),
            obspy.station.Network(
                code="AZ", stations=[
                    obspy.station.Station(
                        code="BZN",
                        latitude=33.4915,
                        longitude=-116.667,
                        elevation=1301.0,
                        channels=[
                            obspy.station.Channel(
                                code="LHZ", location_code="",
                                latitude=33.4915,
                                longitude=-116.667,
                                elevation=1301.0,
                                depth=0.0,
                                azimuth=0.0,
                                dip=-90.0,
                                sample_rate=1.0,
                                sensor=obspy.station.Equipment(
                                    type="Streckeisen STS-2 G1/Quanterra 330 "
                                         "Linear Phase Be"),
                                start_date=obspy.UTCDateTime(
                                    "2010-07-26T17:22:00"),
                                end_date=obspy.UTCDateTime(
                                    "2013-07-15T21:22:23"),
                                response=obspy.station.Response(
                                    instrument_sensitivity=obspy.station
                                    .InstrumentSensitivity(
                                        frequency=0.03, input_units="M/S",
                                        output_units=None, value=6.27252E8))),
                            obspy.station.Channel(
                                code="LHZ", location_code="",
                                latitude=33.4915,
                                longitude=-116.667,
                                elevation=1301.0,
                                depth=0.0,
                                azimuth=0.0,
                                dip=-90.0,
                                sample_rate=1.0,
                                sensor=obspy.station.Equipment(
                                    type="Streckeisen STS-2 G1/Quanterra 330 "
                                         "Linear Phase Be"),
                                start_date=obspy.UTCDateTime(
                                    "2013-07-15T21:22:23"),
                                end_date=obspy.UTCDateTime(
                                    "2013-10-22T19:30:00"),
                                response=obspy.station.Response(
                                    instrument_sensitivity=obspy.station
                                    .InstrumentSensitivity(
                                        frequency=0.03, input_units="M/S",
                                        output_units=None, value=6.27252E8))),
                            obspy.station.Channel(
                                code="LHZ", location_code="",
                                latitude=33.4915,
                                longitude=-116.667,
                                elevation=1301.0,
                                depth=0.0,
                                azimuth=0.0,
                                dip=-90.0,
                                sample_rate=1.0,
                                sensor=obspy.station.Equipment(
                                    type="Streckeisen STS-2 G1/Quanterra 330 "
                                         "Linear Phase Be"),
                                start_date=obspy.UTCDateTime(
                                    "2013-10-22T19:30:00"),
                                end_date=obspy.UTCDateTime(
                                    "2599-12-31T23:59:59"),
                                response=obspy.station.Response(
                                    instrument_sensitivity=obspy.station
                                    .InstrumentSensitivity(
                                        frequency=0.03, input_units="M/S",
                                        output_units=None, value=6.27252E8))),
                        ])
                ])
            ])

        # Read from a filename.
        filename = os.path.join(self.data_dir, "channel_level_fdsn.txt")
        inv = fdsn_text.read_FDSN_station_text_file(filename)
        inv_obs = obspy.read_inventory(filename)

        # Copy creation date as it will be slightly different otherwise.
        inv.created = expected_inv.created
        inv_obs.created = expected_inv.created
        self.assertEqual(inv, expected_inv)
        self.assertEqual(inv_obs, expected_inv)

        # Read from open file in text mode.
        with open(filename, "rt") as fh:
            inv = fdsn_text.read_FDSN_station_text_file(fh)
            fh.seek(0, 0)
            inv_obs = obspy.read_inventory(fh)
        inv.created = expected_inv.created
        inv_obs.created = expected_inv.created
        self.assertEqual(inv, expected_inv)
        self.assertEqual(inv_obs, expected_inv)

        # Read from open file in binary mode.
        with open(filename, "rb") as fh:
            inv = fdsn_text.read_FDSN_station_text_file(fh)
            fh.seek(0, 0)
            inv_obs = obspy.read_inventory(fh)
        inv.created = expected_inv.created
        inv_obs.created = expected_inv.created
        self.assertEqual(inv, expected_inv)
        self.assertEqual(inv_obs, expected_inv)

        # Read from StringIO.
        with open(filename, "rt") as fh:
            with io.StringIO(fh.read()) as buf:
                buf.seek(0, 0)
                inv = fdsn_text.read_FDSN_station_text_file(buf)
                buf.seek(0, 0)
                inv_obs = obspy.read_inventory(buf)
        inv.created = expected_inv.created
        inv_obs.created = expected_inv.created
        self.assertEqual(inv, expected_inv)
        self.assertEqual(inv_obs, expected_inv)

        # Read from BytesIO.
        with open(filename, "rb") as fh:
            with io.BytesIO(fh.read()) as buf:
                buf.seek(0, 0)
                inv = fdsn_text.read_FDSN_station_text_file(buf)
                buf.seek(0, 0)
                inv_obs = obspy.read_inventory(buf)
        inv.created = expected_inv.created
        inv_obs.created = expected_inv.created
        self.assertEqual(inv, expected_inv)
        self.assertEqual(inv_obs, expected_inv)

    def test_reading_unicode_file(self):
        """
        Tests reading a file with non ASCII characters.
        """
        # Manually create an expected Inventory object.
        expected_inv = obspy.station.Inventory(source=None, networks=[
            obspy.station.Network(
                code="PR", stations=[
                    obspy.station.Station(
                        code="CTN1", latitude=18.43718, longitude=-67.1303,
                        elevation=10.0, site=obspy.station.Site(
                            name="CATA¿O DEFENSA CIVIL"),
                        start_date=obspy.UTCDateTime("2004-01-27T00:00:00"),
                        end_date=obspy.UTCDateTime("2599-12-31T23:59:59"))])])

        # Read from a filename.
        filename = os.path.join(self.data_dir, "unicode_example_fdsn.txt")
        inv = fdsn_text.read_FDSN_station_text_file(filename)
        inv_obs = obspy.read_inventory(filename)

        # Copy creation date as it will be slightly different otherwise.
        inv.created = expected_inv.created
        inv_obs.created = expected_inv.created
        self.assertEqual(inv, expected_inv)
        self.assertEqual(inv_obs, expected_inv)

        # Read from open file in text mode.
        with open(filename, "rt") as fh:
            inv = fdsn_text.read_FDSN_station_text_file(fh)
            fh.seek(0, 0)
            inv_obs = obspy.read_inventory(fh)
        inv.created = expected_inv.created
        inv_obs.created = expected_inv.created
        self.assertEqual(inv, expected_inv)
        self.assertEqual(inv_obs, expected_inv)

        # Read from open file in binary mode.
        with open(filename, "rb") as fh:
            inv = fdsn_text.read_FDSN_station_text_file(fh)
            fh.seek(0, 0)
            inv_obs = obspy.read_inventory(fh)
        inv.created = expected_inv.created
        inv_obs.created = expected_inv.created
        self.assertEqual(inv, expected_inv)
        self.assertEqual(inv_obs, expected_inv)

        # Read from StringIO.
        with open(filename, "rt") as fh:
            with io.StringIO(fh.read()) as buf:
                buf.seek(0, 0)
                inv = fdsn_text.read_FDSN_station_text_file(buf)
                buf.seek(0, 0)
                inv_obs = obspy.read_inventory(buf)
        inv.created = expected_inv.created
        inv_obs.created = expected_inv.created
        self.assertEqual(inv, expected_inv)
        self.assertEqual(inv_obs, expected_inv)

        # Read from BytesIO.
        with open(filename, "rb") as fh:
            with io.BytesIO(fh.read()) as buf:
                buf.seek(0, 0)
                inv = fdsn_text.read_FDSN_station_text_file(buf)
                buf.seek(0, 0)
                inv_obs = obspy.read_inventory(buf)
        inv.created = expected_inv.created
        inv_obs.created = expected_inv.created
        self.assertEqual(inv, expected_inv)
        self.assertEqual(inv_obs, expected_inv)


def suite():
    return unittest.makeSuite(StationTextTestCase, "test")


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
