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

        # Copy creation date as it will be slightly different otherwise.
        inv.created = expected_inv.created
        self.assertEqual(inv, expected_inv)

        # Read from open file in text mode.
        with open(filename, "rt") as fh:
            inv = fdsn_text.read_FDSN_station_text_file(fh)
        inv.created = expected_inv.created
        self.assertEqual(inv, expected_inv)

        # Read from open file in binary mode.
        with open(filename, "rb") as fh:
            inv = fdsn_text.read_FDSN_station_text_file(fh)
        inv.created = expected_inv.created
        self.assertEqual(inv, expected_inv)

        # Read from StringIO.
        with open(filename, "rt") as fh:
            with io.StringIO(fh.read()) as buf:
                buf.seek(0, 0)
                inv = fdsn_text.read_FDSN_station_text_file(buf)
        inv.created = expected_inv.created
        self.assertEqual(inv, expected_inv)

        # Read from BytesIO.
        with open(filename, "rb") as fh:
            with io.BytesIO(fh.read()) as buf:
                buf.seek(0, 0)
                inv = fdsn_text.read_FDSN_station_text_file(buf)
        inv.created = expected_inv.created
        self.assertEqual(inv, expected_inv)


def suite():
    return unittest.makeSuite(StationTextTestCase, "test")


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
