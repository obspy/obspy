#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test suite for the StationXML reader and writer.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2013
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""
from io import BytesIO
import inspect
from itertools import izip
import os
import unittest

import obspy
import obspy.station


class StationXMLTestCase(unittest.TestCase):
    """
    """
    def setUp(self):
        # Most generic way to get the actual data directory.
        self.data_dir = os.path.join(os.path.dirname(os.path.abspath(
            inspect.getfile(inspect.currentframe()))), "data")

    def test_is_stationxml(self):
        """
        Tests the is_StationXML() function.
        """
        # Check positives.
        stationxmls = [os.path.join(self.data_dir, "minimal_station.xml")]
        for stat in stationxmls:
            self.assertTrue(obspy.station.stationxml.is_StationXML(stat))

        # Check some negatives.
        not_stationxmls = ["Variations-FDSNSXML-SEED.txt",
            "fdsn-station+availability-1.0.xsd", "fdsn-station-1.0.xsd"]
        not_stationxmls = [os.path.join(self.data_dir, os.path.pardir,
            os.path.pardir, "docs", _i) for _i in not_stationxmls]
        for stat in not_stationxmls:
            self.assertFalse(obspy.station.stationxml.is_StationXML(stat))

    def test_read_and_write_minimal_file(self):
        """
        Test that writing the most basic StationXML document possible works.
        """
        filename = os.path.join(self.data_dir, "minimal_station.xml")
        inv = obspy.station.readInventory(filename)

        # Assert the few values that are set directly.
        self.assertEqual(inv.source, "OBS")
        self.assertEqual(inv.created, obspy.UTCDateTime(2013, 1, 1))
        self.assertEqual(len(inv.networks), 1)
        self.assertEqual(inv.networks[0].code, "PY")

        # Write it again. Also validate it to get more confidence.
        file_buffer = BytesIO()
        inv.write(file_buffer, format="StationXML", validate=True)
        file_buffer.seek(0, 0)
        new_file = file_buffer.read().splitlines()

        with open(filename, "rb") as fh:
            org_file = fh.read().splitlines()

        self.assertEqual(len(new_file), len(org_file))

        for new_line, org_line in izip(new_file, org_file):
            self.assertEqual(new_line.strip(), org_line.strip())


def suite():
    return unittest.makeSuite(StationXMLTestCase, "test")


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
