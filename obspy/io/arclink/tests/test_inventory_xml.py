#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test suite for the sc3ml reader.

Modified after obspy.io.stationXML
    > obspy.obspy.io.stationxml.core.py

:author:
    Mathijs Koymans (koymans@knmi.nl), 11.2015 - [Jollyfant@GitHub]

:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import inspect
import os
import unittest

from obspy.core.inventory import read_inventory
from obspy.io.arclink.inventory import validate_arclink_xml


class ArclinkInventoryTestCase(unittest.TestCase):

    def setUp(self):
        """
        Read example stationXML/sc3ml format to Inventory
        """
        self.data_dir = os.path.join(os.path.dirname(os.path.abspath(
            inspect.getfile(inspect.currentframe()))), "data")
        self.arclink_xml_path = os.path.join(self.data_dir,
                                             "arclink_inventory.xml")
        self.station_xml_path = os.path.join(self.data_dir, "gols_station.xml")

    def test_validate_inventories_against_schema(self):
        self.assertTrue(validate_arclink_xml(self.arclink_xml_path)[0])
        self.assertFalse(validate_arclink_xml(self.station_xml_path)[0])

    def test_auto_read_arclink_xml(self):
        arclink_inv = read_inventory(self.arclink_xml_path)
        self.assertIsNotNone(arclink_inv)

        station_inv = read_inventory(self.station_xml_path)

        arclink_inv = arclink_inv.select(station="GOLS")
        channels = ['HH1', 'HH2', 'HHZ', 'BHZ', 'BH1', 'BH2', 'LH1', 'LH2',
                    'LHZ']

        for channel in channels:
            arclink_cha = arclink_inv.select(channel=channel)
            station_cha = station_inv.select(channel=channel)
            for arc, sc3ml in zip(arclink_cha[0][0], station_cha[0][0]):
                self.assertEqual(arc.code, sc3ml.code)
                self.assertEqual(arc.latitude, sc3ml.latitude)
                self.assertEqual(arc.longitude, sc3ml.longitude)
                self.assertEqual(arc.depth, sc3ml.depth)
                self.assertEqual(arc.azimuth, sc3ml.azimuth)
                self.assertEqual(arc.dip, sc3ml.dip)
                self.assertEqual(arc.sample_rate, sc3ml.sample_rate)
                self.assertEqual(arc.start_date, sc3ml.start_date)
                self.assertEqual(arc.end_date, sc3ml.end_date)
                self.assertEqual(arc.storage_format, sc3ml.storage_format)

                cdisps = "clock_drift_in_seconds_per_sample"
                self.assertEqual(getattr(arc, cdisps), getattr(sc3ml, cdisps))

                # Compare datalogger element
                for arc_log, sc3ml_log in zip(arc.data_logger.__dict__.items(),
                                              sc3ml.data_logger.__dict__
                                              .items()):
                    self.assertEqual(arc_log, sc3ml_log)

                # Compare sensor element
                for arc_sen, sc3ml_sen in zip(arc.sensor.__dict__.items(),
                                              sc3ml.sensor.__dict__.items()):
                    # Skip the type; set for stationXML not for ArclinkXML
                    if arc_sen[0] == 'type':
                        continue
                    self.assertEqual(arc_sen, sc3ml_sen)

                # Same number of response stages
                self.assertEqual(len(arc.response.response_stages),
                                 len(sc3ml.response.response_stages))

                # Check the response stages
                for arc_resp, sta_resp in zip(arc.response.response_stages,
                                              sc3ml.response.response_stages):

                    self.assertEqual(arc_resp.stage_gain, sta_resp.stage_gain)
                    self.assertEqual(arc_resp.stage_sequence_number,
                                     sta_resp.stage_sequence_number)

                    for arc_cs, sta_cs in zip(arc_resp.__dict__.items(),
                                              sta_resp.__dict__.items()):

                        # Name can be different and resource_id is not given
                        # in stationXML
                        if arc_cs[0] in ['name', 'resource_id2']:
                            continue

                        # Floating point precision troubles..
                        # 0.021099999999999997 != 0.0211
                        if isinstance(arc_cs[1], float):
                            self.assertAlmostEqual(arc_cs[1], sta_cs[1])
                        else:
                            self.assertEqual(arc_cs, sta_cs)


def suite():
    return unittest.makeSuite(ArclinkInventoryTestCase, "test")


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
