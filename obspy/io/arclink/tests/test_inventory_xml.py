#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test suite for the arclink inventory reader.

Modified after obspy.io.stationXML
    > obspy.obspy.io.stationxml.core.py

:author:
    Mathijs Koymans (koymans@knmi.nl), 11.2015 - [Jollyfant@GitHub]

:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""

import inspect
import os
import unittest

from obspy.core.inventory import read_inventory
from obspy.io.arclink.inventory import validate_arclink_xml, SCHEMA_NAMESPACE
import pytest


class ArclinkInventoryTestCase(unittest.TestCase):

    def setUp(self):
        """
        Read example stationXML and arclink format to Inventory
        """
        self.data_dir = os.path.join(os.path.dirname(os.path.abspath(
            inspect.getfile(inspect.currentframe()))), "data")
        self.arclink_xml_path = os.path.join(self.data_dir,
                                             "arclink_inventory.xml")
        self.station_xml_path = os.path.join(self.data_dir, "gols_station.xml")
        self.arclink_xml_poly = os.path.join(self.data_dir,
                                             "arclink_inventory_poly.xml")
        self.arclink_afc_path = os.path.join(self.data_dir, "arclink_afc.xml")
        self.station_afc_path = os.path.join(self.data_dir, "station_afc.xml")

    def test_publicid_slash(self):
        v = read_inventory(os.path.join(self.data_dir, "public-id-slash.xml"))
        self.assertEqual(len(v[0][0][0].response.response_stages), 11)

    def test_analogue_filter_chain(self):
        """
        Test analogue filter chain and compare to stationXML equivalent
        """
        arclink_inv = read_inventory(self.arclink_afc_path)
        station_inv = read_inventory(self.station_afc_path)

        arclink_resp = arclink_inv[0][0][0].response.response_stages
        station_resp = station_inv[0][0][0].response.response_stages

        for arclink_stage, station_stage in zip(arclink_resp, station_resp):

            self.assertEqual(arclink_stage.stage_gain,
                             station_stage.stage_gain)
            self.assertEqual(arclink_stage.stage_sequence_number,
                             station_stage.stage_sequence_number)

            for arc_cs, sta_cs in zip(arclink_stage.__dict__.items(),
                                      station_stage.__dict__.items()):

                if arc_cs[0] in ['name', 'resource_id2']:
                    continue

                self.assertEqual(arc_cs, sta_cs)

    def test_validate_inventories_against_schema(self):
        self.assertTrue(validate_arclink_xml(self.arclink_xml_path)[0])
        self.assertFalse(validate_arclink_xml(self.station_xml_path)[0])

    def test_raise_polynomial(self):
        with self.assertRaises(NotImplementedError) as e:
            read_inventory(self.arclink_xml_poly)

        self.assertEqual(e.exception.args[0], "responsePolynomial not"
                         "implemented. Contact the ObsPy developers")

    @pytest.mark.filterwarnings("ignore:Attribute 'storage_format'.*removed")
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
            for arc, st_xml in zip(arclink_cha[0][0], station_cha[0][0]):
                self.assertEqual(arc.code, st_xml.code)
                self.assertEqual(arc.latitude, st_xml.latitude)
                self.assertEqual(arc.longitude, st_xml.longitude)
                self.assertEqual(arc.depth, st_xml.depth)
                self.assertEqual(arc.azimuth, st_xml.azimuth)
                self.assertEqual(arc.dip, st_xml.dip)
                self.assertEqual(arc.sample_rate, st_xml.sample_rate)
                self.assertEqual(arc.start_date, st_xml.start_date)
                self.assertEqual(arc.end_date, st_xml.end_date)
                # reading stationxml will ignore old StationXML 1.0 defined
                # StorageFormat, Arclink Inventory XML and SC3ML get it stored
                # in extra now
                self.assertEqual(st_xml.storage_format, None)
                self.assertEqual(arc.storage_format, None)
                self.assertEqual(
                    arc.extra['format'],
                    {'namespace': SCHEMA_NAMESPACE, 'value': None})

                cdisps = "clock_drift_in_seconds_per_sample"
                self.assertEqual(getattr(arc, cdisps), getattr(st_xml, cdisps))

                # Compare datalogger element
                for arc_el, st_xml_el in zip(arc.data_logger.__dict__.items(),
                                             st_xml.data_logger.__dict__
                                             .items()):
                    self.assertEqual(arc_el, st_xml_el)

                # Compare sensor element
                for arc_sen, st_xml_sen in zip(arc.sensor.__dict__.items(),
                                               st_xml.sensor.__dict__.items()):
                    # Skip the type; set for stationXML not for ArclinkXML
                    if arc_sen[0] == 'type':
                        continue
                    self.assertEqual(arc_sen, st_xml_sen)

                # Same number of response stages
                self.assertEqual(len(arc.response.response_stages),
                                 len(st_xml.response.response_stages))

                # Check the response stages
                for arc_resp, sta_resp in zip(arc.response.response_stages,
                                              st_xml.response.response_stages):

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
