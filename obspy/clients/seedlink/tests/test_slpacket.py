# -*- coding: utf-8 -*-
"""
The obspy.clients.seedlink.slpacket test suite.
"""
import os.path
import unittest

import pytest

from obspy.clients.seedlink.slpacket import SLPacket


pytestmark = pytest.mark.network


class SLPacketTestCase(unittest.TestCase):

    def _read_data_file(self, fn):
        path = os.path.dirname(__file__)
        fn = os.path.join(path, 'data', fn)

        with open(fn, 'rb') as f:
            data = f.read()

        return data

    def test_get_string_payload(self):
        """
        Test parsing of SeedLink MiniSEED payload as XML string.

        The GEOFON and the IRIS Ringserver packets differ in the size of bytes
        used for MiniSEED headers (8 vs. 7 bytes).
        """
        # Check the INFO CAPABILITIES response from GEOFON
        packet = self._read_data_file('info_packet_geofon.slink')
        packet = SLPacket(packet, 0)
        payload = packet.get_string_payload()

        xml = b'<?xml version="1.0"?>'
        self.assertTrue(payload.startswith(xml))
        self.assertEqual(len(payload), 368)

        # Check the INFO CAPABILITIES response from IRIS Ringserver
        packet = self._read_data_file('info_packet_iris.slink')
        packet = SLPacket(packet, 0)
        payload = packet.get_string_payload()

        xml = b'<?xml version="1.0" encoding="utf-8"?>'
        self.assertTrue(payload.startswith(xml))
        self.assertEqual(len(payload), 456)
