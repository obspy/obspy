# -*- coding: utf-8 -*-
"""
The obspy.clients.seedlink.slpacket test suite.
"""
from obspy.clients.seedlink.slpacket import SLPacket


def _read_data_file(path):
    with open(path, 'rb') as f:
        data = f.read()
    return data


class TestSLPacket():

    def test_get_string_payload(self, testdata):
        """
        Test parsing of SeedLink MiniSEED payload as XML string.

        The GEOFON and the IRIS Ringserver packets differ in the size of bytes
        used for MiniSEED headers (8 vs. 7 bytes).
        """
        # Check the INFO CAPABILITIES response from GEOFON
        packet = _read_data_file(testdata['info_packet_geofon.slink'])
        packet = SLPacket(packet, 0)
        payload = packet.get_string_payload()

        xml = b'<?xml version="1.0"?>'
        assert payload.startswith(xml)
        assert len(payload) == 368

        # Check the INFO CAPABILITIES response from IRIS Ringserver
        packet = _read_data_file(testdata['info_packet_iris.slink'])
        packet = SLPacket(packet, 0)
        payload = packet.get_string_payload()

        xml = b'<?xml version="1.0" encoding="utf-8"?>'
        assert payload.startswith(xml)
        assert len(payload) == 456
