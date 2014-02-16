#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test suite for the inventory class.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2013
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""
from __future__ import unicode_literals
import unittest

from obspy.station import Inventory, Network, Station, Channel, Response
from obspy import UTCDateTime


class InventoryTestCase(unittest.TestCase):
    """
    Tests the for :class:`~obspy.station.inventory.Inventory` class.
    """
    def test_initialization(self):
        """
        Some simple sanity tests.
        """
        dt = UTCDateTime()
        inv = Inventory(source="TEST", networks=[])
        # If no time is given, the creation time should be set to the current
        # time. Use a large offset for potentially slow computers and test
        # runs.
        self.assertTrue(inv.created - dt <= 10.0)

    def test_get_response(self):
        responseN1S1 = Response('RESPN1S1')
        responseN1S2 = Response('RESPN1S2')
        responseN2S1 = Response('RESPN2S1')
        channelsN1S1 = [Channel(code='BHZ',
                                location_code='',
                                latitude=0.0,
                                longitude=0.0,
                                elevation=0.0,
                                depth=0.0,
                                response=responseN1S1)]
        channelsN1S2 = [Channel(code='BHZ',
                                location_code='',
                                latitude=0.0,
                                longitude=0.0,
                                elevation=0.0,
                                depth=0.0,
                                response=responseN1S2)]
        channelsN2S1 = [Channel(code='BHZ',
                                location_code='',
                                latitude=0.0,
                                longitude=0.0,
                                elevation=0.0,
                                depth=0.0,
                                response=responseN2S1)]
        stations1 = [Station(code='N1S1',
                             latitude=0.0,
                             longitude=0.0,
                             elevation=0.0,
                             channels=channelsN1S1),
                     Station(code='N1S2',
                             latitude=0.0,
                             longitude=0.0,
                             elevation=0.0,
                             channels=channelsN1S2)]
        stations2 = [Station(code='N2S1',
                             latitude=0.0,
                             longitude=0.0,
                             elevation=0.0,
                             channels=channelsN2S1)]
        networks = [Network('N1', stations=stations1),
                    Network('N2', stations=stations2)]
        inv = Inventory(networks=networks, source='TEST')

        response = inv.get_response('N1.N1S1..BHZ',
                                    UTCDateTime('2010-01-01T12:00'))
        self.assertEqual(response, responseN1S1)
        response = inv.get_response('N1.N1S2..BHZ',
                                    UTCDateTime('2010-01-01T12:00'))
        self.assertEqual(response, responseN1S2)
        response = inv.get_response('N2.N2S1..BHZ',
                                    UTCDateTime('2010-01-01T12:00'))
        self.assertEqual(response, responseN2S1)


def suite():
    return unittest.makeSuite(InventoryTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
