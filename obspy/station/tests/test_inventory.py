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
import unittest

import obspy
from obspy.station import Inventory


class InventoryTestCase(unittest.TestCase):
    """
    Tests the for :class:`~obspy.station.inventory.Inventory` class.
    """
    def test_initialization(self):
        """
        Some simple sanity tests.
        """
        dt = obspy.UTCDateTime()
        inv = Inventory(source="TEST", networks=[])
        # If no time is given, the creation time should be set to the current
        # time. Use a large offset for potentially slow computers and test
        # runs.
        self.assertTrue(inv.created - dt <= 10.0)


def suite():
    return unittest.makeSuite(InventoryTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
