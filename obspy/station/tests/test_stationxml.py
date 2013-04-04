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
import inspect
import os
import unittest

import obspy.station


class StationXMLTestCase(unittest.TestCase):
    """
    """
    def setUp(self):
        # Most generic way to get the actual data directory.
        self.data_dir = os.path.join(os.path.dirname(os.path.abspath(
            inspect.getfile(inspect.currentframe()))), "data")

    def test_read_and_write_minimal_file(self):
        """
        Test that writing the most basic StationXML document possible works.
        """
        filename = os.path.join(self.data_dir, "minimal_station.xml")
        inv = obspy.station.readInventory(filename)

def suite():
    return unittest.makeSuite(StationXMLTestCase,  'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
