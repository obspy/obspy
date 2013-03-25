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
import unittest

import obspy
from obspy.station import SeismicInventory


class StationXMLTestCase(unittest.TestCase):
    """
    """
    def test_writing_simple_file(self):
        """
        Test that writing the most basic StationXML document possible works.
        """
        pass


def suite():
    return unittest.makeSuite(StationXMLTestCase,  'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
