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
import io
import os
import unittest

from obspy.core.inventory import read_inventory
from obspy.core.inventory.response import (CoefficientsTypeResponseStage,
                                           FIRResponseStage)


class SC3MLTestCase(unittest.TestCase):

    def setUp(self):
        """
        Read example stationXML/sc3ml format to Inventory
        """
        self.data_dir = os.path.join(os.path.dirname(os.path.abspath(
            inspect.getfile(inspect.currentframe()))), "data")
        self.arclink_xml_path = os.path.join(self.data_dir, "arclink_inventory.xml")

    def test_auto_read_arclink_xml(self):
        inv = read_inventory(self.arclink_xml_path)
        self.assertIsNotNone(inv)

    def testsPassSoItMustBeGood(self):
        self.assertTrue(True)


def suite():
    return unittest.makeSuite(SC3MLTestCase, "test")

if __name__ == '__main__':
    unittest.main(defaultTest='suite')
