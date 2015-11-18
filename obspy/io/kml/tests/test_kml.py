# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA
from future.utils import native_str

import os
import unittest

from obspy import read_inventory, read_events
from obspy.core.util import NamedTemporaryFile
from obspy.core.util.testing import compare_xml_strings


class KMLTestCase(unittest.TestCase):
    """
    Test KML output of Inventory/Catalog.
    """
    def setUp(self):
        # directory where the test files are located
        self.path = os.path.join(os.path.dirname(__file__), 'data')

    def test_write_Inventory(self):
        """
        Test writing Inventory to KML.
        """
        # write the example inventory to KML and read it into a string
        inv = read_inventory()
        with NamedTemporaryFile(suffix=".kml") as tf:
            inv.write(native_str(tf.name), format="KML")
            with open(tf.name, "rb") as fh:
                got = fh.read()
        # read expected result into string
        filename = os.path.join(self.path, 'inventory.kml')
        with open(filename, "rb") as fh:
            expected = fh.read()
        # compare the two
        compare_xml_strings(expected, got)

    def test_write_Catalog(self):
        """
        Test writing Catalog to KML.
        """
        # write the example catalog to KML and read it into a string
        cat = read_events()
        with NamedTemporaryFile(suffix=".kml") as tf:
            cat.write(native_str(tf.name), format="KML")
            with open(tf.name, "rb") as fh:
                got = fh.read()
        # read expected result into string
        filename = os.path.join(self.path, 'catalog.kml')
        with open(filename, "rb") as fh:
            expected = fh.read()
        # compare the two
        compare_xml_strings(expected, got)


def suite():
    return unittest.makeSuite(KMLTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
