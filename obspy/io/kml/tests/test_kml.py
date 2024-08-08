# -*- coding: utf-8 -*-
from obspy import read_inventory, read_events
from obspy.core.util import NamedTemporaryFile
from obspy.core.util.testing import compare_xml_strings


class TestKML():
    """
    Test KML output of Inventory/Catalog.
    """
    def test_write_inventory(self, testdata):
        """
        Test writing Inventory to KML.
        """
        # write the example inventory to KML and read it into a string
        inv = read_inventory()
        with NamedTemporaryFile(suffix=".kml") as tf:
            inv.write(tf.name, format="KML")
            with open(tf.name, "rb") as fh:
                got = fh.read()
        # read expected result into string
        filename = testdata['inventory.kml']
        with open(filename, "rb") as fh:
            expected = fh.read()
        # compare the two
        compare_xml_strings(expected, got)

    def test_write_catalog(self, testdata):
        """
        Test writing Catalog to KML.
        """
        # write the example catalog to KML and read it into a string
        cat = read_events()
        with NamedTemporaryFile(suffix=".kml") as tf:
            cat.write(tf.name, format="KML")
            with open(tf.name, "rb") as fh:
                got = fh.read()
        # read expected result into string
        filename = testdata['catalog.kml']
        with open(filename, "rb") as fh:
            expected = fh.read()
        # compare the two
        compare_xml_strings(expected, got)
