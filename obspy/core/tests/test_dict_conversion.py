"""
Testing utilities for ObsPy.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import unittest
from glob import glob
from os.path import join, dirname

import obspy
from obspy import read_events
from obspy.core.event.dict import catalog_to_dict, dict_to_catalog
from obspy.core.util.misc import _yield_obj_parent_attr
from obspy.core.util.testing import create_diverse_catalog, WarningsCapture


def load_test_quakeml():
    """
    Load the valid quakeml files from the test module.

    :return: list of catalogs.
    """
    base = dirname(obspy.__file__)
    test_data = join(base, 'io', 'quakeml', 'tests', 'data')
    with WarningsCapture():
        return [read_events(x, 'quakeml') for x in glob(join(test_data, '*'))]


class TestCatalogToDict(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        """
        Create input catalogs, dict list, and output catalogs.
        """
        self.in_catalogs = [read_events(), create_diverse_catalog()]
        self.in_catalogs += load_test_quakeml()  # load quakeml catalogs
        self.catalog_dicts = [catalog_to_dict(x) for x in self.in_catalogs]
        self.out_catalogs = [dict_to_catalog(x) for x in self.catalog_dicts]

    def test_roundtrip_conversion(self):
        """
        Any catalog converted to a dict, then converted back to a catalog,
        should be equal to the original catalog.
        """
        for cat1, cat2 in zip(self.in_catalogs, self.out_catalogs):
            self.assertEqual(cat1, cat2)

    def test_dict_types(self):
        """
        Ensure only basic types are in the catalog dictionaries.
        """
        basic_types = (dict, list, int, str, float, tuple)
        for cat_dict in self.catalog_dicts:
            self.assertIsInstance(cat_dict, dict)
            for obj, parent, attr in _yield_obj_parent_attr(cat_dict):
                self.assertTrue(isinstance(obj, basic_types) or obj is None)


def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestCatalogToDict, 'test'))
    return suite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
