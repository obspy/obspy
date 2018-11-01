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

from obspy import read_events
from obspy.core.event.dictionary import catalog_to_dict, dict_to_catalog
from obspy.core.util.misc import _yield_obj_parent_attr
from obspy.core.util.testing import (
    create_diverse_catalog, read_test_datasets)


class TestCatalogToDict(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """
        Create input catalogs, dict list, and output catalogs.
        """
        cls.in_catalogs = [read_events(), create_diverse_catalog()]
        cls.in_catalogs += read_test_datasets(group="event")
        cls.catalog_dicts = [catalog_to_dict(x) for x in cls.in_catalogs]
        cls.out_catalogs = [dict_to_catalog(x) for x in cls.catalog_dicts]

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
