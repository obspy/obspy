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

import obspy
from obspy.core.event.dict import catalog_to_dict, dict_to_catalog
from obspy.core.util.testing import create_diverse_catalog

TEST_CATALOGS = [obspy.read_events(), create_diverse_catalog()]


class TestCatalogToDict(unittest.TestCase):
    """ class to test json """

    # tests
    def test_roundtrip_conversion(self):
        """
        Any catalog converted to a dict, then converted back to a catalog,
        should be equal to the original catalog.
        """
        for cat in TEST_CATALOGS:
            cat_dict = catalog_to_dict(cat)
            cat2 = dict_to_catalog(cat_dict)
            self.assertEqual(cat, cat2)


def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestCatalogToDict, 'test'))
    return suite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
