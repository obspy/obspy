# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA @UnusedWildImport

import os
import unittest
from obspy.core.event import read_events
import numpy as np

class NIEDCatalogReadingTestCase(unittest.TestCase):
    """
    Test everything related to the reading an NIED moment tensor catalog.
    """
    def setUp(self):
        # Directory where the test files are located
        self.path = os.path.dirname(__file__)

    def test_read_nied_catalog(self):
        testfile = os.path.join(self.path, 'data', 'tohoku.txt')
        cat = read_events(testfile, 'NIED')
        self.assertEqual(len(cat), 1)
        ev = cat[0]
        self.assertEqual(len(ev.origins), 2)
        self.assertEqual(len(ev.magnitudes), 2)


    def test_is_nied_catalog(self):
        pass


def suite():
    return unittest.makeSuite(NIEDCatalogReadingTestCase, 'test')

if __name__ == '__main__':
    unittest.main(defaultTest='suite')
