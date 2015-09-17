# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA @UnusedWildImport

import os
import io
import unittest
from obspy.core.event import read_events
from obspy.io.nied.core import _is_nied_catalog
import numpy as np


class NIEDCatalogReadingTestCase(unittest.TestCase):
    """
    Test everything related to the reading an NIED moment tensor catalog.
    """
    def setUp(self):
        # Directory where the test files are located
        self.path = os.path.dirname(__file__)

    def test_read_nied_catalog(self):
        testfile = os.path.join(self.path, 'data', 'NIEDCATALOG')
        cat = read_events(testfile, 'NIED')
        self.assertEqual(len(cat), 1)
        ev = cat[0]
        self.assertEqual(len(ev.origins), 2)
        self.assertEqual(len(ev.magnitudes), 2)

    def test_read_nied_catalog_from_open_files(self):
        """
        Tests that reading an NIED moment tensor file from an open file works.
        """
        testfile = os.path.join(self.path, 'data', 'NIEDCATALOG')
        with open(testfile, "rb") as fh:
            cat = read_events(fh)

    def test_read_nied_catalog_from_bytes_io(self):
        """
        Tests that reading an NIED moment tensor file from an BytesIO objects
        works.
        """
        testfile = os.path.join(self.path, 'data', 'NIEDCATALOG')
        with open(testfile, "rb") as fh:
            buf = io.BytesIO(fh.read())

        with buf:
            cat = read_events(buf)

    def test_is_nied_catalog(self):
        """
        This tests the _is_nied_catalog method by validating that each file in
        the data directory is a nied catalog file and each file in the working
        directory is not.

        The filenames are hard coded so the test will not fail with future
        changes in the structure of the package.
        """
        # NIED catalog file names.
        nied_filenames = ['NIEDCATALOG']

        # Non NIED file names.
        non_nied_filenames = ['test_core.py',
                               '__init__.py']
        # Loop over NIED files
        for _i in nied_filenames:
            filename = os.path.join(self.path, 'data', _i)
            is_nied = _is_nied_catalog(filename)
            self.assertTrue(is_nied)
        # Loop over non NIED files
        for _i in non_nied_filenames:
            filename = os.path.join(self.path, _i)
            is_nied = _is_nied_catalog(filename)
            self.assertFalse(is_nied)

def suite():
    return unittest.makeSuite(NIEDCatalogReadingTestCase, 'test')

if __name__ == '__main__':
    unittest.main(defaultTest='suite')
