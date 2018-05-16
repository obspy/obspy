# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA @UnusedWildImport

import os
import unittest

from obspy import read_events, Catalog
from obspy.core.event import Event
from obspy.io.focmec.core import _is_focmec, _read_focmec


class FOCMECTestCase(unittest.TestCase):
    """
    Test everything related to reading FOCMEC files
    """
    def setUp(self):
        # Directory where the test files are located
        self.path = os.path.dirname(__file__)
        self.lst_file = os.path.join(self.path, 'data', 'focmec_8sta.lst')
        self.out_file = os.path.join(self.path, 'data', 'focmec_8sta.out')

    def _assert_cat_common_parts(self, cat):
        self.assertTrue(isinstance(cat, Catalog))
        self.assertEqual(len(cat), 1)
        event = cat[0]
        self.assertTrue(isinstance(event, Event))
        self.assertEqual(len(event.focal_mechanisms), 4)

    def _assert_cat_out(self, cat):
        self._assert_cat_common_parts(cat)

    def _assert_cat_lst(self, cat):
        self._assert_cat_common_parts(cat)

    def test_is_focmec(self):
        for file_ in (self.lst_file, self.out_file):
            self.assertTrue(_is_focmec(file_))

    def test_read_focmec_out(self):
        cat = _read_focmec(self.out_file)
        self._assert_cat_out(cat)

    def test_read_focmec_lst(self):
        cat = _read_focmec(self.lst_file)
        self._assert_cat_lst(cat)

    def test_read_focmec_out_through_plugin(self):
        cat = read_events(self.out_file)
        self._assert_cat_out(cat)

    def test_read_focmec_lst_through_plugin(self):
        cat = read_events(self.lst_file)
        self._assert_cat_lst(cat)


def suite():
    return unittest.makeSuite(FOCMECTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
