#!/usr/bin/env python
# -*- coding: utf-8 -*-
from obspy import readEvents
from obspy.ndk.core import read_ndk

import os
import unittest


class NDKTestCase(unittest.TestCase):
    """
    Test suite for obspy.ndk
    """
    def setUp(self):
        self.datapath = os.path.join(os.path.dirname(__file__), 'data')

    def test_read_single_ndk(self):
        """
        Test reading a single event from and NDK file and comparing it to a
        QuakeML file that has been manually checked to contain all the
        information in the NDK file.
        """
        filename = os.path.join(self.datapath, "C200604092050A.ndk")
        cat = read_ndk(filename)

        reference = os.path.join(self.datapath, "C200604092050A.xml")
        ref_cat = readEvents(reference)

        self.assertEqual(cat, ref_cat)


def suite():
    return unittest.makeSuite(NDKTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
