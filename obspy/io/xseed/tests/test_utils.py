# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA @UnusedWildImport

import unittest

from obspy import UTCDateTime
from obspy.xseed.utils import DateTime2String, toTag


class UtilsTestCase(unittest.TestCase):
    """
    Utils test suite.
    """
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_toTag(self):
        name = "Hello World"
        self.assertEqual("hello_world", toTag(name))

    def test_DateTime2String(self):
        dt = UTCDateTime(2008, 12, 23, 0o1, 30, 22, 123456)
        self.assertEqual(DateTime2String(dt), "2008,358,01:30:22.1234")
        dt = UTCDateTime(2008, 12, 23, 0o1, 30, 22, 98765)
        self.assertEqual(DateTime2String(dt), "2008,358,01:30:22.0987")
        dt = UTCDateTime(2008, 12, 23, 0o1, 30, 22, 1234)
        self.assertEqual(DateTime2String(dt), "2008,358,01:30:22.0012")
        dt = UTCDateTime(2008, 12, 23, 0o1, 30, 22, 123)
        self.assertEqual(DateTime2String(dt), "2008,358,01:30:22.0001")
        dt = UTCDateTime(2008, 12, 23, 0o1, 30, 22, 9)
        self.assertEqual(DateTime2String(dt), "2008,358,01:30:22.0000")
        dt = UTCDateTime(2008, 12, 23, 0o1, 30, 21)
        self.assertEqual(DateTime2String(dt), "2008,358,01:30:21.0000")
        dt = UTCDateTime(2008, 12, 23, 0o1, 0, 0, 0)
        self.assertEqual(DateTime2String(dt), "2008,358,01:00:00.0000")
        dt = UTCDateTime(2008, 12, 23)
        self.assertEqual(DateTime2String(dt), "2008,358")

    def test_DateTime2StringCompact(self):
        dt = UTCDateTime(2008, 12, 23, 0o1, 30, 22, 123456)
        self.assertEqual(DateTime2String(dt, True),
                         "2008,358,01:30:22.1234")
        dt = UTCDateTime(2008, 12, 23, 0o1, 30, 22)
        self.assertEqual(DateTime2String(dt, True), "2008,358,01:30:22")
        dt = UTCDateTime(2008, 12, 23, 0o1, 30)
        self.assertEqual(DateTime2String(dt, True), "2008,358,01:30")
        dt = UTCDateTime(2008, 12, 23, 0o1)
        self.assertEqual(DateTime2String(dt, True), "2008,358,01")
        dt = UTCDateTime(2008, 12, 23)
        self.assertEqual(DateTime2String(dt, True), "2008,358")


def suite():
    return unittest.makeSuite(UtilsTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
