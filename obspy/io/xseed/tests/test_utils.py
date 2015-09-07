# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA @UnusedWildImport

import glob
import inspect
import os
import unittest

from obspy import UTCDateTime
from obspy.io.xseed.utils import datetime_2_string, to_tag, is_RESP


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
        self.assertEqual("hello_world", to_tag(name))

    def test_DateTime2String(self):
        dt = UTCDateTime(2008, 12, 23, 0o1, 30, 22, 123456)
        self.assertEqual(datetime_2_string(dt), "2008,358,01:30:22.1234")
        dt = UTCDateTime(2008, 12, 23, 0o1, 30, 22, 98765)
        self.assertEqual(datetime_2_string(dt), "2008,358,01:30:22.0987")
        dt = UTCDateTime(2008, 12, 23, 0o1, 30, 22, 1234)
        self.assertEqual(datetime_2_string(dt), "2008,358,01:30:22.0012")
        dt = UTCDateTime(2008, 12, 23, 0o1, 30, 22, 123)
        self.assertEqual(datetime_2_string(dt), "2008,358,01:30:22.0001")
        dt = UTCDateTime(2008, 12, 23, 0o1, 30, 22, 9)
        self.assertEqual(datetime_2_string(dt), "2008,358,01:30:22.0000")
        dt = UTCDateTime(2008, 12, 23, 0o1, 30, 21)
        self.assertEqual(datetime_2_string(dt), "2008,358,01:30:21.0000")
        dt = UTCDateTime(2008, 12, 23, 0o1, 0, 0, 0)
        self.assertEqual(datetime_2_string(dt), "2008,358,01:00:00.0000")
        dt = UTCDateTime(2008, 12, 23)
        self.assertEqual(datetime_2_string(dt), "2008,358")

    def test_DateTime2StringCompact(self):
        dt = UTCDateTime(2008, 12, 23, 0o1, 30, 22, 123456)
        self.assertEqual(datetime_2_string(dt, True),
                         "2008,358,01:30:22.1234")
        dt = UTCDateTime(2008, 12, 23, 0o1, 30, 22)
        self.assertEqual(datetime_2_string(dt, True), "2008,358,01:30:22")
        dt = UTCDateTime(2008, 12, 23, 0o1, 30)
        self.assertEqual(datetime_2_string(dt, True), "2008,358,01:30")
        dt = UTCDateTime(2008, 12, 23, 0o1)
        self.assertEqual(datetime_2_string(dt, True), "2008,358,01")
        dt = UTCDateTime(2008, 12, 23)
        self.assertEqual(datetime_2_string(dt, True), "2008,358")

    def test_is_RESP(self):
        """
        Checks is_RESP() routine on all files in signal/tests/data.
        """
        signal_test_files = os.path.abspath(os.path.join(
            inspect.getfile(inspect.currentframe()),
            os.path.pardir, os.path.pardir, os.path.pardir, os.path.pardir,
            "signal", "tests", "data", "*"))
        # List of files that are actually RESP files, all other files are
        # considered non-RESP files
        resp_filenames = [
            "IUANMO.resp",
            "RESP.NZ.CRLZ.10.HHZ",
            "RESP.NZ.CRLZ.10.HHZ.mac",
            "RESP.NZ.CRLZ.10.HHZ.windows",
            "RESP.OB.AAA._.BH_",
            ]
        for filename in glob.glob(signal_test_files):
            got = is_RESP(filename)
            expected = os.path.basename(filename) in resp_filenames
            self.assertEqual(
                got, expected,
                "is_RESP() returns %s for file %s" % (got, filename))


def suite():
    return unittest.makeSuite(UtilsTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
