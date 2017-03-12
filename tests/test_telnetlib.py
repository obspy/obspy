# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA @UnusedWildImport
from future.utils import PY2

import os
import telnetlib
import unittest
from unittest import skipIf

from vcr import vcr


class TelnetlibTestCase(unittest.TestCase):
    """
    Test suite using telnetlib
    """
    def setUp(self):
        # Directory where the test files are located
        self.path = os.path.join(os.path.dirname(__file__), 'vcrtapes')

    def tearDown(self):
        # cleanup temporary files
        tempfile = os.path.join(self.path,
                                'test_telnetlib.test_arclink_recording.vcr')
        try:
            os.remove(tempfile)
        except OSError:
            pass

    @vcr
    def test_arclink(self):
        t = telnetlib.Telnet("webdc.eu", 18002, 20)
        t.write(b'HELLO\r\n')
        # read until ")\r\n" - this fails in linux
        out = t.read_until(b')\r\n', timeout=1)
        self.assertIn(b'ArcLink', out)
        self.assertIn(b')', out)
        out = t.read_until(b'\r\n', timeout=1)
        self.assertIn(b'GFZ', out)
        t.write(b'USER test@obspy.org\r\n')
        out = t.read_until(b'\r\n', timeout=1)
        self.assertEqual(out, b'OK\r\n')
        t.write(b'INSTITUTION Anonymous\r\n')
        out = t.read_until(b'OK\r\n', timeout=1)
        self.assertEqual(out, b'OK\r\n')
        t.close()

    @vcr
    def test_read_until_only_linesep(self):
        """
        use Telnet.read_until only with line separator but no extra string
        """
        t = telnetlib.Telnet("webdc.eu", 18002, 20)
        t.write(b'HELLO\r\n')
        out = t.read_until(b'\r\n', timeout=1)
        self.assertIn(b'ArcLink', out)
        out = t.read_until(b'\r\n', timeout=1)
        self.assertIn(b'GFZ', out)
        t.write(b'USER test@obspy.org\r\n')
        out = t.read_until(b'\r\n', timeout=1)
        self.assertEqual(out, b'OK\r\n')
        t.write(b'INSTITUTION Anonymous\r\n')
        out = t.read_until(b'\r\n', timeout=1)
        self.assertEqual(out, b'OK\r\n')
        t.close()

    @skipIf(PY2, 'recording in PY2 is not supported')
    @vcr(overwrite=True)
    def test_arclink_recording(self):
        t = telnetlib.Telnet("webdc.eu", 18002, 20)
        t.write(b'HELLO\r\n')
        out = t.read_until(b')\r\n', timeout=1)
        self.assertIn(b'ArcLink', out)
        out = t.read_until(b'\r\n', timeout=1)
        self.assertIn(b'GFZ', out)
        t.write(b'USER test@obspy.org\r\n')
        out = t.read_until(b'\r\n', timeout=1)
        self.assertEqual(out, b'OK\r\n')
        t.write(b'INSTITUTION Anonymous\r\n')
        out = t.read_until(b'OK\r\n', timeout=1)
        self.assertEqual(out, b'OK\r\n')
        t.close()


if __name__ == '__main__':
    unittest.main()
