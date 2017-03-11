# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA @UnusedWildImport
from future.standard_library import hooks

import io
import os
import unittest

from vcr import vcr
from vcr.utils import stdout_redirector


with hooks():
    from urllib.request import urlopen


class CoreTestCase(unittest.TestCase):
    """
    Test suite for vcr
    """
    def setUp(self):
        # Directory where the test files are located
        self.path = os.path.join(os.path.dirname(__file__), 'vcrtapes')
        self.temp_test_vcr = os.path.join(self.path, 'test_core.temp_test.vcr')
        self.read_test_vcr = os.path.join(self.path, 'test_core.read_test.vcr')

    def tearDown(self):
        # cleanup
        try:
            os.remove(self.temp_test_vcr)
        except OSError:
            pass

    def test_connectivity(self):
        # basic network connection test to exclude network issues
        r = urlopen('https://www.python.org/')
        self.assertEqual(r.status, 200)

    def test_playback(self):
        # define function with decorator
        @vcr
        def read_test():
            r = urlopen('https://www.python.org/')
            self.assertEqual(r.status, 200)

        # run the test
        capture = io.StringIO()
        with stdout_redirector(capture):
            read_test()
        self.assertEqual(capture.getvalue(), '')

    def test_playback_with_debug(self):
        # define function with decorator
        @vcr(debug=True)
        def read_test():
            r = urlopen('https://www.python.org/')
            self.assertEqual(r.status, 200)

        # run the test
        capture = io.StringIO()
        with stdout_redirector(capture):
            read_test()
        self.assertIn('VCR PLAYBACK', capture.getvalue())

    def test_playback_without_debug(self):
        # define function with decorator
        @vcr
        def read_test():
            r = urlopen('https://www.python.org/')
            self.assertEqual(r.status, 200)

        # run the test
        capture = io.StringIO()
        with stdout_redirector(capture):
            read_test()
        self.assertEqual(capture.getvalue(), '')

    def test_life_cycle(self):
        # define function with @vcr decorator and enable debug mode
        @vcr(debug=True)
        def temp_test():
            r = urlopen('https://www.python.org/')
            self.assertEqual(r.status, 200)

        # an initial run of our little test will start in recording mode
        # and auto-generate a .vcr file - however, this file shouldn't exist at
        # the moment
        self.assertFalse(os.path.exists(self.temp_test_vcr))

        # run the test
        capture = io.StringIO()
        with stdout_redirector(capture):
            temp_test()
        # debug mode should state its in recording mode
        self.assertIn('VCR RECORDING', capture.getvalue())

        # now the .vcr file should exist
        self.assertTrue(os.path.exists(self.temp_test_vcr))

        # re-run the test - this time it should be using the recorded file
        capture = io.StringIO()
        with stdout_redirector(capture):
            temp_test()
        # debug mode should state its in playback mode
        self.assertIn('VCR PLAYBACK', capture.getvalue())


if __name__ == '__main__':
    unittest.main()
