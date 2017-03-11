# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA @UnusedWildImport

import unittest

import requests

from vcr import vcr


class RequestsTestCase(unittest.TestCase):
    """
    Test suite using requests
    """
    def test_connectivity(self):
        # basic network connection test to exclude network issues
        r = requests.get('https://www.python.org/')
        self.assertEqual(r.status_code, 200)

    @vcr
    def test_captured_requests_http(self):
        r = requests.get('http://httpstat.us/')
        self.assertEqual(r.status_code, 200)

    @vcr
    def test_captured_requests_https(self):
        r = requests.get('https://www.python.org/')
        self.assertEqual(r.status_code, 200)


if __name__ == '__main__':
    unittest.main()
