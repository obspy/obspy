# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA @UnusedWildImport
from future.standard_library import hooks

import unittest

from vcr import vcr

with hooks():
    import urllib.parse
    import http.client


class RequestsTestCase(unittest.TestCase):
    """
    Test suite using requests
    """
    def test_connectivity(self):
        # basic network connection test to exclude network issues
        conn = http.client.HTTPSConnection("www.python.org")
        conn.request("HEAD", "/")
        response = conn.getresponse()
        self.assertEqual(response.status, 200)
        conn.close()

    @vcr
    def test_http_get(self):
        conn = http.client.HTTPConnection("www.python.org")
        conn.request("HEAD", "/")
        response = conn.getresponse()
        self.assertEqual(response.status, 301)
        conn.close()

    @vcr
    def test_https_get(self):
        conn = http.client.HTTPSConnection("www.python.org")
        conn.request("HEAD", "/")
        response = conn.getresponse()
        self.assertEqual(response.status, 200)
        conn.close()

    @vcr
    def test_redirect_to_https(self):
        conn = http.client.HTTPSConnection("obspy.org")
        conn.request("HEAD", "/")
        response = conn.getresponse()
        self.assertEqual(response.status, 302)
        conn.close()

    @vcr
    def test_http_post(self):
        params = urllib.parse.urlencode({'@number': 12524, '@type': 'issue',
                                         '@action': 'show'})
        headers = {"Content-type": "application/x-www-form-urlencoded",
                   "Accept": "text/plain"}
        conn = http.client.HTTPConnection("bugs.python.org")
        conn.request("POST", "", params, headers)
        response = conn.getresponse()
        self.assertEqual(response.status, 302)
        data = response.read()
        self.assertIn(b'Redirecting', data)
        conn.close()


if __name__ == '__main__':
    unittest.main()
