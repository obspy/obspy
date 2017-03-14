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
        conn.request("GET", "/")
        response = conn.getresponse()
        self.assertEqual(response.status, 200)
        self.assertEqual(response.reason, 'OK')
        conn.close()

    @vcr
    def test_http_get(self):
        conn = http.client.HTTPConnection("www.python.org")
        conn.request("GET", "/")
        response = conn.getresponse()
        self.assertEqual(response.status, 301)
        self.assertEqual(response.reason, 'Moved Permanently')

        conn.close()

    @vcr
    def test_http_get_invalid(self):
        conn = http.client.HTTPConnection("httpstat.us")
        conn.request("GET", "/404")
        response = conn.getresponse()
        self.assertEqual(response.status, 404)
        self.assertEqual(response.reason, 'Not Found')
        conn.close()

    @vcr
    def test_https_get(self):
        conn = http.client.HTTPSConnection("www.python.org")
        conn.request("GET", "/")
        response = conn.getresponse()
        self.assertEqual(response.status, 200)
        self.assertEqual(response.reason, 'OK')
        conn.close()

    @vcr
    def test_https_head(self):
        conn = http.client.HTTPSConnection("www.python.org")
        conn.request("HEAD", "/")
        response = conn.getresponse()
        self.assertEqual(response.status, 200)
        self.assertEqual(response.reason, 'OK')
        data = response.read()
        self.assertEqual(len(data), 0)
        self.assertEqual(data, b'')
        conn.close()

    @vcr
    def test_redirect_to_https(self):
        conn = http.client.HTTPSConnection("obspy.org")
        conn.request("GET", "/")
        response = conn.getresponse()
        self.assertEqual(response.status, 302)
        self.assertEqual(response.reason, 'Moved Temporarily')
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
        self.assertEqual(response.reason, 'Found')
        data = response.read()
        self.assertIn(b'Redirecting', data)
        conn.close()


if __name__ == '__main__':
    unittest.main()
