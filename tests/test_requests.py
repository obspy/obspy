# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA @UnusedWildImport

import json
import tempfile
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
    def test_http_get(self):
        r = requests.get('http://httpbin.org/status/200')
        self.assertEqual(r.status_code, 200)

    @vcr
    def test_http_post(self):
        payload = dict(key1='value1', key2='value2')
        r = requests.post('http://httpbin.org/post', data=payload)
        out = json.loads(r.text)
        self.assertEqual(out['form'], {'key1': 'value1', 'key2': 'value2'})

    @vcr
    def test_http_post_file(self):
        with tempfile.TemporaryFile(mode='wb+') as file:
            file.write(b'test123')
            file.seek(0)
            files = {'file': file}
            r = requests.post('http://httpbin.org/post', files=files)
        out = json.loads(r.text)
        self.assertEqual(out['files']['file'], 'test123')

    @vcr
    def test_cookies(self):
        cookies = dict(cookies_are='working')
        r = requests.get('http://httpbin.org/cookies', cookies=cookies)
        out = json.loads(r.text)
        self.assertEqual(out['cookies'], {"cookies_are": "working"})

    @vcr
    def test_cookie_jar(self):
        jar = requests.cookies.RequestsCookieJar()
        jar.set('tasty_cookie', 'yum', domain='httpbin.org', path='/cookies')
        jar.set('gross_cookie', 'blech', domain='httpbin.org', path='/null')
        r = requests.get('http://httpbin.org/cookies', cookies=jar)
        out = json.loads(r.text)
        self.assertEqual(out['cookies'], {"tasty_cookie": "yum"})

    @vcr
    def test_https_get(self):
        r = requests.get('https://www.python.org/')
        self.assertEqual(r.status_code, 200)

    @vcr
    def test_allow_redirects_false(self):
        # 1
        r = requests.get('http://github.com/', allow_redirects=False)
        self.assertEqual(r.status_code, 301)  # Moved Permanently
        self.assertEqual(r.url, 'http://github.com/')
        self.assertEqual(r.headers['Location'], 'https://github.com/')
        # 2
        r = requests.get('http://obspy.org/', allow_redirects=False)
        self.assertEqual(r.status_code, 302)  # Found (Moved Temporarily)
        self.assertEqual(r.url, 'http://obspy.org/')
        self.assertEqual(r.headers['Location'],
                         'https://github.com/obspy/obspy/wiki/')

    @vcr
    def test_redirect(self):
        r = requests.get('http://github.com/')
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.url, 'https://github.com/')
        self.assertEqual(len(r.history), 1)

    @vcr
    def test_sessions(self):
        s = requests.Session()
        s.get('http://httpbin.org/cookies/set/sessioncookie/123456789')
        r = s.get('http://httpbin.org/cookies')
        out = json.loads(r.text)
        self.assertEqual(out['cookies'], {"sessioncookie": "123456789"})

    @vcr
    def test_sessions2(self):
        s = requests.Session()
        s.auth = ('user', 'pass')
        s.headers.update({'x-test': 'true'})
        r = s.get('http://httpbin.org/headers', headers={'x-test2': 'true'})
        out = json.loads(r.text)
        self.assertEqual(out['headers']['X-Test'], 'true')
        self.assertEqual(out['headers']['X-Test2'], 'true')

    @vcr
    def test_redirect_twice(self):
        # http://obspy.org redirects to https://github.com/obspy/obspy/wiki
        r = requests.get('http://obspy.org/')
        self.assertEqual(r.status_code, 200)
        self.assertEqual(len(r.history), 2)


if __name__ == '__main__':
    unittest.main()
