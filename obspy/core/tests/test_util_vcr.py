# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import unittest

import requests

from obspy.core.util.vcr import vcr


class VCRTestCase(unittest.TestCase):
    """
    Test suite for obspy.core.util.vcr
    """

    @vcr
    def test_http(self):
        r = requests.get('http://tests.obspy.org/')
        self.assertEqual(r.status_code, 200)

    @vcr
    def test_https(self):
        r = requests.get('https://tests.obspy.org/')
        self.assertEqual(r.status_code, 200)


def suite():
    return unittest.makeSuite(VCRTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
