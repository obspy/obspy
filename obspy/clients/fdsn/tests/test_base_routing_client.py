#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import collections
import unittest

from obspy.clients.fdsn.routing.routing_client import BaseRoutingClient


_DummyResponse = collections.namedtuple("_DummyResponse", ["content"])


class BaseRoutingClientTestCase(unittest.TestCase):
    def setUp(self):
        # Need to inherit to add two required method by the abstract base
        # class.
        class _DummyBaseRoutingClient(BaseRoutingClient):  # pragma: no cover
            def get_service_version(self):
                """
                Return a semantic version number of the remote service as a string.
                """
                return "0.0.0"

            def _handle_requests_http_error(self, r):
                raise NotImplementedError

        self._cls_object = _DummyBaseRoutingClient
        self._cls = ("obspy.clients.fdsn.routing.routing_client."
                     "BaseRoutingClient")

    def test_expansion_of_include_and_exclude_providers(self):
        c = self._cls_object(
            include_providers=["IRIS", "http://example.com"],
            exclude_providers=["BGR", "http://example2.com"])
        self.assertEqual(
            c.include_providers, ["service.iris.edu", "example.com"])
        self.assertEqual(
            c.exclude_providers, ["eida.bgr.de", "example2.com"])

        # None are set.
        c = self._cls_object()
        self.assertEqual(c.include_providers, [])
        self.assertEqual(c.exclude_providers, [])

        # Single strings.
        c = self._cls_object(include_providers="IRIS",
                             exclude_providers="BGR")
        self.assertEqual(c.include_providers, ["service.iris.edu"])
        self.assertEqual(c.exclude_providers, ["eida.bgr.de"])

        c = self._cls_object(include_providers="http://example.com/path",
                             exclude_providers="http://example2.com")
        self.assertEqual(c.include_providers, ["example.com/path"])
        self.assertEqual(c.exclude_providers, ["example2.com"])

    def test_request_filtering(self):
        split = {
            # Note that this is HTTPS.
            "https://example.com": "1234",
            "http://example2.com": "1234",
            "http://example3.com": "1234",
            "http://service.iris.edu": "1234"
        }

        c = self._cls_object(include_providers=["IRIS", "http://example.com"])
        self.assertEqual(c._filter_requests(split), {
            "https://example.com": "1234",
            "http://service.iris.edu": "1234"
        })

        c = self._cls_object(exclude_providers=["IRIS", "http://example.com"])
        self.assertEqual(c._filter_requests(split), {
            "http://example2.com": "1234",
            "http://example3.com": "1234"
        })

        # Both filters are always applied - it might result in zero
        # remaining providers.
        c = self._cls_object(include_providers=["IRIS", "http://example.com"],
                             exclude_providers=["IRIS", "http://example.com"])
        self.assertEqual(c._filter_requests(split), {})


def suite():
    return unittest.makeSuite(BaseRoutingClientTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
