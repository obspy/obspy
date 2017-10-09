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
            c.include_providers,
            ["http://service.iris.edu", "http://example.com"])
        self.assertEqual(
            c.exclude_providers,
            ["http://eida.bgr.de", "http://example2.com"])

        # None are set.
        c = self._cls_object()
        self.assertEqual(c.include_providers, [])
        self.assertEqual(c.exclude_providers, [])

        # Single strings.
        c = self._cls_object(include_providers="IRIS",
                             exclude_providers="BGR")
        self.assertEqual(c.include_providers, ["http://service.iris.edu"])
        self.assertEqual(c.exclude_providers, ["http://eida.bgr.de"])

        c = self._cls_object(include_providers="http://example.com",
                             exclude_providers="http://example2.com")
        self.assertEqual(c.include_providers, ["http://example.com"])
        self.assertEqual(c.exclude_providers, ["http://example2.com"])


def suite():
    return unittest.makeSuite(BaseRoutingClientTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
