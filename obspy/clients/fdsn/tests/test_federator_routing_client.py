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
from distutils.version import LooseVersion
import unittest

from obspy.clients.fdsn.routing.federator_routing_client import \
    FederatorRoutingClient


_DummyResponse = collections.namedtuple("_DummyResponse", ["content"])


class EIDAWSRoutingClientTestCase(unittest.TestCase):
    def setUp(self):
        self.client = FederatorRoutingClient()
        self._cls = ("obspy.clients.fdsn.routing.federator_routing_client."
                     "FederatorRoutingClient")

    def test_get_service_version(self):
        # At the time of test writing the version is 1.1.1. Here we just
        # make sure it is larger.
        self.assertGreaterEqual(
            LooseVersion(self.client.get_service_version()),
            LooseVersion("1.1.1"))

def suite():  # pragma: no cover
    return unittest.makeSuite(EIDAWSRoutingClientTestCase, 'test')


if __name__ == '__main__':  # pragma: no cover
    unittest.main(defaultTest='suite')
