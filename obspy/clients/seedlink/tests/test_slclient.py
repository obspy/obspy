# -*- coding: utf-8 -*-
"""
The obspy.seedlink.slclient test suite.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import unittest

from obspy import UTCDateTime
from obspy.core.util.decorator import skipIf
from obspy.seedlink.slclient import SLClient


class SLClientTestCase(unittest.TestCase):

    @skipIf(__name__ != '__main__', 'test must be started manually')
    def test_info(self):
        slClient = SLClient(loglevel='DEBUG')
        slClient.slconn.setSLAddress("geofon.gfz-potsdam.de:18000")
        slClient.infolevel = "ID"
        slClient.verbose = 2
        slClient.initialize()
        slClient.run()

    @skipIf(__name__ != '__main__', 'test must be started manually')
    def test_time_window(self):
        slClient = SLClient()
        slClient.slconn.setSLAddress("geofon.gfz-potsdam.de:18000")
        slClient.multiselect = ("GE_STU:BHZ")
        # set a time window from 2 min - 1 min in the past
        dt = UTCDateTime()
        slClient.begin_time = (dt - 120.0).formatSeedLink()
        slClient.end_time = (dt - 60.0).formatSeedLink()
        slClient.verbose = 2
        slClient.initialize()
        slClient.run()

    @skipIf(__name__ != '__main__', 'test must be started manually')
    def test_issue708(self):
        slClient = SLClient()
        slClient.slconn.setSLAddress("rtserve.iris.washington.edu:18000")
        slClient.multiselect = ("G_FDF:00BHZ, G_SSB:00BHZ")
        # set a time window from 2 min - 1 min in the past
        dt = UTCDateTime()
        slClient.begin_time = (dt - 120.0).formatSeedLink()
        slClient.end_time = (dt - 60.0).formatSeedLink()
        slClient.verbose = 2
        slClient.initialize()
        slClient.run()


def suite():
    return unittest.makeSuite(SLClientTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
