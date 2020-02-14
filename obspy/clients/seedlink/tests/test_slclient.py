# -*- coding: utf-8 -*-
"""
The obspy.clients.seedlink.slclient test suite.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import unittest

from obspy import UTCDateTime
from obspy.clients.seedlink.slclient import SLClient


class SLClientTestCase(unittest.TestCase):

    @unittest.skipIf(__name__ != '__main__', 'test must be started manually')
    def test_info(self):
        sl_client = SLClient(loglevel='DEBUG')
        sl_client.slconn.set_sl_address("geofon.gfz-potsdam.de:18000")
        sl_client.infolevel = "ID"
        sl_client.verbose = 2
        sl_client.initialize()
        sl_client.run()

    @unittest.skipIf(__name__ != '__main__', 'test must be started manually')
    def test_time_window(self):
        sl_client = SLClient()
        sl_client.slconn.set_sl_address("geofon.gfz-potsdam.de:18000")
        sl_client.multiselect = ("GE_STU:BHZ")
        # set a time window from 2 min - 1 min in the past
        dt = UTCDateTime()
        sl_client.begin_time = (dt - 120.0).format_seedlink()
        sl_client.end_time = (dt - 60.0).format_seedlink()
        sl_client.verbose = 2
        sl_client.initialize()
        sl_client.run()

    @unittest.skipIf(__name__ != '__main__', 'test must be started manually')
    def test_issue708(self):
        sl_client = SLClient()
        sl_client.slconn.set_sl_address("rtserve.iris.washington.edu:18000")
        sl_client.multiselect = ("G_FDFM:00BHZ, G_SSB:00BHZ")
        # set a time window from 2 min - 1 min in the past
        dt = UTCDateTime()
        sl_client.begin_time = (dt - 120.0).format_seedlink()
        sl_client.end_time = (dt - 60.0).format_seedlink()
        sl_client.verbose = 2
        sl_client.initialize()
        sl_client.run()


def suite():
    return unittest.makeSuite(SLClientTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
