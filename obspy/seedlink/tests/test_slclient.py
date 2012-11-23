# -*- coding: utf-8 -*-
"""
The obspy.seedlink.slclient test suite.
"""
from obspy import UTCDateTime
from obspy.core.util.decorator import skipIf
from obspy.seedlink.slclient import SLClient
import unittest


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
        # set a time window from 2 min in the past to 5 sec in the future
        dt = UTCDateTime()
        slClient.begin_time = (dt - 120.0).formatSeedLink()
        slClient.end_time = (dt + 5.0).formatSeedLink()
        print "SeedLink date-time range:", slClient.begin_time, " -> ",
        print slClient.end_time
        slClient.verbose = 2
        slClient.initialize()
        slClient.run()


def suite():
    return unittest.makeSuite(SLClientTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
