# -*- coding: utf-8 -*-
"""
The obspyRT.seedlink.slclient test suite.
"""	


import unittest
from obspyRT.seedlink.util import *
from obspyRT.seedlink.slclient import *
from obspy.core.utcdatetime import UTCDateTime


class SLClientTestCase(unittest.TestCase):


    def test_info(self):
        slClient = SLClient()
        slClient.slconn.setSLAddress("geofon.gfz-potsdam.de:18000")
        slClient.infolevel = "ID"
        slClient.verbose = 2
        slClient.initialize()
        slClient.run()
        print
        print


    def test_time_window(self):
        slClient = SLClient()
        slClient.slconn.setSLAddress("geofon.gfz-potsdam.de:18000")
        slClient.multiselect = ("GE_STU:BHZ")
        # set a time window from 2 min in the past to 5 sec in the future
        datetime = UTCDateTime()
        slClient.begin_time = Util.formatSeedLink(datetime - 120.0)
        slClient.end_time = Util.formatSeedLink(datetime + 5.0)
        print "SeedLink date-time range:", slClient.begin_time, " -> ", slClient.end_time
        slClient.verbose = 2
        slClient.initialize()
        slClient.run()


def suite():
    return unittest.makeSuite(SLClientTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')