# -*- coding: utf-8 -*-
"""
The obspy.clients.seedlink.slclient test suite.
"""
import unittest

import pytest

from obspy import UTCDateTime
from obspy.clients.seedlink.slclient import SLClient


pytestmark = pytest.mark.network


class SLClientTestCase(unittest.TestCase):
    """
    These test routines use SLClient, which is actually not expected to be
    used, but expected to be subclassed since at least the packet_handler
    method needs to be reimplemented.
    The original packet_handler method does not do anything with the received
    metadata or waveforms but only prints some information and not even checks
    when the requested waveform data (if any) is fully received to break out of
    the loop that is waiting for more packets. That is why these data requests
    below get stuck and not terminate.
    """

    @unittest.skipIf(__name__ != '__main__', 'test must be started manually')
    def test_info(self):
        sl_client = SLClient()
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
