# -*- coding: utf-8 -*-
"""
The obspy.clients.seedlink.client.seedlinkconnection test suite.
"""

import pytest

from obspy.clients.seedlink.client.seedlinkconnection import SeedLinkConnection
from obspy.clients.seedlink.client.slnetstation import SLNetStation
from obspy.clients.seedlink.seedlinkexception import SeedLinkException


pytestmark = pytest.mark.network


class TestSeedLinkConnection():

    def test_issue777(self):
        """
        Regression tests for Github issue #777
        """
        conn = SeedLinkConnection()

        # Check adding multiple streams (#3)
        conn.add_stream('BW', 'RJOB', 'EHZ', seqnum=-1, timestamp=None)
        conn.add_stream('BW', 'RJOB', 'EHN', seqnum=-1, timestamp=None)
        assert not isinstance(conn.streams[0].get_selectors()[1], list)

        # Check if the correct Exception is raised (#4)
        try:
            conn.negotiate_station(SLNetStation('BW', 'RJOB', None,
                                                None, None))
        except Exception as e:
            assert isinstance(e, SeedLinkException)

        # Test if calling add_stream() with selectors_str=None still raises
        # (#5)
        try:
            conn.add_stream('BW', 'RJOB', None, seqnum=-1, timestamp=None)
        except AttributeError:
            msg = 'Calling add_stream with selectors_str=None raised ' + \
                  'AttributeError'
            self.fail(msg)
