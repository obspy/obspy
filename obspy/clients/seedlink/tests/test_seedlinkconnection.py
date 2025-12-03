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
        conn.add_stream('BW', 'RJOB', 'EHZ', seqnum=-1,
                        timestamp=None)
        conn.add_stream('BW', 'RJOB', 'EHN', seqnum=-1,
                        timestamp=None)
        assert not isinstance(conn.streams[0].get_selectors()[1], list)

        # Check if the correct Exception is raised (#4)
        try:
            conn.negotiate_station(SLNetStation('BW', 'RJOB',
                                                None, None,
                                                None))
        except Exception as e:
            assert isinstance(e, SeedLinkException)

        # Test if calling add_stream() with selectors_str=None still raises
        # (#5)
        try:
            conn.add_stream('BW', 'RJOB', None,
                            seqnum=-1, timestamp=None)
        except AttributeError:
            msg = 'Calling add_stream with selectors_str=None raised ' + \
                  'AttributeError'
            self.fail(msg)

    def test_read_stream_list(self, tmpdir):
        """
        Test reading a stream list from a file.
        """
        # Create a dummy stream list file
        stream_list_file = tmpdir.join("streamlist.txt")
        stream_list_content = (
            "# Comment line\n"
            "GE ISP  BH?.D\n"
            "NL HGN\n"
            "MN AQU  BH?  HH?\n"
        )
        stream_list_file.write(stream_list_content)

        conn = SeedLinkConnection()
        # Note: 'defselect' is currently ignored for lines with no selectors in
        # read_stream_list because empty string is passed instead of None to
        # add_stream logic.
        # We test the current behavior (empty list of selectors).
        count = conn.read_stream_list(str(stream_list_file), "DEF")

        assert count == 3
        assert len(conn.streams) == 3

        # Check first stream
        assert conn.streams[0].net == "GE"
        assert conn.streams[0].station == "ISP"
        assert conn.streams[0].get_selectors() == ["BH?.D"]

        # Check second stream
        assert conn.streams[1].net == "NL"
        assert conn.streams[1].station == "HGN"
        assert conn.streams[1].get_selectors() == []

        # Check third stream
        assert conn.streams[2].net == "MN"
        assert conn.streams[2].station == "AQU"
        assert conn.streams[2].get_selectors() == ["BH?", "HH?"]

        # Test reading a non-existent file
        assert conn.read_stream_list(str(tmpdir.join("missing.txt")),
                                     None) == 0

        # Test reading a non-existent file
        assert conn.read_stream_list(str(tmpdir.join("missing.txt")),
                                     None) == 0

    def test_recover_and_save_state(self, tmpdir):
        """
        Test saving and recovering state to/from a file.
        """
        state_file = tmpdir.join("sl_state.txt")

        # 1. Setup initial connection with some streams and state
        conn1 = SeedLinkConnection()
        # Add stream 1: GE ISP (SEQ 100, specific time)
        conn1.add_stream("GE", "ISP", "BHZ", 100,
                         None)
        # Manually set btime (normally set by update_stream/packets)
        from obspy import UTCDateTime
        time1 = UTCDateTime("2023-01-01T12:00:00.0Z")
        conn1.streams[0].btime = time1

        # Add stream 2: NL HGN (SEQ 200, different time)
        conn1.add_stream("NL", "HGN", None, 200,
                         None)
        time2 = UTCDateTime("2023-01-02T13:30:00.0Z")
        conn1.streams[1].btime = time2

        conn1.statefile = str(state_file)

        # 2. Save state
        saved_count = conn1.save_state(str(state_file))
        # save_state returns 0, not count of streams (based on code)
        assert saved_count == 0
        # Check if file exists and has content
        assert state_file.check()
        content = state_file.read()
        assert "GE ISP 100 2023,1,1,12,0,0" in content
        assert "NL HGN 200 2023,1,2,13,30,0" in content

        # 3. Recover state with a new connection
        conn2 = SeedLinkConnection()
        # Add streams with default/empty state first
        conn2.add_stream("GE", "ISP", "BHZ", -1,
                         None)
        conn2.add_stream("NL", "HGN", None, -1,
                         None)
        # Add a stream that is NOT in the state file (should remain unchanged)
        conn2.add_stream("MN", "AQU", None, -1,
                         None)

        conn2.statefile = str(state_file)
        recovered_count = conn2.recover_state(str(state_file))

        assert recovered_count == 2

        # Verify Stream 1 recovered
        assert conn2.streams[0].net == "GE"
        assert conn2.streams[0].seqnum == 100
        assert conn2.streams[0].btime == time1

        # Verify Stream 2 recovered
        assert conn2.streams[1].net == "NL"
        assert conn2.streams[1].seqnum == 200
        assert conn2.streams[1].btime == time2

        # Verify Stream 3 untouched
        assert conn2.streams[2].net == "MN"
        assert conn2.streams[2].seqnum == -1
        assert conn2.streams[2].btime is None