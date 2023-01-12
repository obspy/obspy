# -*- coding: utf-8 -*-
"""
The obspy.clients.seedlink.basic_client test suite.
"""
from unittest import mock

import pytest

from obspy import UTCDateTime
from obspy.clients.seedlink.basic_client import Client


@pytest.mark.network
class TestClient():

    def init_client(self):
        self.client = Client("rtserver.ipgp.fr")

    @pytest.fixture(autouse=True, scope="function")
    def setup(self):
        self.init_client()

    def test_get_waveform(self):
        def _test_offset_from_realtime(offset):
            t = UTCDateTime() - offset
            request = ["G", "FDFM", "00", "LHN", t, t + 20]
            st = self.client.get_waveforms(*request)
            assert len(st) > 0
            for tr in st:
                assert tr.id == ".".join(request[:4])
            assert any([len(tr) > 0 for tr in st])
            st.merge(1)
            assert abs(tr.stats.starttime - request[4]) < 1
            assert abs(tr.stats.endtime - request[5]) < 1

        # getting a result depends on two things.. how long backwards the ring
        # buffer stores data and how close to realtime the data is available,
        # so check some different offsets and see if we get some data
        for offset in (3600, 2000, 1000, 500):
            try:
                _test_offset_from_realtime(offset)
            except AssertionError:
                continue
            else:
                break
        else:
            raise

    def test_get_info(self):
        """
        Test fetching station information
        """
        client = self.client

        info = client.get_info(station='F*')
        assert ('G', 'FDFM') in info
        # should have at least 7 stations
        assert len(info) > 2
        # only fetch one station
        info = client.get_info(network='G', station='FDFM')
        assert [('G', 'FDFM')] == info
        # check that we have a cache on station level
        assert ('G', 'FDFM') in client._station_cache
        assert len(client._station_cache) > 20
        assert client._station_cache_level == "station"

    def test_multiple_waveform_requests_with_multiple_info_requests(self):
        """
        This test a combination of waveform requests that internally do
        multiple info requests on increasing detail levels
        """
        def _test_offset_from_realtime(offset):
            # need to reinit to clean out any caches
            self.init_client()
            t = UTCDateTime() - offset
            # first do a request that needs an info request on station level
            # only
            st = self.client.get_waveforms("*", "F?FM", "??", "B??", t, t + 5)
            assert len(st) > 2
            assert len(self.client._station_cache) > 20
            station_cache_size = len(self.client._station_cache)
            assert ("G", "FDFM") in self.client._station_cache
            assert self.client._station_cache_level == "station"
            for tr in st:
                assert tr.stats.network == "G"
                assert tr.stats.station == "FDFM"
                assert tr.stats.channel[0] == "B"
            # now make a subsequent request that needs an info request on
            # channel level
            st = self.client.get_waveforms("*", "F?FM", "*", "B*", t, t + 5)
            assert len(st) > 2
            assert len(self.client._station_cache) > station_cache_size
            assert ("G", "FDFM", "00", "BHZ") in self.client._station_cache
            assert self.client._station_cache_level == "channel"
            for tr in st:
                assert tr.stats.network == "G"
                assert tr.stats.station == "FDFM"
                assert tr.stats.channel[0] == "B"

        # getting a result depends on two things.. how long backwards the ring
        # buffer stores data and how close to realtime the data is available,
        # so check some different offsets and see if we get some data
        for offset in (3600, 2000, 1000, 500):
            try:
                _test_offset_from_realtime(offset)
            except AssertionError:
                continue
            else:
                break
        else:
            raise


@mock.patch("obspy.clients.seedlink.basic_client.Client._multiselect_request")
def test_get_waveform_calls_to_get_info(multiselect_mock):
    """
    Make sure get_waveforms() without wildcards does not call get_info()
    Test works without network since connection is only made when
    multiselect request goes out.
    """
    client = Client("abcde")
    t = UTCDateTime(2000, 1, 1)
    with mock.patch(
            "obspy.clients.seedlink.basic_client.Client.get_info") as p:
        client.get_waveforms("GR", "FUR", "", "HHZ", t, t+1)
        assert p.call_count == 0
        # get_info should only be called when wildcards are in SEED ID
        client.get_waveforms("GR", "?UR", "", "HHZ", t, t+1)
        assert p.call_count == 1
        client.get_waveforms("*R", "FUR", "", "HHZ", t, t+1)
        assert p.call_count == 2
        client.get_waveforms("GR", "FUR", "", "HH*", t, t+1)
        assert p.call_count == 3
