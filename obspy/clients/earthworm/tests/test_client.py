# -*- coding: utf-8 -*-
"""
The obspy.clients.earthworm.client test suite.
"""
import pytest

from obspy import read
from obspy.core.utcdatetime import UTCDateTime
from obspy.core.util import NamedTemporaryFile
from obspy.core.util.decorator import skip_on_network_error
from obspy.clients.earthworm import Client

pytestmark = pytest.mark.network


class TestEWClient:
    """
    Test cases for obspy.clients.earthworm.client.Client.
    """
    start = UTCDateTime() - 24 * 3600 * 4
    end = start + 1.0

    def try_get_stream(self, client, kwargs):
        """Try to get the stream with either '' or '--' location codes. """
        # I am not sure why, but sometimes the location code needs to be --
        # and other times '' in order to get a stream of nonzero length.
        # Just try both.
        st = client.get_waveforms(
            location='', starttime=self.start, endtime=self.end, **kwargs
        )
        if len(st) > 0:
            return st
        st = client.get_waveforms(
            location='--', starttime=self.start, endtime=self.end, **kwargs
        )
        return st

    @pytest.fixture(scope='class', autouse=True)
    def set_utc_precision(self):
        """Set UTC precision to 4 for this tests class."""
        current_precision = UTCDateTime.DEFAULT_PRECISION
        UTCDateTime.DEFAULT_PRECISION = 4
        yield
        UTCDateTime.DEFAULT_PRECISION = current_precision

    @pytest.fixture(scope='class')
    def ew_client(self):
        """Return the earthworn client."""
        ew_client = Client("pubavo1.wr.usgs.gov", 16022, timeout=30.0)
        return ew_client

    @pytest.fixture(scope='class')
    def ew_stream(self, ew_client):
        """Return a stream fetched from the test ew client."""

        # example 1 -- 1 channel, cleanup
        kwargs = dict(
            network='AV',
            station='AKV',
            channel='BHE',
        )
        return self.try_get_stream(ew_client, kwargs)

    @pytest.fixture(scope='class')
    def ew_stream_no_cleanup(self, ew_client):
        """Return a stream fetched from the test ew client with no cleanup."""
        kwargs = dict(
            network='AV',
            station='AKV',
            channel='BHE',
            cleanup=False,
        )
        return self.try_get_stream(ew_client, kwargs)

    @pytest.fixture(scope='class')
    def ew_stream_wildcard(self, ew_client):
        """Return a stream fetched from the test ew client with wildcard."""
        kwargs = dict(
            network='AV',
            station='AKV',
            channel='BH?',
        )
        return self.try_get_stream(ew_client, kwargs)

    @skip_on_network_error
    def test_get_waveform(self, ew_client, ew_stream):
        """
        Tests get_waveforms method.
        """
        assert len(ew_stream) == 1
        delta = ew_stream[0].stats.delta
        trace = ew_stream[0]
        assert len(trace) in (50, 51)
        assert trace.stats.starttime >= self.start - delta
        assert trace.stats.starttime <= self.start + delta
        assert trace.stats.endtime >= self.end - delta
        assert trace.stats.endtime <= self.end + delta
        assert trace.stats.network == 'AV'
        assert trace.stats.station == 'AKV'
        assert trace.stats.location in ('--', '')
        assert trace.stats.channel == 'BHE'

    @skip_on_network_error
    def test_get_waveform_no_cleanup(self, ew_client, ew_stream_no_cleanup):
        """
        Tests get_waveforms method again, 1 channel no cleanup.
        """
        # example 2 -- 1 channel, no cleanup
        ew_stream = ew_stream_no_cleanup
        delta = ew_stream[0].stats.delta
        assert len(ew_stream) >= 2
        summed_length = sum(len(tr) for tr in ew_stream)
        assert summed_length in (50, 51)
        assert ew_stream[0].stats.starttime >= self.start - delta
        assert ew_stream[0].stats.starttime <= self.start + delta
        assert ew_stream[-1].stats.endtime >= self.end - delta
        assert ew_stream[-1].stats.endtime <= self.end + delta
        for trace in ew_stream:
            assert trace.stats.network == 'AV'
            assert trace.stats.station == 'AKV'
            assert trace.stats.location in ('--', '')
            assert trace.stats.channel == 'BHE'

    @skip_on_network_error
    def test_get_waveform_widlcard(self, ew_client, ew_stream_wildcard):
        """
        Test example 3 -- component wildcarded with '?'
        """
        # example 3 -- component wildcarded with '?'
        stream = ew_stream_wildcard
        delta = stream[0].stats.delta
        assert len(stream) == 3
        for trace in stream:
            assert len(trace) in (50, 51)
            assert trace.stats.starttime >= self.start - delta
            assert trace.stats.starttime <= self.start + delta
            assert trace.stats.endtime >= self.end - delta
            assert trace.stats.endtime <= self.end + delta
            assert trace.stats.network == 'AV'
            assert trace.stats.station == 'AKV'
            assert trace.stats.location in ('--', '')
        assert stream[0].stats.channel == 'BHZ'
        assert stream[1].stats.channel == 'BHN'
        assert stream[2].stats.channel == 'BHE'

    @skip_on_network_error
    def test_save_waveform(self, ew_client):
        """
        Tests save_waveforms method.
        """
        # initialize client
        with NamedTemporaryFile() as tf:
            testfile = tf.name
            # 1 channel, cleanup (using SLIST to avoid dependencies)
            ew_client.save_waveforms(
                testfile,
                'AV',
                'AKV',
                '--',
                'BHE',
                self.start,
                self.end,
                format="SLIST",
            )
            stream = read(testfile)
        assert len(stream) == 1
        delta = stream[0].stats.delta
        trace = stream[0]
        assert len(trace) == 51
        assert trace.stats.starttime >= self.start - delta
        assert trace.stats.starttime <= self.start + delta
        assert trace.stats.endtime >= self.end - delta
        assert trace.stats.endtime <= self.end + delta
        assert trace.stats.network == 'AV'
        assert trace.stats.station == 'AKV'
        assert trace.stats.location in ('--', '')
        assert trace.stats.channel == 'BHE'

    @skip_on_network_error
    def test_availability(self, ew_client):
        data = ew_client.get_availability()
        seeds = ["%s.%s.%s.%s" % (d[0], d[1], d[2], d[3]) for d in data]
        assert 'AV.AKV.--.BHZ' in seeds
