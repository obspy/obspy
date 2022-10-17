# -*- coding: utf-8 -*-
"""
The obspy.clients.neic.client test suite.
"""
import unittest

import pytest

from obspy.core.utcdatetime import UTCDateTime
from obspy.clients.neic import Client


pytestmark = pytest.mark.network


class ClientTestCase(unittest.TestCase):
    """
    Test cases for obspy.clients.neic.client.Client.
    """
    @classmethod
    def setUpClass(cls):
        cls.client = Client(host="137.227.224.97", port=2061, timeout=8)
        cls.starttime = UTCDateTime() - 5 * 60 * 60

    def test_get_waveform(self):
        """
        Tests get_waveforms method. Tests against get_waveforms_nscl method.
        """
        client = self.client
        t = self.starttime
        duration = 1.0
        st = client.get_waveforms_nscl("IUANMO BH.00", t, duration)
        # try a series of requests, compare against get_waveforms_nscl
        args = [["IU", "ANMO", "00", "BH."],
                ["??", "ANMO", "0?", "BH[Z21]"],
                ["IU", "ANM.*", "00", "B??"],
                ["IU", "ANMO", "0*", "BH."],
                ]
        for args_ in args:
            st2 = client.get_waveforms(*args_, starttime=t,
                                       endtime=t + duration)
            self.assertEqual(st, st2)

    def test_get_waveform_nscl(self):
        """
        Tests get_waveforms_nscl method.
        """
        client = self.client
        t = self.starttime
        duration_long = 3600.0
        duration = 1.0
        components = ["1", "2", "Z"]
        # try one longer request to see if fetching multiple blocks works
        st = client.get_waveforms_nscl("IUANMO BH.00", t, duration_long)
        # merge to avoid failing tests simply due to gaps
        st.merge()
        st.sort()
        self.assertEqual(len(st), 3)
        for tr, component in zip(st, components):
            stats = tr.stats
            self.assertEqual(stats.station, "ANMO")
            self.assertEqual(stats.network, "IU")
            self.assertEqual(stats.location, "00")
            self.assertEqual(stats.channel, "BH" + component)
            # requested data duration has some minor fluctuations sometimes but
            # should be pretty close to the expected duration.
            # it should not be over a delta longer than expected (it should be
            # trimmed correctly if more data is returned) but sometimes it's
            # one delta shorter
            self.assertTrue(
                abs(duration_long - (stats.endtime - stats.starttime)) <=
                tr.stats.delta)
            # if the following fails this is likely due to a change at the
            # requested station and simply has to be adapted
            self.assertEqual(stats.sampling_rate, 40.0)
            self.assertEqual(len(tr), 144001)
        # now use shorter piece, this is faster and less error prone (gaps etc)
        st = client.get_waveforms_nscl("IUANMO BH.00", t, duration)
        st.sort()
        # test returned stream
        self.assertEqual(len(st), 3)
        for tr, component in zip(st, components):
            stats = tr.stats
            self.assertEqual(stats.station, "ANMO")
            self.assertEqual(stats.network, "IU")
            self.assertEqual(stats.location, "00")
            self.assertEqual(stats.channel, "BH" + component)
            # requested data duration has some minor fluctuations sometimes but
            # should be pretty close to the expected duration.
            # it should not be over a delta longer than expected (it should be
            # trimmed correctly if more data is returned) but sometimes it's
            # one delta shorter
            self.assertTrue(
                abs(duration - (stats.endtime - stats.starttime)) <=
                tr.stats.delta)
            # if the following fails this is likely due to a change at the
            # requested station and simply has to be adapted
            self.assertEqual(stats.sampling_rate, 40.0)
            self.assertEqual(len(tr), 41)

        # try a series of regex patterns that should return the same data
        st = client.get_waveforms_nscl("IUANMO BH", t, duration)
        patterns = ["IUANMO BH...",
                    "IUANMO BH.*",
                    "IUANMO BH[Z12].*",
                    "IUANMO BH[Z12]..",
                    "..ANMO BH.*"]
        for pattern in patterns:
            st2 = client.get_waveforms_nscl(pattern, t, duration)
            self.assertEqual(st, st2)
