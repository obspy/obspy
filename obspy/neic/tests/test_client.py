# -*- coding: utf-8 -*-
"""
The obspy.neic.client test suite.
"""

from obspy.core.utcdatetime import UTCDateTime
from obspy.neic import Client
import unittest


class ClientTestCase(unittest.TestCase):
    """
    Test cases for obspy.neic.client.Client.
    """
    def test_getWaveform(self):
        """
        Tests getWaveform method. Tests against getWaveformNSCL method.
        """
        client = Client(host="137.227.224.97", port=2061)
        # now - 5 hours
        t = UTCDateTime() - 5 * 60 * 60
        duration = 1.0
        st = client.getWaveformNSCL("USISCO BH.00", t, duration)
        # try a series of requests, compare against getWaveformNSCL
        args = [["US", "ISCO", "00", "BH."],
                ["??", "ISCO", "??", "BH[Z21]"],
                ["US", "ISC.*", "00", "B??"],
                ["US", "ISCO", ".*", "BH."],
                ]
        for args_ in args:
            st2 = client.getWaveform(*args_, starttime=t, endtime=t + duration)
            self.assertTrue(st == st2)

    def test_getWaveformNSCL(self):
        """
        Tests getWaveformNSCL method.
        """
        client = Client(host="137.227.224.97", port=2061)
        # now - 5 hours
        t = UTCDateTime() - 5 * 60 * 60
        duration_long = 3600.0
        duration = 1.0
        components = ["1", "2", "Z"]
        # try one longer request to see if fetching multiple blocks works
        st = client.getWaveformNSCL("USISCO BH.00", t, duration_long)
        # merge to avoid failing tests simply due to gaps
        st.merge()
        st.sort()
        self.assertTrue(len(st) == 3)
        for tr, component in zip(st, components):
            stats = tr.stats
            self.assertTrue(stats.station == "ISCO")
            self.assertTrue(stats.network == "US")
            self.assertTrue(stats.location == "00")
            self.assertTrue(stats.channel == "BH" + component)
            self.assertTrue(stats.endtime - stats.starttime == duration_long)
            # if the following fails this is likely due to a change at the
            # requested station and simply has to be adapted
            self.assertTrue(stats.sampling_rate == 40)
            self.assertTrue(len(tr) == 144001)
        # now use shorter piece, this is faster and less error prone (gaps etc)
        st = client.getWaveformNSCL("USISCO BH.00", t, duration)
        st.sort()
        # test returned stream
        self.assertTrue(len(st) == 3)
        for tr, component in zip(st, components):
            stats = tr.stats
            self.assertTrue(stats.station == "ISCO")
            self.assertTrue(stats.network == "US")
            self.assertTrue(stats.location == "00")
            self.assertTrue(stats.channel == "BH" + component)
            self.assertTrue(stats.endtime - stats.starttime == duration)
            # if the following fails this is likely due to a change at the
            # requested station and simply has to be adapted
            self.assertTrue(stats.sampling_rate == 40)
            self.assertTrue(len(tr) == 41)

        # try a series of regex patterns that should return the same data
        patterns = ["USISCO BH...",
                    "USISCO BH.*",
                    "USISCO BH[Z12].*",
                    "USISCO BH[Z12]..",
                    "USISCO B.*",
                    "..ISCO B.*"]
        for pattern in patterns:
            st2 = client.getWaveformNSCL(pattern, t, duration)
            self.assertTrue(st == st2)


def suite():
    return unittest.makeSuite(ClientTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
