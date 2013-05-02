# -*- coding: utf-8 -*-
"""
The obspy.neic.client test suite.
"""

from obspy.core.utcdatetime import UTCDateTime
from obspy.neic import Client
from obspy.neic.util import ascdate, asctime
import unittest


class ClientTestCase(unittest.TestCase):
    """
    Test cases for obspy.neic.client.Client.
    """
    def test_getWaveform(self):
        """
        Tests getWaveform method.
        """
        client = Client(host="137.227.224.97", port=2061)
        # now - 5 hours
        dt = UTCDateTime() - 5 * 60 * 60
        print ascdate() + " " + asctime() + " start=" + str(dt)
        st = client.getWaveformNSCL("USISCO BH.00", dt, 3600.0)
        print ascdate() + " " + asctime() + " " + str(st)
        st = client.getWaveform("US", "DUG", "BH.", "00", dt, dt + 3600)
        print ascdate() + " " + asctime() + " " + str(st)


def suite():
    return unittest.makeSuite(ClientTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
