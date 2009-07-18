# -*- coding: utf-8 -*-
"""
The obspy.arclink.client test suite.
"""

from obspy.arclink import Client
import unittest


class ClientTestCase(unittest.TestCase):
    """
    Test cases for L{obspy.arclink.client.Client}.
    """

    def test_getWaveform(self):
        """
        """
        client = Client()
        data = client.getWaveform()
        print len(data)
        data2 = client.saveWaveform('muh.mseed')
        print len(data2)
        print "done"


def suite():
    return unittest.makeSuite(ClientTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
