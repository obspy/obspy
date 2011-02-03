# -*- coding: utf-8 -*-
"""
The obspy.orfeus.client test suite.
"""

from obspy.core.utcdatetime import UTCDateTime
from obspy.orfeus import Client
import unittest


class ClientTestCase(unittest.TestCase):
    """
    Test cases for obspy.orfeus.client.Client.
    """
    def test_getEvents(self):
        """
        Testing event request method.
        """
        pass

def suite():
    return unittest.makeSuite(ClientTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
