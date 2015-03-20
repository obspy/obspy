# -*- coding: utf-8 -*-
"""
The obspy.seedlink.client.seedlinkconnection test suite.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import unittest

from obspy.seedlink.client.seedlinkconnection import SeedLinkConnection
from obspy.seedlink.client.slnetstation import SLNetStation
from obspy.seedlink.seedlinkexception import SeedLinkException


class SeedLinkConnectionTestCase(unittest.TestCase):

    def test_issue777(self):
        """
        Regression tests for Github issue #777
        """
        conn = SeedLinkConnection()

        # Check adding multiple streams (#3)
        conn.addStream('BW', 'RJOB', 'EHZ', seqnum=-1, timestamp=None)
        conn.addStream('BW', 'RJOB', 'EHN', seqnum=-1, timestamp=None)
        self.assertFalse(isinstance(conn.streams[0].getSelectors()[1], list))

        # Check if the correct Exception is raised (#4)
        try:
            conn.negotiateStation(SLNetStation('BW', 'RJOB', None, None, None))
        except Exception as e:
            self.assertTrue(isinstance(e, SeedLinkException))

        # Test if calling addStream() with selectors_str=None still raises (#5)
        try:
            conn.addStream('BW', 'RJOB', None, seqnum=-1, timestamp=None)
        except AttributeError:
            msg = 'Calling addStream with selectors_str=None raised ' + \
                  'AttributeError'
            self.fail(msg)


def suite():
    return unittest.makeSuite(SeedLinkConnectionTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
