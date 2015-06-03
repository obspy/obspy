# -*- coding: utf-8 -*-
"""
The obspy.clients.seedlink.basic_client test suite.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import unittest

from obspy import UTCDateTime
from obspy.clients.seedlink.basic_client import Client


class ClientTestCase(unittest.TestCase):
    def setUp(self):
        self.client = Client("rtserver.ipgp.fr")

    def test_get_waveform(self):
        t = UTCDateTime() - 3600
        for request in [["G", "FDF", "00", "LHN", t, t + 20],
                        ["G", "CLF", "00", "BHZ", t, t + 10]]:
            st = self.client.get_waveforms(*request)
            self.assertGreater(len(st), 0)
            for tr in st:
                self.assertEqual(tr.id, ".".join(request[:4]))
            self.assertTrue(any([len(tr) > 0 for tr in st]))
            st.merge(1)
            self.assertTrue(abs(tr.stats.starttime - request[4]) < 1)
            self.assertTrue(abs(tr.stats.endtime - request[5]) < 1)


def suite():
    return unittest.makeSuite(ClientTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
