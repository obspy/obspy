# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import unittest

import os

from obspy.core.inventory import Response

from obspy.clients.nrl.client import NRL, LocalNRL, RemoteNRL


class NRLTestCase(unittest.TestCase):
    """
    NRL test suite.

    """
    def setUp(self):
        self.local_nrl_root = os.path.join(os.path.dirname(__file__),
                                           'data',
                                           'IRIS',
                                           )
        # Small subset of NRL
        self.nrl_local = NRL(root=self.local_nrl_root)
        self.local_dl_key = [ 'REF TEK', 'RT 130 & 130-SMA', 1, 1]
        self.local_sensor_key = ['Guralp', 'CMG-3T', '120s - 50 Hz', '1500']
        
        #self.nrl_online = NRL(root='http://ds.iris.edu/NRL')
        #self.nrl_default = NRL()

        # For Lloyd delete
        # remove
        self.nrl_full_local = NRL(root=os.path.join(os.path.dirname(__file__),
                                           'data',
                                           'IRIS_full' ))
        
        self.list_of_nrls = [self.nrl_local,
                             #self.nrl_default,
                             #self.nrl_online,
                             self.nrl_full_local,  # remove
                             ]

    def test_nrl_types(self):
        for nrl in self.list_of_nrls:
            self.assertIsInstance(nrl, NRL)
        self.assertIsInstance(self.nrl_local, LocalNRL)
        self.assertIsInstance(self.nrl_online, RemoteNRL)
        self.assertIsInstance(self.nrl_default, RemoteNRL)

    def test_get_response(self):
        resp = self.nrl_full_local.get_response(
            datalogger_keys=self.local_dl_key,
            sensor_keys=self.local_sensor_key)
        self.assertIsInstance(resp, Response)


def suite():
    return unittest.makeSuite(NRLTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
