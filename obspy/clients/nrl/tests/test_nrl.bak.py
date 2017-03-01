# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import unittest

import os

from obspy.clients.nrl.client import NRL


class NRLTestCase(unittest.TestCase):
    """
    NRL test suite.

    """
    def setUp(self):
        self.local_nrl_root = os.path.join(os.path.dirname(__file__),
                                           'data',
                                           'IRIS',
                                           )
        self._nrl_local = NRL(root=self.local_nrl_root)
        self._nrl_online = NRL(root='http://ds.iris.edu/NRL/')

    def test_local_path(self):
        nrl = self._nrl_local
        # Test Datalogger
        path = nrl.datalogger_path(('REF TEK', 'RT 130 & 130-SMA',
                                    nrl.GAIN, nrl.SR),
                                   gain=1, sr=1)
        self.assertEqual(os.path.basename(path), 'RESP.XX.NR001..LHZ.130.1.1')
        # Test sensor
        path = nrl.sensor_path(['Guralp', 'CMG-3T',
                                '120s - 50 Hz', '1500'])
        self.assertEqual(os.path.basename(path),
                         'RESP.XX.NS007..BHZ.CMG3T.120.1500')

    def test_url_path(self):
        nrl = self._nrl_online
        path = nrl.datalogger_path(('REF TEK', 'RT 130 & 130-SMA',
                                    nrl.GAIN, nrl.SR),
                                   gain=1, sr=1)
        self.assertEqual(os.path.basename(path), 'RESP.XX.NR001..LHZ.130.1.1')
        path = nrl.sensor_path(['Guralp', 'CMG-3T',
                                '120s - 50 Hz', '1500'])
        self.assertEqual(os.path.basename(path),
                         'RESP.XX.NS007..BHZ.CMG3T.120.1500')

    def test_resp(self):
        # these functions return the contents of RESPs
        nrl = self._nrl_online
        resp = nrl.datalogger_from_short('q330',
                                         gain=1, sr=1)
        self.assertRegexpMatches(resp, r'B050F03\s*Station:\s*NQ001')
        # self.assert
        nrl = self._nrl_local
        resp = nrl.sensor_from_short('cmg3t')
        self.assertRegexpMatches(resp, r'B050F03\s*Station:\s*NS007')

    def test_all_shortcut_path(self):
        nrl = self._nrl_local
        for sc in nrl.dl_shortcuts:
            resp_path = nrl.datalogger_path_from_short(sc, gain=1, sr=1)
            self.assertTrue(os.path.basename(resp_path).startswith('RESP.XX'))
        for sc in nrl.sensor_shortcuts:
            resp_path = nrl.sensor_path_from_short(sc)
            self.assertTrue(os.path.basename(resp_path).startswith('RESP.XX'))

    def test_get_inv_response(self):
        nrl = self._nrl_online
        response = nrl.get_response(
            sensor_keys=['Nanometrics', 'Trillium Compact', '120 s'],
            datalogger_keys=['REF TEK', 'RT 130 & 130-SMA', '1', '200'])



def suite():
    return unittest.makeSuite(NRLTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
