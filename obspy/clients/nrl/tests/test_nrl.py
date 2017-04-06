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
        # Longer diffs in the test assertions.
        self.maxDiff = None
        # Small subset of NRL included in tests/data
        self.local_nrl_root = os.path.join(
            os.path.dirname(__file__), 'data', 'IRIS')
        self.nrl_local = NRL(root=self.local_nrl_root)
        self.local_dl_key = ['REF TEK', 'RT 130 & 130-SMA', '1', '1']
        self.local_sensor_key = ['Guralp', 'CMG-3T', '120s - 50Hz', '1500']

        # This is also the default URL.
        self.nrl_online = NRL(root='http://ds.iris.edu/NRL')

        self.list_of_nrls = [self.nrl_local, self.nrl_online]

    def test_nrl_types(self):
        for nrl in self.list_of_nrls:
            self.assertIsInstance(nrl, NRL)
        self.assertIsInstance(self.nrl_local, LocalNRL)
        self.assertIsInstance(self.nrl_online, RemoteNRL)

    def test_get_response(self):
        resp = self.nrl_local.get_response(
            datalogger_keys=self.local_dl_key,
            sensor_keys=self.local_sensor_key)
        self.assertIsInstance(resp, Response)

    def test_nrl_class_str_method(self):
        out = str(self.nrl_local)
        # The local NRL is not going to chance so it is fine to test this.
        self.assertEqual(out.strip(), """
NRL library at %s
  Sensors: 20 manufacturers
    'CEA-DASE', 'CME', 'Chaparral Physics', 'Eentec', 'Generic',
    'Geo Space/OYO', 'Geodevice', 'Geotech', 'Guralp', 'Hyperion',
    'IESE', 'Kinemetrics', 'Lennartz', 'Metrozet', 'Nanometrics',
    'REF TEK', 'Sercel/Mark Products', 'SolGeo',
    'Sprengnether (now Eentec)', 'Streckeisen'
  Dataloggers: 13 manufacturers
    'Agecodagis', 'DAQ Systems (NetDAS)', 'Earth Data', 'Eentec',
    'Generic', 'Geodevice', 'Geotech', 'Guralp', 'Kinemetrics',
    'Nanometrics', 'Quanterra', 'REF TEK', 'SolGeo'
        """.strip() % self.local_nrl_root)

    def test_nrl_dict_str_method(self):
        out = str(self.nrl_local.sensors)
        self.assertEqual(out.strip(), """
Select the sensor manufacturer (20 items):
  'CEA-DASE', 'CME', 'Chaparral Physics', 'Eentec', 'Generic',
  'Geo Space/OYO', 'Geodevice', 'Geotech', 'Guralp', 'Hyperion',
  'IESE', 'Kinemetrics', 'Lennartz', 'Metrozet', 'Nanometrics',
  'REF TEK', 'Sercel/Mark Products', 'SolGeo',
  'Sprengnether (now Eentec)', 'Streckeisen'""".strip())


def suite():  # pragma: no cover
    return unittest.makeSuite(NRLTestCase, 'test')


if __name__ == '__main__':  # pragma: no cover
    unittest.main(defaultTest='suite')
