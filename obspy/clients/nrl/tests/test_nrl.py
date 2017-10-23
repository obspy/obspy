# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import os
import unittest

import numpy as np

from obspy.core.inventory import (Response, PolesZerosResponseStage,
                                  ResponseStage, CoefficientsTypeResponseStage)
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
        # Get only the sensor response.
        sensor_resp = self.nrl_local.get_sensor_response(self.local_sensor_key)

        # Get only the datalogger response.
        dl_resp = self.nrl_local.get_datalogger_response(self.local_dl_key)

        # Get full response.
        resp = self.nrl_local.get_response(
            datalogger_keys=self.local_dl_key,
            sensor_keys=self.local_sensor_key)

        # All of them should be Response objects.
        self.assertIsInstance(resp, Response)
        self.assertIsInstance(dl_resp, Response)
        self.assertIsInstance(sensor_resp, Response)

        # The full response is the first stage from the sensor and all
        # following from the datalogger.
        self.assertEqual(resp.response_stages[0],
                         sensor_resp.response_stages[0])
        self.assertEqual(resp.response_stages[1:],
                         dl_resp.response_stages[1:])

        # Test the actual responses. Testing the parsing of the exact values
        # and what not is done in obspy.io.xseed.
        paz = sensor_resp.response_stages[0]
        self.assertIsInstance(paz, PolesZerosResponseStage)
        np.testing.assert_allclose(
            paz.poles, [(-0.037008 + 0.037008j), (-0.037008 - 0.037008j),
                        (-502.65 + 0j), (-1005 + 0j), (-1131 + 0j)])
        np.testing.assert_allclose(paz.zeros, [0j, 0j])

        self.assertEqual(len(dl_resp.response_stages), 15)
        self.assertEqual(len(resp.response_stages), 15)

        self.assertIsInstance(resp.response_stages[1], ResponseStage)
        for _i in range(2, 15):
            self.assertIsInstance(resp.response_stages[_i],
                                  CoefficientsTypeResponseStage)

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

    def test_error_handling_invalid_path(self):
        with self.assertRaises(ValueError) as err:
            NRL("/some/really/random/path")
        self.assertEqual(
            err.exception.args[0],
            "Provided path '/some/really/random/path' seems to be a local "
            "file path but the directory does not exist.")


def suite():  # pragma: no cover
    return unittest.makeSuite(NRLTestCase, 'test')


if __name__ == '__main__':  # pragma: no cover
    unittest.main(defaultTest='suite')
