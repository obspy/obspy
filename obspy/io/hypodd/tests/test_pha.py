# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import os
import unittest
import warnings

from obspy import read_events, read_inventory
from obspy.io.hypodd import pha


class PHATestCase(unittest.TestCase):
    """
    Test suite for obspy.io.hypodd.pha
    """

    def setUp(self):
        self.path = os.path.dirname(__file__)
        self.fname = os.path.join(self.path, 'data', 'example.pha')
        self.fname2 = os.path.join(self.path, 'data', '60s_nan.pha')

    def test_is_pha(self):
        self.assertEqual(pha._is_pha(self.fname), True)

    def test_is_not_pha(self):
        fname = os.path.join(self.path, 'test_pha.py')
        self.assertEqual(pha._is_pha(fname), False)

    def test_read_pha(self):
        cat = read_events(self.fname)
        self.assertEqual(len(cat), 2)
        event = cat[0]
        self.assertEqual(len(event.origins), 1)
        self.assertEqual(len(event.magnitudes), 1)
        ori = event.preferred_origin()
        mag = event.preferred_magnitude()
        self.assertEqual(str(ori.time), '2025-05-14T14:35:35.510000Z')
        self.assertEqual(ori.latitude, 40.2254)
        self.assertEqual(ori.longitude, 10.4496)
        self.assertEqual(ori.depth, 9408.0)
        self.assertEqual(mag.mag, 3.50)
        self.assertEqual(len(event.picks), 2)
        self.assertEqual(len(ori.arrivals), 2)
        target = {'FUR': (3.52199909, 1.0000, 'P'),
                  'WET': (5.86199909, 1.0000, 'S')}
        for arr in ori.arrivals:
            pick = arr.pick_id.get_referred_object()
            sta = pick.waveform_id.station_code
            reltime = pick.time - ori.time
            self.assertAlmostEqual(reltime, target[sta][0], 6)
            self.assertEqual(arr.time_weight, target[sta][1])
            self.assertEqual(arr.phase, target[sta][2])
            self.assertEqual(arr.phase, pick.phase_hint)

    def test_60s_nan(self):
        """
        issue 2627
        """
        cat = read_events(self.fname2)
        event = cat[0]
        self.assertEqual(len(event.origins), 1)
        self.assertEqual(len(event.magnitudes), 1)
        ori = event.preferred_origin()
        self.assertEqual(str(ori.time), '2025-05-14T14:36:00.000000Z')
        event = cat[1]
        self.assertEqual(len(event.origins), 1)
        self.assertEqual(len(event.magnitudes), 0)

    def test_populate_waveform_id(self):
        inv = read_inventory()
        with warnings.catch_warnings(record=True) as ws:
            cat = read_events(self.fname, inventory=inv)
            self.assertEqual(len(ws), 2)
            for w in ws:
                self.assertIn('Multiple', str(w.message))
        # FUR with channels, WET without channels
        inv = read_inventory().select(channel='HH?')
        inv[0][1].channels = []
        cat = read_events(self.fname, inventory=inv,
                          id_default='BLA.{}.11.DH{}',
                          id_map={'UBR': 'BLB.{}.00.BH{}'})
        self.assertEqual(len(cat), 2)
        picks = cat[0].picks + cat[1].picks
        self.assertEqual(len(picks), 4)
        waveform_ids = [p.waveform_id.get_seed_string() for p in picks]
        self.assertIn('GR.FUR..HHZ', waveform_ids)
        self.assertIn('BLA.WET.11.DHN', waveform_ids)
        self.assertIn('BLB.UBR.00.BHZ', waveform_ids)
        self.assertIn('BLA.WERD.11.DH', waveform_ids)


def suite():
    return unittest.makeSuite(PHATestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
