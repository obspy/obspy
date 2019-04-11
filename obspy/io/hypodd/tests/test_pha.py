# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import os
import unittest

from obspy import read_events, read_inventory
from obspy.io.hypodd import pha


class PHATestCase(unittest.TestCase):
    """
    Test suite for obspy.io.hypodd.pha
    """
    def setUp(self):
        path = os.path.dirname(__file__)
        self.fname = os.path.join(path, 'data', 'example.pha')

    def test_is_pha(self):
        self.assertEqual(pha._is_pha(self.fname), True)


    def test_read_pha(self):
        cat = read_events(self.fname)
        self.assertEqual(len(cat), 2)


    def test_populate_waveform_id(self):
        # read invenroty - FUR with channels, WET without channels
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
        self.assertIn('GR.WET.11.DHN', waveform_ids)
        self.assertIn('BLB.UBR.00.BHZ', waveform_ids)
        self.assertIn('BLA.WERD.11.DH', waveform_ids)


def suite():
    return unittest.makeSuite(PHATestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
