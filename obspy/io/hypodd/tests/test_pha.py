# -*- coding: utf-8 -*-
import os
import unittest
import warnings

from obspy import read_events, read_inventory, UTCDateTime
from obspy.core.event import Catalog, Event, Origin, Pick, WaveformStreamID
from obspy.core.util import NamedTemporaryFile
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
            self.assertAlmostEqual(reltime, target[sta][0], 4)
            self.assertEqual(arr.time_weight, target[sta][1])
            self.assertEqual(arr.phase, target[sta][2])
            self.assertEqual(arr.phase, pick.phase_hint)
        event = cat[1]
        ori = event.preferred_origin()
        self.assertAlmostEqual(ori.latitude_errors.uncertainty, 0.0045, 5)
        self.assertLess(ori.latitude_errors.uncertainty,
                        ori.longitude_errors.uncertainty)
        self.assertEqual(ori.depth_errors.uncertainty, 400)
        self.assertEqual(ori.quality.standard_error, 0.3)
        self.assertEqual(ori.quality.associated_phase_count, 2)

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
            warnings.resetwarnings()
            cat = read_events(self.fname, inventory=inv)
            self.assertEqual(len(ws), 4)
        # FUR with channels, WET without channels
        inv = read_inventory().select(channel='HH?')
        inv[0][1].channels = []
        cat = read_events(self.fname, inventory=inv,
                          default_seedid='BLA.{}.11.DH{}',
                          seedid_map={'UBR': 'BLB.{}.00.BH{}'}, warn=False)
        self.assertEqual(len(cat), 2)
        picks = cat[0].picks + cat[1].picks
        self.assertEqual(len(picks), 4)
        waveform_ids = [p.waveform_id.get_seed_string() for p in picks]
        self.assertIn('GR.FUR..HHZ', waveform_ids)
        self.assertIn('BLA.WET.11.DHN', waveform_ids)
        self.assertIn('BLB.UBR.00.BHZ', waveform_ids)
        self.assertIn('BLA.WERD.11.DH', waveform_ids)

    def test_eventid_map(self):
        cat = read_events(self.fname) + read_events(self.fname2)
        cat[0].resource_id = 'X'
        cat[1].resource_id = '2f'
        cat[2].resource_id = 'Y'
        cat[3].resource_id = '1234567890Z'
        with NamedTemporaryFile() as tf:
            tempfile = tf.name
            with self.assertWarnsRegex(UserWarning, 'Missing mag'):
                eventid_map = cat.write(tempfile, 'HYPODDPHA')
            cat2 = read_events(tempfile, eventid_map=eventid_map)
            cat3 = read_events(tempfile)
            with self.assertWarnsRegex(UserWarning, 'Missing mag'):
                eventid_map2 = cat.write(tempfile, 'HYPODDPHA',
                                         eventid_map=eventid_map)
        self.assertEqual(cat2[0].resource_id.id.split('/')[-1], 'X')
        self.assertEqual(cat2[1].resource_id.id.split('/')[-1], '2f')
        self.assertEqual(cat2[2].resource_id.id.split('/')[-1], 'Y')
        self.assertEqual(cat2[3].resource_id.id.split('/')[-1], '1234567890Z')
        self.assertEqual(cat3[0].resource_id.id.split('/')[-1], '1')
        self.assertEqual(cat3[1].resource_id.id.split('/')[-1], '2')
        self.assertEqual(cat3[2].resource_id.id.split('/')[-1], '3')
        self.assertEqual(cat3[3].resource_id.id.split('/')[-1], '123456789')
        self.assertEqual(eventid_map2, eventid_map)

    def test_write_pha(self):
        with open(self.fname) as f:
            filedata = f.read()
        cat = read_events(self.fname)
        cat[0].origins[0].arrivals[0].time_weight = None
        with NamedTemporaryFile() as tf:
            tempfile = tf.name
            cat.write(tempfile, 'HYPODDPHA')
            with open(tempfile) as f:
                filedata2 = f.read()
        self.assertEqual(filedata2.replace(' ', ''), filedata.replace(' ', ''))

    def test_write_pha_minimal(self):
        ori = Origin(time=UTCDateTime(0), latitude=42, longitude=43,
                     depth=10000)
        pick = Pick(time=UTCDateTime(10), phase_hint='S',
                    waveform_id=WaveformStreamID(station_code='STA'))
        del ori.latitude_errors
        del ori.longitude_errors
        del ori.depth_errors
        cat = Catalog([Event(origins=[ori], picks=[pick])])
        with NamedTemporaryFile() as tf:
            tempfile = tf.name
            with self.assertWarnsRegex(UserWarning, 'Missing mag'):
                cat.write(tempfile, 'HYPODDPHA')
            cat2 = read_events(tempfile)
        self.assertEqual(len(cat2), 1)
        self.assertEqual(len(cat2[0].picks), 1)
