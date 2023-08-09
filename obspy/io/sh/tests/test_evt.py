import glob
import os.path
import unittest

from obspy import read_events, read_inventory
from obspy.io.sh.evt import _is_evt


class EvtTestCase(unittest.TestCase):

    def setUp(self):
        # Directory where the test files are located
        self.path = os.path.dirname(__file__)

    def test_is_evt_file(self):
        path = os.path.join(self.path, 'data', '*.evt')
        for fname in glob.glob(path):
            self.assertEqual(_is_evt(fname), True)

    def test_local_event1(self):
        fname = os.path.join(self.path, 'data', 'local1.evt')
        cat = read_events(fname)
        self.assertEqual(len(cat), 2)

    def test_local_event2(self):
        fname = os.path.join(self.path, 'data', 'local2.evt')
        cat = read_events(fname)
        self.assertEqual(len(cat), 1)
        ev = cat[0]
        mag = ev.preferred_magnitude()
        origin = ev.preferred_origin()
        oc = origin.origin_uncertainty
        self.assertEqual(mag.mag, 0.6)
        self.assertEqual(mag.magnitude_type, 'ML')
        self.assertEqual(round(origin.longitude_errors.uncertainty, 3), 0.013)
        self.assertEqual(round(origin.latitude_errors.uncertainty, 3), 0.013)
        self.assertEqual(oc.min_horizontal_uncertainty, 20)
        self.assertEqual(oc.max_horizontal_uncertainty, 20)
        self.assertEqual(oc.azimuth_max_horizontal_uncertainty, 75.70)
        self.assertEqual(origin.arrivals[0].time_weight, 4.0)

    def test_tele_event1(self):
        fname = os.path.join(self.path, 'data', 'tele1.evt')
        cat = read_events(fname)
        self.assertEqual(len(cat), 1)

    def test_tele_event2(self):
        # untested field: sign
        fname = os.path.join(self.path, 'data', 'tele2.evt')
        cat = read_events(fname)
        self.assertEqual(len(cat), 1)
        ev = cat[0]
        origin = ev.preferred_origin()
        mag = ev.preferred_magnitude()
        sta_mag = ev.station_magnitudes[0]
        pick = ev.picks[66]
        arrival = origin.arrivals[66]
        arrival = origin.arrivals[191]
        self.assertEqual(len(ev.picks), 195)
        self.assertEqual(len(ev.station_magnitudes), 38)
        self.assertEqual(len(origin.arrivals), 195)

        self.assertEqual(str(pick.time), '2015-08-10T10:14:32.918000Z')
        self.assertEqual(pick.phase_hint, 'T')
        self.assertEqual(pick.onset, 'emergent')
        self.assertEqual(pick.evaluation_mode, 'manual')
        self.assertEqual(str(pick.filter_id), 'G_WWSSN_SP')
        self.assertEqual(pick.waveform_id.get_seed_string(), '.GRA1..Z')
        self.assertEqual(origin.arrivals[66].pick_id, pick.resource_id)
        self.assertEqual(arrival.phase, 'sS')
        self.assertEqual(arrival.azimuth, 307.6)
        self.assertEqual(arrival.distance, 48.132)
        self.assertEqual(str(origin.time), '2015-08-10T10:05:25.808000Z')
        self.assertEqual(origin.latitude, 36.23)
        self.assertEqual(origin.longitude, 71.38)

        self.assertEqual(origin.depth, 238200.0)
        self.assertEqual(origin.depth_errors.uncertainty, 7180.0)
        self.assertEqual(origin.quality.used_station_count, 30)
        self.assertEqual(origin.region, 'Afghanistan-Tajikistan border region')
        self.assertEqual(ev.event_type, 'earthquake')

        self.assertEqual(mag.mag, 6.1)
        self.assertEqual(mag.magnitude_type, 'Mb')
        self.assertEqual(sta_mag.mag, 6.2)
        self.assertEqual(sta_mag.station_magnitude_type, 'Mb')
        self.assertEqual(sta_mag.waveform_id.get_seed_string(), '.AHRW..Z')

    def test_populate_waveform_id(self):
        fname = os.path.join(self.path, 'data', 'tele2.evt')
        # read invenroty - FUR with channels, WET without channels
        inv = read_inventory().select(channel='HH?')
        inv[0][1].channels = []
        cat = read_events(fname, inventory=inv,
                          default_seedid='BLA.{}.11.DH{}',
                          seedid_map={'UBR': 'BLB.{}.00.BH{}'}, warn=False)
        picks = cat[0].picks
        waveform_ids = [p.waveform_id.get_seed_string() for p in picks]
        self.assertIn('GR.FUR..HHZ', waveform_ids)
        self.assertIn('BLA.WET.11.DHZ', waveform_ids)
        self.assertIn('BLB.UBR.00.BHZ', waveform_ids)
        self.assertIn('BLA.WERD.11.DHZ', waveform_ids)
