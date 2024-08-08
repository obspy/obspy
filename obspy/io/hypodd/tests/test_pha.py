# -*- coding: utf-8 -*-
import warnings

from obspy import read_events, read_inventory, UTCDateTime
from obspy.core.event import Catalog, Event, Origin, Pick, WaveformStreamID
from obspy.core.util import NamedTemporaryFile
from obspy.io.hypodd import pha
import pytest


class TestPHA():
    """
    Test suite for obspy.io.hypodd.pha
    """
    @pytest.fixture(autouse=True, scope="function")
    def setup(self, testdata):
        self.fname = testdata['example.pha']
        self.fname2 = testdata['60s_nan.pha']

    def test_is_pha(self):
        assert pha._is_pha(self.fname)

    def test_is_not_pha(self, datapath):
        fname = datapath.parent / 'test_pha.py'
        assert not pha._is_pha(fname)

    def test_read_pha(self):
        cat = read_events(self.fname)
        assert len(cat) == 2
        event = cat[0]
        assert len(event.origins) == 1
        assert len(event.magnitudes) == 1
        ori = event.preferred_origin()
        mag = event.preferred_magnitude()
        assert str(ori.time) == '2025-05-14T14:35:35.510000Z'
        assert ori.latitude == 40.2254
        assert ori.longitude == 10.4496
        assert ori.depth == 9408.0
        assert mag.mag == 3.50
        assert len(event.picks) == 2
        assert len(ori.arrivals) == 2
        target = {'FUR': (3.52199909, 1.0000, 'P'),
                  'WET': (5.86199909, 1.0000, 'S')}
        for arr in ori.arrivals:
            pick = arr.pick_id.get_referred_object()
            sta = pick.waveform_id.station_code
            reltime = pick.time - ori.time
            assert round(abs(reltime-target[sta][0]), 4) == 0
            assert arr.time_weight == target[sta][1]
            assert arr.phase == target[sta][2]
            assert arr.phase == pick.phase_hint
        event = cat[1]
        ori = event.preferred_origin()
        assert round(abs(ori.latitude_errors.uncertainty-0.0045), 5) == 0
        assert ori.latitude_errors.uncertainty < \
            ori.longitude_errors.uncertainty
        assert ori.depth_errors.uncertainty == 400
        assert ori.quality.standard_error == 0.3
        assert ori.quality.associated_phase_count == 2

    def test_60s_nan(self):
        """
        issue 2627
        """
        cat = read_events(self.fname2)
        event = cat[0]
        assert len(event.origins) == 1
        assert len(event.magnitudes) == 1
        ori = event.preferred_origin()
        assert str(ori.time) == '2025-05-14T14:36:00.000000Z'
        event = cat[1]
        assert len(event.origins) == 1
        assert len(event.magnitudes) == 0

    def test_populate_waveform_id(self):
        inv = read_inventory()
        with warnings.catch_warnings(record=True) as ws:
            warnings.resetwarnings()
            cat = read_events(self.fname, inventory=inv)
            assert len(ws) == 4
        # FUR with channels, WET without channels
        inv = read_inventory().select(channel='HH?')
        inv[0][1].channels = []
        cat = read_events(self.fname, inventory=inv,
                          default_seedid='BLA.{}.11.DH{}',
                          seedid_map={'UBR': 'BLB.{}.00.BH{}'}, warn=False)
        assert len(cat) == 2
        picks = cat[0].picks + cat[1].picks
        assert len(picks) == 4
        waveform_ids = [p.waveform_id.get_seed_string() for p in picks]
        assert 'GR.FUR..HHZ' in waveform_ids
        assert 'BLA.WET.11.DHN' in waveform_ids
        assert 'BLB.UBR.00.BHZ' in waveform_ids
        assert 'BLA.WERD.11.DH' in waveform_ids

    def test_eventid_map(self):
        cat = read_events(self.fname) + read_events(self.fname2)
        cat[0].resource_id = 'X'
        cat[1].resource_id = '2f'
        cat[2].resource_id = 'Y'
        cat[3].resource_id = '1234567890Z'
        with NamedTemporaryFile() as tf:
            tempfile = tf.name
            with pytest.warns(UserWarning, match='Missing mag'):
                eventid_map = cat.write(tempfile, 'HYPODDPHA')
            cat2 = read_events(tempfile, eventid_map=eventid_map)
            cat3 = read_events(tempfile)
            with pytest.warns(UserWarning, match='Missing mag'):
                eventid_map2 = cat.write(tempfile, 'HYPODDPHA',
                                         eventid_map=eventid_map)
        assert cat2[0].resource_id.id.split('/')[-1] == 'X'
        assert cat2[1].resource_id.id.split('/')[-1] == '2f'
        assert cat2[2].resource_id.id.split('/')[-1] == 'Y'
        assert cat2[3].resource_id.id.split('/')[-1] == '1234567890Z'
        assert cat3[0].resource_id.id.split('/')[-1] == '1'
        assert cat3[1].resource_id.id.split('/')[-1] == '2'
        assert cat3[2].resource_id.id.split('/')[-1] == '3'
        assert cat3[3].resource_id.id.split('/')[-1] == '123456789'
        assert eventid_map2 == eventid_map

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
        assert filedata2.replace(' ', '') == filedata.replace(' ', '')

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
            with pytest.warns(UserWarning, match='Missing mag'):
                cat.write(tempfile, 'HYPODDPHA')
            cat2 = read_events(tempfile)
        assert len(cat2) == 1
        assert len(cat2[0].picks) == 1
