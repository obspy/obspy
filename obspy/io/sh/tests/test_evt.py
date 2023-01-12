from obspy import read_events, read_inventory
from obspy.io.sh.evt import _is_evt


class TestEvt():
    def test_is_evt_file(self, datapath):
        for fname in datapath.glob('*.evt'):
            assert _is_evt(fname)

    def test_local_event1(self, testdata):
        fname = testdata['local1.evt']
        cat = read_events(fname)
        assert len(cat) == 2

    def test_local_event2(self, testdata):
        fname = testdata['local2.evt']
        cat = read_events(fname)
        assert len(cat) == 1
        ev = cat[0]
        mag = ev.preferred_magnitude()
        origin = ev.preferred_origin()
        oc = origin.origin_uncertainty
        assert mag.mag == 0.6
        assert mag.magnitude_type == 'ML'
        assert round(origin.longitude_errors.uncertainty, 3) == 0.013
        assert round(origin.latitude_errors.uncertainty, 3) == 0.013
        assert oc.min_horizontal_uncertainty == 20
        assert oc.max_horizontal_uncertainty == 20
        assert oc.azimuth_max_horizontal_uncertainty == 75.70
        assert origin.arrivals[0].time_weight == 4.0

    def test_tele_event1(self, testdata):
        fname = testdata['tele1.evt']
        cat = read_events(fname)
        assert len(cat) == 1

    def test_tele_event2(self, testdata):
        # untested field: sign
        fname = testdata['tele2.evt']
        cat = read_events(fname)
        assert len(cat) == 1
        ev = cat[0]
        origin = ev.preferred_origin()
        mag = ev.preferred_magnitude()
        sta_mag = ev.station_magnitudes[0]
        pick = ev.picks[66]
        arrival = origin.arrivals[66]
        arrival = origin.arrivals[191]
        assert len(ev.picks) == 195
        assert len(ev.station_magnitudes) == 38
        assert len(origin.arrivals) == 195

        assert str(pick.time) == '2015-08-10T10:14:32.918000Z'
        assert pick.phase_hint == 'T'
        assert pick.onset == 'emergent'
        assert pick.evaluation_mode == 'manual'
        assert str(pick.filter_id) == 'G_WWSSN_SP'
        assert pick.waveform_id.get_seed_string() == '.GRA1..Z'
        assert origin.arrivals[66].pick_id == pick.resource_id
        assert arrival.phase == 'sS'
        assert arrival.azimuth == 307.6
        assert arrival.distance == 48.132
        assert str(origin.time) == '2015-08-10T10:05:25.808000Z'
        assert origin.latitude == 36.23
        assert origin.longitude == 71.38

        assert origin.depth == 238200.0
        assert origin.depth_errors.uncertainty == 7180.0
        assert origin.quality.used_station_count == 30
        assert origin.region == 'Afghanistan-Tajikistan border region'
        assert ev.event_type == 'earthquake'

        assert mag.mag == 6.1
        assert mag.magnitude_type == 'Mb'
        assert sta_mag.mag == 6.2
        assert sta_mag.station_magnitude_type == 'Mb'
        assert sta_mag.waveform_id.get_seed_string() == '.AHRW..Z'

    def test_populate_waveform_id(self, testdata):
        fname = testdata['tele2.evt']
        # read invenroty - FUR with channels, WET without channels
        inv = read_inventory().select(channel='HH?')
        inv[0][1].channels = []
        cat = read_events(fname, inventory=inv,
                          default_seedid='BLA.{}.11.DH{}',
                          seedid_map={'UBR': 'BLB.{}.00.BH{}'}, warn=False)
        picks = cat[0].picks
        waveform_ids = [p.waveform_id.get_seed_string() for p in picks]
        assert 'GR.FUR..HHZ' in waveform_ids
        assert 'BLA.WET.11.DHZ' in waveform_ids
        assert 'BLB.UBR.00.BHZ' in waveform_ids
        assert 'BLA.WERD.11.DHZ' in waveform_ids
