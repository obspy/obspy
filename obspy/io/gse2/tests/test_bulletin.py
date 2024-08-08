#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The gse2.bulletin test suite.

:author:
    EOST (École et Observatoire des Sciences de la Terre)
:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""
import warnings

from obspy.core.event import read_events
from obspy.core.utcdatetime import UTCDateTime
from obspy.io.gse2.bulletin import _read_gse2, GSE2BulletinSyntaxError
from obspy.core.event.header import OriginDepthType, PickPolarity
from obspy.core.inventory.inventory import read_inventory
import pytest


class TestBulletin():
    """
    Test suite for obspy.io.gse2.bulletin
    """
    def test_catalog(self, datapath):
        """
        Test Catalog object.
        """
        filename = datapath / 'bulletin' / 'gse_2.0_standard.txt'
        catalog = _read_gse2(filename)
        assert catalog.resource_id == 'smi:local/event/evid'
        assert len(catalog.comments) == 1
        comment = catalog.comments[0]
        text = ("Reviewed Event Bulletin (REB) of the GSE_IDC for January 16, "
                "1995")
        assert comment.text == text
        assert comment.resource_id.id[:10] == 'smi:local/'

    def test_event(self, datapath):
        """
        Test Event object.
        """
        filename = datapath / 'bulletin' / 'gse_2.0_standard.txt'
        catalog = _read_gse2(filename)
        assert len(catalog) == 1
        event = catalog[0]
        assert event.resource_id == 'smi:local/event/280435'
        assert event.event_type == 'earthquake'
        assert event.event_type_certainty == "known"
        assert event.creation_info is not None
        assert event.preferred_origin_id == 'smi:local/origin/282672'
        assert event.preferred_magnitude_id == \
            'smi:local/origin/282672/magnitude/0'
        # event descriptions
        assert len(event.event_descriptions) == 1
        event_description = event.event_descriptions[0]
        assert event_description.text == \
            'GREECE-ALBANIA BORDER REGION'
        assert event_description.type == 'region name'
        # comments
        assert len(event.comments) == 1
        comment = event.comments[0]
        assert comment.text == 'GSE2.0:evtype=ke'
        assert comment.resource_id.id[:10] == 'smi:local/'

    def test_origin(self, datapath):
        """
        Test Origin object.
        """
        filename = datapath / 'bulletin' / 'gse_2.0_standard.txt'
        catalog = _read_gse2(filename)
        assert len(catalog) == 1
        assert len(catalog[0].origins) == 1
        origin = catalog[0].origins[0]
        assert origin.resource_id == 'smi:local/origin/282672'
        assert origin.time == UTCDateTime('1995-01-16T07:26:52.4')
        assert origin.time_errors.uncertainty == 12.69
        assert origin.latitude == 39.45
        assert origin.latitude_errors.uncertainty is None
        assert origin.longitude == 20.44
        assert origin.longitude_errors.uncertainty is None
        assert origin.depth == 66800
        assert origin.depth_errors.uncertainty == 83800
        assert origin.depth_type == OriginDepthType('from location')
        assert not origin.time_fixed
        assert not origin.epicenter_fixed
        assert origin.reference_system_id is None
        assert origin.method_id == 'smi:local/method/inversion'
        assert origin.earth_model_id is None
        assert origin.origin_type is None
        assert origin.region is None
        assert origin.evaluation_mode == "manual"
        assert origin.evaluation_status is None
        # quality
        assert origin.quality.associated_phase_count == 9
        assert origin.quality.used_phase_count == 9
        assert origin.quality.associated_station_count == 8
        assert origin.quality.used_station_count == 8
        assert origin.quality.depth_phase_count is None
        assert origin.quality.standard_error == 0.53
        assert origin.quality.azimuthal_gap == 322
        assert origin.quality.secondary_azimuthal_gap is None
        assert origin.quality.ground_truth_level is None
        assert origin.quality.minimum_distance == 10.56
        assert origin.quality.maximum_distance == 78.21
        assert origin.quality.median_distance is None
        # origin uncertainty
        u = origin.origin_uncertainty
        assert u.horizontal_uncertainty is None
        assert u.min_horizontal_uncertainty == 83700
        assert u.max_horizontal_uncertainty == 93600
        assert u.azimuth_max_horizontal_uncertainty == 27
        assert u.confidence_ellipsoid is None
        assert u.preferred_description == 'uncertainty ellipse'
        assert u.confidence_level is None
        # creation info
        assert origin.creation_info.author == 'GSE_IDC'
        # comments
        assert len(origin.comments) == 0
        # composite times
        assert len(origin.composite_times) == 0

    def test_pick(self, datapath):
        """
        Test Pick object.
        """
        filename = datapath / 'bulletin' / 'gse_2.0_standard.txt'
        catalog = _read_gse2(filename)
        assert len(catalog) == 1
        picks = catalog[0].picks
        assert len(picks) == 9
        # Test first Pick
        pick_1 = picks[0]
        assert pick_1.resource_id == 'smi:local/pick/3586432'
        assert pick_1.time == UTCDateTime('1995-01-16T07:29:20.7')
        # WaveformStreamId
        waveform_1 = pick_1.waveform_id
        assert waveform_1.network_code == 'XX'
        assert waveform_1.station_code == 'GERES'
        assert waveform_1.channel_code is None
        assert waveform_1.location_code is None
        assert waveform_1.resource_uri is None
        assert pick_1.filter_id is None
        assert pick_1.method_id is None
        assert pick_1.horizontal_slowness == 13.8
        assert pick_1.backazimuth == 163.7
        assert pick_1.slowness_method_id is None
        assert pick_1.onset == 'emergent'
        assert pick_1.phase_hint == 'P'
        assert pick_1.polarity is None
        assert pick_1.evaluation_mode == 'manual'
        assert pick_1.evaluation_status is None
        assert pick_1.creation_info is not None
        assert len(pick_1.comments) == 0
        # Test second Pick
        pick_2 = picks[1]
        assert pick_2.resource_id == 'smi:local/pick/3586513'
        assert pick_2.time == UTCDateTime('1995-01-16T07:31:17.5')
        # WaveformStreamId
        waveform_2 = pick_2.waveform_id
        assert waveform_2.network_code == 'XX'
        assert waveform_2.station_code == 'GERES'
        assert waveform_2.channel_code is None
        assert waveform_2.location_code is None
        assert waveform_2.resource_uri is None
        assert pick_2.filter_id is None
        assert pick_2.method_id is None
        assert pick_2.horizontal_slowness == 23.4
        assert pick_2.backazimuth == 153.4
        assert pick_2.slowness_method_id is None
        assert pick_2.onset is None
        assert pick_2.phase_hint == 'S'
        assert pick_2.polarity == PickPolarity.POSITIVE
        assert pick_2.evaluation_mode is None
        assert pick_2.evaluation_status is None
        assert pick_2.creation_info is not None
        assert len(pick_2.comments) == 0

    def test_arrival(self, datapath):
        """
        Test Arrival object.
        """
        filename = datapath / 'bulletin' / 'gse_2.0_standard.txt'
        catalog = _read_gse2(filename)
        assert len(catalog) == 1
        assert len(catalog[0].origins) == 1
        arrivals = catalog[0].origins[0].arrivals
        assert len(arrivals) == 9
        # Test first Arrival
        arrival_1 = arrivals[0]
        assert arrival_1.resource_id == \
            'smi:local/origin/282672/arrival/3586432'
        assert arrival_1.pick_id == 'smi:local/pick/3586432'
        assert arrival_1.phase == 'P'
        assert arrival_1.time_correction is None
        assert arrival_1.azimuth == 150.3
        assert arrival_1.distance == 10.56
        assert arrival_1.takeoff_angle is None
        assert arrival_1.time_residual == -0.2
        assert arrival_1.horizontal_slowness_residual == 0.1
        assert arrival_1.backazimuth_residual == 13.4
        assert arrival_1.time_weight == 1
        assert arrival_1.backazimuth_weight is None
        assert arrival_1.horizontal_slowness_weight is None
        assert arrival_1.earth_model_id is None
        assert arrival_1.creation_info is not None
        assert len(arrival_1.comments) == 0
        # Test second Arrival
        arrival_2 = arrivals[1]
        assert arrival_2.resource_id == \
            'smi:local/origin/282672/arrival/3586513'
        assert arrival_2.pick_id == 'smi:local/pick/3586513'
        assert arrival_2.phase == 'S'
        assert arrival_2.time_correction is None
        assert arrival_2.azimuth == 150.3
        assert arrival_2.distance == 10.56
        assert arrival_2.takeoff_angle is None
        assert arrival_2.time_residual == -0.6
        assert arrival_2.horizontal_slowness_residual == -1.0
        assert arrival_2.backazimuth_residual == 3.1
        assert arrival_2.time_weight is None
        assert arrival_2.backazimuth_weight == 1
        assert arrival_2.horizontal_slowness_weight is None
        assert arrival_2.earth_model_id is None
        assert arrival_2.creation_info is not None
        assert len(arrival_2.comments) == 0

    def test_magnitude(self, datapath):
        """
        Test Magnitude object.
        """
        filename = datapath / 'bulletin' / 'gse_2.0_standard.txt'
        catalog = _read_gse2(filename)
        assert len(catalog) == 1
        magnitudes = catalog[0].magnitudes
        assert len(magnitudes) == 2
        # Test first Magnitude
        mag_1 = magnitudes[0]
        assert mag_1.resource_id == 'smi:local/origin/282672/magnitude/0'
        assert mag_1.mag == 3.6
        assert mag_1.mag_errors.uncertainty == 0.2
        assert mag_1.magnitude_type == 'mb'
        assert mag_1.origin_id == 'smi:local/origin/282672'
        assert mag_1.method_id is None
        assert mag_1.station_count == 3
        assert mag_1.azimuthal_gap is None
        assert mag_1.evaluation_mode is None
        assert mag_1.evaluation_status is None
        assert mag_1.creation_info is not None
        assert len(mag_1.comments) == 0
        assert len(mag_1.station_magnitude_contributions) == 3
        # Test second Magnitude
        mag_2 = magnitudes[1]
        assert mag_2.resource_id == 'smi:local/origin/282672/magnitude/1'
        assert mag_2.mag == 4.0
        assert mag_2.mag_errors.uncertainty is None
        assert mag_2.magnitude_type == 'ML'
        assert mag_2.origin_id == 'smi:local/origin/282672'
        assert mag_2.method_id is None
        assert mag_2.station_count == 1
        assert mag_2.azimuthal_gap is None
        assert mag_2.evaluation_mode is None
        assert mag_2.evaluation_status is None
        assert mag_2.creation_info is not None
        assert len(mag_2.comments) == 0
        assert len(mag_2.station_magnitude_contributions) == 1

    def test_station_magnitude(self, datapath):
        """
        Test StationMagnitude object.
        """
        filename = datapath / 'bulletin' / 'gse_2.0_standard.txt'
        catalog = _read_gse2(filename)
        assert len(catalog) == 1
        station_magnitudes = catalog[0].station_magnitudes
        assert len(station_magnitudes) == 4
        # Test first StationMagnitude
        sta_mag_1 = station_magnitudes[0]
        assert sta_mag_1.resource_id == 'smi:local/magnitude/station/3586432/0'
        assert sta_mag_1.origin_id == 'smi:local/origin/282672'
        assert sta_mag_1.mag == 4.0
        assert sta_mag_1.station_magnitude_type == 'ML'
        assert sta_mag_1.amplitude_id == 'smi:local/amplitude/3586432'
        assert sta_mag_1.method_id is None
        assert sta_mag_1.creation_info is not None
        assert len(sta_mag_1.comments) == 0

        waveform_1 = sta_mag_1.waveform_id
        assert waveform_1.network_code == 'XX'
        assert waveform_1.station_code == 'GERES'
        assert waveform_1.channel_code is None
        assert waveform_1.location_code is None
        assert waveform_1.resource_uri is None

        # Test second StationMagnitude
        sta_mag_2 = station_magnitudes[1]
        assert sta_mag_2.resource_id == 'smi:local/magnitude/station/3586555/0'
        assert sta_mag_2.origin_id == 'smi:local/origin/282672'
        assert sta_mag_2.mag == 3.7
        assert sta_mag_2.station_magnitude_type == 'mb'
        assert sta_mag_2.amplitude_id == 'smi:local/amplitude/3586555'
        assert sta_mag_2.method_id is None
        assert sta_mag_2.creation_info is not None
        assert len(sta_mag_2.comments) == 0
        # Test with a file containig Mag2 but not Mag1
        fields = {
            'line_1': {
                'author': slice(105, 113),
                'id': slice(114, 123),
            },
            'line_2': {
                'az': slice(40, 46),
                'antype': slice(105, 106),
                'loctype': slice(107, 108),
                'evtype': slice(109, 111),
            },
            'arrival': {
                'amp': slice(94, 104),
            },
        }
        filename = datapath / 'bulletin' / 'event.txt'
        catalog = _read_gse2(filename, fields=fields,
                             res_id_prefix="quakeml:ldg",
                             event_point_separator=True)
        station_magnitudes = catalog[0].station_magnitudes
        assert len(station_magnitudes) == 5
        sta_mag_3 = station_magnitudes[0]
        assert sta_mag_3.resource_id.id == \
            'quakeml:ldg/magnitude/station/6867444/1'
        assert sta_mag_3.origin_id.id == 'quakeml:ldg/origin/375628'
        assert sta_mag_3.mag == 1.7
        assert sta_mag_3.station_magnitude_type == 'Md'
        assert sta_mag_3.amplitude_id.id == 'quakeml:ldg/amplitude/6867444'
        assert sta_mag_3.method_id is None
        assert sta_mag_3.waveform_id.get_seed_string() == 'XX.MBDF..'
        assert sta_mag_3.creation_info is not None
        assert len(sta_mag_3.comments) == 0

        waveform_2 = sta_mag_2.waveform_id
        assert waveform_2.network_code == 'XX'
        assert waveform_2.station_code == 'FINES'
        assert waveform_2.channel_code is None
        assert waveform_2.location_code is None
        assert waveform_2.resource_uri is None

    def test_amplitude(self, datapath):
        """
        Test Amplitude object.
        """
        filename = datapath / 'bulletin' / 'gse_2.0_standard.txt'
        catalog = _read_gse2(filename)
        assert len(catalog) == 1
        amplitudes = catalog[0].amplitudes
        # test a new feature: don't store an object amplitude if the magnitude
        # type is not defined
        assert len(amplitudes) == 6
        # Test first amplitude
        amplitude_1 = amplitudes[0]
        assert amplitude_1.resource_id == 'smi:local/amplitude/3586432'
        assert amplitude_1.generic_amplitude == 0.6
        assert amplitude_1.type is None
        assert amplitude_1.category is None
        assert amplitude_1.unit is None
        assert amplitude_1.method_id is None
        assert amplitude_1.period == 0.3
        assert amplitude_1.snr == 6.8
        assert amplitude_1.time_window is None
        assert amplitude_1.pick_id == 'smi:local/pick/3586432'
        # WaveformStreamId
        waveform_1 = amplitude_1.waveform_id
        assert waveform_1.network_code == 'XX'
        assert waveform_1.station_code == 'GERES'
        assert waveform_1.channel_code is None
        assert waveform_1.location_code is None
        assert waveform_1.resource_uri is None
        assert amplitude_1.filter_id is None
        assert amplitude_1.scaling_time is None
        assert amplitude_1.magnitude_hint == 'ML'
        assert amplitude_1.evaluation_mode is None
        assert amplitude_1.evaluation_status is None
        assert amplitude_1.creation_info is not None
        assert len(amplitude_1.comments) == 0
        # Test second amplitude
        amplitude_2 = amplitudes[1]
        assert amplitude_2.resource_id == 'smi:local/amplitude/3586513'
        assert amplitude_2.generic_amplitude == 2.9
        assert amplitude_2.type is None
        assert amplitude_2.category is None
        assert amplitude_2.unit is None
        assert amplitude_2.method_id is None
        assert amplitude_2.period == 0.6
        assert amplitude_2.snr == 4.9
        assert amplitude_2.time_window is None
        assert amplitude_2.pick_id == 'smi:local/pick/3586513'
        # WaveformStreamId
        waveform_2 = amplitude_2.waveform_id
        assert waveform_2.network_code == 'XX'
        assert waveform_2.station_code == 'GERES'
        assert waveform_2.channel_code is None
        assert waveform_2.location_code is None
        assert waveform_2.resource_uri is None
        assert amplitude_2.filter_id is None
        assert amplitude_2.scaling_time is None
        assert amplitude_2.magnitude_hint is None
        assert amplitude_2.evaluation_mode is None
        assert amplitude_2.evaluation_status is None
        assert amplitude_2.creation_info is not None
        assert len(amplitude_2.comments) == 0

    def test_several_events(self, datapath):
        """
        Test with several events.
        """
        filename = datapath / 'bulletin' / 'gse_2.0_2_events.txt'
        catalog = _read_gse2(filename)
        assert len(catalog) == 2
        # Test firt event
        event_1 = catalog[0]
        assert event_1.resource_id == 'smi:local/event/280435'
        assert len(event_1.event_descriptions) == 1
        assert event_1.event_descriptions[0].text == \
            'GREECE-ALBANIA BORDER REGION'
        assert len(event_1.comments) == 1
        assert len(event_1.picks) == 9
        assert len(event_1.amplitudes) == 6
        assert len(event_1.origins) == 1
        assert len(event_1.magnitudes) == 2
        assert len(event_1.station_magnitudes) == 4
        # Test second event
        event_2 = catalog[1]
        assert event_2.resource_id == 'smi:local/event/280436'
        assert len(event_2.event_descriptions) == 1
        assert event_2.event_descriptions[0].text == \
            'VANCOUVER ISLAND REGION'
        assert len(event_2.comments) == 1
        assert len(event_2.picks) == 7
        assert len(event_2.amplitudes) == 5
        assert len(event_2.origins) == 1
        assert len(event_2.magnitudes) == 1
        assert len(event_2.station_magnitudes) == 2

    def test_parameters(self, datapath):
        filename = datapath / 'bulletin' / 'gse_2.0_standard.txt'
        fields = {
            'line_1': {
                'time': slice(0, 21),
                'time_fixf': slice(22, 23),
                'lat': slice(25, 33),
                'lon': slice(34, 43),
                'epicenter_fixf': slice(44, 45),
                'depth': slice(47, 52),
                'depth_fixf': slice(53, 54),
                'n_def': slice(56, 60),
                'n_sta': slice(61, 65),
                'gap': slice(66, 69),
                'mag_type_1': slice(71, 73),
                'mag_1': slice(73, 77),
                'mag_n_sta_1': slice(78, 80),
                'mag_type_2': slice(82, 84),
                'mag_2': slice(84, 88),
                'mag_n_sta_2': slice(89, 91),
                'mag_type_3': slice(93, 95),
                'mag_3': slice(95, 99),
                'mag_n_sta_3': slice(100, 102),
                'author': slice(104, 112),
                'id': slice(114, 122),
            },
            'line_2': {
                'rms': slice(5, 10),
                'ot_error': slice(15, 21),
                's_major': slice(25, 31),
                's_minor': slice(32, 38),
                'az': slice(40, 43),
                'depth_err': slice(49, 54),
                'min_dist': slice(56, 62),
                'max_dist': slice(63, 69),
                'mag_err_1': slice(74, 77),
                'mag_err_2': slice(85, 88),
                'mag_err_3': slice(96, 99),
                'antype': slice(104, 105),
                'loctype': slice(106, 107),
                'evtype': slice(108, 110),
            },
            'arrival': {
                'sta': slice(0, 5),
                'dist': slice(6, 12),
                'ev_az': slice(13, 18),
                'picktype': slice(19, 20),
                'direction': slice(20, 21),
                'detchar': slice(21, 22),
                'phase': slice(23, 30),
                'time': slice(31, 52),
                't_res': slice(53, 58),
                'azim': slice(59, 64),
                'az_res': slice(65, 71),
                'slow': slice(72, 77),
                's_res': slice(78, 83),
                't_def': slice(84, 85),
                'a_def': slice(85, 86),
                's_def': slice(86, 87),
                'snr': slice(88, 93),
                'amp': slice(94, 103),
                'per': slice(104, 109),
                'mag_type_1': slice(110, 112),
                'mag_1': slice(112, 116),
                'mag_type_2': slice(117, 119),
                'mag_2': slice(119, 123),
                'id': slice(124, 132),
            },
        }
        catalog = _read_gse2(filename, None, 'TZ', '00', 'SHZ', 'quakeml:idc',
                             fields, False, 'AGE')
        assert len(catalog) == 1
        # Test Comment ResourceIdentifier
        assert len(catalog.comments) == 1
        comment = catalog.comments[0]
        assert comment.resource_id.id[:12] == 'quakeml:idc/'
        # Test Event ResourceIdentifier
        event = catalog[0]
        assert event.resource_id == 'quakeml:idc/event/280435'
        assert event.creation_info.agency_id == 'AGE'
        # Test Origin ResourceIdentifier
        assert len(event.origins) == 1
        origin = event.origins[0]
        assert origin.resource_id == 'quakeml:idc/origin/282672'
        assert origin.method_id == 'quakeml:idc/method/inversion'
        # Test Pick ResourceIdentifier
        assert len(event.picks) == 9
        pick = event.picks[0]
        assert pick.resource_id == 'quakeml:idc/pick/3586432'
        # Test Arrival ResourceIdentifier
        assert len(origin.arrivals) == 9
        arrival = origin.arrivals[0]
        assert arrival.resource_id == \
            'quakeml:idc/origin/282672/arrival/3586432'
        # Test Magnitude ResourceIdentifier
        assert len(event.magnitudes) == 2
        magnitude = event.magnitudes[0]
        assert magnitude.resource_id == 'quakeml:idc/origin/282672/magnitude/0'
        # Test StationMagnitude ResourceIdentifier
        assert len(event.station_magnitudes) == 4
        sta_mag = event.station_magnitudes[0]
        assert sta_mag.resource_id == 'quakeml:idc/magnitude/station/3586432/0'
        # Test Amplitude ResourceIdentifier
        assert len(event.amplitudes) == 6
        amplitude = event.amplitudes[0]
        assert amplitude.resource_id == 'quakeml:idc/amplitude/3586432'
        # Test network code
        waveform = pick.waveform_id
        assert waveform.network_code == 'TZ'
        assert waveform.channel_code == 'SHZ'
        assert waveform.location_code == '00'

    def test_non_standard_format(self, datapath):
        """
        Test non-standard GSE2 format which can normally be parsed too.
        """
        filename = datapath / 'bulletin' / 'gse_2.0_non_standard.txt'
        fields = {
            'line_1': {
                'author': slice(105, 113),
                'id': slice(114, 123),
            },
            'line_2': {
                'az': slice(40, 46),
                'antype': slice(105, 106),
                'loctype': slice(107, 108),
                'evtype': slice(109, 111),
            },
            'arrival': {
                'amp': slice(94, 104),
            },
        }
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            catalog = _read_gse2(filename, fields=fields,
                                 event_point_separator=True)
        assert len(catalog) == 2
        event = catalog[0]
        assert len(event.origins) == 1
        origin = event.origins[0]
        # Test fields with non-standard indexes
        assert origin.creation_info.author == 'bulletin_test'
        assert origin.resource_id == 'smi:local/origin/282672'
        assert origin.quality.standard_error == 0.55
        uncertainty = origin.origin_uncertainty
        assert uncertainty.azimuth_max_horizontal_uncertainty == 27.05
        assert origin.evaluation_mode == "manual"
        assert origin.method_id == 'smi:local/method/inversion'
        assert event.event_type == 'earthquake'
        assert event.event_type_certainty == "known"
        assert len(event.picks) == 9
        pick = event.picks[0]
        assert pick.phase_hint == 'Pg'
        assert len(event.amplitudes) == 4
        amplitude = event.amplitudes[0]
        assert amplitude.generic_amplitude == 2.9

    def test_inventory(self, datapath):
        filename = datapath / 'bulletin' / 'gse_2.0_non_standard.txt'
        inventory_filename = datapath / 'bulletin' / 'inventory.xml'
        inventory = read_inventory(inventory_filename)
        fields = {
            'line_1': {
                'author': slice(105, 113),
                'id': slice(114, 123),
            },
            'line_2': {
                'az': slice(40, 46),
                'antype': slice(105, 106),
                'loctype': slice(107, 108),
                'evtype': slice(109, 111),
            },
            'arrival': {
                'amp': slice(94, 104),
            },
        }
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            catalog = _read_gse2(filename, inventory, fields=fields,
                                 event_point_separator=True)
        assert len(catalog) == 2
        # Test a station present in the inventory
        event = catalog[0]
        assert len(event.picks) == 9
        pick = event.picks[0]
        waveform = pick.waveform_id
        assert waveform.network_code == 'ZU'
        assert waveform.channel_code == 'SHZ'
        assert waveform.location_code == '1'
        # Test a station not present in the inventory
        pick_2 = event.picks[2]
        waveform_2 = pick_2.waveform_id
        assert waveform_2.network_code == 'XX'
        assert waveform_2.channel_code is None
        assert waveform_2.location_code is None

    def test_inventory_with_multiple_channels(self, datapath):
        filename = datapath / 'bulletin' / 'gse_2.0_non_standard.txt'
        inventory_filename = \
            datapath / 'bulletin' / 'inventory_multiple_channels.xml'
        inventory = read_inventory(inventory_filename)
        fields = {
            'line_1': {
                'author': slice(105, 113),
                'id': slice(114, 123),
            },
            'line_2': {
                'az': slice(40, 46),
                'antype': slice(105, 106),
                'loctype': slice(107, 108),
                'evtype': slice(109, 111),
            },
            'arrival': {
                'amp': slice(94, 104),
            },
        }
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            catalog = _read_gse2(filename, inventory, fields=fields,
                                 event_point_separator=True)
        assert len(catalog) == 2
        # Test a station present in the inventory
        event = catalog[0]
        assert len(event.picks) == 9
        pick = event.picks[0]
        waveform = pick.waveform_id
        assert waveform.network_code == 'ZU'
        assert waveform.channel_code == 'HHZ'
        assert waveform.location_code == '1'
        # Test a station with several channels
        pick_2 = event.picks[2]
        waveform_2 = pick_2.waveform_id
        assert waveform_2.network_code == 'ZU'
        assert waveform_2.channel_code is None
        assert waveform_2.location_code is None
        # Test a station not present in the inventory
        pick_3 = event.picks[3]
        waveform_3 = pick_3.waveform_id
        assert waveform_3.network_code == 'XX'
        assert waveform_3.channel_code is None
        assert waveform_3.location_code is None

    def test_several_begin(self, datapath):
        """
        Test with several events.
        """
        filename = datapath / 'bulletin' / 'gse_2.0_2_begins.txt'
        catalog = _read_gse2(filename)
        assert len(catalog) == 2
        # Test firt event
        event_1 = catalog[0]
        assert event_1.resource_id == 'smi:local/event/280435'
        assert len(event_1.event_descriptions) == 1
        assert event_1.event_descriptions[0].text == \
            'GREECE-ALBANIA BORDER REGION'
        assert len(event_1.comments) == 1
        assert len(event_1.picks) == 9
        assert len(event_1.amplitudes) == 6
        assert len(event_1.origins) == 1
        assert len(event_1.magnitudes) == 2
        assert len(event_1.station_magnitudes) == 4
        # Test second event
        event_2 = catalog[1]
        assert event_2.resource_id == 'smi:local/event/280436'
        assert len(event_2.event_descriptions) == 1
        assert event_2.event_descriptions[0].text == \
            'VANCOUVER ISLAND REGION'
        assert len(event_2.comments) == 1
        assert len(event_2.picks) == 7
        assert len(event_2.amplitudes) == 5
        assert len(event_2.origins) == 1
        assert len(event_2.magnitudes) == 1
        assert len(event_2.station_magnitudes) == 2

    def test_read_events(self, datapath):
        """
        Tests reading a GSE2.0 document via read_events.
        """
        filename = datapath / 'bulletin' / 'gse_2.0_2_events.txt'
        catalog = read_events(filename)
        assert len(catalog) == 2

    def test_incomplete_file(self, datapath):
        filename = datapath / 'bulletin' / 'gse_2.0_incomplete.txt'
        with pytest.raises(GSE2BulletinSyntaxError):
            _read_gse2(filename)
