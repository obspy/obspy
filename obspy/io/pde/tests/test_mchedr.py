# -*- coding: utf-8 -*-
import warnings

import pytest

from obspy.core.event import ResourceIdentifier, read_events
from obspy.core.utcdatetime import UTCDateTime
from obspy.core.util.base import NamedTemporaryFile
from obspy.io.pde.mchedr import _read_mchedr
from obspy.io.quakeml.core import _read_quakeml, _write_quakeml

# lxml < 2.3 seems not to ship with RelaxNG schema parser and namespace support
IS_RECENT_LXML = False
try:
    from lxml.etree import __version__
    version = float(__version__.rsplit('.', 1)[0])
    if version >= 2.3:
        IS_RECENT_LXML = True
except Exception:
    pass


class TestMchedr():
    """
    Test suite for obspy.mchedr
    """
    @pytest.fixture(autouse=True, scope="function")
    def setup(self, testdata):
        # read the mchedr file once for all
        self.catalog = _read_mchedr(testdata['mchedr.dat'])

    def test_catalog(self):
        assert len(self.catalog) == 1
        assert str(self.catalog) == \
            '''1 Event(s) in Catalog:
2012-01-01T05:27:55.980000Z | +31.456, +138.072 | 6.2  Mb'''

    def test_event(self):
        """
        Tests Event object.
        """
        event = self.catalog[0]
        assert event.resource_id == \
            ResourceIdentifier(
                id='quakeml:us.anss.org/event/20120101052755.98')
        # enums
        assert event.event_type is None
        assert event.event_type_certainty is None
        # comments
        assert len(event.comments) == 1
        c = event.comments
        assert c[0].text == 'MW 6.8 (WCMT), 6.8 (UCMT), 6.8 (GCMT). \
Felt (V) at Chiba; (IV) at Fussa, Kawasaki, Saitama, Tokyo, \
Yokohama and Yokosuka; (III) at Ebina, Zama and Zushi; (II) \
at Misawa and Narita, Honshu. Recorded (4 JMA) in Chiba, Fukushima, \
Gumma, Ibaraki, Kanagawa, Miyagi, Saitama, Tochigi and Tokyo.'
        # event descriptions
        assert len(event.event_descriptions) == 2
        d = event.event_descriptions
        assert d[0].text == 'SOUTHEAST OF HONSHU, JAPAN'
        assert d[0].type == 'region name'
        assert d[1].text == '211'
        assert d[1].type == 'Flinn-Engdahl region'
        # creation info
        assert event.creation_info is None

    def test_origin(self):
        """
        Tests Origin object.
        """
        assert len(self.catalog[0].origins) == 4
        origin = self.catalog[0].origins[0]
        assert origin.resource_id == \
            ResourceIdentifier(
                id='quakeml:us.anss.org/origin/20120101052755.98')
        assert origin.origin_type == 'hypocenter'
        assert origin.time == \
            UTCDateTime(2012, 1, 1, 5, 27, 55, 980000)
        assert origin.latitude == 31.456
        assert round(abs(origin.latitude_errors.uncertainty-0.0155), 3) == 0
        assert origin.longitude == 138.072
        assert round(abs(origin.longitude_errors.uncertainty-0.0173), 3) == 0
        assert origin.depth == 365300.0
        assert origin.depth_errors.uncertainty == 2700.0
        assert origin.depth_type == 'from location'
        assert origin.method_id is None
        assert origin.time_fixed is None
        assert origin.epicenter_fixed is None
        assert origin.earth_model_id == \
            ResourceIdentifier(
                id='quakeml:us.anss.org/earthmodel/ak135')
        assert origin.evaluation_mode is None
        assert origin.evaluation_status is None
        assert origin.origin_type == 'hypocenter'
        # composite times
        assert len(origin.composite_times) == 0
        # quality
        assert origin.quality.used_station_count == 628
        assert origin.quality.standard_error == 0.84
        assert origin.quality.azimuthal_gap == 10.8
        assert origin.quality.maximum_distance == 29.1
        assert origin.quality.minimum_distance == 2.22
        assert origin.quality.associated_phase_count == 52
        assert origin.quality.associated_station_count == 628
        assert origin.quality.depth_phase_count == 0
        assert origin.quality.secondary_azimuthal_gap is None
        assert origin.quality.ground_truth_level is None
        assert origin.quality.median_distance is None
        # comments
        assert len(origin.comments) == 0
        # creation info
        assert origin.creation_info.author is None
        assert origin.creation_info.agency_id == 'USGS-NEIC'
        assert origin.creation_info.author_uri is None
        assert origin.creation_info.agency_uri is None
        assert origin.creation_info.creation_time is None
        assert origin.creation_info.version is None
        # origin uncertainty
        u = origin.origin_uncertainty
        assert u.preferred_description == 'confidence ellipsoid'
        assert u.horizontal_uncertainty is None
        assert u.min_horizontal_uncertainty is None
        assert u.max_horizontal_uncertainty is None
        assert u.azimuth_max_horizontal_uncertainty is None
        # confidence ellipsoid
        c = u.confidence_ellipsoid
        assert c.semi_intermediate_axis_length == 2750.0
        # c.major_axis_rotation is computed during file reading:
        assert round(abs(c.major_axis_rotation-170.5), 3) == 0
        assert c.major_axis_plunge == 76.06
        assert c.semi_minor_axis_length == 2210.0
        assert c.semi_major_axis_length == 4220.0
        assert c.major_axis_azimuth == 292.79

    def test_magnitude(self):
        """
        Tests Magnitude object.
        """
        assert len(self.catalog[0].magnitudes) == 3
        mag = self.catalog[0].magnitudes[0]
        assert mag.resource_id == \
            ResourceIdentifier(
                id='quakeml:us.anss.org/magnitude/20120101052755.98/mb')
        assert mag.mag == 6.2
        assert mag.mag_errors.uncertainty is None
        assert mag.magnitude_type == 'Mb'
        assert mag.station_count == 294
        assert mag.evaluation_status is None
        # comments
        assert len(mag.comments) == 0
        # creation info
        assert mag.creation_info.author is None
        assert mag.creation_info.agency_id == 'USGS-NEIC'
        assert mag.creation_info.author_uri is None
        assert mag.creation_info.agency_uri is None
        assert mag.creation_info.creation_time is None
        assert mag.creation_info.version is None

    def test_stationmagnitude(self):
        """
        Tests StationMagnitude object.
        """
        assert len(self.catalog[0].station_magnitudes) == 19
        mag = self.catalog[0].station_magnitudes[0]
        assert mag.mag == 6.6
        assert mag.mag_errors.uncertainty is None
        assert mag.station_magnitude_type == 'Mb'
        assert mag.waveform_id.station_code == 'MDJ'
        assert mag.creation_info is None

    def test_amplitude(self):
        """
        Tests Amplitude object.
        """
        assert len(self.catalog[0].station_magnitudes) == 19
        amp = self.catalog[0].amplitudes[0]
        assert round(abs(amp.generic_amplitude-3.94502e-06), 7) == 0
        assert amp.type == 'AB'
        assert amp.period == 1.3
        assert amp.magnitude_hint == 'Mb'
        assert amp.waveform_id.station_code == 'MDJ'
        assert amp.creation_info is None

    def test_arrival(self):
        """
        Tests Arrival object.
        """
        assert len(self.catalog[0].origins[0].arrivals) == 52
        ar = self.catalog[0].origins[0].arrivals[0]
        assert ar.phase == 'Pn'
        assert ar.azimuth == 41.4
        assert ar.distance == 2.22
        assert ar.takeoff_angle is None
        assert ar.takeoff_angle_errors.uncertainty is None
        assert ar.time_residual == -1.9
        assert ar.horizontal_slowness_residual is None
        assert ar.backazimuth_residual is None
        assert ar.time_weight is None
        assert ar.horizontal_slowness_weight is None
        assert ar.backazimuth_weight is None
        assert ar.earth_model_id == \
            ResourceIdentifier('quakeml:us.anss.org/earthmodel/ak135')
        assert len(ar.comments) == 0

    def test_pick(self):
        """
        Tests Pick object.
        """
        assert len(self.catalog[0].picks) == 52
        pick = self.catalog[0].picks[0]
        assert pick.time == UTCDateTime(2012, 1, 1, 5, 28, 48, 180000)
        assert pick.time_errors.uncertainty is None
        assert pick.waveform_id.station_code == 'JHJ2'
        assert round(abs(pick.backazimuth--138.6), 7) == 0
        assert pick.onset == 'emergent'
        assert pick.phase_hint == 'Pn'
        assert pick.polarity is None
        assert pick.evaluation_mode is None
        assert pick.evaluation_status is None
        assert len(pick.comments) == 0

    def test_focalmechanism(self):
        """
        Tests FocalMechanism object.
        """
        assert len(self.catalog[0].focal_mechanisms) == 4
        fm = self.catalog[0].focal_mechanisms[0]
        assert fm.resource_id == \
            ResourceIdentifier(
                id='quakeml:us.anss.org/focalmechanism/'
                   '20120101052755.98/ucmt/mwc')
        # general
        assert fm.waveform_id == []
        assert fm.triggering_origin_id is None
        assert fm.azimuthal_gap is None
        assert fm.station_polarity_count is None
        assert fm.misfit is None
        assert fm.station_distribution_ratio is None
        assert fm.method_id == \
            ResourceIdentifier(
                id='quakeml:us.anss.org/methodID=CMT')
        # comments
        assert len(fm.comments) == 0
        # creation info
        assert fm.creation_info.author is None
        assert fm.creation_info.agency_id == 'UCMT'
        assert fm.creation_info.author_uri is None
        assert fm.creation_info.agency_uri is None
        assert fm.creation_info.creation_time is None
        assert fm.creation_info.version is None
        # nodalPlanes
        assert round(abs(fm.nodal_planes.nodal_plane_1.strike-5.0), 7) == 0
        assert round(abs(fm.nodal_planes.nodal_plane_1.dip-85.0), 7) == 0
        assert round(abs(fm.nodal_planes.nodal_plane_1.rake--76.0), 7) == 0
        assert round(abs(fm.nodal_planes.nodal_plane_2.strike-116.0), 7) == 0
        assert round(abs(fm.nodal_planes.nodal_plane_2.dip-15.0), 7) == 0
        assert round(abs(fm.nodal_planes.nodal_plane_2.rake--159.0), 7) == 0
        assert fm.nodal_planes.preferred_plane is None
        # principalAxes
        assert round(abs(fm.principal_axes.t_axis.azimuth-82.0), 7) == 0
        assert round(abs(fm.principal_axes.t_axis.plunge-38.0), 7) == 0
        assert round(abs(fm.principal_axes.t_axis.length-1.87e+19), 7) == 0
        assert round(abs(fm.principal_axes.p_axis.azimuth-290.0), 7) == 0
        assert round(abs(fm.principal_axes.p_axis.plunge-49.0), 7) == 0
        assert round(abs(fm.principal_axes.p_axis.length--1.87e+19), 7) == 0
        assert fm.principal_axes.n_axis.azimuth == 184
        assert fm.principal_axes.n_axis.plunge == 14
        assert fm.principal_axes.n_axis.length == 0.0
        # momentTensor
        mt = fm.moment_tensor
        assert mt.resource_id == \
            ResourceIdentifier(
                id='quakeml:us.anss.org/momenttensor/'
                   '20120101052755.98/ucmt/mwc')
        assert round(abs(mt.scalar_moment-1.9e+19), 7) == 0
        assert round(abs(mt.tensor.m_rr--3.4e+18), 7) == 0
        assert round(abs(mt.tensor.m_tt--8e+17), 7) == 0
        assert round(abs(mt.tensor.m_pp-4.2e+18), 7) == 0
        assert round(abs(mt.tensor.m_rt--1.9e+18), 7) == 0
        assert round(abs(mt.tensor.m_rp--1.77e+19), 7) == 0
        assert round(abs(mt.tensor.m_tp--4.2e+18), 7) == 0
        assert mt.clvd is None

    def test_write_quakeml(self):
        """
        Tests writing a QuakeML document.
        """
        with NamedTemporaryFile() as tf:
            _write_quakeml(self.catalog, tf, validate=IS_RECENT_LXML)
            # Read file again. Avoid the (legit) warning about the already used
            # resource identifiers.
            tf.seek(0)
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("ignore")
                catalog2 = _read_quakeml(tf)
        assert len(catalog2), 1

    def test_read_events(self, testdata):
        """
        Tests reading an mchedr document via read_events.
        """
        filename = testdata['mchedr.dat']
        # Read file again. Avoid the (legit) warning about the already used
        # resource identifiers.
        with warnings.catch_warnings(record=True):
            warnings.simplefilter('always')
            catalog = read_events(filename)
            assert len(catalog), 1
