# -*- coding: utf-8 -*-
import os
import unittest
import warnings

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


class MchedrTestCase(unittest.TestCase):
    """
    Test suite for obspy.mchedr
    """

    catalog = None

    def setUp(self):
        # directory where the test files are located
        self.path = os.path.join(os.path.dirname(__file__), 'data')
        filename = os.path.join(self.path, 'mchedr.dat')
        # read the mchedr file once for all
        if self.__class__.catalog is None:
            self.__class__.catalog = _read_mchedr(filename)

    def test_catalog(self):
        self.assertEqual(len(self.catalog), 1)
        self.assertEqual(
            str(self.catalog),
            '''1 Event(s) in Catalog:
2012-01-01T05:27:55.980000Z | +31.456, +138.072 | 6.2  Mb'''
        )

    def test_event(self):
        """
        Tests Event object.
        """
        event = self.catalog[0]
        self.assertEqual(
            event.resource_id,
            ResourceIdentifier(
                id='quakeml:us.anss.org/event/20120101052755.98'))
        # enums
        self.assertEqual(event.event_type, None)
        self.assertEqual(event.event_type_certainty, None)
        # comments
        self.assertEqual(len(event.comments), 1)
        c = event.comments
        self.assertEqual(c[0].text, 'MW 6.8 (WCMT), 6.8 (UCMT), 6.8 (GCMT). \
Felt (V) at Chiba; (IV) at Fussa, Kawasaki, Saitama, Tokyo, \
Yokohama and Yokosuka; (III) at Ebina, Zama and Zushi; (II) \
at Misawa and Narita, Honshu. Recorded (4 JMA) in Chiba, Fukushima, \
Gumma, Ibaraki, Kanagawa, Miyagi, Saitama, Tochigi and Tokyo.')
        # event descriptions
        self.assertEqual(len(event.event_descriptions), 2)
        d = event.event_descriptions
        self.assertEqual(d[0].text, 'SOUTHEAST OF HONSHU, JAPAN')
        self.assertEqual(d[0].type, 'region name')
        self.assertEqual(d[1].text, '211')
        self.assertEqual(d[1].type, 'Flinn-Engdahl region')
        # creation info
        self.assertEqual(event.creation_info, None)

    def test_origin(self):
        """
        Tests Origin object.
        """
        self.assertEqual(len(self.catalog[0].origins), 4)
        origin = self.catalog[0].origins[0]
        self.assertEqual(
            origin.resource_id,
            ResourceIdentifier(
                id='quakeml:us.anss.org/origin/20120101052755.98'))
        self.assertEqual(origin.origin_type, 'hypocenter')
        self.assertEqual(
            origin.time,
            UTCDateTime(2012, 1, 1, 5, 27, 55, 980000))
        self.assertEqual(origin.latitude, 31.456)
        self.assertAlmostEqual(
            origin.latitude_errors.uncertainty, 0.0155, places=3)
        self.assertEqual(origin.longitude, 138.072)
        self.assertAlmostEqual(
            origin.longitude_errors.uncertainty, 0.0173, places=3)
        self.assertEqual(origin.depth, 365300.0)
        self.assertEqual(origin.depth_errors.uncertainty, 2700.0)
        self.assertEqual(origin.depth_type, 'from location')
        self.assertEqual(origin.method_id, None)
        self.assertEqual(origin.time_fixed, None)
        self.assertEqual(origin.epicenter_fixed, None)
        self.assertEqual(
            origin.earth_model_id,
            ResourceIdentifier(
                id='quakeml:us.anss.org/earthmodel/ak135'))
        self.assertEqual(origin.evaluation_mode, None)
        self.assertEqual(origin.evaluation_status, None)
        self.assertEqual(origin.origin_type, 'hypocenter')
        # composite times
        self.assertEqual(len(origin.composite_times), 0)
        # quality
        self.assertEqual(origin.quality.used_station_count, 628)
        self.assertEqual(origin.quality.standard_error, 0.84)
        self.assertEqual(origin.quality.azimuthal_gap, 10.8)
        self.assertEqual(origin.quality.maximum_distance, 29.1)
        self.assertEqual(origin.quality.minimum_distance, 2.22)
        self.assertEqual(origin.quality.associated_phase_count, 52)
        self.assertEqual(origin.quality.associated_station_count, 628)
        self.assertEqual(origin.quality.depth_phase_count, 0)
        self.assertEqual(origin.quality.secondary_azimuthal_gap, None)
        self.assertEqual(origin.quality.ground_truth_level, None)
        self.assertEqual(origin.quality.median_distance, None)
        # comments
        self.assertEqual(len(origin.comments), 0)
        # creation info
        self.assertEqual(origin.creation_info.author, None)
        self.assertEqual(origin.creation_info.agency_id, 'USGS-NEIC')
        self.assertEqual(origin.creation_info.author_uri, None)
        self.assertEqual(origin.creation_info.agency_uri, None)
        self.assertEqual(origin.creation_info.creation_time, None)
        self.assertEqual(origin.creation_info.version, None)
        # origin uncertainty
        u = origin.origin_uncertainty
        self.assertEqual(u.preferred_description, 'confidence ellipsoid')
        self.assertEqual(u.horizontal_uncertainty, None)
        self.assertEqual(u.min_horizontal_uncertainty, None)
        self.assertEqual(u.max_horizontal_uncertainty, None)
        self.assertEqual(u.azimuth_max_horizontal_uncertainty, None)
        # confidence ellipsoid
        c = u.confidence_ellipsoid
        self.assertEqual(c.semi_intermediate_axis_length, 2750.0)
        # c.major_axis_rotation is computed during file reading:
        self.assertAlmostEqual(c.major_axis_rotation, 170.5, places=3)
        self.assertEqual(c.major_axis_plunge, 76.06)
        self.assertEqual(c.semi_minor_axis_length, 2210.0)
        self.assertEqual(c.semi_major_axis_length, 4220.0)
        self.assertEqual(c.major_axis_azimuth, 292.79)

    def test_magnitude(self):
        """
        Tests Magnitude object.
        """
        self.assertEqual(len(self.catalog[0].magnitudes), 3)
        mag = self.catalog[0].magnitudes[0]
        self.assertEqual(
            mag.resource_id,
            ResourceIdentifier(
                id='quakeml:us.anss.org/magnitude/20120101052755.98/mb'))
        self.assertEqual(mag.mag, 6.2)
        self.assertEqual(mag.mag_errors.uncertainty, None)
        self.assertEqual(mag.magnitude_type, 'Mb')
        self.assertEqual(mag.station_count, 294)
        self.assertEqual(mag.evaluation_status, None)
        # comments
        self.assertEqual(len(mag.comments), 0)
        # creation info
        self.assertEqual(mag.creation_info.author, None)
        self.assertEqual(mag.creation_info.agency_id, 'USGS-NEIC')
        self.assertEqual(mag.creation_info.author_uri, None)
        self.assertEqual(mag.creation_info.agency_uri, None)
        self.assertEqual(mag.creation_info.creation_time, None)
        self.assertEqual(mag.creation_info.version, None)

    def test_stationmagnitude(self):
        """
        Tests StationMagnitude object.
        """
        self.assertEqual(len(self.catalog[0].station_magnitudes), 19)
        mag = self.catalog[0].station_magnitudes[0]
        self.assertEqual(mag.mag, 6.6)
        self.assertEqual(mag.mag_errors.uncertainty, None)
        self.assertEqual(mag.station_magnitude_type, 'Mb')
        self.assertEqual(mag.waveform_id.station_code, 'MDJ')
        self.assertEqual(mag.creation_info, None)

    def test_amplitude(self):
        """
        Tests Amplitude object.
        """
        self.assertEqual(len(self.catalog[0].station_magnitudes), 19)
        amp = self.catalog[0].amplitudes[0]
        self.assertAlmostEqual(amp.generic_amplitude, 3.94502e-06)
        self.assertEqual(amp.type, 'AB')
        self.assertEqual(amp.period, 1.3)
        self.assertEqual(amp.magnitude_hint, 'Mb')
        self.assertEqual(amp.waveform_id.station_code, 'MDJ')
        self.assertEqual(amp.creation_info, None)

    def test_arrival(self):
        """
        Tests Arrival object.
        """
        self.assertEqual(len(self.catalog[0].origins[0].arrivals), 52)
        ar = self.catalog[0].origins[0].arrivals[0]
        self.assertEqual(ar.phase, 'Pn')
        self.assertEqual(ar.azimuth, 41.4)
        self.assertEqual(ar.distance, 2.22)
        self.assertEqual(ar.takeoff_angle, None)
        self.assertEqual(ar.takeoff_angle_errors.uncertainty, None)
        self.assertEqual(ar.time_residual, -1.9)
        self.assertEqual(ar.horizontal_slowness_residual, None)
        self.assertEqual(ar.backazimuth_residual, None)
        self.assertEqual(ar.time_weight, None)
        self.assertEqual(ar.horizontal_slowness_weight, None)
        self.assertEqual(ar.backazimuth_weight, None)
        self.assertEqual(
            ar.earth_model_id,
            ResourceIdentifier('quakeml:us.anss.org/earthmodel/ak135'))
        self.assertEqual(len(ar.comments), 0)

    def test_pick(self):
        """
        Tests Pick object.
        """
        self.assertEqual(len(self.catalog[0].picks), 52)
        pick = self.catalog[0].picks[0]
        self.assertEqual(pick.time, UTCDateTime(2012, 1, 1, 5, 28, 48, 180000))
        self.assertEqual(pick.time_errors.uncertainty, None)
        self.assertEqual(pick.waveform_id.station_code, 'JHJ2')
        self.assertAlmostEqual(pick.backazimuth, -138.6)
        self.assertEqual(pick.onset, 'emergent')
        self.assertEqual(pick.phase_hint, 'Pn')
        self.assertEqual(pick.polarity, None)
        self.assertEqual(pick.evaluation_mode, None)
        self.assertEqual(pick.evaluation_status, None)
        self.assertEqual(len(pick.comments), 0)

    def test_focalmechanism(self):
        """
        Tests FocalMechanism object.
        """
        self.assertEqual(len(self.catalog[0].focal_mechanisms), 4)
        fm = self.catalog[0].focal_mechanisms[0]
        self.assertEqual(
            fm.resource_id,
            ResourceIdentifier(
                id='quakeml:us.anss.org/focalmechanism/'
                   '20120101052755.98/ucmt/mwc'))
        # general
        self.assertEqual(fm.waveform_id, [])
        self.assertEqual(fm.triggering_origin_id, None)
        self.assertEqual(fm.azimuthal_gap, None)
        self.assertEqual(fm.station_polarity_count, None)
        self.assertEqual(fm.misfit, None)
        self.assertEqual(fm.station_distribution_ratio, None)
        self.assertEqual(
            fm.method_id,
            ResourceIdentifier(
                id='quakeml:us.anss.org/methodID=CMT'))
        # comments
        self.assertEqual(len(fm.comments), 0)
        # creation info
        self.assertEqual(fm.creation_info.author, None)
        self.assertEqual(fm.creation_info.agency_id, 'UCMT')
        self.assertEqual(fm.creation_info.author_uri, None)
        self.assertEqual(fm.creation_info.agency_uri, None)
        self.assertEqual(fm.creation_info.creation_time, None)
        self.assertEqual(fm.creation_info.version, None)
        # nodalPlanes
        self.assertAlmostEqual(fm.nodal_planes.nodal_plane_1.strike, 5.0)
        self.assertAlmostEqual(fm.nodal_planes.nodal_plane_1.dip, 85.0)
        self.assertAlmostEqual(fm.nodal_planes.nodal_plane_1.rake, -76.0)
        self.assertAlmostEqual(fm.nodal_planes.nodal_plane_2.strike, 116.0)
        self.assertAlmostEqual(fm.nodal_planes.nodal_plane_2.dip, 15.0)
        self.assertAlmostEqual(fm.nodal_planes.nodal_plane_2.rake, -159.0)
        self.assertEqual(fm.nodal_planes.preferred_plane, None)
        # principalAxes
        self.assertAlmostEqual(fm.principal_axes.t_axis.azimuth, 82.0)
        self.assertAlmostEqual(fm.principal_axes.t_axis.plunge, 38.0)
        self.assertAlmostEqual(fm.principal_axes.t_axis.length, 1.87e+19)
        self.assertAlmostEqual(fm.principal_axes.p_axis.azimuth, 290.0)
        self.assertAlmostEqual(fm.principal_axes.p_axis.plunge, 49.0)
        self.assertAlmostEqual(fm.principal_axes.p_axis.length, -1.87e+19)
        self.assertEqual(fm.principal_axes.n_axis.azimuth, 184)
        self.assertEqual(fm.principal_axes.n_axis.plunge, 14)
        self.assertEqual(fm.principal_axes.n_axis.length, 0.0)
        # momentTensor
        mt = fm.moment_tensor
        self.assertEqual(
            mt.resource_id,
            ResourceIdentifier(
                id='quakeml:us.anss.org/momenttensor/'
                   '20120101052755.98/ucmt/mwc'))
        self.assertAlmostEqual(mt.scalar_moment, 1.9e+19)
        self.assertAlmostEqual(mt.tensor.m_rr, -3.4e+18)
        self.assertAlmostEqual(mt.tensor.m_tt, -8e+17)
        self.assertAlmostEqual(mt.tensor.m_pp, 4.2e+18)
        self.assertAlmostEqual(mt.tensor.m_rt, -1.9e+18)
        self.assertAlmostEqual(mt.tensor.m_rp, -1.77e+19)
        self.assertAlmostEqual(mt.tensor.m_tp, -4.2e+18)
        self.assertEqual(mt.clvd, None)

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
        self.assertTrue(len(catalog2), 1)

    def test_read_events(self):
        """
        Tests reading an mchedr document via read_events.
        """
        filename = os.path.join(self.path, 'mchedr.dat')
        # Read file again. Avoid the (legit) warning about the already used
        # resource identifiers.
        with warnings.catch_warnings(record=True):
            warnings.simplefilter('always')
            catalog = read_events(filename)
            self.assertTrue(len(catalog), 1)


def suite():
    return unittest.makeSuite(MchedrTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
