# -*- coding: utf-8 -*-
import io
import math
import os
import unittest
import warnings

from lxml import etree

from obspy.core.event import (Catalog, Event, FocalMechanism, Magnitude,
                              MomentTensor, Origin, Pick, ResourceIdentifier,
                              Tensor, WaveformStreamID, read_events,
                              EventDescription)
from obspy.core.utcdatetime import UTCDateTime
from obspy.core.util import AttribDict
from obspy.core.util.base import NamedTemporaryFile
from obspy.core.util.testing import compare_xml_strings
from obspy.io.quakeml.core import Pickler, _read_quakeml, _write_quakeml


# lxml < 2.3 seems not to ship with RelaxNG schema parser and namespace support
IS_RECENT_LXML = False
version = float(etree.__version__.rsplit('.', 1)[0])
if version >= 2.3:
    IS_RECENT_LXML = True


def assert_no_extras(obj, verbose=False):
    """
    Helper routine to make sure no information ends up in 'extra' when there is
    not supposed to be anything in there. (to make sure, after changes in
    #2466)
    """
    assert getattr(obj, 'extra', None) is None
    if verbose:
        print('no extras in {!s}'.format(obj))
    if isinstance(obj, Catalog):
        for event in obj:
            assert_no_extras(event, verbose=verbose)
        return
    # recurse deeper if an event-type object
    for name, _type in getattr(obj, '_properties', []):
        assert_no_extras(getattr(obj, name), verbose=verbose)


class QuakeMLTestCase(unittest.TestCase):
    """
    Test suite for obspy.io.quakeml
    """
    def setUp(self):
        # directory where the test files are located
        self.path = os.path.join(os.path.dirname(__file__), 'data')
        self.neries_filename = os.path.join(self.path, 'neries_events.xml')
        self.neries_catalog = _read_quakeml(self.neries_filename)

    def test_read_quakeml(self):
        """
        """
        # IRIS
        filename = os.path.join(self.path, 'iris_events.xml')
        catalog = _read_quakeml(filename)
        assert_no_extras(catalog)
        self.assertEqual(len(catalog), 2)
        self.assertEqual(
            catalog[0].resource_id,
            ResourceIdentifier(
                'smi:www.iris.edu/ws/event/query?eventId=3279407'))
        self.assertEqual(
            catalog[1].resource_id,
            ResourceIdentifier(
                'smi:www.iris.edu/ws/event/query?eventId=2318174'))
        # NERIES
        catalog = self.neries_catalog
        assert_no_extras(catalog)
        self.assertEqual(len(catalog), 3)
        self.assertEqual(
            catalog[0].resource_id,
            ResourceIdentifier('quakeml:eu.emsc/event/20120404_0000041'))
        self.assertEqual(
            catalog[1].resource_id,
            ResourceIdentifier('quakeml:eu.emsc/event/20120404_0000038'))
        self.assertEqual(
            catalog[2].resource_id,
            ResourceIdentifier('quakeml:eu.emsc/event/20120404_0000039'))

    def test_usgs_eventype(self):
        filename = os.path.join(self.path, 'usgs_event.xml')
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("ignore")
            catalog = _read_quakeml(filename)
        # there are some custom namespace attributes
        self.assertTrue(len(catalog[0].extra) == 3)
        self.assertTrue(len(catalog[0].origins[0].extra) == 4)
        self.assertEqual(len(catalog), 1)
        self.assertEqual(catalog[0].event_type, 'quarry blast')

    def test_event(self):
        """
        Tests Event object.
        """
        filename = os.path.join(self.path, 'quakeml_1.2_event.xml')
        catalog = _read_quakeml(filename)
        assert_no_extras(catalog)
        self.assertEqual(len(catalog), 1)
        event = catalog[0]
        self.assertEqual(
            event.resource_id,
            ResourceIdentifier('smi:ch.ethz.sed/event/historical/1165'))
        # enums
        self.assertEqual(event.event_type, 'earthquake')
        self.assertEqual(event.event_type_certainty, 'suspected')
        # comments
        self.assertEqual(len(event.comments), 2)
        c = event.comments
        self.assertEqual(c[0].text, 'Relocated after re-evaluation')
        self.assertEqual(c[0].resource_id, None)
        self.assertEqual(c[0].creation_info.agency_id, 'EMSC')
        self.assertEqual(c[1].text, 'Another comment')
        self.assertEqual(
            c[1].resource_id,
            ResourceIdentifier(id="smi:some/comment/id/number_3"))
        self.assertEqual(c[1].creation_info, None)
        # event descriptions
        self.assertEqual(len(event.event_descriptions), 3)
        d = event.event_descriptions
        self.assertEqual(d[0].text, '1906 San Francisco Earthquake')
        self.assertEqual(d[0].type, 'earthquake name')
        self.assertEqual(d[1].text, 'NEAR EAST COAST OF HONSHU, JAPAN')
        self.assertEqual(d[1].type, 'Flinn-Engdahl region')
        self.assertEqual(d[2].text, 'free-form string')
        self.assertEqual(d[2].type, None)
        # creation info
        self.assertEqual(event.creation_info.author, "Erika Mustermann")
        self.assertEqual(event.creation_info.agency_id, "EMSC")
        self.assertEqual(
            event.creation_info.author_uri,
            ResourceIdentifier("smi:smi-registry/organization/EMSC"))
        self.assertEqual(
            event.creation_info.agency_uri,
            ResourceIdentifier("smi:smi-registry/organization/EMSC"))
        self.assertEqual(
            event.creation_info.creation_time,
            UTCDateTime("2012-04-04T16:40:50+00:00"))
        self.assertEqual(event.creation_info.version, "1.0.1")
        # exporting back to XML should result in the same document
        with open(filename, "rt") as fp:
            original = fp.read()
        processed = Pickler().dumps(catalog)
        compare_xml_strings(original, processed)

    def test_origin(self):
        """
        Tests Origin object.
        """
        filename = os.path.join(self.path, 'quakeml_1.2_origin.xml')
        catalog = _read_quakeml(filename)
        assert_no_extras(catalog)
        self.assertEqual(len(catalog), 1)
        self.assertEqual(len(catalog[0].origins), 1)
        origin = catalog[0].origins[0]
        self.assertEqual(
            origin.resource_id,
            ResourceIdentifier(
                'smi:www.iris.edu/ws/event/query?originId=7680412'))
        self.assertEqual(origin.time, UTCDateTime("2011-03-11T05:46:24.1200"))
        self.assertEqual(origin.latitude, 38.297)
        self.assertEqual(origin.latitude_errors.lower_uncertainty, None)
        self.assertEqual(origin.longitude, 142.373)
        self.assertEqual(origin.longitude_errors.uncertainty, None)
        self.assertEqual(origin.depth, 29.0)
        self.assertEqual(origin.depth_errors.confidence_level, 50.0)
        self.assertEqual(origin.depth_type, "from location")
        self.assertEqual(
            origin.method_id,
            ResourceIdentifier(id="smi:some/method/NA"))
        self.assertEqual(origin.time_fixed, None)
        self.assertEqual(origin.epicenter_fixed, False)
        self.assertEqual(
            origin.reference_system_id,
            ResourceIdentifier(id="smi:some/reference/muh"))
        self.assertEqual(
            origin.earth_model_id,
            ResourceIdentifier(id="smi:same/model/maeh"))
        self.assertEqual(origin.region, 'Kalamazoo')
        self.assertEqual(origin.evaluation_mode, "manual")
        self.assertEqual(origin.evaluation_status, "preliminary")
        self.assertEqual(origin.origin_type, "hypocenter")
        # composite times
        self.assertEqual(len(origin.composite_times), 2)
        c = origin.composite_times
        self.assertEqual(c[0].year, 2029)
        self.assertEqual(c[0].month, None)
        self.assertEqual(c[0].day, None)
        self.assertEqual(c[0].hour, 12)
        self.assertEqual(c[0].minute, None)
        self.assertEqual(c[0].second, None)
        self.assertEqual(c[1].year, None)
        self.assertEqual(c[1].month, None)
        self.assertEqual(c[1].day, None)
        self.assertEqual(c[1].hour, 1)
        self.assertEqual(c[1].minute, None)
        self.assertEqual(c[1].second, 29.124234)
        # quality
        self.assertEqual(origin.quality.used_station_count, 16)
        self.assertEqual(origin.quality.standard_error, 0)
        self.assertEqual(origin.quality.azimuthal_gap, 231)
        self.assertEqual(origin.quality.maximum_distance, 53.03)
        self.assertEqual(origin.quality.minimum_distance, 2.45)
        self.assertEqual(origin.quality.associated_phase_count, None)
        self.assertEqual(origin.quality.associated_station_count, None)
        self.assertEqual(origin.quality.depth_phase_count, None)
        self.assertEqual(origin.quality.secondary_azimuthal_gap, None)
        self.assertEqual(origin.quality.ground_truth_level, None)
        self.assertEqual(origin.quality.median_distance, None)
        # comments
        self.assertEqual(len(origin.comments), 2)
        c = origin.comments
        self.assertEqual(c[0].text, 'Some comment')
        self.assertEqual(
            c[0].resource_id,
            ResourceIdentifier(id="smi:some/comment/reference"))
        self.assertEqual(c[0].creation_info.author, 'EMSC')
        self.assertEqual(c[1].resource_id, None)
        self.assertEqual(c[1].creation_info, None)
        self.assertEqual(c[1].text, 'Another comment')
        # creation info
        self.assertEqual(origin.creation_info.author, "NEIC")
        self.assertEqual(origin.creation_info.agency_id, None)
        self.assertEqual(origin.creation_info.author_uri, None)
        self.assertEqual(origin.creation_info.agency_uri, None)
        self.assertEqual(origin.creation_info.creation_time, None)
        self.assertEqual(origin.creation_info.version, None)
        # origin uncertainty
        u = origin.origin_uncertainty
        self.assertEqual(u.preferred_description, "uncertainty ellipse")
        self.assertEqual(u.horizontal_uncertainty, 9000)
        self.assertEqual(u.min_horizontal_uncertainty, 6000)
        self.assertEqual(u.max_horizontal_uncertainty, 10000)
        self.assertEqual(u.azimuth_max_horizontal_uncertainty, 80.0)
        # confidence ellipsoid
        c = u.confidence_ellipsoid
        self.assertEqual(c.semi_intermediate_axis_length, 2.123)
        self.assertEqual(c.major_axis_rotation, 5.123)
        self.assertEqual(c.major_axis_plunge, 3.123)
        self.assertEqual(c.semi_minor_axis_length, 1.123)
        self.assertEqual(c.semi_major_axis_length, 0.123)
        self.assertEqual(c.major_axis_azimuth, 4.123)
        # exporting back to XML should result in the same document
        with open(filename, "rt") as fp:
            original = fp.read()
        processed = Pickler().dumps(catalog)
        compare_xml_strings(original, processed)

    def test_magnitude(self):
        """
        Tests Magnitude object.
        """
        filename = os.path.join(self.path, 'quakeml_1.2_magnitude.xml')
        catalog = _read_quakeml(filename)
        assert_no_extras(catalog)
        self.assertEqual(len(catalog), 1)
        self.assertEqual(len(catalog[0].magnitudes), 1)
        mag = catalog[0].magnitudes[0]
        self.assertEqual(
            mag.resource_id,
            ResourceIdentifier('smi:ch.ethz.sed/magnitude/37465'))
        self.assertEqual(mag.mag, 5.5)
        self.assertEqual(mag.mag_errors.uncertainty, 0.1)
        self.assertEqual(mag.magnitude_type, 'MS')
        self.assertEqual(
            mag.method_id,
            ResourceIdentifier(
                'smi:ch.ethz.sed/magnitude/generic/surface_wave_magnitude'))
        self.assertEqual(mag.station_count, 8)
        self.assertEqual(mag.evaluation_status, 'preliminary')
        # comments
        self.assertEqual(len(mag.comments), 2)
        c = mag.comments
        self.assertEqual(c[0].text, 'Some comment')
        self.assertEqual(
            c[0].resource_id,
            ResourceIdentifier(id="smi:some/comment/id/muh"))
        self.assertEqual(c[0].creation_info.author, 'EMSC')
        self.assertEqual(c[1].creation_info, None)
        self.assertEqual(c[1].text, 'Another comment')
        self.assertEqual(c[1].resource_id, None)
        # creation info
        self.assertEqual(mag.creation_info.author, "NEIC")
        self.assertEqual(mag.creation_info.agency_id, None)
        self.assertEqual(mag.creation_info.author_uri, None)
        self.assertEqual(mag.creation_info.agency_uri, None)
        self.assertEqual(mag.creation_info.creation_time, None)
        self.assertEqual(mag.creation_info.version, None)
        # exporting back to XML should result in the same document
        with open(filename, "rt") as fp:
            original = fp.read()
        processed = Pickler().dumps(catalog)
        compare_xml_strings(original, processed)

    def test_station_magnitude_contribution(self):
        """
        Tests the station magnitude contribution object.
        """
        filename = os.path.join(
            self.path, 'quakeml_1.2_stationmagnitudecontributions.xml')
        catalog = _read_quakeml(filename)
        assert_no_extras(catalog)
        self.assertEqual(len(catalog), 1)
        self.assertEqual(len(catalog[0].magnitudes), 1)
        self.assertEqual(
            len(catalog[0].magnitudes[0].station_magnitude_contributions), 2)
        # Check the first stationMagnitudeContribution object.
        stat_contrib = \
            catalog[0].magnitudes[0].station_magnitude_contributions[0]
        self.assertEqual(
            stat_contrib.station_magnitude_id.id,
            "smi:ch.ethz.sed/magnitude/station/881342")
        self.assertEqual(stat_contrib.weight, 0.77)
        self.assertEqual(stat_contrib.residual, 0.02)
        # Check the second stationMagnitudeContribution object.
        stat_contrib = \
            catalog[0].magnitudes[0].station_magnitude_contributions[1]
        self.assertEqual(
            stat_contrib.station_magnitude_id.id,
            "smi:ch.ethz.sed/magnitude/station/881334")
        self.assertEqual(stat_contrib.weight, 0.)
        self.assertEqual(stat_contrib.residual, 0.)

        # exporting back to XML should result in the same document
        with open(filename, "rt") as fp:
            original = fp.read()
        processed = Pickler().dumps(catalog)
        compare_xml_strings(original, processed)

    def test_station_magnitude(self):
        """
        Tests StationMagnitude object.
        """
        filename = os.path.join(self.path, 'quakeml_1.2_stationmagnitude.xml')
        catalog = _read_quakeml(filename)
        assert_no_extras(catalog)
        self.assertEqual(len(catalog), 1)
        self.assertEqual(len(catalog[0].station_magnitudes), 1)
        mag = catalog[0].station_magnitudes[0]
        # Assert the actual StationMagnitude object. Everything that is not set
        # in the QuakeML file should be set to None.
        self.assertEqual(
            mag.resource_id,
            ResourceIdentifier("smi:ch.ethz.sed/magnitude/station/881342"))
        self.assertEqual(
            mag.origin_id,
            ResourceIdentifier('smi:some/example/id'))
        self.assertEqual(mag.mag, 6.5)
        self.assertEqual(mag.mag_errors.uncertainty, 0.2)
        self.assertEqual(mag.station_magnitude_type, 'MS')
        self.assertEqual(
            mag.amplitude_id,
            ResourceIdentifier("smi:ch.ethz.sed/amplitude/824315"))
        self.assertEqual(
            mag.method_id,
            ResourceIdentifier(
                "smi:ch.ethz.sed/magnitude/generic/surface_wave_magnitude"))
        self.assertEqual(
            mag.waveform_id,
            WaveformStreamID(network_code='BW', station_code='FUR',
                             resource_uri="smi:ch.ethz.sed/waveform/201754"))
        self.assertEqual(mag.creation_info, None)
        # exporting back to XML should result in the same document
        with open(filename, "rt") as fp:
            original = fp.read()
        processed = Pickler().dumps(catalog)
        compare_xml_strings(original, processed)

    def test_data_used_in_moment_tensor(self):
        """
        Tests the data used objects in moment tensors.
        """
        filename = os.path.join(self.path, 'quakeml_1.2_data_used.xml')

        # Test reading first.
        catalog = _read_quakeml(filename)
        assert_no_extras(catalog)
        event = catalog[0]

        self.assertTrue(len(event.focal_mechanisms), 2)
        # First focmec contains only one data used element.
        self.assertEqual(
            len(event.focal_mechanisms[0].moment_tensor.data_used), 1)
        du = event.focal_mechanisms[0].moment_tensor.data_used[0]
        self.assertEqual(du.wave_type, "body waves")
        self.assertEqual(du.station_count, 88)
        self.assertEqual(du.component_count, 166)
        self.assertEqual(du.shortest_period, 40.0)
        # Second contains three. focmec contains only one data used element.
        self.assertEqual(
            len(event.focal_mechanisms[1].moment_tensor.data_used), 3)
        du = event.focal_mechanisms[1].moment_tensor.data_used
        self.assertEqual(du[0].wave_type, "body waves")
        self.assertEqual(du[0].station_count, 88)
        self.assertEqual(du[0].component_count, 166)
        self.assertEqual(du[0].shortest_period, 40.0)
        self.assertEqual(du[1].wave_type, "surface waves")
        self.assertEqual(du[1].station_count, 96)
        self.assertEqual(du[1].component_count, 189)
        self.assertEqual(du[1].shortest_period, 50.0)
        self.assertEqual(du[2].wave_type, "mantle waves")
        self.assertEqual(du[2].station_count, 41)
        self.assertEqual(du[2].component_count, 52)
        self.assertEqual(du[2].shortest_period, 125.0)

        # exporting back to XML should result in the same document
        with open(filename, "rt") as fp:
            original = fp.read()
        processed = Pickler().dumps(catalog)
        compare_xml_strings(original, processed)

    def test_arrival(self):
        """
        Tests Arrival object.
        """
        filename = os.path.join(self.path, 'quakeml_1.2_arrival.xml')
        catalog = _read_quakeml(filename)
        assert_no_extras(catalog)
        self.assertEqual(len(catalog), 1)
        self.assertEqual(len(catalog[0].origins[0].arrivals), 2)
        ar = catalog[0].origins[0].arrivals[0]
        # Test the actual Arrival object. Everything not set in the QuakeML
        # file should be None.
        self.assertEqual(
            ar.pick_id,
            ResourceIdentifier('smi:ch.ethz.sed/pick/117634'))
        self.assertEqual(ar.phase, 'Pn')
        self.assertEqual(ar.azimuth, 12.0)
        self.assertEqual(ar.distance, 0.5)
        self.assertEqual(ar.takeoff_angle, 11.0)
        self.assertEqual(ar.takeoff_angle_errors.uncertainty, 0.2)
        self.assertEqual(ar.time_residual, 1.6)
        self.assertEqual(ar.horizontal_slowness_residual, 1.7)
        self.assertEqual(ar.backazimuth_residual, 1.8)
        self.assertEqual(ar.time_weight, 0.48)
        self.assertEqual(ar.horizontal_slowness_weight, 0.49)
        self.assertEqual(ar.backazimuth_weight, 0.5)
        self.assertEqual(
            ar.earth_model_id,
            ResourceIdentifier('smi:ch.ethz.sed/earthmodel/U21'))
        self.assertEqual(len(ar.comments), 1)
        self.assertEqual(ar.creation_info.author, "Erika Mustermann")
        # exporting back to XML should result in the same document
        with open(filename, "rt") as fp:
            original = fp.read()
        processed = Pickler().dumps(catalog)
        compare_xml_strings(original, processed)

    def test_pick(self):
        """
        Tests Pick object.
        """
        filename = os.path.join(self.path, 'quakeml_1.2_pick.xml')
        catalog = _read_quakeml(filename)
        assert_no_extras(catalog)
        self.assertEqual(len(catalog), 1)
        self.assertEqual(len(catalog[0].picks), 2)
        pick = catalog[0].picks[0]
        self.assertEqual(
            pick.resource_id,
            ResourceIdentifier('smi:ch.ethz.sed/pick/117634'))
        self.assertEqual(pick.time, UTCDateTime('2005-09-18T22:04:35Z'))
        self.assertEqual(pick.time_errors.uncertainty, 0.012)
        self.assertEqual(
            pick.waveform_id,
            WaveformStreamID(network_code='BW', station_code='FUR',
                             resource_uri='smi:ch.ethz.sed/waveform/201754'))
        self.assertEqual(
            pick.filter_id,
            ResourceIdentifier('smi:ch.ethz.sed/filter/lowpass/standard'))
        self.assertEqual(
            pick.method_id,
            ResourceIdentifier('smi:ch.ethz.sed/picker/autopicker/6.0.2'))
        self.assertEqual(pick.backazimuth, 44.0)
        self.assertEqual(pick.onset, 'impulsive')
        self.assertEqual(pick.phase_hint, 'Pn')
        self.assertEqual(pick.polarity, 'positive')
        self.assertEqual(pick.evaluation_mode, "manual")
        self.assertEqual(pick.evaluation_status, "confirmed")
        self.assertEqual(len(pick.comments), 2)
        self.assertEqual(pick.creation_info.author, "Erika Mustermann")
        # exporting back to XML should result in the same document
        with open(filename, "rt") as fp:
            original = fp.read()
        processed = Pickler().dumps(catalog)
        compare_xml_strings(original, processed)

    def test_focalmechanism(self):
        """
        Tests FocalMechanism object.
        """
        filename = os.path.join(self.path, 'quakeml_1.2_focalmechanism.xml')
        catalog = _read_quakeml(filename)
        assert_no_extras(catalog)
        self.assertEqual(len(catalog), 1)
        self.assertEqual(len(catalog[0].focal_mechanisms), 2)
        fm = catalog[0].focal_mechanisms[0]
        # general
        self.assertEqual(
            fm.resource_id,
            ResourceIdentifier('smi:ISC/fmid=292309'))
        self.assertEqual(len(fm.waveform_id), 2)
        self.assertEqual(fm.waveform_id[0].network_code, 'BW')
        self.assertEqual(fm.waveform_id[0].station_code, 'FUR')
        self.assertEqual(
            fm.waveform_id[0].resource_uri,
            ResourceIdentifier(id="smi:ch.ethz.sed/waveform/201754"))
        self.assertTrue(isinstance(fm.waveform_id[0], WaveformStreamID))
        self.assertEqual(
            fm.triggering_origin_id,
            ResourceIdentifier('smi:local/originId=7680412'))
        self.assertAlmostEqual(fm.azimuthal_gap, 0.123)
        self.assertEqual(fm.station_polarity_count, 987)
        self.assertAlmostEqual(fm.misfit, 1.234)
        self.assertAlmostEqual(fm.station_distribution_ratio, 2.345)
        self.assertEqual(
            fm.method_id,
            ResourceIdentifier('smi:ISC/methodID=Best_double_couple'))
        # comments
        self.assertEqual(len(fm.comments), 2)
        c = fm.comments
        self.assertEqual(c[0].text, 'Relocated after re-evaluation')
        self.assertEqual(c[0].resource_id, None)
        self.assertEqual(c[0].creation_info.agency_id, 'MUH')
        self.assertEqual(c[1].text, 'Another MUH')
        self.assertEqual(
            c[1].resource_id,
            ResourceIdentifier(id="smi:some/comment/id/number_3"))
        self.assertEqual(c[1].creation_info, None)
        # creation info
        self.assertEqual(fm.creation_info.author, "Erika Mustermann")
        self.assertEqual(fm.creation_info.agency_id, "MUH")
        self.assertEqual(
            fm.creation_info.author_uri,
            ResourceIdentifier("smi:smi-registry/organization/MUH"))
        self.assertEqual(
            fm.creation_info.agency_uri,
            ResourceIdentifier("smi:smi-registry/organization/MUH"))
        self.assertEqual(
            fm.creation_info.creation_time,
            UTCDateTime("2012-04-04T16:40:50+00:00"))
        self.assertEqual(fm.creation_info.version, "1.0.1")
        # nodalPlanes
        self.assertAlmostEqual(fm.nodal_planes.nodal_plane_1.strike, 346.0)
        self.assertAlmostEqual(fm.nodal_planes.nodal_plane_1.dip, 57.0)
        self.assertAlmostEqual(fm.nodal_planes.nodal_plane_1.rake, 75.0)
        self.assertAlmostEqual(fm.nodal_planes.nodal_plane_2.strike, 193.0)
        self.assertAlmostEqual(fm.nodal_planes.nodal_plane_2.dip, 36.0)
        self.assertAlmostEqual(fm.nodal_planes.nodal_plane_2.rake, 112.0)
        self.assertEqual(fm.nodal_planes.preferred_plane, 2)
        # principalAxes
        self.assertAlmostEqual(fm.principal_axes.t_axis.azimuth, 216.0)
        self.assertAlmostEqual(fm.principal_axes.t_axis.plunge, 73.0)
        self.assertAlmostEqual(fm.principal_axes.t_axis.length, 1.050e+18)
        self.assertAlmostEqual(fm.principal_axes.p_axis.azimuth, 86.0)
        self.assertAlmostEqual(fm.principal_axes.p_axis.plunge, 10.0)
        self.assertAlmostEqual(fm.principal_axes.p_axis.length, -1.180e+18)
        self.assertEqual(fm.principal_axes.n_axis.azimuth, None)
        self.assertEqual(fm.principal_axes.n_axis.plunge, None)
        self.assertEqual(fm.principal_axes.n_axis.length, None)
        # momentTensor
        mt = fm.moment_tensor
        self.assertEqual(
            mt.resource_id,
            ResourceIdentifier('smi:ISC/mtid=123321'))
        self.assertEqual(
            mt.derived_origin_id,
            ResourceIdentifier('smi:ISC/origid=13145006'))
        self.assertAlmostEqual(mt.scalar_moment, 1.100e+18)
        self.assertAlmostEqual(mt.tensor.m_rr, 9.300e+17)
        self.assertAlmostEqual(mt.tensor.m_tt, 1.700e+17)
        self.assertAlmostEqual(mt.tensor.m_pp, -1.100e+18)
        self.assertAlmostEqual(mt.tensor.m_rt, -2.200e+17)
        self.assertAlmostEqual(mt.tensor.m_rp, 4.000e+17)
        self.assertAlmostEqual(mt.tensor.m_tp, 3.000e+16)
        self.assertAlmostEqual(mt.clvd, 0.22)
        # exporting back to XML should result in the same document
        with open(filename, "rb") as fp:
            original = fp.read()
        processed = Pickler().dumps(catalog)
        compare_xml_strings(original, processed)

    def test_write_quakeml(self):
        """
        Tests writing a QuakeML document.
        """
        filename = os.path.join(self.path, 'qml-example-1.2-RC3.xml')
        with NamedTemporaryFile() as tf:
            tmpfile = tf.name
            catalog = _read_quakeml(filename)
            assert_no_extras(catalog)
            self.assertTrue(len(catalog), 1)
            _write_quakeml(catalog, tmpfile, validate=IS_RECENT_LXML)
            # Read file again. Avoid the (legit) warning about the already used
            # resource identifiers.
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("ignore")
                catalog2 = _read_quakeml(tmpfile)
                assert_no_extras(catalog2)
        self.assertTrue(len(catalog2), 1)

    def test_read_events(self):
        """
        Tests reading a QuakeML document via read_events.
        """
        with NamedTemporaryFile() as tf:
            tmpfile = tf.name
            catalog = read_events(self.neries_filename)
            assert_no_extras(catalog)
            self.assertTrue(len(catalog), 3)
            catalog.write(tmpfile, format='QUAKEML')
            # Read file again. Avoid the (legit) warning about the already used
            # resource identifiers.
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("ignore")
                catalog2 = read_events(tmpfile)
                assert_no_extras(catalog2)
        self.assertTrue(len(catalog2), 3)

    @unittest.skipIf(not IS_RECENT_LXML, "lxml >= 2.3 is required")
    def test_enums(self):
        """
        Parses the QuakeML xsd scheme definition and checks if all enums are
        correctly defined.

        This is a very strict test against the xsd scheme file of QuakeML 1.2.
        If obspy.core.event will ever be more loosely coupled to QuakeML this
        test WILL HAVE to be changed.
        """
        from lxml.etree import parse

        xsd_enum_definitions = {}
        xsd_file = os.path.join(
            self.path, os.pardir, os.pardir, "data", "QuakeML-BED-1.2.xsd")
        root = parse(xsd_file).getroot()

        # Get all enums from the xsd file.
        nsmap = {k: v for k, v in root.nsmap.items() if k is not None}
        for stype in root.findall("xs:simpleType", namespaces=nsmap):
            type_name = stype.get("name")
            restriction = stype.find("xs:restriction", namespaces=nsmap)
            if restriction is None:
                continue
            if restriction.get("base") != "xs:string":
                continue
            enums = restriction.findall(
                "xs:enumeration", namespaces=nsmap)
            if not enums:
                continue
            enums = [_i.get("value") for _i in enums]
            xsd_enum_definitions[type_name] = enums

        # Now import all enums and check if they are correct.
        from obspy.core.event import header as event_header
        from obspy.core.util import Enum
        all_enums = {}
        for module_item_name in dir(event_header):
            module_item = getattr(event_header, module_item_name)
            if type(module_item) != Enum:
                continue
            # Assign clearer names.
            enum_name = module_item_name
            enum_values = [_i.lower() for _i in module_item.keys()]
            all_enums[enum_name] = enum_values
        # Now loop over all enums defined in the xsd file and check them.
        for enum_name, enum_items in xsd_enum_definitions.items():
            self.assertIn(enum_name, all_enums.keys())
            # Check that also all enum items are available.
            all_items = all_enums[enum_name]
            all_items = [_i.lower() for _i in all_items]
            for enum_item in enum_items:
                if enum_item.lower() not in all_items:  # pragma: no cover
                    msg = "Value '%s' not in Enum '%s'" % \
                        (enum_item, enum_name)
                    raise Exception(msg)
            # Check if there are too many items.
            if len(all_items) != len(enum_items):  # pragma: no cover
                additional_items = [_i for _i in all_items
                                    if _i.lower() not in enum_items]
                msg = "Enum {enum_name} has the following additional items" + \
                    " not defined in the xsd style sheet:\n\t{enumerations}"
                msg = msg.format(enum_name=enum_name,
                                 enumerations=", ".join(additional_items))
                raise Exception(msg)

    def test_read_string(self):
        """
        Test reading a QuakeML string/unicode object via read_events.
        """
        with open(self.neries_filename, 'rb') as fp:
            data = fp.read()

        catalog = read_events(data)
        assert_no_extras(catalog)
        self.assertEqual(len(catalog), 3)

    def test_preferred_tags(self):
        """
        Testing preferred magnitude, origin and focal mechanism tags
        """
        # testing empty event
        ev = Event()
        self.assertEqual(ev.preferred_origin(), None)
        self.assertEqual(ev.preferred_magnitude(), None)
        self.assertEqual(ev.preferred_focal_mechanism(), None)
        # testing existing event
        filename = os.path.join(self.path, 'preferred.xml')
        catalog = read_events(filename)
        assert_no_extras(catalog)
        self.assertEqual(len(catalog), 1)
        ev_str = "Event:\t2012-12-12T05:46:24.120000Z | +38.297, +142.373 " + \
                 "| 2.0  MW"
        self.assertIn(ev_str, str(catalog.events[0]))
        # testing ids
        ev = catalog.events[0]
        self.assertEqual('smi:orig2', ev.preferred_origin_id)
        self.assertEqual('smi:mag2', ev.preferred_magnitude_id)
        self.assertEqual('smi:fm2', ev.preferred_focal_mechanism_id)
        # testing objects
        self.assertEqual(ev.preferred_origin(), ev.origins[1])
        self.assertEqual(ev.preferred_magnitude(), ev.magnitudes[1])
        self.assertEqual(
            ev.preferred_focal_mechanism(), ev.focal_mechanisms[1])

    def test_creating_minimal_quakeml_with_mt(self):
        """
        Tests the creation of a minimal QuakeML containing origin, magnitude
        and moment tensor.
        """
        # Rotate into physical domain
        lat, lon, depth, org_time = 10.0, -20.0, 12000, UTCDateTime(2012, 1, 1)
        mrr, mtt, mpp, mtr, mpr, mtp = 1E18, 2E18, 3E18, 3E18, 2E18, 1E18
        scalar_moment = math.sqrt(
            mrr ** 2 + mtt ** 2 + mpp ** 2 + mtr ** 2 + mpr ** 2 + mtp ** 2)
        moment_magnitude = 0.667 * (math.log10(scalar_moment) - 9.1)

        # Initialise event
        ev = Event(event_type="earthquake")

        ev_origin = Origin(time=org_time, latitude=lat, longitude=lon,
                           depth=depth, resource_id=ResourceIdentifier())
        ev.origins.append(ev_origin)

        # populate event moment tensor
        ev_tensor = Tensor(m_rr=mrr, m_tt=mtt, m_pp=mpp, m_rt=mtr, m_rp=mpr,
                           m_tp=mtp)

        ev_momenttensor = MomentTensor(tensor=ev_tensor)
        ev_momenttensor.scalar_moment = scalar_moment
        ev_momenttensor.derived_origin_id = ev_origin.resource_id

        ev_focalmechanism = FocalMechanism(moment_tensor=ev_momenttensor)
        ev.focal_mechanisms.append(ev_focalmechanism)

        # populate event magnitude
        ev_magnitude = Magnitude()
        ev_magnitude.mag = moment_magnitude
        ev_magnitude.magnitude_type = 'Mw'
        ev_magnitude.evaluation_mode = 'automatic'
        ev.magnitudes.append(ev_magnitude)

        # write QuakeML file
        cat = Catalog(events=[ev])
        memfile = io.BytesIO()
        cat.write(memfile, format="quakeml", validate=IS_RECENT_LXML)

        memfile.seek(0, 0)
        new_cat = _read_quakeml(memfile)
        assert_no_extras(new_cat)
        self.assertEqual(len(new_cat), 1)
        event = new_cat[0]
        self.assertEqual(len(event.origins), 1)
        self.assertEqual(len(event.magnitudes), 1)
        self.assertEqual(len(event.focal_mechanisms), 1)
        org = event.origins[0]
        mag = event.magnitudes[0]
        fm = event.focal_mechanisms[0]
        self.assertEqual(org.latitude, lat)
        self.assertEqual(org.longitude, lon)
        self.assertEqual(org.depth, depth)
        self.assertEqual(org.time, org_time)
        # Moment tensor.
        mt = fm.moment_tensor.tensor
        self.assertTrue((fm.moment_tensor.scalar_moment - scalar_moment) /
                        scalar_moment < scalar_moment * 1E-10)
        self.assertEqual(mt.m_rr, mrr)
        self.assertEqual(mt.m_pp, mpp)
        self.assertEqual(mt.m_tt, mtt)
        self.assertEqual(mt.m_rt, mtr)
        self.assertEqual(mt.m_rp, mpr)
        self.assertEqual(mt.m_tp, mtp)
        # Mag
        self.assertAlmostEqual(mag.mag, moment_magnitude)
        self.assertEqual(mag.magnitude_type, "Mw")
        self.assertEqual(mag.evaluation_mode, "automatic")

    def test_read_equivalence(self):
        """
        See #662.
        Tests if _read_quakeml() and read_events() return the same results.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            cat1 = read_events(self.neries_filename)
            cat2 = _read_quakeml(self.neries_filename)
        self.assertEqual(cat1, cat2)

    def test_reading_twice_raises_no_warning(self):
        """
        Tests that reading a QuakeML file twice does not raise a warnings.

        Not an extensive test but likely good enough.
        """
        filename = os.path.join(self.path, "qml-example-1.2-RC3.xml")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cat1 = _read_quakeml(filename)
            self.assertEqual(len(w), 0)
            cat2 = _read_quakeml(filename)
            self.assertEqual(len(w), 0)
        assert_no_extras(cat1)

        self.assertEqual(cat1, cat2)

    def test_read_amplitude_time_window(self):
        """
        Tests reading an QuakeML Amplitude with TimeWindow.
        """
        filename = os.path.join(self.path, "qml-example-1.2-RC3.xml")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cat = _read_quakeml(filename)
            self.assertEqual(len(w), 0)

        self.assertEqual(len(cat[0].amplitudes), 1)
        amp = cat[0].amplitudes[0]
        self.assertEqual(amp.type, "A")
        self.assertEqual(amp.category, "point")
        self.assertEqual(amp.unit, "m/s")
        self.assertEqual(amp.generic_amplitude, 1e-08)
        self.assertEqual(amp.time_window.begin, 0.0)
        self.assertEqual(amp.time_window.end, 0.51424)
        self.assertEqual(amp.time_window.reference,
                         UTCDateTime("2007-10-10T14:40:39.055"))

    def test_write_amplitude_time_window(self):
        """
        Tests writing an QuakeML Amplitude with TimeWindow.
        """
        filename = os.path.join(self.path, "qml-example-1.2-RC3.xml")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cat = _read_quakeml(filename)
            self.assertEqual(len(w), 0)

        with NamedTemporaryFile() as tf:
            tmpfile = tf.name
            cat.write(tmpfile, format='QUAKEML')
            with open(tmpfile, "rb") as fh:
                lines = fh.readlines()

            firstline = 45
            while b"<amplitude " not in lines[firstline]:
                firstline += 1

            got = [lines[i_].strip()
                   for i_ in range(firstline, firstline + 13)]
            expected = [
                b'<amplitude publicID="smi:nz.org.geonet/event/2806038g/'
                b'amplitude/1/modified">',
                b'<genericAmplitude>',
                b'<value>1e-08</value>',
                b'</genericAmplitude>',
                b'<type>A</type>',
                b'<category>point</category>',
                b'<unit>m/s</unit>',
                b'<timeWindow>',
                b'<reference>2007-10-10T14:40:39.055000Z</reference>',
                b'<begin>0.0</begin>',
                b'<end>0.51424</end>',
                b'</timeWindow>',
                b'</amplitude>']
            self.assertEqual(got, expected)

    def test_write_with_extra_tags_and_read(self):
        """
        Tests that a QuakeML file with additional custom "extra" tags gets
        written correctly and that when reading it again the extra tags are
        parsed correctly.
        """
        filename = os.path.join(self.path, "quakeml_1.2_origin.xml")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cat = _read_quakeml(filename)
            assert_no_extras(cat)
            self.assertEqual(len(w), 0)

        # add some custom tags to first event:
        #  - tag with explicit namespace but no explicit ns abbreviation
        #  - tag without explicit namespace (gets obspy default ns)
        #  - tag with explicit namespace and namespace abbreviation
        my_extra = AttribDict(
            {'public': {'value': False,
                        'namespace': 'http://some-page.de/xmlns/1.0',
                        'attrib': {'some_attrib': 'some_value',
                                   'another_attrib': 'another_value'}},
             'custom': {'value': 'True',
                        'namespace': 'http://test.org/xmlns/0.1'},
             'new_tag': {'value': 1234,
                         'namespace': 'http://test.org/xmlns/0.1'},
             'tX': {'value': UTCDateTime('2013-01-02T13:12:14.600000Z'),
                    'namespace': 'http://test.org/xmlns/0.1'},
             'dataid': {'namespace': 'http://anss.org/xmlns/catalog/0.1',
                        'type': 'attribute', 'value': '00999999'},
             # some nested tags :
             'quantity': {'namespace': 'http://some-page.de/xmlns/1.0',
                          'attrib': {'attrib1': 'attrib_value1',
                                     'attrib2': 'attrib_value2'},
                          'value': {
                              'my_nested_tag1': {
                                  'namespace': 'http://some-page.de/xmlns/1.0',
                                  'value': 1.23E10},
                              'my_nested_tag2': {
                                  'namespace': 'http://some-page.de/xmlns/1.0',
                                  'value': False}}}})
        nsmap = {'ns0': 'http://test.org/xmlns/0.1',
                 'catalog': 'http://anss.org/xmlns/catalog/0.1'}
        cat[0].extra = my_extra.copy()
        # insert a pick with an extra field
        p = Pick()
        p.extra = {'weight': {'value': 2,
                              'namespace': 'http://test.org/xmlns/0.1'}}
        cat[0].picks.append(p)

        with NamedTemporaryFile() as tf:
            tmpfile = tf.name
            # write file
            cat.write(tmpfile, format='QUAKEML', nsmap=nsmap)
            # check contents
            with open(tmpfile, 'rb') as fh:
                # enforce reproducible attribute orders through write_c14n
                obj = etree.fromstring(fh.read()).getroottree()
                buf = io.BytesIO()
                obj.write_c14n(buf)
                buf.seek(0, 0)
                content = buf.read()
            # check namespace definitions in root element
            expected = [b'<q:quakeml',
                        b'xmlns:catalog="http://anss.org/xmlns/catalog/0.1"',
                        b'xmlns:ns0="http://test.org/xmlns/0.1"',
                        b'xmlns:ns1="http://some-page.de/xmlns/1.0"',
                        b'xmlns:q="http://quakeml.org/xmlns/quakeml/1.2"',
                        b'xmlns="http://quakeml.org/xmlns/bed/1.2"']
            for line in expected:
                self.assertIn(line, content)
            # check additional tags
            expected = [
                b'<ns0:custom>True</ns0:custom>',
                b'<ns0:new_tag>1234</ns0:new_tag>',
                b'<ns0:tX>2013-01-02T13:12:14.600000Z</ns0:tX>',
                b'<ns1:public '
                b'another_attrib="another_value" '
                b'some_attrib="some_value">false</ns1:public>'
            ]
            for line in expected:
                self.assertIn(line, content)
            # now, read again to test if it's parsed correctly..
            cat = _read_quakeml(tmpfile)
        # when reading..
        #  - namespace abbreviations should be disregarded
        #  - we always end up with a namespace definition, even if it was
        #    omitted when originally setting the custom tag
        #  - custom namespace abbreviations should attached to Catalog
        self.assertTrue(hasattr(cat[0], 'extra'))

        def _tostr(x):
            if isinstance(x, bool):
                if x:
                    return str('true')
                else:
                    return str('false')
            elif isinstance(x, AttribDict):
                for key, value in x.items():
                    x[key].value = _tostr(value['value'])
                return x
            else:
                return str(x)

        for key, value in my_extra.items():
            my_extra[key]['value'] = _tostr(value['value'])
        self.assertEqual(cat[0].extra, my_extra)
        self.assertTrue(hasattr(cat[0].picks[0], 'extra'))
        self.assertEqual(
            cat[0].picks[0].extra,
            {'weight': {'value': '2',
                        'namespace': 'http://test.org/xmlns/0.1'}})
        self.assertTrue(hasattr(cat, 'nsmap'))
        self.assertEqual(getattr(cat, 'nsmap')['ns0'], nsmap['ns0'])

    def test_read_same_file_twice_to_same_variable(self):
        """
        Reading the same file twice to the same variable should not raise a
        warning.
        """
        sio = io.BytesIO(b"""<?xml version='1.0' encoding='utf-8'?>
        <q:quakeml xmlns:q="http://quakeml.org/xmlns/quakeml/1.2"
                   xmlns="http://quakeml.org/xmlns/bed/1.2">
          <eventParameters publicID="smi:local/catalog">
            <event publicID="smi:local/event">
              <origin publicID="smi:local/origin">
                <time>
                  <value>1970-01-01T00:00:00.000000Z</value>
                </time>
                <latitude>
                  <value>0.0</value>
                </latitude>
                <longitude>
                  <value>0.0</value>
                </longitude>
                <depth>
                  <value>0.0</value>
                </depth>
                <arrival publicID="smi:local/arrival">
                  <pickID>smi:local/pick</pickID>
                  <phase>P</phase>
                </arrival>
              </origin>
            </event>
          </eventParameters>
        </q:quakeml>
        """)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cat1 = read_events(sio)  # NOQA
            cat2 = read_events(sio)  # NOQA

        assert_no_extras(cat1)
        # No warning should have been raised.
        self.assertEqual(len(w), 0)

    def test_focal_mechanism_write_read(self):
        """
        Test for a bug in reading a FocalMechanism without MomentTensor from
        QuakeML file. Makes sure that FocalMechanism.moment_tensor stays None
        if no MomentTensor is in the file.
        """
        memfile = io.BytesIO()
        # create virtually empty FocalMechanism
        fm = FocalMechanism()
        event = Event(focal_mechanisms=[fm])
        cat = Catalog(events=[event])
        cat.write(memfile, format="QUAKEML", validate=True)
        # now read again, and make sure there's no stub MomentTensor, but
        # rather `None`
        memfile.seek(0)
        cat = read_events(memfile, format="QUAKEML")
        assert_no_extras(cat)
        self.assertEqual(cat[0].focal_mechanisms[0].moment_tensor, None)

    def test_avoid_empty_stub_elements(self):
        """
        Test for a bug in reading QuakeML. Makes sure that some subelements do
        not get assigned stub elements, but rather stay None.
        """
        # Test 1: Test subelements of moment_tensor
        memfile = io.BytesIO()
        # create virtually empty FocalMechanism
        mt = MomentTensor(derived_origin_id='smi:local/abc')
        fm = FocalMechanism(moment_tensor=mt)
        event = Event(focal_mechanisms=[fm])
        cat = Catalog(events=[event])
        cat.write(memfile, format="QUAKEML", validate=True)
        # now read again, and make sure there's no stub subelements on
        # MomentTensor, but rather `None`
        memfile.seek(0)
        cat = read_events(memfile, format="QUAKEML")
        assert_no_extras(cat)
        self.assertEqual(cat[0].focal_mechanisms[0].moment_tensor.tensor, None)
        self.assertEqual(
            cat[0].focal_mechanisms[0].moment_tensor.source_time_function,
            None)
        # Test 2: Test subelements of focal_mechanism
        memfile = io.BytesIO()
        # create virtually empty FocalMechanism
        fm = FocalMechanism()
        event = Event(focal_mechanisms=[fm])
        cat = Catalog(events=[event])
        cat.write(memfile, format="QUAKEML", validate=True)
        # now read again, and make sure there's no stub MomentTensor, but
        # rather `None`
        memfile.seek(0)
        cat = read_events(memfile, format="QUAKEML")
        assert_no_extras(cat)
        self.assertEqual(cat[0].focal_mechanisms[0].nodal_planes, None)
        self.assertEqual(cat[0].focal_mechanisms[0].principal_axes, None)

    def test_writing_invalid_quakeml_id(self):
        """
        Some ids might be invalid. We still want to write them to not mess
        with any external tools relying on the ids. But we also raise a
        warning of course.
        """
        filename = os.path.join(self.path, 'invalid_id.xml')
        cat = read_events(filename)
        self.assertEqual(
            cat[0].resource_id.id,
            "smi:org.gfz-potsdam.de/geofon/RMHP(60)>>ITAPER(3)>>BW(4,5,15)")
        with NamedTemporaryFile() as tf:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                cat.write(tf.name, format="quakeml")
                cat2 = read_events(tf.name)
        self.assertEqual(len(w), 19)
        self.assertEqual(
            w[0].message.args[0],
            "'smi:org.gfz-potsdam.de/geofon/RMHP(60)>>ITAPER(3)>>BW(4,5,15)' "
            "is not a valid QuakeML URI. It will be in the final file but "
            "note that the file will not be a valid QuakeML file.")
        self.assertEqual(
            cat2[0].resource_id.id,
            "smi:org.gfz-potsdam.de/geofon/RMHP(60)>>ITAPER(3)>>BW(4,5,15)")

    def test_reading_invalid_enums(self):
        """
        Raise a warning when an invalid enum value is attempted to be read.
        """
        filename = os.path.join(self.path, "invalid_enum.xml")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cat = read_events(filename)
        self.assertEqual(len(w), 1)
        self.assertEqual(
            w[0].message.args[0],
            'Setting attribute "depth_type" failed. Value "randomized" could '
            'not be converted to type "Enum(["from location", "from moment '
            'tensor inversion", ..., "operator assigned", "other"])". The '
            'attribute "depth_type" will not be set and will be missing in '
            'the resulting object.')
        # It should of course not be set.
        self.assertIsNone(cat[0].origins[0].depth_type)

    def test_issue_2339(self):
        """
        Make sure an empty EventDescription object does not prevent a catalog
        from being saved to disk and re-read, while still being equal.
        """
        # create a catalog  with an empty event description
        empty_description = EventDescription()
        cat1 = Catalog(events=[read_events()[0]])
        cat1[0].event_descriptions.append(empty_description)
        # serialize the catalog using quakeml and re-read
        bio = io.BytesIO()
        cat1.write(bio, 'quakeml')
        bio.seek(0)
        cat2 = read_events(bio)
        assert_no_extras(cat2)
        # the text of the empty EventDescription instances should be equal
        text1 = cat1[0].event_descriptions[-1].text
        text2 = cat2[0].event_descriptions[-1].text
        self.assertEqual(text1, text2)
        # the two catalogs should be equal
        self.assertEqual(cat1, cat2)

    def test_native_namespace_in_extra(self):
        """
        Make sure that QuakeML tags that are not the same as the document
        root's namespaces still are handled as custom tags (coming
        after any expected/mandatory tags) and get parsed into extras section
        properly.
        """
        custom1 = {
            'value': u'11111',
            'namespace': 'http://quakeml.org/xmlns/bed/9.99'}
        custom2 = {
            'value': u'22222',
            'namespace': 'http://quakeml.org/xmlns/quakeml/8.87'}
        extra = {'custom1': custom1, 'custom2': custom2}

        cat = Catalog()
        cat.extra = extra

        with io.BytesIO() as buf:
            cat.write(buf, format='QUAKEML')
            buf.seek(0)
            cat2 = read_events(buf, format='QUAKEML')

        self.assertEqual(extra, cat2.extra)
        self.assertIn(('custom1', custom1), cat2.extra.items())
        self.assertIn(('custom2', custom2), cat2.extra.items())


def suite():
    return unittest.makeSuite(QuakeMLTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
