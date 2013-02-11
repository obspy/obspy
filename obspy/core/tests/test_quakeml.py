# -*- coding: utf-8 -*-

from obspy.core.event import ResourceIdentifier, WaveformStreamID, \
    readEvents, Event
from obspy.core.quakeml import readQuakeML, Pickler, writeQuakeML
from obspy.core.utcdatetime import UTCDateTime
from obspy.core.util.base import NamedTemporaryFile
from xml.etree.ElementTree import tostring, fromstring
import os
import unittest
import warnings


class QuakeMLTestCase(unittest.TestCase):
    """
    Test suite for obspy.core.quakeml
    """
    def setUp(self):
        # directory where the test files are located
        self.path = os.path.join(os.path.dirname(__file__), 'data')

    def _compareStrings(self, doc1, doc2):
        """
        Simple helper function to compare two XML strings.
        """
        obj1 = fromstring(doc1)
        str1 = ''.join([s.strip() for s in tostring(obj1).splitlines()])
        obj2 = fromstring(doc2)
        str2 = ''.join([s.strip() for s in tostring(obj2).splitlines()])
        if str1 != str2:
            print
            print str1
            print str2
        self.assertEquals(str1, str2)

    def test_readQuakeML(self):
        """
        """
        # IRIS
        filename = os.path.join(self.path, 'iris_events.xml')
        catalog = readQuakeML(filename)
        self.assertEquals(len(catalog), 2)
        self.assertEquals(catalog[0].resource_id,
            ResourceIdentifier(\
                'smi:www.iris.edu/ws/event/query?eventId=3279407'))
        self.assertEquals(catalog[1].resource_id,
            ResourceIdentifier(\
                'smi:www.iris.edu/ws/event/query?eventId=2318174'))
        # NERIES
        filename = os.path.join(self.path, 'neries_events.xml')
        catalog = readQuakeML(filename)
        self.assertEquals(len(catalog), 3)
        self.assertEquals(catalog[0].resource_id,
            ResourceIdentifier('quakeml:eu.emsc/event/20120404_0000041'))
        self.assertEquals(catalog[1].resource_id,
            ResourceIdentifier('quakeml:eu.emsc/event/20120404_0000038'))
        self.assertEquals(catalog[2].resource_id,
            ResourceIdentifier('quakeml:eu.emsc/event/20120404_0000039'))

    def test_event(self):
        """
        Tests Event object.
        """
        filename = os.path.join(self.path, 'quakeml_1.2_event.xml')
        catalog = readQuakeML(filename)
        self.assertEquals(len(catalog), 1)
        event = catalog[0]
        self.assertEquals(event.resource_id,
            ResourceIdentifier('smi:ch.ethz.sed/event/historical/1165'))
        # enums
        self.assertEquals(event.event_type, 'earthquake')
        self.assertEquals(event.event_type_certainty, 'suspected')
        # comments
        self.assertEquals(len(event.comments), 2)
        c = event.comments
        self.assertEquals(c[0].text, 'Relocated after re-evaluation')
        self.assertEquals(c[0].resource_id, None)
        self.assertEquals(c[0].creation_info.agency_id, 'EMSC')
        self.assertEquals(c[1].text, 'Another comment')
        self.assertEquals(c[1].resource_id,
            ResourceIdentifier(resource_id="smi:some/comment/id/number_3"))
        self.assertEquals(c[1].creation_info, None)
        # event descriptions
        self.assertEquals(len(event.event_descriptions), 3)
        d = event.event_descriptions
        self.assertEquals(d[0].text, '1906 San Francisco Earthquake')
        self.assertEquals(d[0].type, 'earthquake name')
        self.assertEquals(d[1].text, 'NEAR EAST COAST OF HONSHU, JAPAN')
        self.assertEquals(d[1].type, 'Flinn-Engdahl region')
        self.assertEquals(d[2].text, 'free-form string')
        self.assertEquals(d[2].type, None)
        # creation info
        self.assertEquals(event.creation_info.author, "Erika Mustermann")
        self.assertEquals(event.creation_info.agency_id, "EMSC")
        self.assertEquals(event.creation_info.author_uri,
            ResourceIdentifier("smi:smi-registry/organization/EMSC"))
        self.assertEquals(event.creation_info.agency_uri,
            ResourceIdentifier("smi:smi-registry/organization/EMSC"))
        self.assertEquals(event.creation_info.creation_time,
            UTCDateTime("2012-04-04T16:40:50+00:00"))
        self.assertEquals(event.creation_info.version, "1.0.1")
        # exporting back to XML should result in the same document
        original = open(filename, "rt").read()
        processed = Pickler().dumps(catalog)
        self._compareStrings(original, processed)

    def test_origin(self):
        """
        Tests Origin object.
        """
        filename = os.path.join(self.path, 'quakeml_1.2_origin.xml')
        catalog = readQuakeML(filename)
        self.assertEquals(len(catalog), 1)
        self.assertEquals(len(catalog[0].origins), 1)
        origin = catalog[0].origins[0]
        self.assertEquals(origin.resource_id,
            ResourceIdentifier(\
            'smi:www.iris.edu/ws/event/query?originId=7680412'))
        self.assertEquals(origin.time, UTCDateTime("2011-03-11T05:46:24.1200"))
        self.assertEquals(origin.latitude, 38.297)
        self.assertEquals(origin.latitude_errors.lower_uncertainty, None)
        self.assertEquals(origin.longitude, 142.373)
        self.assertEquals(origin.longitude_errors.uncertainty, None)
        self.assertEquals(origin.depth, 29.0)
        self.assertEquals(origin.depth_errors.confidence_level, 50.0)
        self.assertEquals(origin.depth_type, "from location")
        self.assertEquals(origin.method_id,
            ResourceIdentifier(resource_id="smi:some/method/NA"))
        self.assertEquals(origin.time_fixed, None)
        self.assertEquals(origin.epicenter_fixed, False)
        self.assertEquals(origin.reference_system_id,
            ResourceIdentifier(resource_id="smi:some/reference/muh"))
        self.assertEquals(origin.earth_model_id,
            ResourceIdentifier(resource_id="smi:same/model/maeh"))
        self.assertEquals(origin.evaluation_mode, "manual")
        self.assertEquals(origin.evaluation_status, "preliminary")
        self.assertEquals(origin.origin_type, "hypocenter")
        # composite times
        self.assertEquals(len(origin.composite_times), 2)
        c = origin.composite_times
        self.assertEquals(c[0].year, 2029)
        self.assertEquals(c[0].month, None)
        self.assertEquals(c[0].day, None)
        self.assertEquals(c[0].hour, 12)
        self.assertEquals(c[0].minute, None)
        self.assertEquals(c[0].second, None)
        self.assertEquals(c[1].year, None)
        self.assertEquals(c[1].month, None)
        self.assertEquals(c[1].day, None)
        self.assertEquals(c[1].hour, 1)
        self.assertEquals(c[1].minute, None)
        self.assertEquals(c[1].second, 29.124234)
        # quality
        self.assertEquals(origin.quality.used_station_count, 16)
        self.assertEquals(origin.quality.standard_error, 0)
        self.assertEquals(origin.quality.azimuthal_gap, 231)
        self.assertEquals(origin.quality.maximum_distance, 53.03)
        self.assertEquals(origin.quality.minimum_distance, 2.45)
        self.assertEquals(origin.quality.associated_phase_count, None)
        self.assertEquals(origin.quality.associated_station_count, None)
        self.assertEquals(origin.quality.depth_phase_count, None)
        self.assertEquals(origin.quality.secondary_azimuthal_gap, None)
        self.assertEquals(origin.quality.ground_truth_level, None)
        self.assertEquals(origin.quality.median_distance, None)
        # comments
        self.assertEquals(len(origin.comments), 2)
        c = origin.comments
        self.assertEquals(c[0].text, 'Some comment')
        self.assertEquals(c[0].resource_id,
            ResourceIdentifier(resource_id="smi:some/comment/reference"))
        self.assertEquals(c[0].creation_info.author, 'EMSC')
        self.assertEquals(c[1].resource_id, None)
        self.assertEquals(c[1].creation_info, None)
        self.assertEquals(c[1].text, 'Another comment')
        # creation info
        self.assertEquals(origin.creation_info.author, "NEIC")
        self.assertEquals(origin.creation_info.agency_id, None)
        self.assertEquals(origin.creation_info.author_uri, None)
        self.assertEquals(origin.creation_info.agency_uri, None)
        self.assertEquals(origin.creation_info.creation_time, None)
        self.assertEquals(origin.creation_info.version, None)
        # origin uncertainty
        u = origin.origin_uncertainty
        self.assertEquals(u.preferred_description, "uncertainty ellipse")
        self.assertEquals(u.horizontal_uncertainty, 9000)
        self.assertEquals(u.min_horizontal_uncertainty, 6000)
        self.assertEquals(u.max_horizontal_uncertainty, 10000)
        self.assertEquals(u.azimuth_max_horizontal_uncertainty, 80.0)
        # confidence ellipsoid
        c = u.confidence_ellipsoid
        self.assertEquals(c.semi_intermediate_axis_length, 2.123)
        self.assertEquals(c.major_axis_rotation, 5.123)
        self.assertEquals(c.major_axis_plunge, 3.123)
        self.assertEquals(c.semi_minor_axis_length, 1.123)
        self.assertEquals(c.semi_major_axis_length, 0.123)
        self.assertEquals(c.major_axis_azimuth, 4.123)
        # exporting back to XML should result in the same document
        original = open(filename, "rt").read()
        processed = Pickler().dumps(catalog)
        self._compareStrings(original, processed)

    def test_magnitude(self):
        """
        Tests Magnitude object.
        """
        filename = os.path.join(self.path, 'quakeml_1.2_magnitude.xml')
        catalog = readQuakeML(filename)
        self.assertEquals(len(catalog), 1)
        self.assertEquals(len(catalog[0].magnitudes), 1)
        mag = catalog[0].magnitudes[0]
        self.assertEquals(mag.resource_id,
            ResourceIdentifier('smi:ch.ethz.sed/magnitude/37465'))
        self.assertEquals(mag.mag, 5.5)
        self.assertEquals(mag.mag_errors.uncertainty, 0.1)
        self.assertEquals(mag.magnitude_type, 'MS')
        self.assertEquals(mag.method_id,
            ResourceIdentifier(\
            'smi:ch.ethz.sed/magnitude/generic/surface_wave_magnitude'))
        self.assertEquals(mag.station_count, 8)
        self.assertEquals(mag.evaluation_status, 'preliminary')
        # comments
        self.assertEquals(len(mag.comments), 2)
        c = mag.comments
        self.assertEquals(c[0].text, 'Some comment')
        self.assertEquals(c[0].resource_id,
            ResourceIdentifier(resource_id="smi:some/comment/id/muh"))
        self.assertEquals(c[0].creation_info.author, 'EMSC')
        self.assertEquals(c[1].creation_info, None)
        self.assertEquals(c[1].text, 'Another comment')
        self.assertEquals(c[1].resource_id, None)
        # creation info
        self.assertEquals(mag.creation_info.author, "NEIC")
        self.assertEquals(mag.creation_info.agency_id, None)
        self.assertEquals(mag.creation_info.author_uri, None)
        self.assertEquals(mag.creation_info.agency_uri, None)
        self.assertEquals(mag.creation_info.creation_time, None)
        self.assertEquals(mag.creation_info.version, None)
        # exporting back to XML should result in the same document
        original = open(filename, "rt").read()
        processed = Pickler().dumps(catalog)
        self._compareStrings(original, processed)

    def test_stationmagnitudecontribution(self):
        """
        Tests the station magnitude contribution object.
        """
        filename = os.path.join(self.path,
            'quakeml_1.2_stationmagnitudecontributions.xml')
        catalog = readQuakeML(filename)
        self.assertEquals(len(catalog), 1)
        self.assertEquals(len(catalog[0].magnitudes), 1)
        self.assertEquals(
            len(catalog[0].magnitudes[0].station_magnitude_contributions), 2)
        # Check the first stationMagnitudeContribution object.
        stat_contrib = \
            catalog[0].magnitudes[0].station_magnitude_contributions[0]
        self.assertEqual(stat_contrib.station_magnitude_id.resource_id,
            "smi:ch.ethz.sed/magnitude/station/881342")
        self.assertEqual(stat_contrib.weight, 0.77)
        self.assertEqual(stat_contrib.residual, 0.02)
        # Check the second stationMagnitudeContribution object.
        stat_contrib = \
            catalog[0].magnitudes[0].station_magnitude_contributions[1]
        self.assertEqual(stat_contrib.station_magnitude_id.resource_id,
            "smi:ch.ethz.sed/magnitude/station/881334")
        self.assertEqual(stat_contrib.weight, 0.55)
        self.assertEqual(stat_contrib.residual, 0.11)

        # exporting back to XML should result in the same document
        original = open(filename, "rt").read()
        processed = Pickler().dumps(catalog)
        self._compareStrings(original, processed)

    def test_stationmagnitude(self):
        """
        Tests StationMagnitude object.
        """
        filename = os.path.join(self.path, 'quakeml_1.2_stationmagnitude.xml')
        catalog = readQuakeML(filename)
        self.assertEquals(len(catalog), 1)
        self.assertEquals(len(catalog[0].station_magnitudes), 1)
        mag = catalog[0].station_magnitudes[0]
        # Assert the actual StationMagnitude object. Everything that is not set
        # in the QuakeML file should be set to None.
        self.assertEqual(mag.resource_id,
            ResourceIdentifier("smi:ch.ethz.sed/magnitude/station/881342"))
        self.assertEquals(mag.origin_id,
            ResourceIdentifier('smi:some/example/id'))
        self.assertEquals(mag.mag, 6.5)
        self.assertEquals(mag.mag_errors.uncertainty, 0.2)
        self.assertEquals(mag.station_magnitude_type, 'MS')
        self.assertEqual(mag.amplitude_id,
            ResourceIdentifier("smi:ch.ethz.sed/amplitude/824315"))
        self.assertEqual(mag.method_id,
            ResourceIdentifier(\
                "smi:ch.ethz.sed/magnitude/generic/surface_wave_magnitude"))
        self.assertEqual(mag.waveform_id,
            WaveformStreamID(network_code='BW', station_code='FUR',
                             resource_uri="smi:ch.ethz.sed/waveform/201754"))
        self.assertEqual(mag.creation_info, None)
        # exporting back to XML should result in the same document
        original = open(filename, "rt").read()
        processed = Pickler().dumps(catalog)
        self._compareStrings(original, processed)

    def test_arrival(self):
        """
        Tests Arrival object.
        """
        filename = os.path.join(self.path, 'quakeml_1.2_arrival.xml')
        catalog = readQuakeML(filename)
        self.assertEquals(len(catalog), 1)
        self.assertEquals(len(catalog[0].origins[0].arrivals), 2)
        ar = catalog[0].origins[0].arrivals[0]
        # Test the actual Arrival object. Everything not set in the QuakeML
        # file should be None.
        self.assertEquals(ar.pick_id,
            ResourceIdentifier('smi:ch.ethz.sed/pick/117634'))
        self.assertEquals(ar.phase, 'Pn')
        self.assertEquals(ar.azimuth, 12.0)
        self.assertEquals(ar.distance, 0.5)
        self.assertEquals(ar.takeoff_angle, 11.0)
        self.assertEquals(ar.takeoff_angle_errors.uncertainty, 0.2)
        self.assertEquals(ar.time_residual, 1.6)
        self.assertEquals(ar.horizontal_slowness_residual, 1.7)
        self.assertEquals(ar.backazimuth_residual, 1.8)
        self.assertEquals(ar.time_weight, 0.48)
        self.assertEquals(ar.horizontal_slowness_weight, 0.49)
        self.assertEquals(ar.backazimuth_weight, 0.5)
        self.assertEquals(ar.earth_model_id,
            ResourceIdentifier('smi:ch.ethz.sed/earthmodel/U21'))
        self.assertEquals(len(ar.comments), 1)
        self.assertEquals(ar.creation_info.author, "Erika Mustermann")
        # exporting back to XML should result in the same document
        original = open(filename, "rt").read()
        processed = Pickler().dumps(catalog)
        self._compareStrings(original, processed)

    def test_pick(self):
        """
        Tests Pick object.
        """
        filename = os.path.join(self.path, 'quakeml_1.2_pick.xml')
        catalog = readQuakeML(filename)
        self.assertEquals(len(catalog), 1)
        self.assertEquals(len(catalog[0].picks), 2)
        pick = catalog[0].picks[0]
        self.assertEquals(pick.resource_id,
            ResourceIdentifier('smi:ch.ethz.sed/pick/117634'))
        self.assertEquals(pick.time, UTCDateTime('2005-09-18T22:04:35Z'))
        self.assertEquals(pick.time_errors.uncertainty, 0.012)
        self.assertEquals(pick.waveform_id,
            WaveformStreamID(network_code='BW', station_code='FUR',
                             resource_uri='smi:ch.ethz.sed/waveform/201754'))
        self.assertEquals(pick.filter_id,
            ResourceIdentifier('smi:ch.ethz.sed/filter/lowpass/standard'))
        self.assertEquals(pick.method_id,
            ResourceIdentifier('smi:ch.ethz.sed/picker/autopicker/6.0.2'))
        self.assertEquals(pick.backazimuth, 44.0)
        self.assertEquals(pick.onset, 'impulsive')
        self.assertEquals(pick.phase_hint, 'Pn')
        self.assertEquals(pick.polarity, 'positive')
        self.assertEquals(pick.evaluation_mode, "manual")
        self.assertEquals(pick.evaluation_status, "confirmed")
        self.assertEquals(len(pick.comments), 2)
        self.assertEquals(pick.creation_info.author, "Erika Mustermann")
        # exporting back to XML should result in the same document
        original = open(filename, "rt").read()
        processed = Pickler().dumps(catalog)
        self._compareStrings(original, processed)

    def test_focalmechanism(self):
        """
        Tests FocalMechanism object.
        """
        filename = os.path.join(self.path, 'quakeml_1.2_focalmechanism.xml')
        catalog = readQuakeML(filename)
        self.assertEquals(len(catalog), 1)
        self.assertEquals(len(catalog[0].focal_mechanisms), 2)
        fm = catalog[0].focal_mechanisms[0]
        # general
        self.assertEquals(fm.resource_id,
            ResourceIdentifier('smi:ISC/fmid=292309'))
        self.assertEquals(fm.waveform_id.network_code, 'BW')
        self.assertEquals(fm.waveform_id.station_code, 'FUR')
        self.assertEquals(fm.waveform_id.resource_uri,
            ResourceIdentifier(resource_id="smi:ch.ethz.sed/waveform/201754"))
        self.assertTrue(isinstance(fm.waveform_id, WaveformStreamID))
        self.assertEquals(fm.triggering_origin_id,
            ResourceIdentifier('smi:originId=7680412'))
        self.assertAlmostEquals(fm.azimuthal_gap, 0.123)
        self.assertEquals(fm.station_polarity_count, 987)
        self.assertAlmostEquals(fm.misfit, 1.234)
        self.assertAlmostEquals(fm.station_distribution_ratio, 2.345)
        self.assertEquals(fm.method_id,
            ResourceIdentifier('smi:ISC/methodID=Best_double_couple'))
        # comments
        self.assertEquals(len(fm.comments), 2)
        c = fm.comments
        self.assertEquals(c[0].text, 'Relocated after re-evaluation')
        self.assertEquals(c[0].resource_id, None)
        self.assertEquals(c[0].creation_info.agency_id, 'MUH')
        self.assertEquals(c[1].text, 'Another MUH')
        self.assertEquals(c[1].resource_id,
            ResourceIdentifier(resource_id="smi:some/comment/id/number_3"))
        self.assertEquals(c[1].creation_info, None)
        # creation info
        self.assertEquals(fm.creation_info.author, "Erika Mustermann")
        self.assertEquals(fm.creation_info.agency_id, "MUH")
        self.assertEquals(fm.creation_info.author_uri,
            ResourceIdentifier("smi:smi-registry/organization/MUH"))
        self.assertEquals(fm.creation_info.agency_uri,
            ResourceIdentifier("smi:smi-registry/organization/MUH"))
        self.assertEquals(fm.creation_info.creation_time,
            UTCDateTime("2012-04-04T16:40:50+00:00"))
        self.assertEquals(fm.creation_info.version, "1.0.1")
        # nodalPlanes
        self.assertAlmostEqual(fm.nodal_planes.nodal_plane_1.strike, 346.0)
        self.assertAlmostEqual(fm.nodal_planes.nodal_plane_1.dip, 57.0)
        self.assertAlmostEqual(fm.nodal_planes.nodal_plane_1.rake, 75.0)
        self.assertAlmostEqual(fm.nodal_planes.nodal_plane_2.strike, 193.0)
        self.assertAlmostEqual(fm.nodal_planes.nodal_plane_2.dip, 36.0)
        self.assertAlmostEqual(fm.nodal_planes.nodal_plane_2.rake, 112.0)
        self.assertEquals(fm.nodal_planes.preferred_plane, 2)
        # principalAxes
        self.assertAlmostEqual(fm.principal_axes.t_axis.azimuth, 216.0)
        self.assertAlmostEqual(fm.principal_axes.t_axis.plunge, 73.0)
        self.assertAlmostEqual(fm.principal_axes.t_axis.length, 1.050e+18)
        self.assertAlmostEqual(fm.principal_axes.p_axis.azimuth, 86.0)
        self.assertAlmostEqual(fm.principal_axes.p_axis.plunge, 10.0)
        self.assertAlmostEqual(fm.principal_axes.p_axis.length, -1.180e+18)
        self.assertEquals(fm.principal_axes.n_axis.azimuth, None)
        self.assertEquals(fm.principal_axes.n_axis.plunge, None)
        self.assertEquals(fm.principal_axes.n_axis.length, None)
        # momentTensor
        mt = fm.moment_tensor
        self.assertEquals(mt.derived_origin_id,
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
        original = open(filename, "rt").read()
        processed = Pickler().dumps(catalog)
        self._compareStrings(original, processed)

    def test_writeQuakeML(self):
        """
        Tests writing a QuakeML document.
        """
        filename = os.path.join(self.path, 'qml-example-1.2-RC3.xml')
        tmpfile = NamedTemporaryFile().name
        catalog = readQuakeML(filename)
        self.assertTrue(len(catalog), 1)
        writeQuakeML(catalog, tmpfile)
        # Read file again. Avoid the (legit) warning about the already used
        # resource identifiers.
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("ignore")
            catalog2 = readQuakeML(tmpfile)
        self.assertTrue(len(catalog2), 1)
        # clean up
        os.remove(tmpfile)

    def test_readEvents(self):
        """
        Tests reading a QuakeML document via readEvents.
        """
        filename = os.path.join(self.path, 'neries_events.xml')
        tmpfile = NamedTemporaryFile().name
        catalog = readEvents(filename)
        self.assertTrue(len(catalog), 3)
        catalog.write(tmpfile, format='QUAKEML')
        # Read file again. Avoid the (legit) warning about the already used
        # resource identifiers.
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("ignore")
            catalog2 = readEvents(tmpfile)
        self.assertTrue(len(catalog2), 3)
        # clean up
        os.remove(tmpfile)

    def test_enums(self):
        """
        Parses the QuakeML xsd scheme definition and checks if all enums are
        correctly defined.

        This is a very strict test against the xsd scheme file of QuakeML
        1.2RC4. If obspy.core.event will ever be more loosely coupled to
        QuakeML this test WILL HAVE to be changed.
        """
        # Currently only works with lxml.
        try:
            from lxml.etree import parse
        except:
            return
        xsd_enum_definitions = {}
        xsd_file = os.path.join(self.path, "QuakeML-BED-1.2.xsd")
        root = parse(xsd_file).getroot()
        for elem in root.getchildren():
            # All enums are simple types.
            if not elem.tag.endswith("simpleType"):
                continue
            # They only have one child, a restriction to strings.
            children = elem.getchildren()
            if len(children) > 1 or \
                not children[0].tag.endswith("restriction") \
                or (children[0].items()[0] != ('base', 'xs:string')):
                continue
            # Furthermore all children of the restriction should be
            # enumerations.
            enums = children[0].getchildren()
            all_enums = [_i.tag.endswith("enumeration") for _i in enums]
            if not all(all_enums):
                continue
            enum_name = elem.get('name')
            xsd_enum_definitions[enum_name] = []
            for enum in enums:
                xsd_enum_definitions[enum_name].append(enum.get('value'))
        # Now import all enums and check if they are correct.
        from obspy.core import event_header
        from obspy.core.util.types import Enum
        available_enums = {}
        for module_item_name in dir(event_header):
            module_item = getattr(event_header, module_item_name)
            if type(module_item) != Enum:
                continue
            # Assign clearer names.
            enum_name = module_item_name
            enum_values = [_i.lower() for _i in module_item.keys()]
            available_enums[enum_name] = enum_values
        # Now loop over all enums defined in the xsd file and check them.
        for enum_name, enum_items in xsd_enum_definitions.iteritems():
            self.assertTrue(enum_name in available_enums.keys())
            # Check that also all enum items are available.
            available_items = available_enums[enum_name]
            available_items = [_i.lower() for _i in available_items]
            for enum_item in enum_items:
                if enum_item.lower() not in available_items:
                    msg = "Value '%s' not in Enum '%s'" % (enum_item,
                        enum_name)
                    raise Exception(msg)
            # Check if there are too many items.
            if len(available_items) != len(enum_items):
                additional_items = [_i for _i in available_items \
                    if _i.lower() not in enum_items]
                msg = "Enum {enum_name} has the following additional items" + \
                    " not defined in the xsd style sheet:\n\t{enumerations}"
                msg = msg.format(enum_name=enum_name,
                    enumerations=", ".join(additional_items))
                raise Exception(msg)

    def test_read_string(self):
        """
        Test reading a QuakeML string/unicode object via readEvents.
        """
        filename = os.path.join(self.path, 'neries_events.xml')
        data = open(filename, 'rt').read()
        catalog = readEvents(data)
        self.assertEquals(len(catalog), 3)

    def test_preferred_tags(self):
        """
        Testing preferred magnitude, origin and focal mechanism tags
        """
        # testing empty event
        ev = Event()
        self.assertEquals(ev.preferred_origin(), None)
        self.assertEquals(ev.preferred_magnitude(), None)
        self.assertEquals(ev.preferred_focal_mechanism(), None)
        # testing existing event
        filename = os.path.join(self.path, 'preferred.xml')
        catalog = readEvents(filename)
        self.assertEquals(len(catalog), 1)
        ev_str = "Event:\t2012-12-12T05:46:24.120000Z | +38.297, +142.373 " + \
                 "| 2.0 MW"
        self.assertTrue(ev_str in str(catalog.events[0]))
        # testing ids
        ev = catalog.events[0]
        self.assertEquals('smi:orig2', ev.preferred_origin_id)
        self.assertEquals('smi:mag2', ev.preferred_magnitude_id)
        self.assertEquals('smi:fm2', ev.preferred_focal_mechanism_id)
        # testing objects
        self.assertEquals(ev.preferred_origin(), ev.origins[1])
        self.assertEquals(ev.preferred_magnitude(), ev.magnitudes[1])
        self.assertEquals(ev.preferred_focal_mechanism(),
            ev.focal_mechanisms[1])


def suite():
    return unittest.makeSuite(QuakeMLTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
