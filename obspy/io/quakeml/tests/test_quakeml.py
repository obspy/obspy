# -*- coding: utf-8 -*-
import io
import math
import warnings

import pytest
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


class TestQuakeML():
    """
    Test suite for obspy.io.quakeml
    """
    def test_read_quakeml(self, testdata):
        """
        """
        # IRIS
        filename = testdata['iris_events.xml']
        catalog = _read_quakeml(filename)
        assert_no_extras(catalog)
        assert len(catalog) == 2
        assert catalog[0].resource_id == \
            ResourceIdentifier(
                'smi:www.iris.edu/ws/event/query?eventId=3279407')
        assert catalog[1].resource_id == \
            ResourceIdentifier(
                'smi:www.iris.edu/ws/event/query?eventId=2318174')
        # NERIES
        catalog = _read_quakeml(testdata['neries_events.xml'])
        assert_no_extras(catalog)
        assert len(catalog) == 3
        assert catalog[0].resource_id == \
            ResourceIdentifier('quakeml:eu.emsc/event/20120404_0000041')
        assert catalog[1].resource_id == \
            ResourceIdentifier('quakeml:eu.emsc/event/20120404_0000038')
        assert catalog[2].resource_id == \
            ResourceIdentifier('quakeml:eu.emsc/event/20120404_0000039')

    def test_usgs_eventype(self, testdata):
        filename = testdata['usgs_event.xml']
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("ignore")
            catalog = _read_quakeml(filename)
        # there are some custom namespace attributes
        assert len(catalog[0].extra) == 3
        assert len(catalog[0].origins[0].extra) == 4
        assert len(catalog) == 1
        assert catalog[0].event_type == 'quarry blast'

    def test_event(self, testdata):
        """
        Tests Event object.
        """
        filename = testdata['quakeml_1.2_event.xml']
        catalog = _read_quakeml(filename)
        assert_no_extras(catalog)
        assert len(catalog) == 1
        event = catalog[0]
        assert event.resource_id == \
            ResourceIdentifier('smi:ch.ethz.sed/event/historical/1165')
        # enums
        assert event.event_type == 'earthquake'
        assert event.event_type_certainty == 'suspected'
        # comments
        assert len(event.comments) == 2
        c = event.comments
        assert c[0].text == 'Relocated after re-evaluation'
        assert c[0].resource_id is None
        assert c[0].creation_info.agency_id == 'EMSC'
        assert c[1].text == 'Another comment'
        assert c[1].resource_id == \
            ResourceIdentifier(id="smi:some/comment/id/number_3")
        assert c[1].creation_info is None
        # event descriptions
        assert len(event.event_descriptions) == 3
        d = event.event_descriptions
        assert d[0].text == '1906 San Francisco Earthquake'
        assert d[0].type == 'earthquake name'
        assert d[1].text == 'NEAR EAST COAST OF HONSHU, JAPAN'
        assert d[1].type == 'Flinn-Engdahl region'
        assert d[2].text == 'free-form string'
        assert d[2].type is None
        # creation info
        assert event.creation_info.author == "Erika Mustermann"
        assert event.creation_info.agency_id == "EMSC"
        assert event.creation_info.author_uri == \
            ResourceIdentifier("smi:smi-registry/organization/EMSC")
        assert event.creation_info.agency_uri == \
            ResourceIdentifier("smi:smi-registry/organization/EMSC")
        assert event.creation_info.creation_time == \
            UTCDateTime("2012-04-04T16:40:50+00:00")
        assert event.creation_info.version == "1.0.1"
        # exporting back to XML should result in the same document
        with open(filename, "rb") as fp:
            original = fp.read()
        processed = Pickler().dumps(catalog)
        compare_xml_strings(original, processed)

    def test_origin(self, testdata):
        """
        Tests Origin object.
        """
        filename = testdata['quakeml_1.2_origin.xml']
        catalog = _read_quakeml(filename)
        assert_no_extras(catalog)
        assert len(catalog) == 1
        assert len(catalog[0].origins) == 1
        origin = catalog[0].origins[0]
        assert origin.resource_id == \
            ResourceIdentifier(
                'smi:www.iris.edu/ws/event/query?originId=7680412')
        assert origin.time == UTCDateTime("2011-03-11T05:46:24.1200")
        assert origin.latitude == 38.297
        assert origin.latitude_errors.lower_uncertainty is None
        assert origin.longitude == 142.373
        assert origin.longitude_errors.uncertainty is None
        assert origin.depth == 29.0
        assert origin.depth_errors.confidence_level == 50.0
        assert origin.depth_type == "from location"
        assert origin.method_id == \
            ResourceIdentifier(id="smi:some/method/NA")
        assert origin.time_fixed is None
        assert not origin.epicenter_fixed
        assert origin.reference_system_id == \
            ResourceIdentifier(id="smi:some/reference/muh")
        assert origin.earth_model_id == \
            ResourceIdentifier(id="smi:same/model/maeh")
        assert origin.region == 'Kalamazoo'
        assert origin.evaluation_mode == "manual"
        assert origin.evaluation_status == "preliminary"
        assert origin.origin_type == "hypocenter"
        # composite times
        assert len(origin.composite_times) == 2
        c = origin.composite_times
        assert c[0].year == 2029
        assert c[0].month is None
        assert c[0].day is None
        assert c[0].hour == 12
        assert c[0].minute is None
        assert c[0].second is None
        assert c[1].year is None
        assert c[1].month is None
        assert c[1].day is None
        assert c[1].hour == 1
        assert c[1].minute is None
        assert c[1].second == 29.124234
        # quality
        assert origin.quality.used_station_count == 16
        assert origin.quality.standard_error == 0
        assert origin.quality.azimuthal_gap == 231
        assert origin.quality.maximum_distance == 53.03
        assert origin.quality.minimum_distance == 2.45
        assert origin.quality.associated_phase_count is None
        assert origin.quality.associated_station_count is None
        assert origin.quality.depth_phase_count is None
        assert origin.quality.secondary_azimuthal_gap is None
        assert origin.quality.ground_truth_level is None
        assert origin.quality.median_distance is None
        # comments
        assert len(origin.comments) == 2
        c = origin.comments
        assert c[0].text == 'Some comment'
        assert c[0].resource_id == \
            ResourceIdentifier(id="smi:some/comment/reference")
        assert c[0].creation_info.author == 'EMSC'
        assert c[1].resource_id is None
        assert c[1].creation_info is None
        assert c[1].text == 'Another comment'
        # creation info
        assert origin.creation_info.author == "NEIC"
        assert origin.creation_info.agency_id is None
        assert origin.creation_info.author_uri is None
        assert origin.creation_info.agency_uri is None
        assert origin.creation_info.creation_time is None
        assert origin.creation_info.version is None
        # origin uncertainty
        u = origin.origin_uncertainty
        assert u.preferred_description == "uncertainty ellipse"
        assert u.horizontal_uncertainty == 9000
        assert u.min_horizontal_uncertainty == 6000
        assert u.max_horizontal_uncertainty == 10000
        assert u.azimuth_max_horizontal_uncertainty == 80.0
        # confidence ellipsoid
        c = u.confidence_ellipsoid
        assert c.semi_intermediate_axis_length == 2.123
        assert c.major_axis_rotation == 5.123
        assert c.major_axis_plunge == 3.123
        assert c.semi_minor_axis_length == 1.123
        assert c.semi_major_axis_length == 0.123
        assert c.major_axis_azimuth == 4.123
        # exporting back to XML should result in the same document
        with open(filename, "rb") as fp:
            original = fp.read()
        processed = Pickler().dumps(catalog)
        compare_xml_strings(original, processed)

    def test_magnitude(self, testdata):
        """
        Tests Magnitude object.
        """
        filename = testdata['quakeml_1.2_magnitude.xml']
        catalog = _read_quakeml(filename)
        assert_no_extras(catalog)
        assert len(catalog) == 1
        assert len(catalog[0].magnitudes) == 1
        mag = catalog[0].magnitudes[0]
        assert mag.resource_id == \
            ResourceIdentifier('smi:ch.ethz.sed/magnitude/37465')
        assert mag.mag == 5.5
        assert mag.mag_errors.uncertainty == 0.1
        assert mag.magnitude_type == 'MS'
        assert mag.method_id == \
            ResourceIdentifier(
                'smi:ch.ethz.sed/magnitude/generic/surface_wave_magnitude')
        assert mag.station_count == 8
        assert mag.evaluation_status == 'preliminary'
        # comments
        assert len(mag.comments) == 2
        c = mag.comments
        assert c[0].text == 'Some comment'
        assert c[0].resource_id == \
            ResourceIdentifier(id="smi:some/comment/id/muh")
        assert c[0].creation_info.author == 'EMSC'
        assert c[1].creation_info is None
        assert c[1].text == 'Another comment'
        assert c[1].resource_id is None
        # creation info
        assert mag.creation_info.author == "NEIC"
        assert mag.creation_info.agency_id is None
        assert mag.creation_info.author_uri is None
        assert mag.creation_info.agency_uri is None
        assert mag.creation_info.creation_time is None
        assert mag.creation_info.version is None
        # exporting back to XML should result in the same document
        with open(filename, "rb") as fp:
            original = fp.read()
        processed = Pickler().dumps(catalog)
        compare_xml_strings(original, processed)

    def test_station_magnitude_contribution(self, testdata):
        """
        Tests the station magnitude contribution object.
        """
        filename = testdata['quakeml_1.2_stationmagnitudecontributions.xml']
        catalog = _read_quakeml(filename)
        assert_no_extras(catalog)
        assert len(catalog) == 1
        assert len(catalog[0].magnitudes) == 1
        assert len(
            catalog[0].magnitudes[0].station_magnitude_contributions) == 2
        # Check the first stationMagnitudeContribution object.
        stat_contrib = \
            catalog[0].magnitudes[0].station_magnitude_contributions[0]
        assert stat_contrib.station_magnitude_id.id == \
            "smi:ch.ethz.sed/magnitude/station/881342"
        assert stat_contrib.weight == 0.77
        assert stat_contrib.residual == 0.02
        # Check the second stationMagnitudeContribution object.
        stat_contrib = \
            catalog[0].magnitudes[0].station_magnitude_contributions[1]
        assert stat_contrib.station_magnitude_id.id == \
            "smi:ch.ethz.sed/magnitude/station/881334"
        assert stat_contrib.weight == 0.
        assert stat_contrib.residual == 0.

        # exporting back to XML should result in the same document
        with open(filename, "rb") as fp:
            original = fp.read()
        processed = Pickler().dumps(catalog)
        compare_xml_strings(original, processed)

    def test_station_magnitude(self, testdata):
        """
        Tests StationMagnitude object.
        """
        filename = testdata['quakeml_1.2_stationmagnitude.xml']
        catalog = _read_quakeml(filename)
        assert_no_extras(catalog)
        assert len(catalog) == 1
        assert len(catalog[0].station_magnitudes) == 1
        mag = catalog[0].station_magnitudes[0]
        # Assert the actual StationMagnitude object. Everything that is not set
        # in the QuakeML file should be set to None.
        assert mag.resource_id == \
            ResourceIdentifier("smi:ch.ethz.sed/magnitude/station/881342")
        assert mag.origin_id == \
            ResourceIdentifier('smi:some/example/id')
        assert mag.mag == 6.5
        assert mag.mag_errors.uncertainty == 0.2
        assert mag.station_magnitude_type == 'MS'
        assert mag.amplitude_id == \
            ResourceIdentifier("smi:ch.ethz.sed/amplitude/824315")
        assert mag.method_id == \
            ResourceIdentifier(
                "smi:ch.ethz.sed/magnitude/generic/surface_wave_magnitude")
        assert mag.waveform_id == \
            WaveformStreamID(network_code='BW', station_code='FUR',
                             resource_uri="smi:ch.ethz.sed/waveform/201754")
        assert mag.creation_info is None
        # exporting back to XML should result in the same document
        with open(filename, "rb") as fp:
            original = fp.read()
        processed = Pickler().dumps(catalog)
        compare_xml_strings(original, processed)

    def test_data_used_in_moment_tensor(self, testdata):
        """
        Tests the data used objects in moment tensors.
        """
        filename = testdata['quakeml_1.2_data_used.xml']

        # Test reading first.
        catalog = _read_quakeml(filename)
        assert_no_extras(catalog)
        event = catalog[0]

        assert len(event.focal_mechanisms), 2
        # First focmec contains only one data used element.
        assert len(event.focal_mechanisms[0].moment_tensor.data_used) == 1
        du = event.focal_mechanisms[0].moment_tensor.data_used[0]
        assert du.wave_type == "body waves"
        assert du.station_count == 88
        assert du.component_count == 166
        assert du.shortest_period == 40.0
        # Second contains three. focmec contains only one data used element.
        assert len(event.focal_mechanisms[1].moment_tensor.data_used) == 3
        du = event.focal_mechanisms[1].moment_tensor.data_used
        assert du[0].wave_type == "body waves"
        assert du[0].station_count == 88
        assert du[0].component_count == 166
        assert du[0].shortest_period == 40.0
        assert du[1].wave_type == "surface waves"
        assert du[1].station_count == 96
        assert du[1].component_count == 189
        assert du[1].shortest_period == 50.0
        assert du[2].wave_type == "mantle waves"
        assert du[2].station_count == 41
        assert du[2].component_count == 52
        assert du[2].shortest_period == 125.0

        # exporting back to XML should result in the same document
        with open(filename, "rb") as fp:
            original = fp.read()
        processed = Pickler().dumps(catalog)
        compare_xml_strings(original, processed)

    def test_arrival(self, testdata):
        """
        Tests Arrival object.
        """
        filename = testdata['quakeml_1.2_arrival.xml']
        catalog = _read_quakeml(filename)
        assert_no_extras(catalog)
        assert len(catalog) == 1
        assert len(catalog[0].origins[0].arrivals) == 2
        ar = catalog[0].origins[0].arrivals[0]
        # Test the actual Arrival object. Everything not set in the QuakeML
        # file should be None.
        assert ar.pick_id == \
            ResourceIdentifier('smi:ch.ethz.sed/pick/117634')
        assert ar.phase == 'Pn'
        assert ar.azimuth == 12.0
        assert ar.distance == 0.5
        assert ar.takeoff_angle == 11.0
        assert ar.takeoff_angle_errors.uncertainty == 0.2
        assert ar.time_residual == 1.6
        assert ar.horizontal_slowness_residual == 1.7
        assert ar.backazimuth_residual == 1.8
        assert ar.time_weight == 0.48
        assert ar.horizontal_slowness_weight == 0.49
        assert ar.backazimuth_weight == 0.5
        assert ar.earth_model_id == \
            ResourceIdentifier('smi:ch.ethz.sed/earthmodel/U21')
        assert len(ar.comments) == 1
        assert ar.creation_info.author == "Erika Mustermann"
        # exporting back to XML should result in the same document
        with open(filename, "rb") as fp:
            original = fp.read()
        processed = Pickler().dumps(catalog)
        compare_xml_strings(original, processed)

    def test_pick(self, testdata):
        """
        Tests Pick object.
        """
        filename = testdata['quakeml_1.2_pick.xml']
        catalog = _read_quakeml(filename)
        assert_no_extras(catalog)
        assert len(catalog) == 1
        assert len(catalog[0].picks) == 2
        pick = catalog[0].picks[0]
        assert pick.resource_id == \
            ResourceIdentifier('smi:ch.ethz.sed/pick/117634')
        assert pick.time == UTCDateTime('2005-09-18T22:04:35Z')
        assert pick.time_errors.uncertainty == 0.012
        assert pick.waveform_id == \
            WaveformStreamID(network_code='BW', station_code='FUR',
                             resource_uri='smi:ch.ethz.sed/waveform/201754')
        assert pick.filter_id == \
            ResourceIdentifier('smi:ch.ethz.sed/filter/lowpass/standard')
        assert pick.method_id == \
            ResourceIdentifier('smi:ch.ethz.sed/picker/autopicker/6.0.2')
        assert pick.backazimuth == 44.0
        assert pick.onset == 'impulsive'
        assert pick.phase_hint == 'Pn'
        assert pick.polarity == 'positive'
        assert pick.evaluation_mode == "manual"
        assert pick.evaluation_status == "confirmed"
        assert len(pick.comments) == 2
        assert pick.creation_info.author == "Erika Mustermann"
        # exporting back to XML should result in the same document
        with open(filename, "rb") as fp:
            original = fp.read()
        processed = Pickler().dumps(catalog)
        compare_xml_strings(original, processed)

    def test_focalmechanism(self, testdata):
        """
        Tests FocalMechanism object.
        """
        filename = testdata['quakeml_1.2_focalmechanism.xml']
        catalog = _read_quakeml(filename)
        assert_no_extras(catalog)
        assert len(catalog) == 1
        assert len(catalog[0].focal_mechanisms) == 2
        fm = catalog[0].focal_mechanisms[0]
        # general
        assert fm.resource_id == \
            ResourceIdentifier('smi:ISC/fmid=292309')
        assert len(fm.waveform_id) == 2
        assert fm.waveform_id[0].network_code == 'BW'
        assert fm.waveform_id[0].station_code == 'FUR'
        assert fm.waveform_id[0].resource_uri == \
            ResourceIdentifier(id="smi:ch.ethz.sed/waveform/201754")
        assert isinstance(fm.waveform_id[0], WaveformStreamID)
        assert fm.triggering_origin_id == \
            ResourceIdentifier('smi:local/originId=7680412')
        assert round(abs(fm.azimuthal_gap-0.123), 7) == 0
        assert fm.station_polarity_count == 987
        assert round(abs(fm.misfit-1.234), 7) == 0
        assert round(abs(fm.station_distribution_ratio-2.345), 7) == 0
        assert fm.method_id == \
            ResourceIdentifier('smi:ISC/methodID=Best_double_couple')
        # comments
        assert len(fm.comments) == 2
        c = fm.comments
        assert c[0].text == 'Relocated after re-evaluation'
        assert c[0].resource_id is None
        assert c[0].creation_info.agency_id == 'MUH'
        assert c[1].text == 'Another MUH'
        assert c[1].resource_id == \
            ResourceIdentifier(id="smi:some/comment/id/number_3")
        assert c[1].creation_info is None
        # creation info
        assert fm.creation_info.author == "Erika Mustermann"
        assert fm.creation_info.agency_id == "MUH"
        assert fm.creation_info.author_uri == \
            ResourceIdentifier("smi:smi-registry/organization/MUH")
        assert fm.creation_info.agency_uri == \
            ResourceIdentifier("smi:smi-registry/organization/MUH")
        assert fm.creation_info.creation_time == \
            UTCDateTime("2012-04-04T16:40:50+00:00")
        assert fm.creation_info.version == "1.0.1"
        # nodalPlanes
        assert round(abs(fm.nodal_planes.nodal_plane_1.strike-346.0), 7) == 0
        assert round(abs(fm.nodal_planes.nodal_plane_1.dip-57.0), 7) == 0
        assert round(abs(fm.nodal_planes.nodal_plane_1.rake-75.0), 7) == 0
        assert round(abs(fm.nodal_planes.nodal_plane_2.strike-193.0), 7) == 0
        assert round(abs(fm.nodal_planes.nodal_plane_2.dip-36.0), 7) == 0
        assert round(abs(fm.nodal_planes.nodal_plane_2.rake-112.0), 7) == 0
        assert fm.nodal_planes.preferred_plane == 2
        # principalAxes
        assert round(abs(fm.principal_axes.t_axis.azimuth-216.0), 7) == 0
        assert round(abs(fm.principal_axes.t_axis.plunge-73.0), 7) == 0
        assert round(abs(fm.principal_axes.t_axis.length-1.050e+18), 7) == 0
        assert round(abs(fm.principal_axes.p_axis.azimuth-86.0), 7) == 0
        assert round(abs(fm.principal_axes.p_axis.plunge-10.0), 7) == 0
        assert round(abs(fm.principal_axes.p_axis.length--1.180e+18), 7) == 0
        assert fm.principal_axes.n_axis.azimuth is None
        assert fm.principal_axes.n_axis.plunge is None
        assert fm.principal_axes.n_axis.length is None
        # momentTensor
        mt = fm.moment_tensor
        assert mt.resource_id == \
            ResourceIdentifier('smi:ISC/mtid=123321')
        assert mt.derived_origin_id == \
            ResourceIdentifier('smi:ISC/origid=13145006')
        assert round(abs(mt.scalar_moment-1.100e+18), 7) == 0
        assert round(abs(mt.tensor.m_rr-9.300e+17), 7) == 0
        assert round(abs(mt.tensor.m_tt-1.700e+17), 7) == 0
        assert round(abs(mt.tensor.m_pp--1.100e+18), 7) == 0
        assert round(abs(mt.tensor.m_rt--2.200e+17), 7) == 0
        assert round(abs(mt.tensor.m_rp-4.000e+17), 7) == 0
        assert round(abs(mt.tensor.m_tp-3.000e+16), 7) == 0
        assert round(abs(mt.clvd-0.22), 7) == 0
        # exporting back to XML should result in the same document
        with open(filename, "rb") as fp:
            original = fp.read()
        processed = Pickler().dumps(catalog)
        compare_xml_strings(original, processed)

    def test_write_quakeml(self, testdata):
        """
        Tests writing a QuakeML document.
        """
        filename = testdata['qml-example-1.2-RC3.xml']
        with NamedTemporaryFile() as tf:
            tmpfile = tf.name
            catalog = _read_quakeml(filename)
            assert_no_extras(catalog)
            assert len(catalog), 1
            _write_quakeml(catalog, tmpfile, validate=IS_RECENT_LXML)
            # Read file again. Avoid the (legit) warning about the already used
            # resource identifiers.
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("ignore")
                catalog2 = _read_quakeml(tmpfile)
                assert_no_extras(catalog2)
        assert len(catalog2), 1

    def test_read_events(self, testdata):
        """
        Tests reading a QuakeML document via read_events.
        """
        with NamedTemporaryFile() as tf:
            tmpfile = tf.name
            catalog = read_events(testdata['neries_events.xml'])
            assert_no_extras(catalog)
            assert len(catalog), 3
            catalog.write(tmpfile, format='QUAKEML')
            # Read file again. Avoid the (legit) warning about the already used
            # resource identifiers.
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("ignore")
                catalog2 = read_events(tmpfile)
                assert_no_extras(catalog2)
        assert len(catalog2), 3

    @pytest.mark.skipif(not IS_RECENT_LXML, reason="lxml >= 2.3 is required")
    def test_enums(self, datapath):
        """
        Parses the QuakeML xsd scheme definition and checks if all enums are
        correctly defined.

        This is a very strict test against the xsd scheme file of QuakeML 1.2.
        If obspy.core.event will ever be more loosely coupled to QuakeML this
        test WILL HAVE to be changed.
        """
        from lxml.etree import parse

        xsd_enum_definitions = {}
        xsd_file = datapath.parent.parent / "data" / "QuakeML-BED-1.2.xsd"
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
            if not isinstance(module_item, Enum):
                continue
            # Assign clearer names.
            enum_name = module_item_name
            enum_values = [_i.lower() for _i in module_item.keys()]
            all_enums[enum_name] = enum_values
        # Now loop over all enums defined in the xsd file and check them.
        for enum_name, enum_items in xsd_enum_definitions.items():
            assert enum_name in all_enums.keys()
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

    def test_read_string(self, testdata):
        """
        Test reading a QuakeML string/unicode object via read_events.
        """
        with open(testdata['neries_events.xml'], 'rb') as fp:
            data = fp.read()

        catalog = read_events(data)
        assert_no_extras(catalog)
        assert len(catalog) == 3

    def test_preferred_tags(self, testdata):
        """
        Testing preferred magnitude, origin and focal mechanism tags
        """
        # testing empty event
        ev = Event()
        assert ev.preferred_origin() is None
        assert ev.preferred_magnitude() is None
        assert ev.preferred_focal_mechanism() is None
        # testing existing event
        filename = testdata['preferred.xml']
        catalog = read_events(filename)
        assert_no_extras(catalog)
        assert len(catalog) == 1
        ev_str = "Event:\t2012-12-12T05:46:24.120000Z | +38.297, +142.373 " + \
                 "| 2.0  MW"
        assert ev_str in str(catalog.events[0])
        # testing ids
        ev = catalog.events[0]
        assert 'smi:orig2' == ev.preferred_origin_id
        assert 'smi:mag2' == ev.preferred_magnitude_id
        assert 'smi:fm2' == ev.preferred_focal_mechanism_id
        # testing objects
        assert ev.preferred_origin() == ev.origins[1]
        assert ev.preferred_magnitude() == ev.magnitudes[1]
        assert ev.preferred_focal_mechanism() == ev.focal_mechanisms[1]

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
        assert len(new_cat) == 1
        event = new_cat[0]
        assert len(event.origins) == 1
        assert len(event.magnitudes) == 1
        assert len(event.focal_mechanisms) == 1
        org = event.origins[0]
        mag = event.magnitudes[0]
        fm = event.focal_mechanisms[0]
        assert org.latitude == lat
        assert org.longitude == lon
        assert org.depth == depth
        assert org.time == org_time
        # Moment tensor.
        mt = fm.moment_tensor.tensor
        assert (fm.moment_tensor.scalar_moment - scalar_moment) / \
            scalar_moment < scalar_moment * 1E-10
        assert mt.m_rr == mrr
        assert mt.m_pp == mpp
        assert mt.m_tt == mtt
        assert mt.m_rt == mtr
        assert mt.m_rp == mpr
        assert mt.m_tp == mtp
        # Mag
        assert round(abs(mag.mag-moment_magnitude), 7) == 0
        assert mag.magnitude_type == "Mw"
        assert mag.evaluation_mode == "automatic"

    def test_read_equivalence(self, testdata):
        """
        See #662.
        Tests if _read_quakeml() and read_events() return the same results.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            cat1 = read_events(testdata['neries_events.xml'])
            cat2 = _read_quakeml(testdata['neries_events.xml'])
        assert cat1 == cat2

    def test_reading_twice_raises_no_warning(self, testdata):
        """
        Tests that reading a QuakeML file twice does not raise a warnings.

        Not an extensive test but likely good enough.
        """
        filename = testdata['qml-example-1.2-RC3.xml']

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cat1 = _read_quakeml(filename)
            assert len(w) == 0
            cat2 = _read_quakeml(filename)
            assert len(w) == 0
        assert_no_extras(cat1)

        assert cat1 == cat2

    def test_read_amplitude_time_window(self, testdata):
        """
        Tests reading an QuakeML Amplitude with TimeWindow.
        """
        filename = testdata['qml-example-1.2-RC3.xml']

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cat = _read_quakeml(filename)
            assert len(w) == 0

        assert len(cat[0].amplitudes) == 1
        amp = cat[0].amplitudes[0]
        assert amp.type == "A"
        assert amp.category == "point"
        assert amp.unit == "m/s"
        assert amp.generic_amplitude == 1e-08
        assert amp.time_window.begin == 0.0
        assert amp.time_window.end == 0.51424
        assert amp.time_window.reference == \
            UTCDateTime("2007-10-10T14:40:39.055")

    def test_write_amplitude_time_window(self, testdata):
        """
        Tests writing an QuakeML Amplitude with TimeWindow.
        """
        filename = testdata['qml-example-1.2-RC3.xml']

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cat = _read_quakeml(filename)
            assert len(w) == 0

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
            assert got == expected

    def test_write_with_extra_tags_and_read(self, testdata):
        """
        Tests that a QuakeML file with additional custom "extra" tags gets
        written correctly and that when reading it again the extra tags are
        parsed correctly.
        """
        filename = testdata['quakeml_1.2_origin.xml']

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cat = _read_quakeml(filename)
            assert_no_extras(cat)
            assert len(w) == 0

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
                assert line in content
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
                assert line in content
            # now, read again to test if it's parsed correctly..
            cat = _read_quakeml(tmpfile)
        # when reading..
        #  - namespace abbreviations should be disregarded
        #  - we always end up with a namespace definition, even if it was
        #    omitted when originally setting the custom tag
        #  - custom namespace abbreviations should attached to Catalog
        assert hasattr(cat[0], 'extra')

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
        assert cat[0].extra == my_extra
        assert hasattr(cat[0].picks[0], 'extra')
        assert cat[0].picks[0].extra == \
            {'weight': {'value': '2',
                        'namespace': 'http://test.org/xmlns/0.1'}}
        assert hasattr(cat, 'nsmap')
        assert getattr(cat, 'nsmap')['ns0'] == nsmap['ns0']

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
        assert len(w) == 0

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
        assert cat[0].focal_mechanisms[0].moment_tensor is None

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
        assert cat[0].focal_mechanisms[0].moment_tensor.tensor is None
        assert \
            cat[0].focal_mechanisms[0].moment_tensor.source_time_function \
            is None
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
        assert cat[0].focal_mechanisms[0].nodal_planes is None
        assert cat[0].focal_mechanisms[0].principal_axes is None

    def test_writing_invalid_quakeml_id(self, testdata):
        """
        Some ids might be invalid. We still want to write them to not mess
        with any external tools relying on the ids. But we also raise a
        warning of course.
        """
        filename = testdata['invalid_id.xml']
        cat = read_events(filename)
        assert cat[0].resource_id.id == \
            "smi:org.gfz-potsdam.de/geofon/RMHP(60)>>ITAPER(3)>>BW(4,5,15)"
        with NamedTemporaryFile() as tf:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                cat.write(tf.name, format="quakeml")
                cat2 = read_events(tf.name)
        assert len(w) == 19
        assert w[0].message.args[0] == \
            ("'smi:org.gfz-potsdam.de/geofon/RMHP(60)>>ITAPER(3)>>BW(4,5,15)' "
             "is not a valid QuakeML URI. It will be in the final file but "
             "note that the file will not be a valid QuakeML file.")
        assert cat2[0].resource_id.id == \
            "smi:org.gfz-potsdam.de/geofon/RMHP(60)>>ITAPER(3)>>BW(4,5,15)"

    def test_reading_invalid_enums(self, testdata):
        """
        Raise a warning when an invalid enum value is attempted to be read.
        """
        filename = testdata['invalid_enum.xml']
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cat = read_events(filename)
        assert len(w) == 1
        assert w[0].message.args[0] == \
            ('Setting attribute "depth_type" failed. Value "randomized" could '
             'not be converted to type "Enum(["from location", "from moment '
             'tensor inversion", ..., "operator assigned", "other"])". The '
             'attribute "depth_type" will not be set and will be missing in '
             'the resulting object.')
        # It should of course not be set.
        assert cat[0].origins[0].depth_type is None

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
        assert text1 == text2
        # the two catalogs should be equal
        assert cat1 == cat2

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

        assert extra == cat2.extra
        assert ('custom1', custom1) in cat2.extra.items()
        assert ('custom2', custom2) in cat2.extra.items()
