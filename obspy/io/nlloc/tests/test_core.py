#!/usr/bin/env python
# -*- coding: utf-8 -*-
import io
import re
import warnings

from obspy import UTCDateTime, read_events
from obspy.core.inventory import Inventory, Network, Station, Channel
from obspy.core.util import NamedTemporaryFile
from obspy.core.util.testing import compare_xml_strings, remove_unique_ids
from obspy.io.nlloc.core import is_nlloc_hyp, read_nlloc_hyp, write_nlloc_obs


def _mock_coordinate_converter(x, y, z):
    """
    Mocks the following pyproj based converter function for the values
    encountered in the test. Mocks the following function::

        import pyproj
        proj_wgs84 = pyproj.Proj("epsg:4326")
        proj_gk4 = pyproj.Proj("epsg:31468")
        def my_conversion(x, y, z):
            x, y = pyproj.transform(proj_gk4, proj_wgs84, x * 1e3, y * 1e3)
            return x, y, z
    """
    if (x, y, z) == (4473.68, 5323.28, 4.57949):
        return (11.6455375456446, 48.04707051747388, 4.57949)
    else:
        raise Exception("Unexpected values during test run.")


class TestNLLOC():
    """
    Test suite for obspy.io.nlloc
    """
    def test_write_nlloc_obs(self, testdata):
        """
        Test writing nonlinloc observations phase file.
        """
        # load nlloc_custom.qml QuakeML file to generate OBS file from it
        filename = testdata["nlloc_custom.qml"]
        cat = read_events(filename, "QUAKEML")
        # adjust one pick time that got cropped by nonlinloc in NLLOC HYP file
        # due to less precision in hypocenter file (that we used to create the
        # reference QuakeML file)
        for pick in cat[0].picks:
            if pick.waveform_id.station_code == "UH4" and \
               pick.phase_hint == "P":
                pick.time -= 0.005

        # read expected OBS file output
        filename = testdata["nlloc.obs"]
        with open(filename, "rb") as fh:
            expected = fh.read().decode()

        # write via plugin
        with NamedTemporaryFile() as tf:
            cat.write(tf, format="NLLOC_OBS")
            tf.seek(0)
            got = tf.read().decode()

        assert expected == got

        # write manually
        with NamedTemporaryFile() as tf:
            write_nlloc_obs(cat, tf)
            tf.seek(0)
            got = tf.read().decode()

        assert expected == got

    def test_read_nlloc_hyp(self, testdata):
        """
        Test reading nonlinloc hypocenter phase file.
        """
        filename = testdata["nlloc_custom.hyp"]
        cat = read_nlloc_hyp(filename,
                             coordinate_converter=_mock_coordinate_converter)
        # reset pick channel codes, these got automatically mapped upon reading
        for pick in cat[0].picks:
            pick.waveform_id.channel_code = None
        with open(testdata["nlloc_custom.qml"], 'rb') as tf:
            quakeml_expected = tf.read().decode()
        with NamedTemporaryFile() as tf:
            cat.write(tf, format="QUAKEML")
            tf.seek(0)
            quakeml_got = tf.read().decode()

        # test creation times manually as they get omitted in the overall test
        creation_time = UTCDateTime("2014-10-17T16:30:08.000000Z")
        assert cat[0].creation_info.creation_time == creation_time
        assert cat[0].origins[0].creation_info.creation_time == creation_time

        quakeml_expected = remove_unique_ids(quakeml_expected,
                                             remove_creation_time=True)
        quakeml_got = remove_unique_ids(quakeml_got, remove_creation_time=True)
        # In python 3 float.__str__ outputs 5 decimals of precision more.
        # We use it in writing QuakeML, so files look different on Py2/3.
        # We use regex to cut off floats in the xml such that we only compare
        # 7 digits.
        pattern = r'(<.*?>[0-9]*?\.[0-9]{7})[0-9]*?(</.*?>)'
        quakeml_expected = re.sub(pattern, r'\1\2', quakeml_expected)
        quakeml_got = re.sub(pattern, r'\1\2', quakeml_got)

        # remove (changing) obspy version number from output
        re_pattern = '<version>ObsPy .*?</version>'
        quakeml_expected = re.sub(re_pattern, '', quakeml_expected, 1)
        quakeml_got = re.sub(re_pattern, '', quakeml_got, 1)

        compare_xml_strings(quakeml_expected.encode(), quakeml_got.encode())

    def test_read_nlloc_hyp_with_builtin_projection(self, testdata):
        """
        Test reading nonlinloc hyp file without a coordinate_converter.
        """
        cat = read_nlloc_hyp(testdata["nlloc.hyp"])
        cat_expected = read_events(testdata["nlloc.qml"])

        # test event
        ev = cat[0]
        ev_expected = cat_expected[0]
        assert round(abs(
            ev.creation_info.creation_time -
            ev_expected.creation_info.creation_time), 7) == 0

        # test origin
        orig = ev.origins[0]
        orig_expected = ev_expected.origins[0]
        assert round(abs(orig.time-orig_expected.time), 7) == 0
        assert round(abs(orig.longitude-orig_expected.longitude), 7) == 0
        assert round(abs(
            orig.longitude_errors.uncertainty -
            orig_expected.longitude_errors.uncertainty), 7) == 0
        assert round(abs(orig.latitude-orig_expected.latitude), 7) == 0
        assert round(abs(
            orig.latitude_errors.uncertainty -
            orig_expected.latitude_errors.uncertainty), 7) == 0
        assert round(abs(orig.depth-orig_expected.depth), 7) == 0
        assert round(abs(
            orig.depth_errors.uncertainty -
            orig_expected.depth_errors.uncertainty), 7) == 0
        assert round(abs(
            orig.depth_errors.confidence_level -
            orig_expected.depth_errors.confidence_level), 7) == 0
        assert orig.depth_type == orig_expected.depth_type
        assert orig.quality.associated_phase_count == \
            orig_expected.quality.associated_phase_count
        assert orig.quality.used_phase_count == \
            orig_expected.quality.used_phase_count
        assert orig.quality.associated_station_count == \
            orig_expected.quality.associated_station_count
        assert orig.quality.used_station_count == \
            orig_expected.quality.used_station_count
        assert round(abs(
            orig.quality.standard_error -
            orig_expected.quality.standard_error), 7) == 0
        assert round(abs(
            orig.quality.azimuthal_gap -
            orig_expected.quality.azimuthal_gap), 7) == 0
        assert round(abs(
            orig.quality.secondary_azimuthal_gap -
            orig_expected.quality.secondary_azimuthal_gap), 7) == 0
        assert orig.quality.ground_truth_level == \
            orig_expected.quality.ground_truth_level
        assert round(abs(
            orig.quality.minimum_distance -
            orig_expected.quality.minimum_distance), 7) == 0
        assert round(abs(
            orig.quality.maximum_distance -
            orig_expected.quality.maximum_distance), 7) == 0
        assert round(abs(
            orig.quality.median_distance -
            orig_expected.quality.median_distance), 7) == 0
        assert round(abs(
            orig.origin_uncertainty.min_horizontal_uncertainty -
            orig_expected.origin_uncertainty.min_horizontal_uncertainty), 7) \
            == 0
        assert round(abs(
            orig.origin_uncertainty.max_horizontal_uncertainty -
            orig_expected.origin_uncertainty.max_horizontal_uncertainty), 7) \
            == 0
        assert round(abs(
            orig.origin_uncertainty.azimuth_max_horizontal_uncertainty -
            getattr(orig_expected.origin_uncertainty,
                    "azimuth_max_horizontal_uncertainty")), 7) == 0
        assert orig.origin_uncertainty.preferred_description == \
            orig_expected.origin_uncertainty.preferred_description
        assert round(abs(
            orig.origin_uncertainty.confidence_level -
            orig_expected.origin_uncertainty.confidence_level), 7) == 0
        assert orig.creation_info.creation_time == \
            orig_expected.creation_info.creation_time
        assert orig.comments[0].text == orig_expected.comments[0].text

        # test a couple of arrivals
        for n in range(2):
            arriv = orig.arrivals[n]
            arriv_expected = orig_expected.arrivals[n]
            assert arriv.phase == arriv_expected.phase
            assert round(abs(arriv.azimuth-arriv_expected.azimuth), 7) == 0
            assert round(abs(arriv.distance-arriv_expected.distance), 7) == 0
            assert arriv.takeoff_angle is None
            assert arriv_expected.takeoff_angle is None
            assert round(
                abs(arriv.time_residual-arriv_expected.time_residual), 7) == 0
            assert round(
                abs(arriv.time_weight-arriv_expected.time_weight), 7) == 0

        # test a couple of picks
        for n in range(2):
            pick = ev.picks[n]
            pick_expected = ev_expected.picks[n]
            assert round(abs(pick.time-pick_expected.time), 7) == 0
            assert pick.waveform_id.station_code == \
                pick_expected.waveform_id.station_code
            assert pick.onset == pick_expected.onset
            assert pick.phase_hint == pick_expected.phase_hint
            assert pick.polarity == pick_expected.polarity

    def test_read_nlloc_hyp_via_plugin(self, testdata):
        filename = testdata["nlloc_custom.hyp"]
        cat = read_events(filename)
        assert len(cat) == 1
        cat = read_events(filename, format="NLLOC_HYP")
        assert len(cat) == 1

    def test_read_nlloc_with_pick_seed_id_lookup(self, testdata):
        # create some bogus metadata for lookup
        cha = Channel('HHZ', '00', 0, 0, 0, 0)
        sta = Station('HM02', 0, 0, 0, channels=[cha])
        cha = Channel('HHZ', '10', 0, 0, 0, 0)
        sta2 = Station('YYYY', 0, 0, 0, channels=[cha])
        net = Network('XX', stations=[sta, sta2])
        # second network with non matching data
        cha = Channel('HHZ', '00', 0, 0, 0, 0)
        sta = Station('ABCD', 0, 0, 0, channels=[cha])
        cha = Channel('HHZ', '10', 0, 0, 0, 0)
        sta2 = Station('EFGH', 0, 0, 0, channels=[cha])
        net2 = Network('YY', stations=[sta, sta2])

        inv = Inventory(networks=[net, net2], source='')

        filename = testdata["nlloc_custom.hyp"]
        # we get some warnings since we only provide sufficient metadata for
        # one pick
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cat = read_events(filename, format="NLLOC_HYP", inventory=inv)
        assert len(cat) == 1
        for pick in cat[0].picks:
            wid = pick.waveform_id
            if wid.station_code == 'HM02':
                assert wid.network_code == 'XX'
                assert wid.location_code == ''
            else:
                assert wid.network_code == ''
                assert wid.location_code is None

    def test_is_nlloc_hyp(self, testdata):
        # test positive
        filename = testdata["nlloc_custom.hyp"]
        assert is_nlloc_hyp(filename)
        # test some negatives
        for filenames in ["nlloc_custom.qml", "nlloc.obs", "gaps.mseed",
                          "BW_RJOB.xml", "QFILE-TEST-ASC.ASC", "LMOW.BHE.SAC"]:
            filename = testdata["nlloc_custom.qml"]
            assert not is_nlloc_hyp(filename)

    def test_read_nlloc_with_picks(self, testdata):
        """
        Test correct resource ID linking when reading NLLOC_HYP file with
        providing original picks.
        """
        picks = read_events(testdata["nlloc_custom.qml"])[0].picks
        arrivals = read_events(
            testdata["nlloc_custom.hyp"], format="NLLOC_HYP",
            picks=picks)[0].origins[0].arrivals
        expected = [p.resource_id for p in picks]
        got = [a.pick_id for a in arrivals]
        assert expected == got

    def test_read_nlloc_with_multiple_events(self, testdata):
        """
        Test reading a NLLOC_HYP file with multiple hypocenters in it.
        """
        got = read_events(testdata["vanua.sum.grid0.loc.hyp"],
                          format="NLLOC_HYP")
        assert len(got) == 3
        assert got[0].origins[0].longitude == 167.049
        assert got[1].origins[0].longitude == 166.905
        assert got[2].origins[0].longitude == 166.858
        assert got[0].origins[0].latitude == -14.4937
        assert got[1].origins[0].latitude == -15.0823
        assert got[2].origins[0].latitude == -15.1529
        for item in got.events + [e.origins[0] for e in got.events]:
            assert item.creation_info.author == u'Océane Foix'
        for event in got.events:
            assert event.comments[0].text == \
                             "Central Vanuatu (3D tomo 2016)"

    def test_read_nlloc_6_beta_signature(self, testdata):
        """
        SIGNATURE field of nlloc hypocenter output file was somehow changed at
        some point after version 6.0 (it appears in 6.0.3 beta release for
        example).
        Date is now seemingly always prepended with 'run:' without a space
        afterwards.
        """
        filename = testdata['nlloc_post_version_6.hyp']
        cat = read_nlloc_hyp(filename)
        # check that signature time-of-run part is correctly read
        # (actually before the fix the above reading already fails..)
        assert cat[0].creation_info.creation_time == \
            UTCDateTime(2017, 5, 9, 11, 0, 22)

    def test_issue_2222(self, testdata):
        """
        Test that hour values of 24 don't break parser.
        """

        # modify the example file to contain an hour 24 and second 60
        with open(testdata['nlloc.hyp']) as f:
            nll_str = f.read().splitlines()
        # first add a line with hour 24
        str_list = list(nll_str[-3])
        str_list[37:41] = '2400'
        nll_str[-3] = ''.join(str_list)
        # then add a line with second 60
        str_list = list(nll_str[-4])
        str_list[46:48] = '60'
        nll_str[-4] = ''.join(str_list)
        # write to string io and read into catalog object
        str_io = io.StringIO()
        str_io.write('\n'.join(nll_str))
        str_io.seek(0)
        cat = read_nlloc_hyp(str_io)
        # check catalog is populated and pick times are right
        assert len(cat) == 1
        pick1, pick2 = cat[0].picks[-1], cat[0].picks[-2]
        assert pick1.time.hour == 0
        assert pick2.time.second == 0

    def test_reading_nlloc_v7_hyp_file(self, testdata):
        """
        Tests that we are getting the positioning of items in phase lines
        right. Values for arrivals are shifted by one index to the right in hyp
        files written by newer nonlinloc versions, see #3223
        """
        path = testdata['nlloc_v7.hyp']
        cat = read_nlloc_hyp(path)
        assert cat[0].origins[0].arrivals[0].azimuth == 107.42
