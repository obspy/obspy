#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA @UnusedWildImport

import inspect
import io
import os
import re
import unittest

from obspy import UTCDateTime, read_events
from obspy.core.util import NamedTemporaryFile, get_example_file
from obspy.core.util.testing import compare_xml_strings, remove_unique_ids
from obspy.io.nlloc.core import is_nlloc_hyp, read_nlloc_hyp, write_nlloc_obs


def _mock_coordinate_converter(x, y, z):
    """
    Mocks the following pyproj based converter function for the values
    encountered in the test. Mocks the following function::

        import pyproj
        proj_wgs84 = pyproj.Proj(init="epsg:4326")
        proj_gk4 = pyproj.Proj(init="epsg:31468")
        def my_conversion(x, y, z):
            x, y = pyproj.transform(proj_gk4, proj_wgs84, x * 1e3, y * 1e3)
            return x, y, z
    """
    if (x, y, z) == (4473.68, 5323.28, 4.57949):
        return (11.6455375456446, 48.04707051747388, 4.57949)
    else:
        raise Exception("Unexpected values during test run.")


class NLLOCTestCase(unittest.TestCase):
    """
    Test suite for obspy.io.nlloc
    """
    def setUp(self):
        self.path = os.path.dirname(os.path.abspath(inspect.getfile(
            inspect.currentframe())))
        self.datapath = os.path.join(self.path, "data")

    def test_write_nlloc_obs(self):
        """
        Test writing nonlinloc observations phase file.
        """
        # load nlloc_custom.qml QuakeML file to generate OBS file from it
        filename = get_example_file("nlloc_custom.qml")
        cat = read_events(filename, "QUAKEML")
        # adjust one pick time that got cropped by nonlinloc in NLLOC HYP file
        # due to less precision in hypocenter file (that we used to create the
        # reference QuakeML file)
        for pick in cat[0].picks:
            if pick.waveform_id.station_code == "UH4" and \
               pick.phase_hint == "P":
                pick.time -= 0.005

        # read expected OBS file output
        filename = get_example_file("nlloc.obs")
        with open(filename, "rb") as fh:
            expected = fh.read().decode()

        # write via plugin
        with NamedTemporaryFile() as tf:
            cat.write(tf, format="NLLOC_OBS")
            tf.seek(0)
            got = tf.read().decode()

        self.assertEqual(expected, got)

        # write manually
        with NamedTemporaryFile() as tf:
            write_nlloc_obs(cat, tf)
            tf.seek(0)
            got = tf.read().decode()

        self.assertEqual(expected, got)

    def test_read_nlloc_hyp(self):
        """
        Test reading nonlinloc hypocenter phase file.
        """
        filename = get_example_file("nlloc_custom.hyp")
        cat = read_nlloc_hyp(filename,
                             coordinate_converter=_mock_coordinate_converter)
        with open(get_example_file("nlloc_custom.qml"), 'rb') as tf:
            quakeml_expected = tf.read().decode()
        with NamedTemporaryFile() as tf:
            cat.write(tf, format="QUAKEML")
            tf.seek(0)
            quakeml_got = tf.read().decode()

        # test creation times manually as they get omitted in the overall test
        creation_time = UTCDateTime("2014-10-17T16:30:08.000000Z")
        self.assertEqual(cat[0].creation_info.creation_time, creation_time)
        self.assertEqual(cat[0].origins[0].creation_info.creation_time,
                         creation_time)

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

        compare_xml_strings(quakeml_expected, quakeml_got)

    def test_read_nlloc_hyp_with_builtin_projection(self):
        """
        Test reading nonlinloc hyp file without a coordinate_converter.
        """
        cat = read_nlloc_hyp(get_example_file("nlloc.hyp"))
        cat_expected = read_events(get_example_file("nlloc.qml"))

        # test event
        ev = cat[0]
        ev_expected = cat_expected[0]
        self.assertAlmostEqual(ev.creation_info.creation_time,
                               ev_expected.creation_info.creation_time)

        # test origin
        orig = ev.origins[0]
        orig_expected = ev_expected.origins[0]
        self.assertAlmostEqual(orig.time, orig_expected.time)
        self.assertAlmostEqual(orig.longitude, orig_expected.longitude)
        self.assertAlmostEqual(orig.longitude_errors.uncertainty,
                               orig_expected.longitude_errors.uncertainty)
        self.assertAlmostEqual(orig.latitude, orig_expected.latitude)
        self.assertAlmostEqual(orig.latitude_errors.uncertainty,
                               orig_expected.latitude_errors.uncertainty)
        self.assertAlmostEqual(orig.depth, orig_expected.depth)
        self.assertAlmostEqual(orig.depth_errors.uncertainty,
                               orig_expected.depth_errors.uncertainty)
        self.assertAlmostEqual(orig.depth_errors.confidence_level,
                               orig_expected.depth_errors.confidence_level)
        self.assertEqual(orig.depth_type, orig_expected.depth_type)
        self.assertEqual(orig.quality.associated_phase_count,
                         orig_expected.quality.associated_phase_count)
        self.assertEqual(orig.quality.used_phase_count,
                         orig_expected.quality.used_phase_count)
        self.assertEqual(orig.quality.associated_station_count,
                         orig_expected.quality.associated_station_count)
        self.assertEqual(orig.quality.used_station_count,
                         orig_expected.quality.used_station_count)
        self.assertAlmostEqual(orig.quality.standard_error,
                               orig_expected.quality.standard_error)
        self.assertAlmostEqual(orig.quality.azimuthal_gap,
                               orig_expected.quality.azimuthal_gap)
        self.assertAlmostEqual(orig.quality.secondary_azimuthal_gap,
                               orig_expected.quality.secondary_azimuthal_gap)
        self.assertEqual(orig.quality.ground_truth_level,
                         orig_expected.quality.ground_truth_level)
        self.assertAlmostEqual(orig.quality.minimum_distance,
                               orig_expected.quality.minimum_distance)
        self.assertAlmostEqual(orig.quality.maximum_distance,
                               orig_expected.quality.maximum_distance)
        self.assertAlmostEqual(orig.quality.median_distance,
                               orig_expected.quality.median_distance)
        self.assertAlmostEqual(
            orig.origin_uncertainty.min_horizontal_uncertainty,
            orig_expected.origin_uncertainty.min_horizontal_uncertainty)
        self.assertAlmostEqual(
            orig.origin_uncertainty.max_horizontal_uncertainty,
            orig_expected.origin_uncertainty.max_horizontal_uncertainty)
        self.assertAlmostEqual(
            orig.origin_uncertainty.azimuth_max_horizontal_uncertainty,
            orig_expected.origin_uncertainty.
            azimuth_max_horizontal_uncertainty)
        self.assertEqual(
            orig.origin_uncertainty.preferred_description,
            orig_expected.origin_uncertainty.preferred_description)
        self.assertAlmostEqual(
            orig.origin_uncertainty.confidence_level,
            orig_expected.origin_uncertainty.confidence_level)
        self.assertEqual(orig.creation_info.creation_time,
                         orig_expected.creation_info.creation_time)
        self.assertEqual(orig.comments[0].text, orig_expected.comments[0].text)

        # test a couple of arrivals
        for n in range(2):
            arriv = orig.arrivals[n]
            arriv_expected = orig_expected.arrivals[n]
            self.assertEqual(arriv.phase, arriv_expected.phase)
            self.assertAlmostEqual(arriv.azimuth, arriv_expected.azimuth)
            self.assertAlmostEqual(arriv.distance, arriv_expected.distance)
            self.assertAlmostEqual(arriv.takeoff_angle,
                                   arriv_expected.takeoff_angle)
            self.assertAlmostEqual(arriv.time_residual,
                                   arriv_expected.time_residual)
            self.assertAlmostEqual(arriv.time_weight,
                                   arriv_expected.time_weight)

        # test a couple of picks
        for n in range(2):
            pick = ev.picks[n]
            pick_expected = ev_expected.picks[n]
            self.assertAlmostEqual(pick.time, pick_expected.time)
            self.assertEqual(pick.waveform_id.station_code,
                             pick_expected.waveform_id.station_code)
            self.assertEqual(pick.onset, pick_expected.onset)
            self.assertEqual(pick.phase_hint, pick_expected.phase_hint)
            self.assertEqual(pick.polarity, pick_expected.polarity)

    def test_read_nlloc_hyp_via_plugin(self):
        filename = get_example_file("nlloc_custom.hyp")
        cat = read_events(filename)
        self.assertEqual(len(cat), 1)
        cat = read_events(filename, format="NLLOC_HYP")
        self.assertEqual(len(cat), 1)

    def test_is_nlloc_hyp(self):
        # test positive
        filename = get_example_file("nlloc_custom.hyp")
        self.assertEqual(is_nlloc_hyp(filename), True)
        # test some negatives
        for filenames in ["nlloc_custom.qml", "nlloc.obs", "gaps.mseed",
                          "BW_RJOB.xml", "QFILE-TEST-ASC.ASC", "LMOW.BHE.SAC"]:
            filename = get_example_file("nlloc_custom.qml")
            self.assertEqual(is_nlloc_hyp(filename), False)

    def test_read_nlloc_with_picks(self):
        """
        Test correct resource ID linking when reading NLLOC_HYP file with
        providing original picks.
        """
        picks = read_events(get_example_file("nlloc_custom.qml"))[0].picks
        arrivals = read_events(
            get_example_file("nlloc_custom.hyp"), format="NLLOC_HYP",
            picks=picks)[0].origins[0].arrivals
        expected = [p.resource_id for p in picks]
        got = [a.pick_id for a in arrivals]
        self.assertEqual(expected, got)

    def test_read_nlloc_with_multiple_events(self):
        """
        Test reading a NLLOC_HYP file with multiple hypocenters in it.
        """
        got = read_events(get_example_file("vanua.sum.grid0.loc.hyp"),
                          format="NLLOC_HYP")
        self.assertEqual(len(got), 3)
        self.assertEqual(got[0].origins[0].longitude, 167.049)
        self.assertEqual(got[1].origins[0].longitude, 166.905)
        self.assertEqual(got[2].origins[0].longitude, 166.858)
        self.assertEqual(got[0].origins[0].latitude, -14.4937)
        self.assertEqual(got[1].origins[0].latitude, -15.0823)
        self.assertEqual(got[2].origins[0].latitude, -15.1529)
        for item in got.events + [e.origins[0] for e in got.events]:
            self.assertEqual(item.creation_info.author, u'Océane Foix')
        for event in got.events:
            self.assertEqual(event.comments[0].text,
                             "Central Vanuatu (3D tomo 2016)")

    def test_read_nlloc_6_beta_signature(self):
        """
        SIGNATURE field of nlloc hypocenter output file was somehow changed at
        some point after version 6.0 (it appears in 6.0.3 beta release for
        example).
        Date is now seemingly always prepended with 'run:' without a space
        afterwards.
        """
        filename = os.path.join(self.datapath, 'nlloc_post_version_6.hyp')
        cat = read_nlloc_hyp(filename)
        # check that signature time-of-run part is correctly read
        # (actually before the fix the above reading already fails..)
        self.assertEqual(
            cat[0].creation_info.creation_time,
            UTCDateTime(2017, 5, 9, 11, 0, 22))

    def test_issue_2222(self):
        """
        Test that hour values of 24 don't break parser.
        """

        # modify the example file to contain an hour 24 and second 60
        nll_str = open(get_example_file('nlloc.hyp')).read().splitlines()
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
        self.assertEqual(len(cat), 1)
        pick1, pick2 = cat[0].picks[-1], cat[0].picks[-2]
        self.assertEqual(pick1.time.hour, 0)
        self.assertEqual(pick2.time.second, 0)


def suite():
    return unittest.makeSuite(NLLOCTestCase, "test")


if __name__ == "__main__":
    unittest.main(defaultTest="suite")
