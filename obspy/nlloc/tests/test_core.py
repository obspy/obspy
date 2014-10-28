#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA @UnusedWildImport

import os
import re
import inspect
import unittest
from obspy import readEvents, UTCDateTime
from obspy.nlloc.core import is_nlloc_hyp, read_nlloc_hyp, write_nlloc_obs
from obspy.core.util import getExampleFile, NamedTemporaryFile
from obspy.core.util.testing import compare_xml_strings, remove_unique_IDs


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
    Test suite for obspy.nlloc
    """
    def setUp(self):
        self.path = os.path.dirname(os.path.abspath(inspect.getfile(
            inspect.currentframe())))
        self.datapath = os.path.join(self.path, "data")

    def test_write_nlloc_obs(self):
        """
        Test writing nonlinloc observations phase file.
        """
        # load nlloc.qml QuakeML file to generate OBS file from it
        filename = getExampleFile("nlloc.qml")
        cat = readEvents(filename, "QUAKEML")
        # adjust one pick time that got cropped by nonlinloc in NLLOC HYP file
        # due to less precision in hypocenter file (that we used to create the
        # reference QuakeML file)
        for pick in cat[0].picks:
            if pick.waveform_id.station_code == "UH4" and \
               pick.phase_hint == "P":
                pick.time -= 0.005

        # read expected OBS file output
        filename = getExampleFile("nlloc.obs")
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
        filename = getExampleFile("nlloc.hyp")
        cat = read_nlloc_hyp(filename,
                             coordinate_converter=_mock_coordinate_converter)
        with open(getExampleFile("nlloc.qml"), 'rb') as tf:
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

        quakeml_expected = remove_unique_IDs(quakeml_expected,
                                             remove_creation_time=True)
        quakeml_got = remove_unique_IDs(quakeml_got, remove_creation_time=True)
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

    def test_read_nlloc_hyp_via_plugin(self):
        filename = getExampleFile("nlloc.hyp")
        cat = readEvents(filename)
        self.assertEqual(len(cat), 1)
        cat = readEvents(filename, format="NLLOC_HYP")
        self.assertEqual(len(cat), 1)

    def test_is_nlloc_hyp(self):
        # test positive
        filename = getExampleFile("nlloc.hyp")
        self.assertEqual(is_nlloc_hyp(filename), True)
        # test some negatives
        for filenames in ["nlloc.qml", "nlloc.obs", "gaps.mseed",
                          "BW_RJOB.xml", "QFILE-TEST-ASC.ASC", "LMOW.BHE.SAC"]:
            filename = getExampleFile("nlloc.qml")
            self.assertEqual(is_nlloc_hyp(filename), False)

    def test_read_nlloc_with_picks(self):
        """
        Test correct resource ID linking when reading NLLOC_HYP file with
        providing original picks.
        """
        picks = readEvents(getExampleFile("nlloc.qml"))[0].picks
        arrivals = readEvents(getExampleFile("nlloc.hyp"), format="NLLOC_HYP",
                              picks=picks)[0].origins[0].arrivals
        expected = [p.resource_id for p in picks]
        got = [a.pick_id for a in arrivals]
        self.assertEqual(expected, got)


def suite():
    return unittest.makeSuite(NLLOCTestCase, "test")


if __name__ == "__main__":
    unittest.main(defaultTest="suite")
