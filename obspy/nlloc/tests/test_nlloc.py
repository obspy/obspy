#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA @UnusedWildImport

import os
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
        with open(filename) as fh:
            expected = fh.read()

        # write via plugin
        with NamedTemporaryFile() as tf:
            cat.write(tf, format="NLLOC_OBS")
            tf.seek(0)
            got = tf.read()

        self.assertEqual(expected, got)

        # write manually
        with NamedTemporaryFile() as tf:
            write_nlloc_obs(cat, tf)
            tf.seek(0)
            got = tf.read()

        self.assertEqual(expected, got)

    def test_read_nlloc_hyp(self):
        """
        Test reading nonlinloc hypocenter phase file.
        """
        filename = getExampleFile("nlloc.hyp")
        cat = read_nlloc_hyp(filename,
                             coordinate_converter=_mock_coordinate_converter)
        with open(getExampleFile("nlloc.qml")) as tf:
            quakeml_expected = tf.read()
        with NamedTemporaryFile() as tf:
            cat.write(tf, format="QUAKEML")
            tf.seek(0)
            quakeml_got = tf.read()

        # test creation times manually as they get omitted in the overall test
        creation_time = UTCDateTime("2014-10-17T16:30:08.000000Z")
        self.assertEqual(cat[0].creation_info.creation_time, creation_time)
        self.assertEqual(cat[0].origins[0].creation_info.creation_time,
                         creation_time)

        quakeml_expected = remove_unique_IDs(quakeml_expected,
                                             remove_creation_time=True)
        quakeml_got = remove_unique_IDs(quakeml_got, remove_creation_time=True)
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


def suite():
    return unittest.makeSuite(NLLOCTestCase, "test")


if __name__ == "__main__":
    unittest.main(defaultTest="suite")
