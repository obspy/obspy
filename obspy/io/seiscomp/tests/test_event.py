#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
seiscomp.event test suite.

:author:
    EOST (École et Observatoire des Sciences de la Terre)
:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import filecmp
import os
import unittest

from obspy.core.util.base import NamedTemporaryFile
from obspy.io.quakeml.core import _read_quakeml


class EventTestCase(unittest.TestCase):
    """
    Test suite for obspy.io.seiscomp.event
    """
    def setUp(self):
        # directory where the test files are located
        io_directory = \
            os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        self.quakeml_path = \
            os.path.join(io_directory, 'quakeml', 'tests', 'data')
        self.path = os.path.join(os.path.dirname(__file__), 'data')

    def cmp_file(self, quakeml_file, sc3ml_file, path=None,
                 validate=True, event_removal=False):
        """
        Check if the generated sc3ml file is the same than the one in the data
        folder.
        """
        if path is None:
            path = self.quakeml_path

        filename = os.path.join(path, quakeml_file)
        catalog = _read_quakeml(filename)

        with NamedTemporaryFile() as tf:
            tmpfile = tf.name
            catalog.write(tmpfile, format='SC3ML', validate=validate,
                          verbose=True, event_removal=event_removal)
            filepath_cmp = os.path.join(self.path, sc3ml_file)
            self.assertTrue(filecmp.cmp(filepath_cmp, tmpfile))

    def test_event(self):
        self.cmp_file('quakeml_1.2_event.xml', 'quakeml_1.2_event.sc3ml')

    def test_origin(self):
        self.cmp_file('quakeml_1.2_origin.xml', 'quakeml_1.2_origin.sc3ml')

    def test_magnitude(self):
        # Missing origin in original QuakeML test case.
        self.cmp_file('quakeml_1.2_magnitude.xml',
                      'quakeml_1.2_magnitude.sc3ml',
                      path=self.path)

    def test_station_magnitude_contribution(self):
        # Missing origin in original QuakeML test case.
        self.cmp_file('quakeml_1.2_stationmagnitudecontributions.xml',
                      'quakeml_1.2_stationmagnitudecontributions.sc3ml',
                      path=self.path)

    def test_station_magnitude(self):
        # Missing origin in original QuakeML test case.
        self.cmp_file('quakeml_1.2_magnitude.xml',
                      'quakeml_1.2_magnitude.sc3ml',
                      path=self.path)

    def test_data_used_in_moment_tensor(self):
        # Can't validate because of missing element derivedOriginID in QuakeML.
        self.cmp_file('quakeml_1.2_data_used.xml',
                      'quakeml_1.2_data_used.sc3ml', validate=False)

    def test_arrival(self):
        self.cmp_file('quakeml_1.2_arrival.xml', 'quakeml_1.2_arrival.sc3ml')

    def test_pick(self):
        self.cmp_file('quakeml_1.2_pick.xml', 'quakeml_1.2_pick.sc3ml')

    def test_focalmechanism(self):
        self.cmp_file('quakeml_1.2_focalmechanism.xml',
                      'quakeml_1.2_focalmechanism.sc3ml')

    def test_iris_events(self):
        self.cmp_file('iris_events.xml', 'iris_events.sc3ml')

    def test_neries_events(self):
        # Some ID are generated automatically. File comparison can't be done.
        filename = os.path.join(self.quakeml_path, 'neries_events.xml')
        catalog = _read_quakeml(filename)

        with NamedTemporaryFile() as tf:
            tmpfile = tf.name
            try:
                catalog.write(tmpfile, format='SC3ML', validate=True,
                              verbose=True)
            except AssertionError as e:
                self.fail(e)

    def test_usgs_events(self):
        self.cmp_file('usgs_event.xml', 'usgs_event.sc3ml')

    def test_example(self):
        self.cmp_file('qml-example-1.2-RC3.xml', 'qml-example-1.2-RC3.sc3ml')

    def test_remove_events(self):
        self.cmp_file('qml-example-1.2-RC3.xml',
                      'qml-example-1.2-RC3_no_events.sc3ml',
                      event_removal=True)


def suite():
    return unittest.makeSuite(EventTestCase, "test")


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
