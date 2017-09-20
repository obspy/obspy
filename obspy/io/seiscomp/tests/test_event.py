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

from lxml import etree

from obspy import read_events
from obspy.core.util.base import NamedTemporaryFile
from obspy.io.quakeml.core import _read_quakeml
from obspy.io.quakeml.core import _validate as _validate_quakeml
from obspy.io.seiscomp.core import validate as validate_sc3ml
from obspy.io.seiscomp.event import _read_sc3ml


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
        self.read_xslt_filename = os.path.join(
            io_directory, 'seiscomp', 'data', 'sc3ml_0.9__quakeml_1.2.xsl')
        self.write_xslt_filename = os.path.join(
            io_directory, 'seiscomp', 'data', 'quakeml_1.2__sc3ml_0.9.xsl')

    def cmp_read_xslt_file(self, sc3ml_file, quakeml_file, validate=True):
        """
        Check if the QuakeML file generated with the XSLT file is the
        same than the one in the data folder.
        """
        transform = etree.XSLT(etree.parse(self.read_xslt_filename))
        filename = os.path.join(self.path, sc3ml_file)
        quakeml_doc = transform(etree.parse(filename))

        with NamedTemporaryFile() as tf:
            tf.write(quakeml_doc)
            if validate:
                self.assertTrue(_validate_quakeml(tf.name))
            filepath_cmp = os.path.join(self.path, quakeml_file)
            self.assertTrue(filecmp.cmp(filepath_cmp, tf.name))

    def cmp_write_xslt_file(self, quakeml_file, sc3ml_file, validate=True,
                            path=None):
        """
        Check if the SC3ML file generated with the XSLT file is the
        same than the one in the data folder.
        """
        if path is None:
            path = self.path

        transform = etree.XSLT(etree.parse(self.write_xslt_filename))
        filename = os.path.join(path, quakeml_file)
        sc3ml_doc = transform(etree.parse(filename))

        with NamedTemporaryFile() as tf:
            tf.write(sc3ml_doc)
            if validate:
                self.assertTrue(validate_sc3ml(tf.name))
            filepath_cmp = os.path.join(self.path, sc3ml_file)
            self.assertTrue(filecmp.cmp(filepath_cmp, tf.name))

    def test_sc3ml_versions(self):
        """
        Test multiple schema versions
        """
        for version in ['0.5', '0.6', '0.7', '0.8', '0.9']:
            filename = os.path.join(self.path, 'version%s' % version)
            read_events(filename)

        filename = os.path.join(self.path, 'version0.3')
        with self.assertRaises(ValueError) as e:
            read_events(filename)

        expected_message = ("Can't read SC3ML version 0.3, ObsPy can deal "
                            "with versions [0.5, 0.6, 0.7, 0.8, 0.9].")
        self.assertEqual(e.exception.args[0], expected_message)

        filename = os.path.join(self.path, 'version0.10')
        with self.assertRaises(ValueError) as e:
            read_events(filename)

        expected_message = ("Can't read SC3ML version 0.10, ObsPy can deal "
                            "with versions [0.5, 0.6, 0.7, 0.8, 0.9].")
        self.assertEqual(e.exception.args[0], expected_message)

    def test_read_xslt_event(self):
        self.cmp_read_xslt_file('quakeml_1.2_event.sc3ml',
                                'quakeml_1.2_event_res.xml')

    def test_read_xslt_origin(self):
        self.cmp_read_xslt_file('quakeml_1.2_origin.sc3ml',
                                'quakeml_1.2_origin_res.xml')

    def test_read_xslt_magnitude(self):
        self.cmp_read_xslt_file('quakeml_1.2_magnitude.sc3ml',
                                'quakeml_1.2_magnitude.xml')

    def test_read_xslt_station_magnitude_contribution(self):
        self.cmp_read_xslt_file(
            'quakeml_1.2_stationmagnitudecontributions.sc3ml',
            'quakeml_1.2_stationmagnitudecontributions.xml')

    def test_read_xslt_station_magnitude(self):
        self.cmp_read_xslt_file('quakeml_1.2_stationmagnitude.sc3ml',
                                'quakeml_1.2_stationmagnitude.xml')

    def test_read_xslt_data_used_in_moment_tensor(self):
        self.cmp_read_xslt_file('quakeml_1.2_data_used.sc3ml',
                                'quakeml_1.2_data_used.xml')

    def test_read_xslt_arrival(self):
        self.cmp_read_xslt_file('quakeml_1.2_arrival.sc3ml',
                                'quakeml_1.2_arrival_res.xml')

    def test_read_xslt_pick(self):
        self.cmp_read_xslt_file('quakeml_1.2_pick.sc3ml',
                                'quakeml_1.2_pick.xml')

    def test_read_xslt_focalmechanism(self):
        self.cmp_read_xslt_file('quakeml_1.2_focalmechanism.sc3ml',
                                'quakeml_1.2_focalmechanism_res.xml')

    def test_read_xslt_iris_events(self):
        # Magnitude lost during conversion
        self.cmp_read_xslt_file('iris_events.sc3ml', 'iris_events_res.xml')

    def test_read_xslt_example(self):
        self.cmp_read_xslt_file('qml-example-1.2-RC3.sc3ml',
                                'qml-example-1.2-RC3.xml')

    def test_read_example(self):
        filename = os.path.join(self.path, 'qml-example-1.2-RC3.sc3ml')
        catalog = _read_sc3ml(filename)

        self.assertEqual(len(catalog.events), 1)
        self.assertEqual(len(catalog.events[0].origins), 1)

    def test_read_id_prefix(self):
        filename = \
            os.path.join(self.path, 'qml-example-1.2-RC3_wrong_id.sc3ml')
        catalog = _read_sc3ml(filename, id_prefix='quakeml:obspy.org/')

        self.assertEqual(len(catalog.events), 1)
        event = catalog.events[0]
        self.assertEqual(event.resource_id, 'quakeml:obspy.org/test_event_id')

        self.assertEqual(len(event.origins), 1)
        origin = event.origins[0]
        self.assertEqual(origin.resource_id,
                         'quakeml:obspy.org/test_origin_id')
        self.assertEqual(origin.reference_system_id,
                         'quakeml:obspy.org/test_reference_system_id')

        self.assertEqual(len(event.amplitudes), 1)
        amplitude = event.amplitudes[0]
        self.assertEqual(amplitude.resource_id,
                         'quakeml:obspy.org/test_amplitude_id')

        self.assertEqual(len(event.magnitudes), 1)
        magnitude = event.magnitudes[0]
        self.assertEqual(magnitude.resource_id,
                         'quakeml:obspy.org/test_magnitude_id')

        self.assertEqual(len(event.station_magnitudes), 1)
        station_magnitude = event.station_magnitudes[0]
        self.assertEqual(station_magnitude.resource_id,
                         'quakeml:obspy.org/test_station_magnitude_id')

    def test_read_string(self):
        """
        Test reading a SC3ML string/unicode object via read_events.
        """
        filename = \
            os.path.join(self.path, 'qml-example-1.2-RC3.sc3ml')
        with open(filename, 'rb') as fp:
            data = fp.read()

            catalog = read_events(data)
            self.assertEqual(len(catalog), 1)

    def test_read_quakeml(self):
        """
        Test reading a QuakeML file via read_events.
        """
        filename = os.path.join(self.path, 'qml-example-1.2-RC3.xml')
        with self.assertRaises(ValueError) as e:
            read_events(filename, format='SC3ML')

        expected_message = "Not a SC3ML compatible file or string."
        self.assertEqual(e.exception.args[0], expected_message)

    def test_write_xslt_event(self):
        self.cmp_write_xslt_file('quakeml_1.2_event.xml',
                                 'quakeml_1.2_event.sc3ml',
                                 path=self.quakeml_path)

    def test_write_xslt_origin(self):
        self.cmp_write_xslt_file('quakeml_1.2_origin.xml',
                                 'quakeml_1.2_origin.sc3ml',
                                 path=self.quakeml_path)

    def test_write_xslt_magnitude(self):
        # Missing origin in original QuakeML test case.
        self.cmp_write_xslt_file('quakeml_1.2_magnitude.xml',
                                 'quakeml_1.2_magnitude.sc3ml')

    def test_write_xslt_station_magnitude_contribution(self):
        # Missing origin in original QuakeML test case.
        self.cmp_write_xslt_file(
            'quakeml_1.2_stationmagnitudecontributions.xml',
            'quakeml_1.2_stationmagnitudecontributions.sc3ml')

    def test_write_xslt_station_magnitude(self):
        # Missing origin in original QuakeML test case.
        self.cmp_write_xslt_file('quakeml_1.2_stationmagnitude.xml',
                                 'quakeml_1.2_stationmagnitude.sc3ml')

    def test_write_xslt_data_used_in_moment_tensor(self):
        self.cmp_write_xslt_file('quakeml_1.2_data_used.xml',
                                 'quakeml_1.2_data_used.sc3ml')

    def test_write_xslt_arrival(self):
        self.cmp_write_xslt_file('quakeml_1.2_arrival.xml',
                                 'quakeml_1.2_arrival.sc3ml',
                                 path=self.quakeml_path)

    def test_write_xslt_pick(self):
        self.cmp_write_xslt_file('quakeml_1.2_pick.xml',
                                 'quakeml_1.2_pick.sc3ml')

    def test_write_xslt_focalmechanism(self):
        self.cmp_write_xslt_file('quakeml_1.2_focalmechanism.xml',
                                 'quakeml_1.2_focalmechanism.sc3ml',
                                 path=self.quakeml_path)

    def test_write_xslt_iris_events(self):
        self.cmp_write_xslt_file('iris_events.xml',
                                 'iris_events.sc3ml',
                                 path=self.quakeml_path)

    def test_write_xslt_neries_events(self):
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

    def test_write_xslt_usgs_events(self):
        # Can't validate due to wrong event types (QuakeML was already
        # unvalid)
        self.cmp_write_xslt_file('usgs_event.xml',
                                 'usgs_event.sc3ml',
                                 path=self.quakeml_path,
                                 validate=False)

    def test_write_xslt_example(self):
        self.cmp_write_xslt_file('qml-example-1.2-RC3.xml',
                                 'qml-example-1.2-RC3.sc3ml')

    def test_write_example(self):
        filename = os.path.join(self.path, 'qml-example-1.2-RC3.xml')
        catalog = _read_quakeml(filename)

        with NamedTemporaryFile() as tf:
            catalog.write(tf, format='SC3ML', validate=True)
            filepath_cmp = \
                os.path.join(self.path, 'qml-example-1.2-RC3_write.sc3ml')
            self.assertTrue(filecmp.cmp(filepath_cmp, tf.name))

    def test_write_remove_events(self):
        filename = os.path.join(self.path, 'qml-example-1.2-RC3.xml')
        catalog = _read_quakeml(filename)

        with NamedTemporaryFile() as tf:
            catalog.write(tf, format='SC3ML', validate=True,
                          event_removal=True)
            filepath_cmp = \
                os.path.join(self.path, 'qml-example-1.2-RC3_no_events.sc3ml')
            self.assertTrue(filecmp.cmp(filepath_cmp, tf.name))

    def test_read_and_write(self):
        filename = os.path.join(self.path, 'qml-example-1.2-RC3_write.sc3ml')
        catalog = read_events(filename)

        with NamedTemporaryFile() as tf:
            catalog.write(tf, format='SC3ML', validate=True)
            self.assertTrue(filecmp.cmp(filename, tf.name))


def suite():
    return unittest.makeSuite(EventTestCase, "test")


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
