#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
seiscomp.core test suite.

:author:
    EOST (École et Observatoire des Sciences de la Terre)
:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""
import os
import unittest

from obspy.io.seiscomp.core import _is_sc3ml, validate


class CoreTestCase(unittest.TestCase):
    """
    Test suite for obspy.io.seiscomp.event
    """
    def setUp(self):
        self.data_dir = os.path.join(os.path.dirname(__file__), 'data')

    def test_sc3ml_versions(self):
        """
        Test multiple schema versions
        """
        for version in ['0.10', '0.11', '0.12']:
            filename = os.path.join(self.data_dir, 'version%s' % version)
            self.assertTrue(_is_sc3ml(filename))

    def test_sc3ml_no_version_attribute(self):
        filename = os.path.join(self.data_dir, 'no_version_attribute.sc3ml')
        self.assertTrue(_is_sc3ml(filename))

    def test_validate(self):
        filename = os.path.join(self.data_dir, 'qml-example-1.2-RC3.sc3ml')
        self.assertTrue(validate(filename))
        self.assertFalse(validate(filename, version='0.8'))

        with self.assertRaises(ValueError) as e:
            validate(filename, version='0.99')

        expected_error = ("0.99 is not a supported version. Use one of these "
                          "versions: [0.6, 0.7, 0.8, 0.9, 0.10, 0.11, 0.12].")
        self.assertEqual(e.exception.args[0], expected_error)
