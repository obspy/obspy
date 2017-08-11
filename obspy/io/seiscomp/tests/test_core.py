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
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import os
import unittest
import warnings

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
        filename_07 = os.path.join(self.data_dir, 'version0.7')
        filename_08 = os.path.join(self.data_dir, 'version0.8')
        filename_09 = os.path.join(self.data_dir, 'version0.9')

        self.assertTrue(_is_sc3ml(filename_07, ['0.7', '0.8']))
        self.assertTrue(_is_sc3ml(filename_08, ['0.7', '0.8']))

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            self.assertTrue(_is_sc3ml(filename_09, ['0.7', '0.8']))

            self.assertEqual(len(w), 1)
            expected_message = ("The sc3ml file has version 0.9, ObsPy can "
                                "deal with versions [0.7, 0.8]. Proceed "
                                "with caution.")
            self.assertEqual(str(w[0].message), expected_message)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            self.assertTrue(_is_sc3ml(filename_09, []))

            self.assertEqual(len(w), 1)
            expected_message = ("The sc3ml file has version 0.9, ObsPy can "
                                "deal with versions []. Proceed with caution.")
            self.assertEqual(str(w[0].message), expected_message)

    def test_sc3ml_no_version_attribute(self):
        filename = os.path.join(self.data_dir, 'no_version_attribute.sc3ml')
        self.assertTrue(_is_sc3ml(filename, ['0.9']))

    def test_validate(self):
        filename = os.path.join(self.data_dir, 'qml-example-1.2-RC3.sc3ml')
        self.assertTrue(validate(filename))
        self.assertFalse(validate(filename, version='0.8'))

        with self.assertRaises(ValueError) as e:
            validate(filename, version='0.10')
            expected_error = ("0.10 is not a supported version. Use one of "
                              "these versions: ['0.7', '0.8', '0.9'].")
            self.assertEqual(e.exception.args[0], expected_error)


def suite():
    return unittest.makeSuite(CoreTestCase, "test")


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
