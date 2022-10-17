#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test suite for the CSS station writer.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""
import fnmatch
import inspect
import os
import shutil
import tempfile
import unittest

import obspy


class CSSStationTestCase(unittest.TestCase):
    """
    Test cases for css station interface
    """
    def setUp(self):
        # Most generic way to get the actual data directory.
        self.data_dir = os.path.join(os.path.dirname(os.path.abspath(
            inspect.getfile(inspect.currentframe()))), 'data', 'station')

    def _run_test(self, inv, fname):
        tempdir = tempfile.mkdtemp(prefix='obspy-')

        try:
            inv.write(os.path.join(tempdir, fname), format='CSS')

            expected_files = sorted(name for name in os.listdir(self.data_dir)
                                    if fnmatch.fnmatch(name, fname + '.*'))
            actual_files = sorted(os.listdir(tempdir))
            self.assertEqual(expected_files, actual_files)

            for expected, actual in zip(expected_files, actual_files):
                with open(os.path.join(self.data_dir, expected), 'rt') as f:
                    expected_text = f.readlines()
                with open(os.path.join(tempdir, actual), 'rt') as f:
                    actual_text = f.readlines()

                self.assertEqual(expected_text, actual_text)

        finally:
            shutil.rmtree(tempdir)

    def test_default_write(self):
        """
        Test that writing of a CSS station database with all possible
        relations works.
        """

        fname = 'default'

        inv = obspy.core.inventory.read_inventory()
        inv[0].comments = [
            obspy.core.inventory.Comment('Comment 1'),
            obspy.core.inventory.Comment('Comment 2'),
        ]
        inv[1].comments = [
            obspy.core.inventory.Comment('Comment\n3'),
        ]

        self._run_test(inv, fname)
