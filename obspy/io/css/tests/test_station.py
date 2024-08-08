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
import os
import pytest
import shutil
import tempfile

import obspy


class TestCSSStation():
    """
    Test cases for css station interface
    """
    @pytest.fixture(autouse=True, scope="function")
    def setup(self, datapath):
        self.data_dir = datapath / 'station'

    def _run_test(self, inv, fname):
        tempdir = tempfile.mkdtemp(prefix='obspy-')

        try:
            inv.write(os.path.join(tempdir, fname), format='CSS')

            expected_files = sorted(
                path.name for path in self.data_dir.glob(fname + '.*'))
            actual_files = sorted(os.listdir(tempdir))
            assert expected_files == actual_files

            for expected, actual in zip(expected_files, actual_files):
                with open(os.path.join(self.data_dir, expected), 'rt') as f:
                    expected_text = f.readlines()
                with open(os.path.join(tempdir, actual), 'rt') as f:
                    actual_text = f.readlines()

                assert expected_text == actual_text

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
