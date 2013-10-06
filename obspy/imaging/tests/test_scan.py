# -*- coding: utf-8 -*-
"""
The obspy.imaging.scripts.scan / obspy-scan test suite.
"""

from obspy.core.util.base import HAS_COMPARE_IMAGE, \
    ImageComparison, getMatplotlibVersion
from obspy.core.util.decorator import skipIf
from obspy.imaging.scripts.scan import main as obspy_scan
from os.path import dirname, abspath, join, pardir
import sys
import os
import unittest


MATPLOTLIB_VERSION = getMatplotlibVersion()


class ScanTestCase(unittest.TestCase):
    """
    Test cases for obspy-scan
    """
    def setUp(self):
        # directory where the test files are located
        self.root = abspath(join(dirname(__file__), pardir, pardir))
        self.path = join(self.root, 'imaging', 'tests', 'images')

    @skipIf(not HAS_COMPARE_IMAGE, 'nose not installed or matplotlib to old')
    def test_scan(self):
        """
        Run obspy-scan on selected tests/data directories
        """
        reltol = 1
        if MATPLOTLIB_VERSION < [1, 3, 0]:
            reltol = 60
        # using mseed increases test time by factor 2
        waveform_dirs = [join(self.root, n, 'tests', 'data')
                         for n in ('sac', 'gse2')]
        with ImageComparison(self.path, 'scan.png', reltol=reltol) as ic:
            try:
                tmp_stdout = sys.stdout
                sys.stdout = open(os.devnull, 'wb')
                obspy_scan(waveform_dirs + ['--output', ic.name])
            finally:
                sys.stdout.close()
                sys.stdout = tmp_stdout


def suite():
    return unittest.makeSuite(ScanTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
