# -*- coding: utf-8 -*-
"""
The obspy.imaging.scripts.scan / obspy-scan test suite.
"""

from obspy.core.util.base import getMatplotlibVersion, NamedTemporaryFile
from obspy.core.util.testing import HAS_COMPARE_IMAGE, ImageComparison
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

    @skipIf(not HAS_COMPARE_IMAGE, 'nose not installed or matplotlib too old')
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

    @skipIf(not HAS_COMPARE_IMAGE, 'nose not installed or matplotlib too old')
    def test_multipleSamplingrates(self):
        """
        Check for multiple sampling rates
        """
        lines = [
            "TIMESERIES XX_TEST__BHZ_R, 200 samples, 200 sps, "
            "2008-01-15T00:00:00.000000, SLIST, INTEGER, Counts",
            "TIMESERIES XX_TEST__BHZ_R,  50 samples,  50 sps, "
            "2008-01-15T00:00:01.000000, SLIST, INTEGER, Counts",
            "TIMESERIES XX_TEST__BHZ_R, 200 samples, 200 sps, "
            "2008-01-15T00:00:02.000000, SLIST, INTEGER, Counts",
        ]
        reltol = 1
        if MATPLOTLIB_VERSION < [1, 3, 0]:
            reltol = 60
        files = []
        with NamedTemporaryFile() as f1:
            with NamedTemporaryFile() as f2:
                with NamedTemporaryFile() as f3:
                    for i, fp in enumerate([f1, f2, f3]):
                        fp.write("%s\n" % lines[i])
                        fp.flush()
                        fp.seek(0)
                        files.append(fp.name)
                    with ImageComparison(self.path, 'scan_mult_sampl.png',
                                         reltol=reltol) as ic:
                        try:
                            tmp_stdout = sys.stdout
                            sys.stdout = open(os.devnull, 'wb')
                            obspy_scan(files + ['--output', ic.name])
                        finally:
                            sys.stdout.close()
                            sys.stdout = tmp_stdout


def suite():
    return unittest.makeSuite(ScanTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
