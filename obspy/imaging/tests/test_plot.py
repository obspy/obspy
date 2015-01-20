# -*- coding: utf-8 -*-
"""
The obspy.imaging.scripts.plot / obspy-plot test suite.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

from obspy.core.util.misc import TemporaryWorkingDirectory
from obspy.core.util.testing import ImageComparison
from obspy.imaging.scripts.plot import main as obspy_plot
from os.path import dirname, abspath, join, pardir, basename
import shutil
import os
import unittest


class PlotTestCase(unittest.TestCase):
    """
    Test cases for obspy-plot
    """
    def setUp(self):
        # directory where the test files are located
        self.root = abspath(join(dirname(__file__), pardir, pardir))
        self.path = join(self.root, 'imaging', 'tests', 'images')
        all_files = [join(self.root, 'core', 'tests', 'data', i)
                     for i in ['slist.ascii', 'slist_2_traces.ascii']]
        self.all_files = all_files

    def test_plot(self):
        """
        Run obspy-plot on selected tests
        """
        # Copy files to a temp folder to avoid wildcard scans.
        with TemporaryWorkingDirectory():
            all_files = []
            for filename in self.all_files:
                newname = join(os.curdir, basename(filename))
                shutil.copy(filename, newname)
                all_files += [newname]

            with ImageComparison(self.path, 'plot.png') as ic:
                obspy_plot(['--outfile', ic.name] + all_files)

    def test_plotNoMerge(self):
        """
        Run obspy-plot without trace merging
        """
        # Copy files to a temp folder to avoid wildcard scans.
        with TemporaryWorkingDirectory():
            all_files = []
            for filename in self.all_files:
                newname = join(os.curdir, basename(filename))
                shutil.copy(filename, newname)
                all_files += [newname]

            with ImageComparison(self.path, 'plot_nomerge.png') as ic:
                obspy_plot(['--no-automerge', '--outfile', ic.name] +
                           all_files)


def suite():
    return unittest.makeSuite(PlotTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
