# -*- coding: utf-8 -*-
"""
The obspy.imaging.scripts.scan / obspy-scan test suite.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import os
import shutil
import unittest
from os.path import abspath, dirname, join, pardir
import warnings

import matplotlib.pyplot as plt

from obspy import read, UTCDateTime
from obspy.core.util.base import NamedTemporaryFile, get_example_file
from obspy.core.util.misc import TemporaryWorkingDirectory, CatchOutput
from obspy.core.util.testing import ImageComparison
from obspy.imaging.scripts.scan import main as obspy_scan
from obspy.imaging.scripts.scan import scan, Scanner


class ScanTestCase(unittest.TestCase):
    """
    Test cases for obspy-scan
    """
    def setUp(self):
        # directory where the test files are located
        self.root = abspath(join(dirname(__file__), pardir, pardir))
        self.path = join(self.root, 'imaging', 'tests', 'images')
        sac_files = ['LMOW.BHE.SAC', 'seism.sac', 'dis.G.SCZ.__.BHE_short',
                     'null_terminated.sac', 'test.sac', 'seism-longer.sac',
                     'test.sac.swap', 'seism-shorter.sac', 'testxy.sac']
        gse2_files = ['STA2.testlines', 'STA2.testlines_out', 'acc.gse',
                      'loc_RJOB20050831023349.z',
                      'loc_RJOB20050831023349_first100_dos.z',
                      'loc_RNON20040609200559.z', 'loc_STAU20031119011659.z',
                      'sta2.gse2', 'twiceCHK2.gse2', 'y2000.gse']
        all_files = [join(self.root, 'io', 'sac', 'tests', 'data', i)
                     for i in sac_files]
        all_files.extend([join(self.root, 'io', 'gse2', 'tests', 'data', i)
                          for i in gse2_files])
        self.all_files = all_files

    def test_scan_main_method(self):
        """
        Run obspy-scan on selected tests/data directories
        """
        # Copy files to a temp folder to avoid wildcard scans.
        with TemporaryWorkingDirectory():
            for filename in self.all_files:
                shutil.copy(filename, os.curdir)

            with ImageComparison(self.path, 'scan.png') as ic:
                obspy_scan([os.curdir] + ['--output', ic.name])

    def test_scan_function_and_scanner_class(self):
        """
        Test scan function and Scanner class (in one test to keep overhead of
        copying files down)
        """
        scanner = Scanner()
        # Copy files to a temp folder to avoid wildcard scans.
        with TemporaryWorkingDirectory():
            for filename in self.all_files:
                shutil.copy(filename, os.curdir)

            scanner.parse(os.curdir)

            with ImageComparison(self.path, 'scan.png') as ic:
                scan(paths=os.curdir, plot=ic.name)

        with ImageComparison(self.path, 'scan.png') as ic:
            scanner.plot(ic.name)

    def test_scan_plot_by_id_with_wildcard(self):
        """
        Test selecting what to plot after scanning with wildcards in selected
        SEED IDs
        """
        files = [
            "BW.UH1._.EHZ.D.2010.147.a.slist.gz",
            "BW.UH1._.EHZ.D.2010.147.b.slist.gz",
            "BW.UH1._.SHZ.D.2010.147.cut.slist.gz",
            "BW.UH2._.SHZ.D.2010.147.cut.slist.gz",
            "BW.UH3._.SHE.D.2010.147.cut.slist.gz",
            "BW.UH3._.SHN.D.2010.147.cut.slist.gz",
            "BW.UH3._.SHZ.D.2010.147.cut.slist.gz",
            "BW.UH4._.EHZ.D.2010.147.cut.slist.gz",
            "IUANMO.seed"]

        scanner = Scanner()

        for filename in files:
            scanner.parse(get_example_file(filename))

        expected = [
            ('*.UH[12]*',
             ['BW.UH2..SHZ\n100.0%',
              'BW.UH1..SHZ\n100.0%',
              'BW.UH1..EHZ\n10.7%']),
            ('*Z',
             ['IU.ANMO.00.LHZ\n100.0%',
              'BW.UH4..EHZ\n100.0%',
              'BW.UH3..SHZ\n100.0%',
              'BW.UH2..SHZ\n100.0%',
              'BW.UH1..SHZ\n100.0%',
              'BW.UH1..EHZ\n10.7%'])]

        for seed_id, expected_labels in expected:
            fig, ax = plt.subplots()
            fig = scanner.plot(fig=fig, show=False, seed_ids=[seed_id])
            got = [label.get_text() for label in ax.get_yticklabels()]
            self.assertEqual(got, expected_labels)
            plt.close(fig)

    def test_scanner_manually_add_streams(self):
        """
        Test Scanner class, manually adding streams of read data files
        """
        scanner = Scanner()
        # Copy files to a temp folder to avoid wildcard scans.
        with TemporaryWorkingDirectory():
            for filename in self.all_files:
                shutil.copy(filename, os.curdir)

            for file_ in os.listdir(os.curdir):
                # some files used in the test cases actually can not
                # be read with obspy..
                if file_ in ('STA2.testlines_out', 'STA2.testlines',
                             'seism-shorter.sac', 'seism-longer.sac'):
                    continue
                st = read(file_, headonly=True)
                scanner.add_stream(st)

        with ImageComparison(self.path, 'scan.png') as ic:
            scanner.plot(ic.name)

    def test_scan_save_load_npz(self):
        """
        Run obspy-scan on selected tests/data directories, saving/loading
        to/from npz.

        Tests both the command line script and the Scanner class.
        """
        scanner = Scanner()
        # Copy files to a temp folder to avoid wildcard scans.
        with TemporaryWorkingDirectory():
            for filename in self.all_files:
                shutil.copy(filename, os.curdir)

            # save via command line
            obspy_scan([os.curdir, '--write', 'scan.npz'])

            # save via Python
            scanner.parse(os.curdir)
            scanner.save_npz('scanner.npz')
            scanner = Scanner()

            # version string of '0.0.0+archive' raises UserWarning - ignore
            with warnings.catch_warnings(record=True):
                warnings.simplefilter('ignore', UserWarning)

                # load via Python
                scanner.load_npz('scanner.npz')
                with ImageComparison(self.path, 'scan.png') as ic:
                    scanner.plot(ic.name)

                # load via command line
                with ImageComparison(self.path, 'scan.png') as ic:
                    obspy_scan(['--load', 'scan.npz', '--output', ic.name])

    def test_scan_times(self):
        """
        Checks for timing related options
        """
        # Copy files to a temp folder to avoid wildcard scans.
        with TemporaryWorkingDirectory():
            for filename in self.all_files:
                shutil.copy(filename, os.curdir)

            with ImageComparison(self.path, 'scan_times.png') as ic:
                obspy_scan([os.curdir] + ['--output', ic.name] +
                           ['--start-time', '2004-01-01'] +
                           ['--end-time', '2004-12-31'] +
                           ['--event-time', '2004-03-14T15:09:26'] +
                           ['--event-time', '2004-02-07T18:28:18'])

    def test_multiple_sampling_rates(self):
        """
        Check for multiple sampling rates
        """
        lines = [
            "TIMESERIES XX_TEST__BHZ_R, 200 samples, 200 sps, "
            "2008-01-15T00:00:00.000000, SLIST, INTEGER, Counts",
            "TIMESERIES XX_TEST__BHZ_R,  50 samples,  50 sps, "
            "2008-01-15T00:00:00.900000, SLIST, INTEGER, Counts",
            "TIMESERIES XX_TEST__BHZ_R, 200 samples, 200 sps, "
            "2008-01-15T00:00:02.000000, SLIST, INTEGER, Counts",
        ]
        files = []
        expected = [
            "XX.TEST..BHZ 2008-01-15T00:00:01.000000Z "
            "2008-01-15T00:00:00.899995Z -0.100",
            "XX.TEST..BHZ 2008-01-15T00:00:01.899999Z "
            "2008-01-15T00:00:02.000000Z 0.100"
        ]
        with NamedTemporaryFile() as f1, NamedTemporaryFile() as f2, \
                NamedTemporaryFile() as f3:
            for i, fp in enumerate([f1, f2, f3]):
                fp.write(("%s\n" % lines[i]).encode('ascii',
                                                    'strict'))
                fp.flush()
                fp.seek(0)
                files.append(fp.name)

            # make image comparison instance and set manual rms (see #2089)
            image_comp = ImageComparison(self.path, 'scan_mult_sampl.png')
            image_comp.tol = 50

            with image_comp as ic:

                obspy_scan(files + ['--output', ic.name, '--print-gaps'])

                with CatchOutput() as out:
                    obspy_scan(files + ['--output', ic.name, '--print-gaps'])

                # read output and compare with expected
                # only check if datetime objects are close, not exact
                output = out.stdout.splitlines()
                for ex_line, out_line in zip(expected, output):
                    ex_split = ex_line.split(' ')
                    out_split = out_line.split(' ')
                    for ex_str, out_str in zip(ex_split, out_split):
                        try:
                            utc1 = UTCDateTime(ex_str)
                            utc2 = UTCDateTime(out_str)
                        except (ValueError, TypeError):
                            # if str is not a datetime it should be equal
                            self.assertEqual(ex_str, out_str)
                        else:
                            # datetimes just need to be close
                            t1, t2 = utc1.timestamp, utc2.timestamp
                            self.assertTrue(abs(t1 - t2) < .001)


def suite():
    return unittest.makeSuite(ScanTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
