# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import os
import shutil
import tempfile
import unittest

import numpy as np

from obspy import UTCDateTime, Trace, Stream
from obspy.clients.filesystem.sds import SDS_FMTSTR, Client


class TemporarySDSDirectory(object):
    """
    Handles creation and deletion of a temporary SDS directory structure.
    To be used with "with" statement.
    """
    def __init__(self, year, doy):
        """
        Set which day's midnight (00:00 hours) is used as a day break in the
        testing (to split the test data into two files).
        """
        self.time = UTCDateTime("%d-%03dT00:00:00" % (year, doy))
        sampling_rate = 0.1
        delta = 1 / sampling_rate
        networks = ("AB", "CD")
        stations = ("XYZ", "ZZZ3")
        locations = ("", "00")
        channels = ("HHZ", "HHN", "HHE", "BHZ", "BHN", "BHE")

        self.stream = Stream()
        for net in networks:
            for sta in stations:
                for loc in locations:
                    for cha in channels:
                        tr = Trace(
                            data=np.arange(100, dtype=np.int32),
                            header=dict(
                                network=net, station=sta, location=loc,
                                channel=cha, sampling_rate=sampling_rate,
                                starttime=self.time - 30 * delta))

                        # cut into two seamless traces
                        tr1 = tr.slice(endtime=self.time + 5 * delta)
                        tr2 = tr.slice(starttime=self.time + 6 * delta)
                        self.stream.append(tr1)
                        self.stream.append(tr2)

    def __enter__(self):
        self.old_dir = os.getcwd()
        self.tempdir = tempfile.mkdtemp(prefix='obspy-sdstest-')
        for tr_ in self.stream:
            t_ = tr_.stats.starttime
            full_path = SDS_FMTSTR.format(year=t_.year, doy=t_.julday,
                                          type="D", **tr_.stats)
            full_path = os.path.join(self.tempdir, full_path)
            dirname, filename = os.path.split(full_path)
            if not os.path.isdir(dirname):
                os.makedirs(dirname)
            tr_.write(full_path, format="MSEED")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):  # @UnusedVariable
        os.chdir(self.old_dir)
        shutil.rmtree(self.tempdir)


class SDSTestCase(unittest.TestCase):
    """
    Test reading data from SDS file structure.
    """
    def test_read_from_SDS(self):
        """
        Test reading data across year and day breaks from SDS directory
        structure. Also tests looking for data on the wrong side of a day
        break (data usually get written for some seconds into the previous or
        next file around midnight).
        """
        # test for one specific SEED ID, without wildcards
        seed_id = "AB.XYZ..HHZ"
        net, sta, loc, cha = seed_id.split(".")
        # use three different day breaks in the testing:
        #  - normal day break during one year
        #     (same directory, separate filenames)
        #  - day break at end of year
        #     (separate directories, separate filenames)
        #      - leap-year
        #      - non-leap-year
        for year, doy in ((2015, 123), (2015, 1), (2012, 1)):
            t = UTCDateTime("%d-%03dT00:00:00" % (year, doy))
            with TemporarySDSDirectory(year=year, doy=doy) as temp_sds:
                # normal test reading across the day break
                client = Client(temp_sds.tempdir)
                st = client.get_waveforms(net, sta, loc, cha, t-20, t+20)
                self.assertEqual(len(st), 1)
                self.assertEqual(st[0].stats.starttime, t-20)
                self.assertEqual(st[0].stats.endtime, t+20)
                self.assertEqual(len(st[0]), 5)
                # test merge option
                st = client.get_waveforms(net, sta, loc, cha, t-200, t+200,
                                          merge=False)
                self.assertEqual(len(st), 2)
                st = client.get_waveforms(net, sta, loc, cha, t-200, t+200,
                                          merge=None)
                self.assertEqual(len(st), 2)
                st = client.get_waveforms(net, sta, loc, cha, t-200, t+200,
                                          merge=0)
                self.assertEqual(len(st), 1)
                # test reading data from a single day file
                # (data is in the file where it's expected)
                st = client.get_waveforms(net, sta, loc, cha, t-80, t-30)
                self.assertEqual(len(st), 1)
                # test reading data from a single day file
                # (data is in the dayfile of the previous day)
                st = client.get_waveforms(net, sta, loc, cha, t+20, t+40)
                self.assertEqual(len(st), 1)
                # test that format autodetection with `format=None` works
                client = Client(temp_sds.tempdir, format=None)
                st = client.get_waveforms(net, sta, loc, cha, t-200, t+200)
                self.assertEqual(len(st), 1)

    def test_read_from_SDS_with_wildcarded_seed_ids(self):
        """
        Test reading data with wildcarded SEED IDs.
        """
        year, doy = 2015, 1
        t = UTCDateTime("%d-%03dT00:00:00" % (year, doy))
        with TemporarySDSDirectory(year=year, doy=doy) as temp_sds:
            # test different wildcard combinations in SEED ID
            client = Client(temp_sds.tempdir)
            for wildcarded_seed_id, num_matching_ids in zip(
                    ("AB.ZZZ3..HH?", "AB.ZZZ3..HH*", "*.*..HHZ",
                     "*.*.*.HHZ", "*.*.*.*"),
                    (3, 3, 4, 8, 48)):
                net, sta, loc, cha = wildcarded_seed_id.split(".")
                st = client.get_waveforms(net, sta, loc, cha, t-200, t+200)
                self.assertEqual(len(st), num_matching_ids)
            # test with SDS type wildcards
            for type_wildcard in ("*", "?"):
                net, sta, loc, cha = wildcarded_seed_id.split(".")
                st = client.get_waveforms(net, sta, loc, cha, t-200, t+200)
                self.assertEqual(len(st), num_matching_ids)


def suite():
    return unittest.makeSuite(SDSTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
