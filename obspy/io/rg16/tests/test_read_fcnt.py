"""
tests for reading fcnt files
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import glob
import io
import unittest
from os.path import dirname, join, basename

import numpy as np
import obspy
from obspy.io.rg16.core import _read_rg16, _is_rg16

TEST_FCNT_DIRECTORY = join(dirname(__file__), 'data')
FCNT_FILES = glob.glob(join(TEST_FCNT_DIRECTORY, '*'))
FCNT_DICT = {basename(x): _read_rg16(x) for x in FCNT_FILES}
FCNT_STREAMS = list(FCNT_DICT.values())

assert len(FCNT_FILES), 'No test files found'


class TestReadRG16(unittest.TestCase):

    supported_samps = {250, 500, 1000, 2000}
    supported_number_of_components = {1, 3}

    def test_rg16_files_identified(self):
        """
        Ensure the rg16 files are correctly labeled as such.
        """
        for fcnt_file in FCNT_FILES:
            self.assertTrue(_is_rg16(fcnt_file))

    def test_empty_buffer(self):
        """
        Ensure an empty buffer returns false.
        """
        buff = io.BytesIO()
        self.assertFalse(_is_rg16(buff))

    def test_supported_samps(self):
        """
        Ensure all the sampling rates are supported.
        """
        for fcnt_stream in FCNT_STREAMS:
            for tr in fcnt_stream:
                self.assertIn(tr.stats.sampling_rate, self.supported_samps)

    def test_component_number(self):
        """
        Ensure there are either 1 type of channel or 3.
        """
        for fcnt_stream in FCNT_STREAMS:
            seed_ids = len({tr.id for tr in fcnt_stream})
            self.assertIn(seed_ids, self.supported_number_of_components)

    def test_channel_code(self):
        """
        Ensure the channel code is seed compliant.
        """
        expected_components = {'2', '3', '4'}

        for fcnt_stream in FCNT_STREAMS:
            for tr in fcnt_stream:
                channel = tr.stats.channel
                component = channel[-1]
                self.assertEqual(len(channel), 3)
                self.assertIn(component, expected_components)
            seed_ids = len({tr.id for tr in fcnt_stream})
            self.assertIn(seed_ids, self.supported_number_of_components)

    def test_contacts_north(self):
        """
        Ensure the contacts north option maps channels and flips Z trace data.
        """
        components = {'Z', 'N', 'E'}
        for filename, st_default in zip(FCNT_FILES, FCNT_STREAMS):
            st_mapped = _read_rg16(filename, contacts_north=True)
            # make sure components have been mapped to principal directions
            for tr in st_mapped:
                self.assertIn(tr.stats.channel[-1], components)
            # make sure z component is reverse of 2
            tr_2 = st_default.select(component='2')
            tr_z = st_mapped.select(component='Z')
            # apparently the one component test file only has channel 3 so
            # we need to make sure a z component is found in each
            if len(tr_2) and len(tr_z):
                self.assertTrue(np.all(tr_2[0].data == -tr_z[0].data))

    def test_contacts_north_and_merge(self):
        """
        Ensure the "contacts_north" and "merge" parameters can be used
        together. See #2198.
        """
        for filename in FCNT_FILES:
            st = _read_rg16(filename, contacts_north=True, merge=True)
            assert isinstance(st, obspy.Stream)

    def test_can_write(self):
        """
        Ensure the resulting stream can be written as mseed.
        """
        for fcnt_stream in FCNT_STREAMS:
            bytstr = io.BytesIO()
            # test passes if this doesn't raise
            try:
                fcnt_stream.write(bytstr, 'mseed')
            except Exception:
                self.fail('Failed to write to mseed!')

    def test_can_read_from_buffer(self):
        """
        Ensure each stream can be read from a buffer.
        """
        for fcnt_file in FCNT_FILES:
            with open(fcnt_file, 'rb') as fi:
                buff = io.BytesIO(fi.read())
            buff.seek(0)
            try:
                _read_rg16(buff, 'mseed')
            except Exception:
                self.fail('failed to read from bytesIO')

    def test_no_empty_streams(self):
        """
        There should be no empty streams.
        """
        for st in FCNT_STREAMS:
            for tr in st:
                self.assertGreater(len(tr.data), 0)

    def test_no_data(self):
        """
        Ensure no data is returned when the option is used.
        """
        for fcnt_file in FCNT_FILES:
            st = _read_rg16(fcnt_file, headonly=True)
            for tr in st:
                self.assertEqual(len(tr.data), 0)
                self.assertNotEqual(tr.stats.npts, 0)

    def test_starttime_endtime(self):
        """
        Ensure starttimes and endtimes filter traces returned.
        """
        for fcnt_file in FCNT_FILES:
            # get good times to filter on
            st = _read_rg16(fcnt_file, headonly=True)
            stats = st[0].stats
            t1, t2 = stats.starttime.timestamp, stats.endtime.timestamp
            tpoint = obspy.UTCDateTime((t1 + t2) / 2.)
            # this should only return one trace for each channel
            st = _read_rg16(fcnt_file, starttime=tpoint, endtime=tpoint)
            ids = {tr.id for tr in st}
            self.assertEqual(len(st), len(ids))
            # make sure tpoint is in the time range
            start = st[0].stats.starttime
            end = st[0].stats.endtime
            self.assertLess(start, tpoint)
            self.assertLess(tpoint, end)

    def test_merge(self):
        """
        Ensure the merge option of read_rg16 merges all contiguous traces
        together.
        """
        for fcnt_file in FCNT_FILES:
            st_merged = _read_rg16(fcnt_file, merge=True)
            st = _read_rg16(fcnt_file).merge()
            self.assertEqual(len(st), len(st_merged))
            self.assertEqual(st, st_merged)

    def test_values(self):
        """
        Test the first few values of one of the files.
        """
        st = FCNT_DICT['one_channel_many_traces.fcnt']
        expected = np.array([
            -0.00163913, -0.00409782, -0.00834465, -0.00439584, -0.00394881,
            -0.00238419, -0.0013411, 0.00052154, 0.00789762, 0.00841916,
            0.00670552, 0.00655651, 0.00759959, 0.00685454, 0.00715256
        ])
        all_close = np.allclose(st[0].data[:len(expected)], expected)
        self.assertTrue(all_close)


def suite():
    return unittest.makeSuite(TestReadRG16, 'test')


if __name__ == '__main__':
    unittest.main()
