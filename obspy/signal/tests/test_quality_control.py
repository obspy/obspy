# -*- coding: utf-8 -*-
"""
The Quality Control test suite.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import os
import unittest

import obspy
from obspy.signal.quality_control import MSEEDMetadata


class QualityControlTestCase(unittest.TestCase):
    """
    Test cases for Quality Control.
    """
    def setUp(self):
        # directory where the test files are located
        self.path = os.path.join(os.path.dirname(__file__), 'data')

    def test_populate_metadata_null(self):
        mseed_filenames = []
        files = list()
        for _i in mseed_filenames:
            files.append(os.path.join(self.path, _i))
        start_time = obspy.UTCDateTime('2012-07-30T00:00:00')
        end_time = obspy.UTCDateTime('2012-07-30T23:59:59')
        mseed_metadata = MSEEDMetadata()

        with self.assertRaises(ValueError) as e:
            mseed_metadata.populate_metadata(files, start_time, end_time)

        self.assertEqual(
            e.exception.args[0],
            "Nothing added - no data within the given temporal constraints "
            "found.")

    def test_populate_metadata(self):
        mseed_filenames = ['VEN1.NL.HHZ.D.2012.07.30.212.0000']
        files = list()
        for _i in mseed_filenames:
            files.append(os.path.join(self.path, _i))
        start_time = obspy.UTCDateTime('2012-07-30T00:00:00')
        end_time = obspy.UTCDateTime('2012-07-31T00:00:00')
        mseed_metadata = MSEEDMetadata()
        mseed_metadata.populate_metadata(files, start_time, end_time)
        self.assertEqual(mseed_metadata._ms_meta['num_gaps'], 3)

    def test_populate_metadata_multiple_files(self):
        mseed_filenames = ['LLW.BHZ.BN.1989.172', 'LLW.BHZ.BN.1989.173']
        files = list()
        for _i in mseed_filenames:
            files.append(os.path.join(self.path, _i))
        start_time = '1989-06-22T00:00:00'
        end_time = '1989-06-22T23:59:59'
        mseed_metadata = MSEEDMetadata()
        mseed_metadata.populate_metadata(files, start_time,
                                         end_time, c_seg=False)
        self.assertEqual(mseed_metadata._ms_meta['num_gaps'], 1)
        self.assertNotIn("c_segments", mseed_metadata._ms_meta)
        mseed_more_filenames = ['NA.SEUT..BHZ.D.2015.289',
                                'NA.SEUT..BHZ.D.2015.290']
        start_time = '2015-10-17T00:00:00'
        end_time = '2015-10-17T23:59:00'
        more_files = list()
        mseed_metadata2 = MSEEDMetadata()
        for _i in mseed_more_filenames:
            more_files.append(os.path.join(self.path, _i))
        mseed_metadata2.populate_metadata(more_files, start_time,
                                          end_time, c_seg=False)
        self.assertEqual(mseed_metadata2._ms_meta['telemetry_sync_error'], 0)
        self.assertEqual(mseed_metadata2._ms_meta['suspect_time_tag'], 6)

    def test_get_json_meta_no_tq(self):
        mseed_filenames = ['fdsnws-dataselect_2015-10-21T11_32_21.mseed']
        files = list()
        for _i in mseed_filenames:
            files.append(os.path.join(self.path, _i))
        start_time = '2015-01-01T00:00:00'
        end_time = '2015-01-02T00:00:00'
        mseed_metadata = MSEEDMetadata()
        mseed_metadata.populate_metadata(files, start_time, end_time)
        self.assertEqual(mseed_metadata._ms_meta['timing_quality_max'], None)

    def test_get_json_meta(self):
        mseed_filenames = ['SFRA.HGE.CH.2011.101']
        files = list()
        for _i in mseed_filenames:
            files.append(os.path.join(self.path, _i))
        start_time = '2011-04-11T00:00:00'
        end_time = '2011-04-11T23:59:59'
        mseed_metadata = MSEEDMetadata()
        mseed_metadata.populate_metadata(files, start_time, end_time)
        self.assertGreater(mseed_metadata._ms_meta['num_gaps'], 10)


def suite():
    return unittest.makeSuite(QualityControlTestCase, 'test')

if __name__ == '__main__':
    unittest.main(defaultTest='suite')
