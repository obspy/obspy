# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import unittest
from obspy.core.compatibility import mock

from obspy import UTCDateTime
from obspy.clients.filesystem.tsindex import Client
from collections import namedtuple


class TSIndexTestCase(unittest.TestCase):
    
    def test_get_nslc(self):
        client = Client("")
        
        NamedRow = namedtuple('NamedRow',
                              ['network', 'station', 'location', 'channel',
                               'earliest', 'latest'])
        mocked_summary_rows = [NamedRow("AK", "ANM", "", "VM2", "2018-08-10T21:52:50.000000", "2018-08-10T22:12:39.999991"),
                               NamedRow("AK", "ANM", "", "VM3", "2018-08-10T21:52:50.000000", "2018-08-10T22:12:39.999991"),
                               NamedRow("AK", "ANM", "", "VM4", "2018-08-10T21:52:50.000000", "2018-08-10T22:12:39.999991"),
                               NamedRow("AK", "ANM", "", "VM5", "2018-08-10T21:52:50.000000", "2018-08-10T22:12:39.999991"),
                               NamedRow("N4", "H43A", "", "VM2", "2018-08-10T21:09:39.000000", "2018-08-10T22:09:28.890415"),
                               NamedRow("N4", "H43A", "", "VM3", "2018-08-10T21:09:39.000000", "2018-08-10T22:09:28.890415"),
                              ]
        client.get_summary_rows = mock.MagicMock(
                                            return_value=mocked_summary_rows)
        
        expected_avail_extents = [("AK", "ANM", "", "VM2"),
                                  ("AK", "ANM", "", "VM3"),
                                  ("AK", "ANM", "", "VM4"),
                                  ("AK", "ANM", "", "VM5"),
                                  ("N4", "H43A", "", "VM2"),
                                  ("N4", "H43A", "", "VM3")
                                 ]
        
        self.assertEqual(client.get_nslc("AK,N4",
                                         "ANM,H43A",
                                         "",
                                         "VM2,VM3,VM4,VM5",
                                         "2018-08-10T21:09:39.000000",
                                         "2018-08-10T22:09:28.890415"),
                                         expected_avail_extents)
    
    def test_get_availability_extent(self):
        client = Client("")
        
        NamedRow = namedtuple('NamedRow',
                              ['network', 'station', 'location', 'channel',
                               'earliest', 'latest'])
        mocked_summary_rows = [NamedRow("AK", "ANM", "", "VM2", "2018-08-10T21:52:50.000000", "2018-08-10T22:12:39.999991"),
                               NamedRow("AK", "ANM", "", "VM3", "2018-08-10T21:52:50.000000", "2018-08-10T22:12:39.999991"),
                               NamedRow("AK", "ANM", "", "VM4", "2018-08-10T21:52:50.000000", "2018-08-10T22:12:39.999991"),
                               NamedRow("AK", "ANM", "", "VM5", "2018-08-10T21:52:50.000000", "2018-08-10T22:12:39.999991"),
                               NamedRow("N4", "H43A", "", "VM2", "2018-08-10T21:09:39.000000", "2018-08-10T22:09:28.890415"),
                               NamedRow("N4", "H43A", "", "VM3", "2018-08-10T21:09:39.000000", "2018-08-10T22:09:28.890415"),
                              ]
        client.get_summary_rows = mock.MagicMock(
                                            return_value=mocked_summary_rows)
        
        expected_avail_extents = [("AK", "ANM", "", "VM2", UTCDateTime("2018-08-10T21:52:50.000000"), UTCDateTime("2018-08-10T22:12:39.999991")),
                                  ("AK", "ANM", "", "VM3", UTCDateTime("2018-08-10T21:52:50.000000"), UTCDateTime("2018-08-10T22:12:39.999991")),
                                  ("AK", "ANM", "", "VM4", UTCDateTime("2018-08-10T21:52:50.000000"), UTCDateTime("2018-08-10T22:12:39.999991")),
                                  ("AK", "ANM", "", "VM5", UTCDateTime("2018-08-10T21:52:50.000000"), UTCDateTime("2018-08-10T22:12:39.999991")),
                                  ("N4", "H43A", "", "VM2", UTCDateTime("2018-08-10T21:09:39.000000"), UTCDateTime("2018-08-10T22:09:28.890415")),
                                  ("N4", "H43A", "", "VM3", UTCDateTime("2018-08-10T21:09:39.000000"), UTCDateTime("2018-08-10T22:09:28.890415"))
                                 ]
        
        self.assertEqual(client.get_availability_extent("AK,N4",
                                                        "ANM,H43A", "",
                                                        "VM2,VM3,VM4,VM5",
                                                        "2018-08-10T21:09:39.000000",
                                                        "2018-08-10T22:09:28.890415"),
                                                        expected_avail_extents)
        
    def test__are_timespans_adjacent(self):
        client = Client("")
        sample_rate = 40 
        # sample_period = 1/40 = 0.025 sec = 25000 ms
        # and tolerance = 0.5 so an adjacent sample is +/-0.0125 sec = 12500 ms 
        
        # 1 sample period later
        ts1 = client._create_timespan(UTCDateTime(2018, 8, 10, 22, 0, 0, 0),
                                      UTCDateTime(2018, 8, 10, 22, 0, 0, 25000))
        ts2 = client._create_timespan(UTCDateTime(2018, 8, 10, 22, 0, 0, 50000),
                                      UTCDateTime(2018, 8, 10, 22, 0, 0, 75000))
        self.assertTrue(client._are_timespans_adjacent(ts1, ts2, sample_rate))
        
        # 1ms after nearest tolerance boundary (next sample - 12500ms)
        ts1 = client._create_timespan(UTCDateTime(2018, 8, 10, 22, 0, 0, 0),
                                      UTCDateTime(2018, 8, 10, 22, 0, 0, 25000))
        ts2 = client._create_timespan(UTCDateTime(2018, 8, 10, 22, 0, 0, 37501),
                                      UTCDateTime(2018, 8, 10, 22, 0, 0, 75000))
        self.assertTrue(client._are_timespans_adjacent(ts1, ts2, sample_rate))
        
        # exactly on nearest tolerance boundary (next sample - 12500ms)
        ts1 = client._create_timespan(UTCDateTime(2018, 8, 10, 22, 0, 0, 0),
                                      UTCDateTime(2018, 8, 10, 22, 0, 0, 25000))
        ts2 = client._create_timespan(UTCDateTime(2018, 8, 10, 22, 0, 0, 37500),
                                      UTCDateTime(2018, 8, 10, 22, 0, 0, 75000))
        self.assertFalse(client._are_timespans_adjacent(ts1, ts2, sample_rate))
        
        # 1ms before nearest tolerance boundary (next sample - 12500ms)
        ts1 = client._create_timespan(UTCDateTime(2018, 8, 10, 22, 0, 0, 0),
                                      UTCDateTime(2018, 8, 10, 22, 0, 0, 25000))
        ts2 = client._create_timespan(UTCDateTime(2018, 8, 10, 22, 0, 0, 37499),
                                      UTCDateTime(2018, 8, 10, 22, 0, 0, 75000))
        self.assertFalse(client._are_timespans_adjacent(ts1, ts2, sample_rate))
        
        # 1ms after farthest tolerance boundary (next sample + 12500ms)
        ts1 = client._create_timespan(UTCDateTime(2018, 8, 10, 22, 0, 0, 0),
                                      UTCDateTime(2018, 8, 10, 22, 0, 0, 25000))
        ts2 = client._create_timespan(UTCDateTime(2018, 8, 10, 22, 0, 0, 62501),
                                      UTCDateTime(2018, 8, 10, 22, 0, 0, 100000))
        self.assertFalse(client._are_timespans_adjacent(ts1, ts2, sample_rate))
        
        # on farthest tolerance boundary (next sample + 12500ms)
        ts1 = client._create_timespan(UTCDateTime(2018, 8, 10, 22, 0, 0, 0),
                                      UTCDateTime(2018, 8, 10, 22, 0, 0, 25000))
        ts2 = client._create_timespan(UTCDateTime(2018, 8, 10, 22, 0, 0, 62500),
                                      UTCDateTime(2018, 8, 10, 22, 0, 0, 100000))
        self.assertFalse(client._are_timespans_adjacent(ts1, ts2, sample_rate))
        
        # 1ms before farthest tolerance boundary (next sample + 12500ms)
        ts1 = client._create_timespan(UTCDateTime(2018, 8, 10, 22, 0, 0, 0),
                                      UTCDateTime(2018, 8, 10, 22, 0, 0, 25000))
        ts2 = client._create_timespan(UTCDateTime(2018, 8, 10, 22, 0, 0, 62499),
                                      UTCDateTime(2018, 8, 10, 22, 0, 0, 100000))
        self.assertTrue(client._are_timespans_adjacent(ts1, ts2, sample_rate))
        
        
    def test_get_availability(self):
        client = Client("")
        
        NamedRow = namedtuple('NamedRow',
                              ['network', 'station', 'location', 'channel',
                               'samplerate', 'starttime', 'endtime',
                               'timespans'])
        # Each row contains one gap. The first rows latest time is adjacent
        # to the second rows earliest time. The third row overlaps with the 
        # second row.
        mocked_tsindex_rows = \
                [
                   NamedRow(network=u'AK', station=u'BAGL',
                            location=u'', channel=u'LCC',
                            starttime=u'2018-08-10T22:00:54.000000',
                            endtime=u'2018-08-10T22:20:53.000000',
                            samplerate=1.0,
                            timespans=u'[1533938454.000000:1533939353.000000],'
                                       '[1533938754.000000:1533939653.000000]'),
                   NamedRow(network=u'AK', station=u'BAGL',
                            location=u'', channel=u'LCC',
                            starttime=u'2018-08-10T22:20:53.999000',
                            endtime=u'2018-09-01T23:20:53.000000',
                            samplerate=1.0,
                            timespans=u'[1533939653.999000:1534112453.000000],'
                                       '[1534116053.000000:1535844053.000000]'),
                   NamedRow(network=u'AK', station=u'BAGL',
                            location=u'', channel=u'LCC',
                            starttime=u'2018-08-27T00:00:00.000000',
                            endtime=u'2018-09-11T00:00:00.000000',
                            samplerate=1.0,
                            timespans=u'[1535328000.0:1536624000.0]')
                ]
                   

        client.get_tsindex_rows = mock.MagicMock(
                                            return_value=mocked_tsindex_rows)
        
        expected_unmerged_avail = [("AK", "BAGL", "", "LCC", UTCDateTime(2018, 8, 10, 22, 0, 54), UTCDateTime(2018, 8, 10, 22, 15, 53)),
                                   ("AK", "BAGL", "", "LCC", UTCDateTime(2018, 8, 10, 22, 5, 54), UTCDateTime(2018, 8, 12, 22, 20, 53)),
                                   ("AK", "BAGL", "", "LCC", UTCDateTime(2018, 8, 12, 23, 20, 53), UTCDateTime(2018, 9, 1, 23, 20, 53)),
                                   ("AK", "BAGL", "", "LCC", UTCDateTime(2018, 8, 27, 0, 0), UTCDateTime(2018, 9, 11, 0, 0, 0))
                                  ]

        avail = client.get_availability("AK",
                                       "BAGL",
                                       "",
                                       "LCC",
                                       UTCDateTime(2018, 8, 10, 22, 0, 54),
                                       UTCDateTime(2018, 9, 1, 23, 20, 53))

        self.assertEqual(client.get_availability("AK",
                                                "BAGL", "",
                                                "LCC",
                                                UTCDateTime(2018, 8, 10, 22, 0, 54),
                                                UTCDateTime(2018, 8, 10, 22, 9, 28, 890415)),
                                                expected_unmerged_avail)
        self.assertEqual(client.get_availability("AK",
                                                "BAGL", "",
                                                
                                                "LCC",
                                                UTCDateTime(2018, 8, 10, 22, 0, 54),
                                                UTCDateTime(2018, 8, 10, 22, 9, 28, 890415),
                                                merge_overlap=False),
                                                expected_unmerged_avail)
        
        expected_merged_avail = [("AK", "BAGL", "", "LCC", UTCDateTime(2018, 8, 10, 22, 0, 54), UTCDateTime(2018, 8, 10, 22, 15, 53)),
                                 ("AK", "BAGL", "", "LCC", UTCDateTime(2018, 8, 10, 22, 5, 54), UTCDateTime(2018, 8, 12, 22, 20, 53)),
                                 ("AK", "BAGL", "", "LCC", UTCDateTime(2018, 8, 12, 23, 20, 53), UTCDateTime(2018, 9, 11, 0, 0, 0))
                                ]

        self.assertNotEqual(client.get_availability("AK",
                                                    "BAGL", "",
                                                    "LCC",
                                                    UTCDateTime(2018, 8, 10, 22, 0, 54),
                                                    UTCDateTime(2018, 8, 10, 22, 9, 28, 890415),
                                                    merge_overlap=True),
                                                    expected_merged_avail)
