#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The obspy.io.gcf.core test suite.
"""
import numpy as np
import pytest

from obspy import read
from obspy.core import Stream, Trace, AttribDict
from obspy.core.util import NamedTemporaryFile
from obspy.core.utcdatetime import UTCDateTime
from obspy.io.gcf.core import _read_gcf, _write_gcf, merge_gcf_stream


EXPECTED = np.array([-49378, -49213, -49273, -49277, -49341, -49415, -49289,
                     -49309, -49277, -49381, -49441, -49276, -49331, -49268,
                     -49250, -49407, -49421, -49282, -49224, -49281],
                    dtype=np.int32)


class TestCore():
    """
    Test cases for gcf core interface
    """
    def test_read_via_obspy(self, testdata):
        """
        Read files via obspy.core.stream.read function.
        """
        filename = testdata['20160603_1955n.gcf']
        # 1
        st = read(filename)
        st.verify()
        assert len(st) == 1
        assert st[0].stats.starttime == \
            UTCDateTime('2016-06-03T19:55:00.000000Z')
        assert st[0].stats.endtime == \
            UTCDateTime('2016-06-03T19:55:02.990000Z')
        assert len(st[0]) == 300
        assert round(abs(st[0].stats.sampling_rate-100.0), 7) == 0
        assert st[0].stats.channel == 'HHN'
        assert st[0].stats.station == '6018'
        np.testing.assert_array_equal(EXPECTED, st[0].data[:20])

    def test_read_head_via_obspy(self, testdata):
        """
        Read files via obspy.core.stream.read function.
        """
        filename = testdata['20160603_1955n.gcf']
        # 1
        st = read(filename, headonly=True)
        assert len(st) == 1
        assert st[0].stats.starttime == \
            UTCDateTime('2016-06-03T19:55:00.000000Z')
        assert st[0].stats.endtime == \
            UTCDateTime('2016-06-03T19:55:02.990000Z')
        assert st[0].stats.npts == 300
        assert round(abs(st[0].stats.sampling_rate-100.0), 7) == 0
        assert st[0].stats.channel == 'HHN'
        assert st[0].stats.station == '6018'

    def test_read_via_module(self, testdata):
        """
        Read files via obspy.io.gcf.core._read_gcf function.
        """
        filename = testdata['20160603_1955n.gcf']
        # 1
        st = _read_gcf(filename)
        st.verify()
        assert len(st) == 1
        assert st[0].stats.starttime == \
            UTCDateTime('2016-06-03T19:55:00.000000Z')
        assert st[0].stats.endtime == \
            UTCDateTime('2016-06-03T19:55:02.990000Z')
        assert len(st[0]) == 300
        assert round(abs(st[0].stats.sampling_rate-100.0), 7) == 0
        assert st[0].stats.channel == 'HHN'
        assert st[0].stats.station == '6018'
        np.testing.assert_array_equal(EXPECTED, st[0].data[:20])

    def test_read_head_via_module(self, testdata):
        """
        Read files via obspy.io.gcf.core._read_gcf function.
        """
        filename = testdata['20160603_1955n.gcf']
        # 1
        st = _read_gcf(filename, headonly=True)
        assert len(st) == 1
        assert st[0].stats.starttime == \
            UTCDateTime('2016-06-03T19:55:00.000000Z')
        assert st[0].stats.endtime == \
            UTCDateTime('2016-06-03T19:55:02.990000Z')
        assert st[0].stats.npts == 300
        assert round(abs(st[0].stats.sampling_rate-100.0), 7) == 0
        assert st[0].stats.channel == 'HHN'
        assert st[0].stats.station == '6018'

    def test_read_channel_prefix_via_obspy(self, testdata):
        """
        Read files via obspy.core.stream.read function.
        """
        filename = testdata['20160603_1955n.gcf']
        # 1
        st = read(filename, headonly=True, channel_prefix="HN")
        assert len(st) == 1
        assert st[0].stats.starttime == \
            UTCDateTime('2016-06-03T19:55:00.000000Z')
        assert st[0].stats.endtime == \
            UTCDateTime('2016-06-03T19:55:02.990000Z')
        assert st[0].stats.npts == 300
        assert round(abs(st[0].stats.sampling_rate-100.0), 7) == 0
        assert st[0].stats.channel == 'HNN'
        assert st[0].stats.station == '6018'

    def test_merge_gcf_stream(self, testdata):
        """
        Read files via obspy.core.stream.read function.
        """
        filename = testdata['20160603_1955n.gcf']
        # 1
        st1 = read(filename, headonly=True, channel_prefix="HN")
        st2 = st1.copy()
        tr1 = st1[0].copy()
        tr2 = st1[0].copy()
        tr1.stats.starttime = UTCDateTime('2016-06-03T19:55:02.000000Z')
        tr1.stats.npts = 100
        tr2.stats.npts = 200
        st2.traces = [tr1, tr2]
        st2 = merge_gcf_stream(st2)
        assert len(st1) == len(st2)
        assert st2[0].stats.starttime == \
            UTCDateTime('2016-06-03T19:55:00.000000Z')
        assert st2[0].stats.endtime == \
            UTCDateTime('2016-06-03T19:55:02.990000Z')
        assert st2[0].stats.npts == 300
        assert round(abs(st2[0].stats.sampling_rate-100.0), 7) == 0
        assert st2[0].stats.channel == 'HNN'
        assert st2[0].stats.station == '6018'

    def test_sps_d(self, testdata):
        """
        Read files via obspy.core.stream.read function.
        """
        filename = testdata['20160603_1910n.gcf']
        # 1
        st = read(filename, headonly=True, channel_prefix="HN")
        assert len(st) == 1
        assert st[0].stats.starttime == \
            UTCDateTime('2016-06-03T19:10:00.000000Z')
        assert st[0].stats.endtime == \
            UTCDateTime('2016-06-03T19:10:01.998000Z')
        assert st[0].stats.npts == 1000
        assert round(abs(st[0].stats.sampling_rate-500.0), 7) == 0
        assert st[0].stats.channel == 'HNN'
        assert st[0].stats.station == '6018'

    def test_read_no_merge(self, testdata):
        """
        test preserving individual blocks in file as is, i.e. do not
        merge traces
        """
        filename = testdata['20160603_1910n.gcf']
        st = read(filename, blockmerge=False, channel_prefix="HN")
        assert len(st) == 2
        # 1
        assert st[0].stats.starttime == \
            UTCDateTime('2016-06-03T19:10:00.000000Z')
        assert st[0].stats.endtime == \
            UTCDateTime('2016-06-03T19:10:00.998000Z')
        assert st[0].stats.npts == 500
        assert round(abs(st[0].stats.sampling_rate-500.0), 7) == 0
        assert st[0].stats.channel == 'HNN'
        assert st[0].stats.station == '6018'
        assert st[0].stats.gcf.FIC == -49345
        assert st[0].stats.gcf.RIC == -49952
        # 2
        assert st[1].stats.starttime == \
            UTCDateTime('2016-06-03T19:10:01.000000Z')
        assert st[1].stats.endtime == \
            UTCDateTime('2016-06-03T19:10:01.998000Z')
        assert st[1].stats.npts == 500
        assert round(abs(st[1].stats.sampling_rate-500.0), 7) == 0
        assert st[1].stats.channel == 'HNN'
        assert st[1].stats.station == '6018'
        assert st[1].stats.gcf.FIC == -49519
        assert st[1].stats.gcf.RIC == -49625

    def test_write_read_fractional_start(self):
        """
        writes and (if succsesfull) re-reads a file with fractional
        (non-integer) start time
        """
        # 1
        # set up a stream object with supported proper non-integer start time
        sps = 1000
        duration = 60
        sysType = 1
        nsamples = int(sps*duration)
        data = np.random.randint(-3600, 3600, nsamples, dtype=np.int32)
        gcf_stat = AttribDict({
                    "system_id": ("ABCDZ2")[:6-sysType],
                    "stream_id": 'XXXXZ2',
                    "sys_type": sysType,
                    "t_leap": 0,
                    "gain": -1 if sysType == 0 else 2,
                    "digi": 0,
                    "ttl": 27,
                    "blk": 0,
                    "FIC": data[0],
                    "RIC": data[-1],
                    "stat": 0
            })
        stats = {
                "network": "XY",
                "station": "ABCD",
                "channel": "HHZ",
                "sampling_rate": sps,
                "starttime": UTCDateTime("20220301000000.250"),
                "gcf": gcf_stat
            }
        out_stream = Stream(traces=Trace(data, header=stats))

        # Write to temporary file, then re-read
        with NamedTemporaryFile() as tf:
            filename = tf.name
            _write_gcf(out_stream, filename)

            # Read temporary file
            in_stream = _read_gcf(filename, network='XY',
                                  station="ABCD", errorret=True)
        # compare
        assert out_stream == in_stream

        # 2
        # adjust start time in stream object to be miss-aligned more
        #  than set tolerance
        stats["starttime"] = UTCDateTime("20220301000000.25016")
        out_stream = Stream(traces=Trace(data, header=stats))

        # Try to write to temporary file, this should fail
        with NamedTemporaryFile() as tf:
            filename = tf.name
            # should raise a ValueError, otherwise write_gcf erroneously passed
            # a miss-aligned start time above set tolerance
            with pytest.raises(ValueError):
                _write_gcf(out_stream, filename, misalign=0.15)

        # 2
        # adjust start time in stream object to be miss-aligned within
        # set tolerance
        stats["starttime"] = UTCDateTime("20220301000000.25015")
        out_stream = Stream(traces=Trace(data, header=stats))

        # Try to write to temporary file, this should fail
        with NamedTemporaryFile() as tf:
            filename = tf.name
            _write_gcf(out_stream, filename, misalign=0.15)

    def test_write_read(self):
        """
        Writes a file then re-reads it and compares
        """
        # Tests, write and read a series of 60 sec gcf files
        for sysType in [0, 1, 2]:
            for sps in [5000, 2000, 625, 250, 134, 100, 1, 0.1]:
                # Set up a Stream object
                duration = 60
                nsamples = int(sps*duration)
                data = np.random.randint(-3600, 3600, nsamples, dtype=np.int32)
                gcf_stat = AttribDict({
                         "system_id": ("ABCDZ2")[:6-sysType],
                         "stream_id": 'XXXXZ2',
                         "sys_type": sysType,
                         "t_leap": 0,
                         "gain": -1 if sysType == 0 else 2,
                         "digi": 0,
                         "ttl": 27,
                         "blk": 0,
                         "FIC": data[0],
                         "RIC": data[-1],
                         "stat": 0
                  })
                stats = {
                     "network": "XY",
                     "station": "ABCD",
                     "channel": "HHZ",
                     "sampling_rate": sps,
                     "starttime": UTCDateTime("20220301000000"),
                     "gcf": gcf_stat
                  }
                out_stream = Stream(traces=Trace(data, header=stats))

                # Write to temporary file, then re-read
                with NamedTemporaryFile() as tf:
                    filename = tf.name
                    _write_gcf(out_stream, filename)

                    # Read temporary file
                    in_stream = _read_gcf(filename, network='XY',
                                          station="ABCD", errorret=True)

                # compare
                assert out_stream == in_stream

    def test_split_last_block(self):
        """
        test writing a file where number of data would fit into a
        single block but number of data will not be properly
        set due to high compression level
        """
        head = {'starttime': UTCDateTime("2020-10-01 00:00:00"),
                'sampling_rate': 100,
                'station': 'XXXX',
                'channel': 'HHZ',
                'gcf': AttribDict({
                    'system_id': 'XXXXZ0',
                    'stream_id': 'XXXXZ0',
                    'sys_type': 0,
                    't_leap': False,
                    'gain': -1,
                    'digi': 0,
                    'ttl': 0,
                    'blk': 0,
                    'FIC': 0,
                    'RIC': 718,
                    'stat': 0})}
        data = np.arange(719)
        tr = Trace(data=data, header=head)
        with NamedTemporaryFile() as tf:
            filename = tf.name
            _write_gcf(tr, filename)
            st = _read_gcf(filename)
        assert tr == st[0]
