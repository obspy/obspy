#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The gse2.core test suite.
"""
import copy
import warnings

import numpy as np

from obspy import Stream, Trace, UTCDateTime, read
from obspy.core.util import NamedTemporaryFile
from obspy.io.gse2.libgse2 import ChksumError
import pytest


class TestCore():
    """
    Test cases for libgse2 core interface
    """
    def test_read_via_obspy(self, testdata):
        """
        Read files via L{obspy.Trace}
        """
        gse2file = testdata['loc_RJOB20050831023349.z']
        testdata = [12, -10, 16, 33, 9, 26, 16, 7, 17, 6, 1, 3, -2]
        # read
        st = read(gse2file, verify_checksum=True)
        st.verify()
        tr = st[0]
        assert tr.stats['station'] == 'RJOB'
        assert tr.stats.npts == 12000
        assert tr.stats['sampling_rate'] == 200
        assert tr.stats.get('channel') == 'Z'
        assert round(abs(tr.stats.get('calib')-9.49e-02), 7) == 0
        assert tr.stats.gse2.get('vang') == -1.0
        assert tr.stats.gse2.get('hang') == -1.0
        assert tr.stats.gse2.get('calper') == 1.0
        assert tr.stats.gse2.get('instype') == ''
        assert round(abs(tr.stats.starttime.timestamp-1125455629.850), 6) == 0
        assert tr.data[0:13].tolist() == testdata

    def test_read_head_via_obspy(self, testdata):
        """
        Read header of files via L{obspy.Trace}
        """
        gse2file = testdata['loc_RJOB20050831023349.z']
        # read
        st = read(gse2file, headonly=True)
        tr = st[0]
        assert tr.stats['station'] == 'RJOB'
        assert tr.stats.npts == 12000
        assert tr.stats['sampling_rate'] == 200
        assert tr.stats.get('channel') == 'Z'
        assert round(abs(tr.stats.get('calib')-9.49e-02), 7) == 0
        assert tr.stats.gse2.get('vang') == -1.0
        assert tr.stats.gse2.get('calper') == 1.0
        assert round(abs(tr.stats.starttime.timestamp-1125455629.850), 6) == 0
        assert str(tr.data) == '[]'

    def test_read_and_write_via_obspy(self, testdata):
        """
        Read and Write files via L{obspy.Trace}
        """
        gse2file = testdata['loc_RNON20040609200559.z']
        # read trace
        st1 = read(gse2file)
        st1.verify()
        tr1 = st1[0]
        # write comparison trace
        with NamedTemporaryFile() as tf:
            tempfile = tf.name
            st2 = Stream()
            st2.traces.append(Trace())
            tr2 = st2[0]
            tr2.data = copy.deepcopy(tr1.data)
            tr2.stats = copy.deepcopy(tr1.stats)
            # raises "UserWarning: Bad value in GSE2 header field"
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', UserWarning)
                st2.write(tempfile, format='GSE2')
            # read comparison trace
            st3 = read(tempfile)
        st3.verify()
        tr3 = st3[0]
        # check if equal
        assert tr3.stats['station'] == tr1.stats['station']
        assert tr3.stats.npts == tr1.stats.npts
        assert tr3.stats['sampling_rate'] == tr1.stats['sampling_rate']
        assert tr3.stats.get('channel') == tr1.stats.get('channel')
        assert tr3.stats.get('starttime') == tr1.stats.get('starttime')
        assert tr3.stats.get('calib') == tr1.stats.get('calib')
        assert tr3.stats.gse2.get('vang') == tr1.stats.gse2.get('vang')
        assert tr3.stats.gse2.get('calper') == tr1.stats.gse2.get('calper')
        np.testing.assert_equal(tr3.data, tr1.data)

    def test_read_and_write_streams_via_obspy(self, testdata):
        """
        Read and Write files containing multiple GSE2 parts via L{obspy.Trace}
        """
        files = [testdata['loc_RNON20040609200559.z'],
                 testdata['loc_RJOB20050831023349.z']]
        testdata = [12, -10, 16, 33, 9, 26, 16, 7, 17, 6, 1, 3, -2]
        # write test file containing multiple GSE2 parts
        with NamedTemporaryFile() as tf:
            for filename in files:
                with open(filename, 'rb') as f1:
                    # raises "UserWarning: Bad value in GSE2 header field"
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore', UserWarning)
                        tf.write(f1.read())
            tf.flush()
            st1 = read(tf.name)
        st1.verify()
        assert len(st1) == 2
        tr11 = st1[0]
        tr12 = st1[1]
        assert tr11.stats['station'] == 'RNON'
        assert tr12.stats['station'] == 'RJOB'
        assert tr12.data[0:13].tolist() == testdata
        # write and read
        with NamedTemporaryFile() as tf:
            tmpfile = tf.name
            # raises "UserWarning: Bad value in GSE2 header field"
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', UserWarning)
                st1.write(tmpfile, format='GSE2')
            st2 = read(tmpfile)
        st2.verify()
        assert len(st2) == 2
        tr21 = st1[0]
        tr22 = st1[1]
        assert tr21.stats['station'] == 'RNON'
        assert tr22.stats['station'] == 'RJOB'
        assert tr22.data[0:13].tolist() == testdata
        np.testing.assert_equal(tr21.data, tr11.data)
        np.testing.assert_equal(tr22.data, tr12.data)

    def test_write_integers_via_obspy(self):
        """
        Write file test via L{obspy.Trace}.
        """
        npts = 1000
        # data cloud of integers - float won't work!
        np.random.seed(815)  # make test reproducible
        data = np.random.randint(-1000, 1000, npts).astype(np.int32)
        stats = {'network': 'BW', 'station': 'TEST', 'location': '',
                 'channel': 'EHE', 'npts': npts, 'sampling_rate': 200.0}
        start = UTCDateTime(2000, 1, 1)
        stats['starttime'] = start
        tr = Trace(data=data, header=stats)
        st = Stream([tr])
        st.verify()
        # write
        with NamedTemporaryFile() as tf:
            tempfile = tf.name
            st.write(tempfile, format="GSE2")
            # read again
            stream = read(tempfile)
        stream.verify()
        np.testing.assert_equal(data, stream[0].data)
        # test default attributes
        assert 'CM6' == stream[0].stats.gse2.datatype
        assert -1 == stream[0].stats.gse2.vang
        assert 1.0 == stream[0].stats.gse2.calper
        assert 1.0 == stream[0].stats.calib

    def test_tab_complete_stats(self, testdata):
        """
        Read files via L{obspy.Trace}
        """
        gse2file = testdata['loc_RJOB20050831023349.z']
        # read
        tr = read(gse2file)[0]
        assert 'station' in dir(tr.stats)
        assert 'npts' in dir(tr.stats)
        assert 'sampling_rate' in dir(tr.stats)
        assert tr.stats['station'] == 'RJOB'
        assert tr.stats.npts == 12000
        assert tr.stats['sampling_rate'] == 200

    def test_write_wrong_format(self):
        """
        Write floating point encoded data
        """
        np.random.seed(815)
        st = Stream([Trace(data=np.random.randn(1000))])
        with NamedTemporaryFile() as tf:
            tmpfile = tf.name
            with pytest.raises(Exception):
                st.write(tmpfile, format="GSE2")

    def test_read_with_wrong_checksum(self, testdata):
        """
        Test if additional kwarg verify_chksum can be given
        """
        # read original file
        gse2file = testdata['loc_RJOB20050831023349.z.wrong_chksum']
        # should not fail
        read(gse2file, verify_chksum=False)
        # should fail
        with pytest.raises(ChksumError):
            read(gse2file, verify_chksum=True)

    def test_read_with_wrong_parameters(self, testdata):
        """
        Test if additional kwargs can be given
        """
        # read original file
        gse2file = testdata['loc_RJOB20050831023349.z.wrong_chksum']
        # add wrong starttime flag of mseed, should also not fail
        read(gse2file, verify_chksum=False, starttime=None)

    def test_read_gse1_via_obspy(self, testdata):
        """
        Read files via L{obspy.Trace}
        """
        gse1file = testdata['loc_STAU20031119011659.z']
        testdata = [-818, -814, -798, -844, -806, -818, -800, -790, -818, -780]
        # read
        st = read(gse1file, verify_checksum=True)
        st.verify()
        tr = st[0]
        assert tr.stats['station'] == 'LE0083'
        assert tr.stats.npts == 3000
        assert round(abs(tr.stats['sampling_rate']-124.9999924), 7) == 0
        assert tr.stats.get('channel') == '  Z'
        assert round(abs(tr.stats.get('calib')-16.0000001), 7) == 0
        assert str(tr.stats.starttime) == '2003-11-19T01:16:59.990000Z'
        assert tr.data[0:10].tolist() == testdata

    def test_read_gse1_head_via_obspy(self, testdata):
        """
        Read header via L{obspy.Trace}
        """
        gse1file = testdata['loc_STAU20031119011659.z']
        # read
        st = read(gse1file, headonly=True)
        tr = st[0]
        assert tr.stats['station'] == 'LE0083'
        assert tr.stats.npts == 3000
        assert round(abs(tr.stats['sampling_rate']-124.9999924), 7) == 0
        assert tr.stats.get('channel') == '  Z'
        assert round(abs(tr.stats.get('calib')-16.0000001), 7) == 0
        assert str(tr.stats.starttime) == '2003-11-19T01:16:59.990000Z'

    def test_read_intv_gse1_via_obspy(self, testdata):
        """
        Read file via L{obspy.Trace}
        """
        gse1file = testdata['GRF_031102_0225.GSE.wrong_chksum']
        data1 = [-334, -302, -291, -286, -266, -252, -240, -214]
        data2 = [-468, -480, -458, -481, -481, -435, -432, -389]

        # verify checksum fails
        with pytest.raises(ChksumError):
            read(gse1file, verify_chksum=True)
        # reading header only
        st = read(gse1file, headonly=True)
        assert len(st) == 2
        # reading without checksum verification
        st = read(gse1file, verify_chksum=False)
        st.verify()
        # first trace
        assert len(st) == 2
        assert st[0].stats['station'] == 'GRA1'
        assert st[0].stats.npts == 6000
        assert round(abs(st[0].stats['sampling_rate']-19.9999997), 7) == 0
        assert st[0].stats.get('channel') == ' BZ'
        assert round(abs(st[0].stats.get('calib')-0.9900001), 7) == 0
        assert st[0].stats.starttime == \
            UTCDateTime('2003-11-02T02:25:00.000000Z')
        # second trace
        assert len(st) == 2
        assert st[1].stats['station'] == 'GRA1'
        assert st[1].stats.npts == 6000
        assert round(abs(st[1].stats['sampling_rate']-19.9999997), 7) == 0
        assert st[1].stats.get('channel') == ' BN'
        assert round(abs(st[1].stats.get('calib')-0.9200001), 7) == 0
        assert st[1].stats.starttime == \
            UTCDateTime('2003-11-02T02:25:00.000000Z')
        # check first 8 samples
        assert st[0].data[0:8].tolist() == data1
        # check last 8 samples
        assert st[1].data[-8:].tolist() == data2

    def test_read_dos(self, testdata):
        """
        Read file with dos newlines / encoding, that is
        Line Feed (LF) and Carriage Return (CR)
        see #355
        """
        filedos = testdata['loc_RJOB20050831023349_first100_dos.z']
        fileunix = testdata['loc_RJOB20050831023349.z']
        st = read(filedos, verify_chksum=True)
        st2 = read(fileunix, verify_chksum=True)
        np.testing.assert_equal(st[0].data, st2[0].data[:100])
        assert st[0].stats['station'] == 'RJOB'

    def test_read_apply_calib(self, testdata):
        """
        Tests apply_calib parameter in read method.
        """
        gse2file = testdata['loc_RJOB20050831023349.z']
        testdata = [12, -10, 16, 33, 9, 26, 16, 7, 17, 6, 1, 3, -2]
        # read w/ apply_calib = False
        st = read(gse2file, apply_calib=False)
        tr = st[0]
        assert tr.data[0:13].tolist() == testdata
        # read w/ apply_calib = True
        st = read(gse2file, apply_calib=True)
        tr = st[0]
        testdata = [n * tr.stats.calib for n in testdata]
        assert tr.data[0:13].tolist() == testdata

    def test_write_and_read_correct_network(self):
        """
        Tests that writing and reading the STA2 line works (otherwise the
        network code of the data is missing), even if some details like e.g.
        latitude are not present.
        """
        tr = Trace(np.arange(5, dtype=np.int32))
        tr.stats.network = "BW"
        with NamedTemporaryFile() as tf:
            tmpfile = tf.name
            tr.write(tmpfile, format='GSE2')
            tr = read(tmpfile)[0]
        assert tr.stats.network == "BW"

    def test_read_gse2_int_datatype(self, testdata):
        """
        Test reading of GSE2 files with data type INT.
        """
        gse2file = testdata['boa___00_07a.gse']
        testdata = [-4, -4, 1, 3, 2, -3, -6, -4, 2, 5]
        # read
        st = read(gse2file, verify_checksum=True)
        st.verify()
        tr = st[0]
        assert tr.stats['station'] == 'BBOA'
        assert tr.stats.npts == 6784
        assert round(abs(tr.stats['sampling_rate']-50.0), 7) == 0
        assert tr.stats.get('channel') == 'CPZ'
        assert round(abs(tr.stats.get('calib')-0.313), 7) == 0
        assert str(tr.stats.starttime) == '1990-04-07T00:07:33.000000Z'
        assert tr.data[0:10].tolist() == testdata
