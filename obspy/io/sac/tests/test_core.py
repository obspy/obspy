# -*- coding: utf-8 -*-
"""
The sac.core test suite.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import copy
import io
import os
import unittest

import numpy as np

from obspy import Stream, Trace, UTCDateTime, read
from obspy.core.util import NamedTemporaryFile
from obspy.io.sac import SacError, SACTrace, SacIOError
from obspy.io.sac.core import (_is_sac, _is_sacXY, _read_sac, _read_sacXY,
                               _write_sac, _write_sacXY)


class CoreTestCase(unittest.TestCase):
    """
    Test cases for sac core interface
    """
    def setUp(self):
        # directory where the test files are located
        self.path = os.path.dirname(__file__)
        self.file = os.path.join(self.path, 'data', 'test.sac')
        self.filexy = os.path.join(self.path, 'data', 'testxy.sac')
        self.filebe = os.path.join(self.path, 'data', 'test.sac.swap')
        self.testdata = np.array(
            [-8.74227766e-08, -3.09016973e-01,
             -5.87785363e-01, -8.09017122e-01, -9.51056600e-01,
             -1.00000000e+00, -9.51056302e-01, -8.09016585e-01,
             -5.87784529e-01, -3.09016049e-01], dtype=np.float32)

    def test_readViaObsPy(self):
        """
        Read files via L{obspy.Stream}
        """
        tr = read(self.file, format='SAC')[0]
        self.assertEqual(tr.stats['station'], 'STA')
        self.assertEqual(tr.stats.npts, 100)
        self.assertEqual(tr.stats['sampling_rate'], 1.0)
        self.assertEqual(tr.stats.get('channel'), 'Q')
        self.assertEqual(tr.stats.starttime.timestamp, 269596810.0)
        self.assertEqual(tr.stats.sac.get('nvhdr'), 6)
        self.assertEqual(tr.stats.sac.b, 10.0)
        np.testing.assert_array_almost_equal(self.testdata[0:10],
                                             tr.data[0:10])

    def test_readwriteViaObsPy(self):
        """
        Write/Read files via L{obspy.Stream}
        """
        tr = read(self.file, format='SAC')[0]
        with NamedTemporaryFile() as tf:
            tempfile = tf.name
            tr.write(tempfile, format='SAC')
            tr1 = read(tempfile)[0]
        np.testing.assert_array_equal(tr.data, tr1.data)
        # this tests failed because SAC calculates the seismogram's
        # mean value in single precision and python in double
        # precision resulting in different values. The following line
        # is therefore just a fix until we have come to a conclusive
        # solution how to handle the two different approaches
        tr1.stats.sac['depmen'] = tr.stats.sac['depmen']
        self.assertEqual(tr, tr1)

    def test_readXYwriteXYViaObsPy(self):
        """
        Write/Read files via L{obspy.Stream}
        """
        tr = read(self.filexy, format='SACXY')[0]
        with NamedTemporaryFile() as tf:
            tempfile = tf.name
            tr.write(tempfile, format='SACXY')
            tr1 = read(tempfile)[0]
        self.assertEqual(tr, tr1)

    def test_readwriteXYViaObsPy(self):
        """
        Read files via L{obspy.Stream}
        """
        tr = read(self.file, format='SAC')[0]
        with NamedTemporaryFile() as tf:
            tempfile = tf.name
            tr.write(tempfile, format='SACXY')
            tr1 = read(tempfile)[0]
        self.assertEqual(tr1.stats['station'], 'STA')
        self.assertEqual(tr1.stats.npts, 100)
        self.assertEqual(tr1.stats['sampling_rate'], 1.0)
        self.assertEqual(tr1.stats.get('channel'), 'Q')
        self.assertEqual(tr1.stats.starttime.timestamp, 269596810.0)
        self.assertEqual(tr1.stats.sac.get('nvhdr'), 6)
        self.assertEqual(tr1.stats.sac.b, 10.0)
        np.testing.assert_array_almost_equal(self.testdata[0:10],
                                             tr1.data[0:10])

    def test_readBigEndianViaObsPy(self):
        """
        Read files via L{obspy.Stream}
        """
        tr = read(self.filebe, format='SAC')[0]
        self.assertEqual(tr.stats['station'], 'STA')
        self.assertEqual(tr.stats.npts, 100)
        self.assertEqual(tr.stats['sampling_rate'], 1.0)
        self.assertEqual(tr.stats.get('channel'), 'Q')
        self.assertEqual(tr.stats.starttime.timestamp, 269596810.0)
        self.assertEqual(tr.stats.sac.get('nvhdr'), 6)
        self.assertEqual(tr.stats.sac.b, 10.0)
        np.testing.assert_array_almost_equal(self.testdata[0:10],
                                             tr.data[0:10])

    def test_swapbytesViaObsPy(self):
        with NamedTemporaryFile() as tf:
            tempfile = tf.name
            trbe = read(self.filebe, format='SAC')[0]
            trbe.write(tempfile, format='SAC', byteorder='<')
            tr = read(tempfile, format='SAC')[0]
            trle = read(self.file, format='SAC')[0]
            self.assertEqual(tr.stats.station, trle.stats.station)
            self.assertEqual(tr.stats.npts, trle.stats.npts)
            self.assertEqual(tr.stats.delta, trle.stats.delta)
            self.assertEqual(tr.stats.sac.b, trle.stats.sac.b)
            np.testing.assert_array_almost_equal(tr.data[0:10],
                                                 trle.data[0:10])
        with NamedTemporaryFile() as tf:
            tempfile = tf.name
            trle = read(self.file, format='SAC')[0]
            trle.write(tempfile, format='SAC', byteorder='>')
            tr = read(tempfile, format='SAC')[0]
            trbe = read(self.filebe, format='SAC')[0]
            self.assertEqual(tr.stats.station, trbe.stats.station)
            self.assertEqual(tr.stats.npts, trbe.stats.npts)
            self.assertEqual(tr.stats.delta, trbe.stats.delta)
            self.assertEqual(tr.stats.sac.b, trbe.stats.sac.b)
            np.testing.assert_array_almost_equal(tr.data[0:10],
                                                 trbe.data[0:10])

    def test_readHeadViaObsPy(self):
        """
        Read files via L{obspy.Stream}
        """
        tr = read(self.file, format='SAC', headonly=True)[0]
        self.assertEqual(tr.stats['station'], 'STA')
        self.assertEqual(tr.stats.npts, 100)
        self.assertEqual(tr.stats['sampling_rate'], 1.0)
        self.assertEqual(tr.stats.get('channel'), 'Q')
        self.assertEqual(tr.stats.starttime.timestamp, 269596810.0)
        self.assertEqual(tr.stats.sac.get('nvhdr'), 6)
        self.assertEqual(tr.stats.sac.b, 10.0)
        self.assertEqual(str(tr.data), '[]')

    def test_writeViaObsPy(self):
        """
        Writing artificial files via L{obspy.Stream}
        """
        st = Stream(traces=[Trace(header={'sac': {}}, data=self.testdata)])
        with NamedTemporaryFile() as tf:
            tempfile = tf.name
            st.write(tempfile, format='SAC')
            tr = read(tempfile)[0]
        np.testing.assert_array_almost_equal(self.testdata, tr.data)

    def test_setVersion(self):
        """
        Tests if SAC version is set when writing
        """
        np.random.seed(815)
        st = Stream([Trace(data=np.random.randn(1000))])
        with NamedTemporaryFile() as tf:
            tempfile = tf.name
            st.write(tempfile, format="SAC")
            st2 = read(tempfile, format="SAC")
        self.assertEqual(st2[0].stats['sac'].nvhdr, 6)

    def test_readAndWriteViaObsPy(self):
        """
        Read and Write files via L{obspy.Stream}
        """
        # read trace
        tr = read(self.file)[0]
        # write comparison trace
        st2 = Stream()
        st2.traces.append(Trace())
        tr2 = st2[0]
        tr2.data = copy.deepcopy(tr.data)
        tr2.stats = copy.deepcopy(tr.stats)
        with NamedTemporaryFile() as tf:
            tempfile = tf.name
            st2.write(tempfile, format='SAC')
            # read comparison trace
            tr3 = read(tempfile)[0]
        # check if equal
        self.assertEqual(tr3.stats['station'], tr.stats['station'])
        self.assertEqual(tr3.stats.npts, tr.stats.npts)
        self.assertEqual(tr.stats['sampling_rate'], tr.stats['sampling_rate'])
        self.assertEqual(tr.stats.get('channel'), tr.stats.get('channel'))
        self.assertEqual(tr.stats.get('starttime'), tr.stats.get('starttime'))
        self.assertEqual(tr.stats.sac.get('nvhdr'), tr.stats.sac.get('nvhdr'))
        np.testing.assert_equal(tr.data, tr3.data)

    def test_convert2Sac(self):
        """
        Test that an obspy trace is correctly written to SAC.
        All the header variables which are tagged as required by
        http://www.iris.edu/manuals/sac/SAC_Manuals/FileFormatPt2.html
        are controlled in this test
        also see http://www.iris.edu/software/sac/manual/file_format.html
        """
        # setUp is called before every test, not only once at the
        # beginning, that is we allocate the data just here
        # generate artificial mseed data
        np.random.seed(815)
        head = {'network': 'NL', 'station': 'HGN', 'location': '00',
                'channel': 'BHZ', 'calib': 1.0, 'sampling_rate': 40.0,
                'starttime': UTCDateTime(2003, 5, 29, 2, 13, 22, 43400)}
        data = np.random.randint(0, 5000, 11947).astype(np.int32)
        st = Stream([Trace(header=head, data=data)])
        # write them as SAC
        with NamedTemporaryFile() as tf:
            tmpfile = tf.name
            st.write(tmpfile, format="SAC")
            st2 = read(tmpfile, format="SAC")
        # check all the required entries (see url in docstring)
        self.assertEqual(st2[0].stats.starttime, st[0].stats.starttime)
        self.assertEqual(st2[0].stats.npts, st[0].stats.npts)
        self.assertEqual(st2[0].stats.sac.nvhdr, 6)
        self.assertAlmostEqual(st2[0].stats.sac.b, 0.000400)
        # compare with correct digit size (nachkommastellen)
        self.assertAlmostEqual((0.0004 + (st[0].stats.npts - 1) *
                               st[0].stats.delta) / st2[0].stats.sac.e, 1.0)
        self.assertEqual(st2[0].stats.sac.iftype, 1)
        self.assertEqual(st2[0].stats.sac.leven, 1)
        self.assertAlmostEqual(st2[0].stats.sampling_rate /
                               st[0].stats.sampling_rate, 1.0)

    def test_iztype11(self):
        # test that iztype 11 is read correctly
        sod_file = os.path.join(self.path, 'data', 'dis.G.SCZ.__.BHE_short')
        tr = read(sod_file)[0]
        with open(sod_file, "rb") as fh:
            sac = SACTrace.read(fh)
        t1 = tr.stats.starttime - float(tr.stats.sac.b)
        t2 = sac.reftime
        self.assertAlmostEqual(t1.timestamp, t2.timestamp, 5)
        # see that iztype is written correctly
        with NamedTemporaryFile() as tf:
            tempfile = tf.name
            tr.write(tempfile, format="SAC")
            with open(tempfile, "rb") as fh:
                sac2 = SACTrace.read(fh)
        self.assertEqual(sac2._header['iztype'], 11)
        self.assertAlmostEqual(tr.stats.sac.b, sac2.b)
        self.assertAlmostEqual(t2.timestamp, sac2.reftime.timestamp, 5)

    def test_defaultValues(self):
        tr = read(self.file)[0]
        self.assertEqual(tr.stats.calib, 1.0)
        self.assertEqual(tr.stats.location, '')
        self.assertEqual(tr.stats.network, '')

    def test_referenceTime(self):
        """
        Test case for bug #107. The SAC reference time is specified by the
        iztype. However is seems no matter what iztype is given, the
        starttime of the seismogram is calculated by adding the B header
        (in seconds) to the SAC reference time.
        """
        file = os.path.join(self.path, "data", "seism.sac")
        tr = read(file)[0]
        # see that starttime is set correctly (#107)
        self.assertAlmostEqual(tr.stats.sac.iztype, 9)
        self.assertAlmostEqual(tr.stats.sac.b, 9.4599991)
        self.assertEqual(tr.stats.starttime,
                         UTCDateTime("1981-03-29 10:38:23.459999"))
        # check that if we rewrite the file, nothing changed
        with NamedTemporaryFile() as tf:
            tmpfile = tf.name
            tr.write(tmpfile, format="SAC")
            tr2 = read(tmpfile)[0]
            self.assertEqual(tr.stats.station, tr2.stats.station)
            self.assertEqual(tr.stats.npts, tr2.stats.npts)
            self.assertEqual(tr.stats.delta, tr2.stats.delta)
            self.assertEqual(tr.stats.starttime, tr2.stats.starttime)
            self.assertEqual(tr.stats.sac.b, tr2.stats.sac.b)
            np.testing.assert_array_equal(tr.data, tr2.data)
        # test some more entries, I can see from the plot
        self.assertEqual(tr.stats.station, "CDV")
        self.assertEqual(tr.stats.channel, "Q")

    def test_undefinedB(self):
        """
        Test that an undefined B value (-12345.0) is not messing up the
        starttime
        """
        # read in the test file an see that sac reference time and
        # starttime of seismogram are correct
        tr = read(self.file)[0]
        self.assertEqual(tr.stats.starttime.timestamp, 269596810.0)
        self.assertEqual(tr.stats.sac.b, 10.0)
        with open(self.file, 'rb') as fh:
            sac_ref_time = SACTrace.read(fh).reftime
        self.assertEqual(sac_ref_time.timestamp, 269596800.0)
        # change b to undefined and write (same case as if b == 0.0)
        # now sac reference time and reftime of seismogram must be the
        # same
        tr.stats.sac.b = -12345.0
        with NamedTemporaryFile() as tf:
            tmpfile = tf.name
            tr.write(tmpfile, format="SAC")
            tr2 = read(tmpfile)[0]
            self.assertEqual(tr2.stats.starttime.timestamp, 269596810.0)
            self.assertEqual(tr2.stats.sac.b, -12345.0)
            with open(tmpfile, "rb") as fh:
                sac_ref_time2 = SACTrace.read(fh).reftime
        self.assertEqual(sac_ref_time2.timestamp, 269596810.0)

    def test_issue156(self):
        """
        Test case for issue #156.
        """
        # 1
        tr = Trace()
        tr.stats.delta = 0.01
        tr.data = np.arange(0, 3000)
        with NamedTemporaryFile() as tf:
            sac_file = tf.name
            tr.write(sac_file, 'SAC')
            st = read(sac_file)
        self.assertEqual(st[0].stats.delta, 0.01)
        self.assertEqual(st[0].stats.sampling_rate, 100.0)
        # 2
        tr = Trace()
        tr.stats.delta = 0.005
        tr.data = np.arange(0, 2000)
        with NamedTemporaryFile() as tf:
            sac_file = tf.name
            tr.write(sac_file, 'SAC')
            st = read(sac_file)
        self.assertEqual(st[0].stats.delta, 0.005)
        self.assertEqual(st[0].stats.sampling_rate, 200.0)

    def test_write_sacXYWithMinimumStats(self):
        """
        Write SACXY with minimal stats header, no inhereted from SAC file
        """
        tr = Trace()
        tr.stats.delta = 0.01
        tr.data = np.arange(0, 3000)
        with NamedTemporaryFile() as tf:
            sac_file = tf.name
            tr.write(sac_file, 'SACXY')
            st = read(sac_file)
        self.assertEqual(st[0].stats.delta, 0.01)
        self.assertEqual(st[0].stats.sampling_rate, 100.0)

    def test_notUsedButGivenHeaders(self):
        """
        Test case for #188
        """
        tr1 = read(self.file)[0]
        not_used = ['xminimum', 'xmaximum', 'yminimum', 'ymaximum',
                    'unused6', 'unused7', 'unused8', 'unused9', 'unused10',
                    'unused11', 'unused12']
        for i, header_value in enumerate(not_used):
            tr1.stats.sac[header_value] = i
        with NamedTemporaryFile() as tf:
            sac_file = tf.name
            tr1.write(sac_file, 'SAC')
            tr2 = read(sac_file)[0]
        for i, header_value in enumerate(not_used):
            self.assertEqual(int(tr2.stats.sac[header_value]), i)

    def test_writingMicroSeconds(self):
        """
        Test case for #194. Check that microseconds are written to
        the SAC header b
        """
        np.random.seed(815)
        head = {'network': 'NL', 'station': 'HGN', 'channel': 'BHZ',
                'sampling_rate': 200.0,
                'starttime': UTCDateTime(2003, 5, 29, 2, 13, 22, 999999)}
        data = np.random.randint(0, 5000, 100).astype(np.int32)
        st = Stream([Trace(header=head, data=data)])
        # write them as SAC
        with NamedTemporaryFile() as tf:
            tmpfile = tf.name
            st.write(tmpfile, format="SAC")
            st2 = read(tmpfile, format="SAC")
        # check all the required entries (see url in docstring)
        self.assertEqual(st2[0].stats.starttime, st[0].stats.starttime)
        self.assertAlmostEqual(st2[0].stats.sac.b, 0.000999)

    def test_nullTerminatedStrings(self):
        """
        Test case for #374. Check that strings stop at the position
        of null termination '\x00'
        """
        null_file = os.path.join(self.path, 'data', 'null_terminated.sac')
        tr = read(null_file)[0]
        self.assertEqual(tr.stats.station, 'PIN1')
        self.assertEqual(tr.stats.network, 'GD')
        self.assertEqual(tr.stats.channel, 'LYE')

    def test_writeSmallTrace(self):
        """
        Tests writing Traces containing 0, 1, 2, 3, 4 samples only.
        """
        for format in ['SAC', 'SACXY']:
            for num in range(5):
                tr = Trace(data=np.arange(num))
                with NamedTemporaryFile() as tf:
                    tempfile = tf.name
                    tr.write(tempfile, format=format)
                    # test results
                    st = read(tempfile, format=format)
                self.assertEqual(len(st), 1)
                np.testing.assert_array_equal(tr.data, st[0].data)

    def test_issue390(self):
        """
        Read all SAC headers if debug_headers flag is enabled.
        """
        # 1 - binary SAC
        tr = read(self.file, headonly=True, debug_headers=True)[0]
        self.assertEqual(tr.stats.sac.nzyear, 1978)
        self.assertEqual(tr.stats.sac.nzjday, 199)
        self.assertEqual(tr.stats.sac.nzhour, 8)
        self.assertEqual(tr.stats.sac.nzmin, 0)
        self.assertEqual(tr.stats.sac.nzsec, 0)
        self.assertEqual(tr.stats.sac.nzmsec, 0)
        self.assertEqual(tr.stats.sac.delta, 1.0)
        self.assertEqual(tr.stats.sac.scale, -12345.0)
        self.assertEqual(tr.stats.sac.npts, 100)
        self.assertEqual(tr.stats.sac.knetwk, '-12345  ')
        self.assertEqual(tr.stats.sac.kstnm, 'STA     ')
        self.assertEqual(tr.stats.sac.kcmpnm, 'Q       ')
        # 2 - ASCII SAC
        tr = read(self.filexy, headonly=True, debug_headers=True)[0]
        self.assertEqual(tr.stats.sac.nzyear, -12345)
        self.assertEqual(tr.stats.sac.nzjday, -12345)
        self.assertEqual(tr.stats.sac.nzhour, -12345)
        self.assertEqual(tr.stats.sac.nzmin, -12345)
        self.assertEqual(tr.stats.sac.nzsec, -12345)
        self.assertEqual(tr.stats.sac.nzmsec, -12345)
        self.assertEqual(tr.stats.sac.delta, 1.0)
        self.assertEqual(tr.stats.sac.scale, -12345.0)
        self.assertEqual(tr.stats.sac.npts, 100)
        self.assertEqual(tr.stats.sac.knetwk, '-12345  ')
        self.assertEqual(tr.stats.sac.kstnm, 'sta     ')
        self.assertEqual(tr.stats.sac.kcmpnm, 'Q       ')

    def test_read_with_fsize(self):
        """
        Testing fsize option on read()
        """
        # reading sac file with wrong file size should raise error
        longer_file = os.path.join(self.path, 'data', 'seism-longer.sac')
        shorter_file = os.path.join(self.path, 'data', 'seism-shorter.sac')
        # default
        self.assertRaises(SacError, read, longer_file)
        self.assertRaises(SacError, read, shorter_file)
        # fsize=True
        self.assertRaises(SacError, read, longer_file, fsize=True)
        self.assertRaises(SacError, read, shorter_file, fsize=True)
        # using fsize=False should not work for shorter file
        # (this is not supported by SAC) ...
        self.assertRaises(SacIOError, read, shorter_file, fsize=False)
        # ...but it should work for longer file
        tr = read(longer_file, fsize=False, debug_headers=True)[0]
        # checking trace
        self.assertEqual(tr.stats.sac.nzyear, 1981)
        self.assertEqual(tr.stats.sac.nzjday, 88)
        self.assertEqual(tr.stats.sac.nzhour, 10)
        self.assertEqual(tr.stats.sac.nzmin, 38)
        self.assertEqual(tr.stats.sac.nzsec, 14)
        self.assertEqual(tr.stats.sac.nzmsec, 0)
        # we should never test equality for float values:
        self.assertLessEqual(abs(tr.stats.sac.delta - 0.01), 1e-9)
        self.assertEqual(tr.stats.sac.scale, -12345.0)
        self.assertEqual(tr.stats.sac.npts, 998)
        self.assertEqual(tr.stats.sac.knetwk, '-12345  ')
        self.assertEqual(tr.stats.sac.kstnm, 'CDV     ')
        self.assertEqual(tr.stats.sac.kcmpnm, 'Q       ')

    def test_read_sac_from_bytes_io(self):
        """
        Tests reading from a BytesIO object.
        """
        with io.BytesIO() as buf:
            # Read file to BytesIO.
            with open(self.file, "rb") as fh:
                buf.write(fh.read())
            buf.seek(0, 0)

            # Attempt to read from it.
            tr = _read_sac(buf)[0]

        # Open file normally and make sure the results are identical.
        tr2 = _read_sac(self.file)[0]
        np.testing.assert_array_equal(tr.data, tr2.data)
        self.assertEqual(tr, tr2)

    def test_read_sac_from_open_file(self):
        """
        Tests reading from an open file.
        """
        with open(self.file, "rb") as fh:
            # Attempt to read from it.
            tr = _read_sac(fh)[0]

        # Open file normally and make sure the results are identical.
        tr2 = _read_sac(self.file)[0]
        np.testing.assert_array_equal(tr.data, tr2.data)
        self.assertEqual(tr, tr2)

    def test_read_write_bytes_io(self):
        """
        Tests reading and writing to and from BytesIO.
        """
        st = _read_sac(self.file)
        with io.BytesIO() as buf:
            _write_sac(st, buf)
            buf.seek(0, 0)
            # Attempt to read from it.
            st2 = _read_sac(buf)

        tr = st[0]
        tr2 = st2[0]
        # depmen is different as it is actually calculated on the fly.
        del tr.stats.sac.depmen
        del tr2.stats.sac.depmen
        np.testing.assert_array_equal(tr.data, tr2.data)
        self.assertEqual(tr, tr2)

    def test_read_write_open_file(self):
        st = _read_sac(self.file)

        with NamedTemporaryFile() as tf_out:
            _write_sac(st, tf_out)
            tf_out.seek(0, 0)
            st2 = _read_sac(tf_out)

        tr = st[0]
        tr2 = st2[0]
        # depmen is different as it is actually calculated on the fly.
        del tr.stats.sac.depmen
        del tr2.stats.sac.depmen
        np.testing.assert_array_equal(tr.data, tr2.data)
        self.assertEqual(tr, tr2)

    def test_writing_to_obj_with_multiple_traces_fails(self):
        """
        Writing to a buf with multiple trace objects should fail. The SAC
        format cannot deal with that.
        """
        st = read()
        with io.BytesIO() as fh:
            self.assertRaises(ValueError, st.write, fh, format="sac")

    def test_writing_to_io_string_io_fails(self):
        """
        Writing to io.StringIO should fail on all platforms.
        """
        st = read()[:1]
        with io.StringIO() as fh:
            self.assertRaises(ValueError, st.write, fh, format="sac")

    def test_read_via_obspy_from_bytes_io(self):
        """
        Read sac files from a BytesIO object via ObsPy.
        """
        with io.BytesIO() as buf:
            # Read file to BytesIO.
            with open(self.file, "rb") as fh:
                buf.write(fh.read())
            buf.seek(0, 0)

            # Attempt to read from it.
            tr = read(buf)[0]

        # Open file normally and make sure the results are identical.
        tr2 = read(self.file)[0]
        np.testing.assert_array_equal(tr.data, tr2.data)
        self.assertEqual(tr, tr2)

    def test_write_via_obspy_to_bytes_io(self):
        """
        Read sac files from a BytesIO object via ObsPy.
        """
        tr = read(self.file)[0]
        with io.BytesIO() as buf:
            tr.write(buf, format="sac")
            buf.seek(0, 0)
            # Attempt to read from it.
            tr2 = read(buf)[0]

        # depmen is different as it is actually calculated on the fly.
        del tr.stats.sac.depmen
        del tr2.stats.sac.depmen
        np.testing.assert_array_equal(tr.data, tr2.data)
        self.assertEqual(tr, tr2)

    def test_readXYwriteXY_from_BytesIO(self):
        """
        Reading/writing XY sac files from/to io.BytesIO. It's alphanumeric
        so bytes should also do the trick.
        """
        # Original.
        st = _read_sacXY(self.filexy)

        with io.BytesIO() as fh:
            _write_sacXY(st, fh)
            fh.seek(0, 0)
            st2 = _read_sacXY(fh)

        self.assertEqual(st, st2)

    def test_readXYwriteXY_from_open_file_binary_mode(self):
        """
        Reading/writing XY sac files to open files in binary mode.
        """
        # Original.
        st = _read_sacXY(self.filexy)

        with NamedTemporaryFile() as tf:
            _write_sacXY(st, tf)
            tf.seek(0, 0)
            st2 = _read_sacXY(tf)

        self.assertEqual(st, st2)

    def test_is_sac_bytes_io(self):
        """
        Tests the _is_sac function for BytesIO objects.
        """
        with io.BytesIO() as buf:
            # Read file to BytesIO.
            with open(self.file, "rb") as fh:
                buf.write(fh.read())
            buf.seek(0, 0)
            self.assertTrue(_is_sac(buf))

        # Should naturally fail for an XY file.
        with io.BytesIO() as buf:
            # Read file to BytesIO.
            with open(self.filexy, "rb") as fh:
                buf.write(fh.read())
            buf.seek(0, 0)
            self.assertFalse(_is_sac(buf))

    def test_is_sac_string_io_raises(self):
        """
        Should raise a ValueError.
        """
        with io.StringIO() as buf:
            buf.write("abcdefghijklmnopqrstuvwxyz")
            buf.seek(0, 0)
            self.assertRaises(ValueError, _is_sac, buf)

    def test_is_sac_open_file(self):
        """
        Tests the _is_sac function for open files.
        """
        with open(self.file, "rb") as fh:
            self.assertTrue(_is_sac(fh))

    def test_is_sacxy_bytes_io(self):
        """
        Tests the _is_sacXY function for BytesIO objects.
        """
        with io.BytesIO() as buf:
            # Read file to BytesIO.
            with open(self.filexy, "rb") as fh:
                buf.write(fh.read())
            buf.seek(0, 0)
            self.assertTrue(_is_sacXY(buf))

        # Should naturally fail for a normal sac file.
        with io.BytesIO() as buf:
            # Read file to BytesIO.
            with open(self.file, "rb") as fh:
                buf.write(fh.read())
            buf.seek(0, 0)
            self.assertFalse(_is_sacXY(buf))

    def test_is_sacxy_string_io_raises(self):
        """
        Tests the _is_sacXY function for StringIO objects where it should
        raise. I/O is binary only.
        """
        with io.StringIO() as buf:
            # Read file to BytesIO.
            with open(self.filexy, "rt") as fh:
                buf.write(fh.read())
            buf.seek(0, 0)
            self.assertRaises(ValueError, _is_sacXY, buf)

    def test_is_sacxy_open_file_binary_mode(self):
        """
        Tests the _is_sacXY function for open files in binary mode.
        """
        with open(self.filexy, "rb") as fh:
            self.assertTrue(_is_sacXY(fh))

        with open(__file__, "rb") as fh:
            self.assertFalse(_is_sacXY(fh))

    def test_is_sacxy_open_file_text_mode_fails(self):
        """
        Tests that the _is_sacXY function for open files in text mode fails.
        """
        with open(self.filexy, "rt") as fh:
            self.assertRaises(ValueError, _is_sacXY, fh)

    def test_writing_to_file_like_objects_with_obspy(self):
        """
        Very simple test just executing a common operation and making sure
        it does not fail.
        """
        st = read()[:1]

        with io.BytesIO() as fh:
            st.write(fh, format="sac")

        with io.BytesIO() as fh:
            st.write(fh, format="sacxy")

        # Will fail if the stream contains more than one trace.
        st = read()
        self.assertGreater(len(st), 1)
        with io.BytesIO() as fh:
            self.assertRaises(ValueError, st.write, fh, format="sac")

        with io.BytesIO() as fh:
            self.assertRaises(ValueError, st.write, fh, format="sacxy")


def suite():
    return unittest.makeSuite(CoreTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
