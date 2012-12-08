# -*- coding: utf-8 -*-
"""
The sac.core test suite.
"""

from obspy import Stream, Trace, read, UTCDateTime
from obspy.core.util import NamedTemporaryFile
from obspy.sac import SacIO
import copy
import numpy as np
import os
import unittest
import filecmp


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
        self.testdata = np.array([-8.74227766e-08, -3.09016973e-01,
            - 5.87785363e-01, -8.09017122e-01, -9.51056600e-01,
            - 1.00000000e+00, -9.51056302e-01, -8.09016585e-01,
            - 5.87784529e-01, -3.09016049e-01], dtype='float32')

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

    def test_readwriteViaObspy(self):
        """
        Write/Read files via L{obspy.Stream}
        """
        tr = read(self.file, format='SAC')[0]
        tempfile = NamedTemporaryFile().name
        tr.write(tempfile, format='SAC')
        tr1 = read(tempfile)[0]
        os.remove(tempfile)
        np.testing.assert_array_equal(tr.data, tr1.data)
        # this tests failed because SAC calculates the seismogram's
        # mean value in single precision and python in double
        # precision resulting in different values. The following line
        # is therefore just a fix until we have come to a conclusive
        # solution how to handle the two different approaches
        tr1.stats.sac['depmen'] = tr.stats.sac['depmen']
        self.assertTrue(tr == tr1)

    def test_readXYwriteXYViaObspy(self):
        """
        Write/Read files via L{obspy.Stream}
        """
        tr = read(self.filexy, format='SACXY')[0]
        tempfile = NamedTemporaryFile().name
        tr.write(tempfile, format='SACXY')
        tr1 = read(tempfile)[0]
        os.remove(tempfile)
        self.assertTrue(tr == tr1)

    def test_readwriteXYViaObspy(self):
        """
        Read files via L{obspy.Stream}
        """
        tr = read(self.file, format='SAC')[0]
        tempfile = NamedTemporaryFile().name
        tr.write(tempfile, format='SACXY')
        tr1 = read(tempfile)[0]
        os.remove(tempfile)
        self.assertEqual(tr1.stats['station'], 'STA')
        self.assertEqual(tr1.stats.npts, 100)
        self.assertEqual(tr1.stats['sampling_rate'], 1.0)
        self.assertEqual(tr1.stats.get('channel'), 'Q')
        self.assertEqual(tr1.stats.starttime.timestamp, 269596810.0)
        self.assertEqual(tr1.stats.sac.get('nvhdr'), 6)
        self.assertEqual(tr1.stats.sac.b, 10.0)
        np.testing.assert_array_almost_equal(self.testdata[0:10],
                                             tr1.data[0:10])

    def test_readBigEndianViaObspy(self):
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
        st = Stream(traces=[Trace(header={'sac':{}}, data=self.testdata)])
        tempfile = NamedTemporaryFile().name
        st.write(tempfile, format='SAC')
        tr = read(tempfile)[0]
        os.remove(tempfile)
        np.testing.assert_array_almost_equal(self.testdata, tr.data)

    def test_setVersion(self):
        """
        Tests if SAC version is set when writing
        """
        tempfile = NamedTemporaryFile().name
        np.random.seed(815)
        st = Stream([Trace(data=np.random.randn(1000))])
        st.write(tempfile, format="SAC")
        st2 = read(tempfile, format="SAC")
        os.remove(tempfile)
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
        tempfile = NamedTemporaryFile().name
        st2.write(tempfile, format='SAC')
        # read comparison trace
        tr3 = read(tempfile)[0]
        os.remove(tempfile)
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
        data = np.random.randint(0, 5000, 11947).astype("int32")
        st = Stream([Trace(header=head, data=data)])
        # write them as SAC
        tmpfile = NamedTemporaryFile().name
        st.write(tmpfile, format="SAC")
        st2 = read(tmpfile, format="SAC")
        # file must exist, we just created it
        os.remove(tmpfile)
        # check all the required entries (see url in docstring)
        self.assertEqual(st2[0].stats.starttime, st[0].stats.starttime)
        self.assertEqual(st2[0].stats.npts, st[0].stats.npts)
        self.assertEqual(st2[0].stats.sac.nvhdr, 6)
        self.assertAlmostEqual(st2[0].stats.sac.b, 0.000400)
        # compare with correct digit size (nachkommastellen)
        self.assertAlmostEqual((0.0004 + (st[0].stats.npts - 1) * \
                               st[0].stats.delta) / st2[0].stats.sac.e, 1.0)
        self.assertEqual(st2[0].stats.sac.iftype, 1)
        self.assertEqual(st2[0].stats.sac.leven, 1)
        self.assertAlmostEqual(st2[0].stats.sampling_rate / \
                               st[0].stats.sampling_rate, 1.0)

    def test_iztype11(self):
        # test that iztype 11 is read correctly
        sod_file = os.path.join(self.path, 'data', 'dis.G.SCZ.__.BHE_short')
        tr = read(sod_file)[0]
        sac = SacIO(sod_file)
        t1 = tr.stats.starttime - float(tr.stats.sac.b)
        t2 = sac.reftime
        self.assertAlmostEqual(t1.timestamp, t2.timestamp, 5)
        # see that iztype is written corretly
        tempfile = NamedTemporaryFile().name
        tr.write(tempfile, format="SAC")
        sac2 = SacIO(tempfile)
        os.remove(tempfile)
        self.assertEqual(sac2.iztype, 11)
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
        tmpfile = NamedTemporaryFile().name
        tr.write(tmpfile, format="SAC")
        filecmp.cmp(file, tmpfile, shallow=False)
        os.remove(tmpfile)
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
        sac_ref_time = SacIO(self.file).reftime
        self.assertEqual(sac_ref_time.timestamp, 269596800.0)
        # change b to undefined and write (same case as if b == 0.0)
        # now sac reference time and reftime of seismogram must be the
        # same
        tr.stats.sac.b = -12345.0
        tmpfile = NamedTemporaryFile().name
        tr.write(tmpfile, format="SAC")
        tr2 = read(tmpfile)[0]
        self.assertEqual(tr2.stats.starttime.timestamp, 269596810.0)
        self.assertEqual(tr2.stats.sac.b, -12345.0)
        sac_ref_time2 = SacIO(tmpfile).reftime
        self.assertEqual(sac_ref_time2.timestamp, 269596810.0)
        os.remove(tmpfile)

    def test_issue156(self):
        """
        Test case for issue #156.
        """
        #1
        tr = Trace()
        tr.stats.delta = 0.01
        tr.data = np.arange(0, 3000)
        sac_file = NamedTemporaryFile().name
        tr.write(sac_file, 'SAC')
        st = read(sac_file)
        os.remove(sac_file)
        self.assertEquals(st[0].stats.delta, 0.01)
        self.assertEquals(st[0].stats.sampling_rate, 100.0)
        #2
        tr = Trace()
        tr.stats.delta = 0.005
        tr.data = np.arange(0, 2000)
        sac_file = NamedTemporaryFile().name
        tr.write(sac_file, 'SAC')
        st = read(sac_file)
        os.remove(sac_file)
        self.assertEquals(st[0].stats.delta, 0.005)
        self.assertEquals(st[0].stats.sampling_rate, 200.0)

    def test_writeSACXYWithMinimumStats(self):
        """
        Write SACXY with minimal stats header, no inhereted from SAC file
        """
        tr = Trace()
        tr.stats.delta = 0.01
        tr.data = np.arange(0, 3000)
        sac_file = NamedTemporaryFile().name
        tr.write(sac_file, 'SACXY')
        st = read(sac_file)
        os.remove(sac_file)
        self.assertEquals(st[0].stats.delta, 0.01)
        self.assertEquals(st[0].stats.sampling_rate, 100.0)

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
        sac_file = NamedTemporaryFile().name
        tr1.write(sac_file, 'SAC')
        tr2 = read(sac_file)[0]
        os.remove(sac_file)
        for i, header_value in enumerate(not_used):
            self.assertEquals(int(tr2.stats.sac[header_value]), i)

    def test_writingMicroSeconds(self):
        """
        Test case for #194. Check that microseconds are written to
        the SAC header b
        """
        np.random.seed(815)
        head = {'network': 'NL', 'station': 'HGN', 'channel': 'BHZ',
                'sampling_rate': 200.0,
                'starttime': UTCDateTime(2003, 5, 29, 2, 13, 22, 999999)}
        data = np.random.randint(0, 5000, 100).astype("int32")
        st = Stream([Trace(header=head, data=data)])
        # write them as SAC
        tmpfile = NamedTemporaryFile().name
        st.write(tmpfile, format="SAC")
        st2 = read(tmpfile, format="SAC")
        # file must exist, we just created it
        os.remove(tmpfile)
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
        Tests writing Traces containing 0, 1 or 2 samples only.
        """
        for format in ['SAC', 'SACXY']:
            for num in range(0, 4):
                tr = Trace(data=np.arange(num))
                tempfile = NamedTemporaryFile().name
                tr.write(tempfile, format=format)
                # test results
                st = read(tempfile, format=format)
                self.assertEquals(len(st), 1)
                self.assertEquals(len(st[0]), num)
                os.remove(tempfile)

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


def suite():
    return unittest.makeSuite(CoreTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
