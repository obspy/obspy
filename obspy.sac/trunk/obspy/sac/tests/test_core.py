# -*- coding: utf-8 -*-
"""
The sac.core test suite.
"""

from obspy.core import Stream, Trace, read, AttribDict
from obspy.core import UTCDateTime
from obspy.core.util import NamedTemporaryFile
from obspy.sac import ReadSac
import copy
import inspect
import numpy as np
import os
import unittest


class CoreTestCase(unittest.TestCase):
    """
    Test cases for sac core interface
    """
    def setUp(self):
        # directory where the test files are located
        self.path = os.path.dirname(inspect.getsourcefile(self.__class__))
        self.file = os.path.join(self.path, 'data', 'test.sac')
        self.filebe = os.path.join(self.path, 'data', 'test.sac.swap')
        self.testdata = np.array([-8.74227766e-08, -3.09016973e-01,
            - 5.87785363e-01, -8.09017122e-01, -9.51056600e-01,
            - 1.00000000e+00, -9.51056302e-01, -8.09016585e-01,
            - 5.87784529e-01, -3.09016049e-01], dtype='float32')

    def test_readViaObspy(self):
        """
        Read files via L{obspy.Stream}
        """
        tr = read(self.file, format='SAC')[0]
        self.assertEqual(tr.stats['station'], 'STA')
        self.assertEqual(tr.stats.npts, 100)
        self.assertEqual(tr.stats['sampling_rate'], 1.0)
        self.assertEqual(tr.stats.get('channel'), 'Q')
        self.assertEqual(tr.stats.starttime.timestamp, 269596800.0)
        self.assertEqual(tr.stats.sac.get('nvhdr'), 6)
        np.testing.assert_array_almost_equal(self.testdata[0:10],
                                             tr.data[0:10])

    def test_readBigEndianViaObspy(self):
        """
        Read files via L{obspy.Stream}
        """
        tr = read(self.filebe, format='SAC')[0]
        self.assertEqual(tr.stats['station'], 'STA')
        self.assertEqual(tr.stats.npts, 100)
        self.assertEqual(tr.stats['sampling_rate'], 1.0)
        self.assertEqual(tr.stats.get('channel'), 'Q')
        self.assertEqual(tr.stats.starttime.timestamp, 269596800.0)
        self.assertEqual(tr.stats.sac.get('nvhdr'), 6)
        np.testing.assert_array_almost_equal(self.testdata[0:10],
                                             tr.data[0:10])

    def test_readHeadViaObspy(self):
        """
        Read files via L{obspy.Stream}
        """
        tr = read(self.file, format='SAC', headonly=True)[0]
        self.assertEqual(tr.stats['station'], 'STA')
        self.assertEqual(tr.stats.npts, 100)
        self.assertEqual(tr.stats['sampling_rate'], 1.0)
        self.assertEqual(tr.stats.get('channel'), 'Q')
        self.assertEqual(tr.stats.starttime.timestamp, 269596800.0)
        self.assertEqual(tr.stats.sac.get('nvhdr'), 6)
        self.assertEqual(str(tr.data), '[]')

    def test_writeViaObspy(self):
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
        self.assertEqual(st2[0].stats['sac'].nvhdr, 6)

    def test_readAndWriteViaObspy(self):
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

    def test_convertMseed2Sac(self):
        # setUp is called before every test, not only once at the
        # beginning, that is we allocate the data just here
        # generate artificial mseed data
        np.random.seed(815)
        head = {'network': 'NL', 'station': 'HGN', 'location': '00',
                'channel': 'BHZ', 'calib': 1.0, 'sampling_rate': 40.0,
                'starttime': UTCDateTime(2003, 5, 29, 2, 13, 22, 43400),
                'mseed': {'dataquality': 'R'}}
        data = np.random.randint(0, 5000, 11947).astype("int32")
        st = Stream([Trace(header=head, data=data)])
        # write them as SAC
        tmpfile = NamedTemporaryFile().name
        st.write(tmpfile, format="SAC")
        st2 = read(tmpfile, format="SAC")
        self.assertEqual(st2[0].stats['npts'], st[0].stats['npts'])
        # file must exist, we just created it
        os.remove(tmpfile)


    def test_iztype11(self):
        # test that iztype 11 is read correctly
        sod_file = os.path.join(self.path, 'data', 'dis.G.SCZ.__.BHE_short')
        tr = read(sod_file)[0]
        sac = ReadSac(sod_file)
        t1 = tr.stats.starttime - float(tr.stats.sac.b)
        t2 = sac.starttime
        self.assertAlmostEqual(t1.timestamp, t2.timestamp, 5)
        # see that iztype is written corretly
        tempfile = NamedTemporaryFile().name
        tr.write(tempfile, format="SAC")
        sac2 = ReadSac(tempfile)
        self.assertEqual(sac2.iztype, 11)
        self.assertAlmostEqual(tr.stats.sac.b, sac2.b)
        self.assertAlmostEqual(t2.timestamp, sac2.starttime.timestamp, 5)
        os.remove(tempfile)

    def test_defaultvalues(self):
        tr = read(self.file)[0]
        self.assertEqual(tr.stats.calib,1.0)
        self.assertEqual(tr.stats.location,'')
        self.assertEqual(tr.stats.network,'')
        
def suite():
    return unittest.makeSuite(CoreTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
