#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The SacIO test suite.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA
from future.utils import native_str

import io
import os
import unittest

import numpy as np

from obspy import Trace, read
from obspy.core.util import NamedTemporaryFile
from obspy.io.sac import SacError, SacIO, SacIOError, attach_paz, attach_resp


class SacIOTestCase(unittest.TestCase):
    """
    Test cases for SacIO.
    """
    def setUp(self):
        # directory where the test files are located
        self.path = os.path.join(os.path.dirname(__file__), 'data')

    def test_Date(self):
        """
        Test for SacIO '_get_date_'-function to calculate timestamp
        """
        fn = os.path.join(os.path.dirname(__file__), 'data', 'test.sac')
        with open(fn, "rb") as fh:
            t = SacIO(fh)
        self.assertEqual(t.reftime.timestamp, 269596800.0)
        diff = t.get_header_value('npts')
        self.assertEqual(int(t.endtime - t.starttime), diff)

    def test_read(self):
        """
        Tests for SacIO read and write
        """
        data = np.array([-8.7422776573475858e-08, -0.30901697278022766,
                         -0.58778536319732666, -0.8090171217918396,
                         -0.95105659961700439, -1.0, -0.95105630159378052,
                         -0.80901658535003662, -0.5877845287322998,
                         -0.30901604890823364, 1.1285198979749111e-06],
                        dtype=native_str('<f4'))
        sacfile = os.path.join(self.path, 'test.sac')
        t = SacIO()
        with open(sacfile, "rb") as fh:
            t.read_sac_file(fh)
        np.testing.assert_array_equal(t.seis[0:11], data)
        self.assertEqual(t.get_header_value('npts'), 100)
        self.assertEqual(t.get_header_value("kstnm"), "STA     ")

    def test_readWrite(self):
        """
        Tests for SacIO read and write
        """
        sacfile = os.path.join(self.path, 'test.sac')
        with NamedTemporaryFile() as tf:
            tempfile = tf.name
            t = SacIO()
            with open(sacfile, "rb") as fh:
                t.read_sac_file(fh)
            self.assertEqual(t.get_header_value('npts'), 100)
            self.assertEqual(t.get_header_value("kcmpnm"), "Q       ")
            self.assertEqual(t.get_header_value("kstnm"), "STA     ")
            t.set_header_value("kstnm", "spiff")
            self.assertEqual(t.get_header_value('kstnm'), 'spiff   ')
            with open(tempfile, "wb") as fh:
                t.write_sac_binary(fh)
            self.assertEqual(os.stat(sacfile)[6], os.stat(tempfile)[6])
            self.assertEqual(os.path.exists(tempfile), True)
            with open(tempfile, "rb") as fh:
                t.read_sac_header(fh)
            self.assertEqual((t.hf is not None), True)
            t.set_header_value("kstnm", "spoff")
            self.assertEqual(t.get_header_value('kstnm'), 'spoff   ')
            # Open with modification!
            with open(tempfile, "rb+") as fh:
                t.write_sac_header(fh)
            t.set_header_value_in_file(tempfile, "kcmpnm", 'Z       ')
            self.assertEqual(t.get_header_value_from_file(tempfile, "kcmpnm"),
                             'Z       ')
            with open(tempfile, "rb") as fh:
                self.assertEqual(
                    SacIO(fh, headonly=True).get_header_value('kcmpnm'),
                    'Z       ')
            with open(tempfile, "rb") as fh:
                self.assertEqual(t.is_valid_sac_file(fh), True)
            self.assertEqual(t.is_valid_xy_sac_file(tempfile), False)
            self.assertEqual(
                SacIO().get_header_value_from_file(sacfile, 'npts'), 100)
            with open(sacfile, "rb") as fh:
                self.assertEqual(SacIO(fh).get_header_value('npts'), 100)

    def test_readWriteXY(self):
        """
        Tests for ascii sac io
        """
        with NamedTemporaryFile() as tf:
            tempfile = tf.name
            tfile = os.path.join(os.path.dirname(__file__), 'data', 'test.sac')
            with open(tfile, "rb") as fh:
                t = SacIO(fh)
            with open(tempfile, "wb") as fh:
                t.write_sac_xy(fh)
            with open(tempfile, "rb") as fh:
                d = SacIO(fh, alpha=True)
            e = SacIO()
            with open(tempfile, "rb") as fh:
                e.read_sac_xy(fh)
            self.assertEqual(e.get_header_value('npts'),
                             d.get_header_value('npts'))
            with open(tempfile, "rb") as fh:
                self.assertEqual(e.is_valid_xy_sac_file(fh), True)
            with open(tempfile, "rb") as fh:
                self.assertEqual(e.is_valid_sac_file(fh), False)
        with NamedTemporaryFile() as tf:
            tempfile = tf.name
            with open(tempfile, "wb") as fh:
                d.write_sac_binary(fh)
            size1 = os.stat(tempfile)[6]
            size2 = os.stat(tfile)[6]
        self.assertEqual(size1, size2)
        np.testing.assert_array_almost_equal(t.seis, d.seis, decimal=5)

    def test_readXYheader(self):
        tfile = os.path.join(os.path.dirname(__file__), 'data', 'test.sac')
        with NamedTemporaryFile() as tf:
            tempfile = tf.name
            with open(tfile, "rb") as fh:
                t = SacIO(fh)
            with open(tempfile, 'wb') as fh:
                t.write_sac_xy(fh)
            with open(tempfile, "rb") as fh:
                d = SacIO(fh, alpha=True)
            e = SacIO()
            with open(tempfile, "rb") as fh:
                e.read_sac_xy_header(fh)
            self.assertEqual(e.get_header_value('npts'),
                             d.get_header_value('npts'))
            self.assertEqual(e.get_header_value('depmen'),
                             d.get_header_value('depmen'))
            self.assertEqual(e.starttime, d.starttime)
            self.assertNotEqual(e.seis.size, d.seis.size)
            with open(tempfile, "rb") as fh:
                c = SacIO(fh, alpha=True, headonly=True)
        self.assertEqual(e.seis.size, c.seis.size)

    def test_readBigEnd(self):
        """
        Test reading big endian binary files
        """
        tfilel = os.path.join(os.path.dirname(__file__), 'data', 'test.sac')
        tfileb = os.path.join(os.path.dirname(__file__), 'data',
                              'test.sac.swap')
        with open(tfilel, "rb") as fh:
            tl = SacIO(fh)
        with open(tfileb, "rb") as fh:
            tb = SacIO(fh)
        self.assertEqual(tl.get_header_value('kevnm'),
                         tb.get_header_value('kevnm'))
        self.assertEqual(tl.get_header_value('npts'),
                         tb.get_header_value('npts'))
        self.assertEqual(tl.get_header_value('delta'),
                         tb.get_header_value('delta'))
        np.testing.assert_array_equal(tl.seis, tb.seis)

    def test_swapbytes(self):
        tfilel = os.path.join(os.path.dirname(__file__), 'data', 'test.sac')
        tfileb = os.path.join(os.path.dirname(__file__), 'data',
                              'test.sac.swap')
        with NamedTemporaryFile() as tf:
            tempfile = tf.name
            with open(tfileb, "rb") as fh:
                tb = SacIO(fh)
            tb.swap_byte_order()
            with open(tempfile, "wb") as fh:
                tb.write_sac_binary(fh)
            with open(tempfile, "rb") as fh:
                t = SacIO(fh)
            with open(tfilel, "rb") as fh:
                tl = SacIO(fh)
            self.assertEqual(t.get_header_value('kevnm'),
                             tl.get_header_value('kevnm'))
            self.assertEqual(t.get_header_value('npts'),
                             tl.get_header_value('npts'))
            self.assertEqual(t.get_header_value('delta'),
                             tl.get_header_value('delta'))
            np.testing.assert_array_equal(t.seis, tl.seis)
        with NamedTemporaryFile() as tf:
            tempfile = tf.name
            with open(tfilel, "rb") as fh:
                tl = SacIO(fh)
            tl.swap_byte_order()
            with open(tempfile, "wb") as fh:
                tl.write_sac_binary(fh)
            with open(tempfile, "rb") as fh:
                t = SacIO(fh)
            with open(tfileb, "rb") as fh:
                tb = SacIO(fh)
            self.assertEqual(t.get_header_value('kevnm'),
                             tb.get_header_value('kevnm'))
            self.assertEqual(t.get_header_value('npts'),
                             tb.get_header_value('npts'))
            self.assertEqual(t.get_header_value('delta'),
                             tb.get_header_value('delta'))
            np.testing.assert_array_equal(t.seis, tb.seis)

    def test_getdist(self):
        tfile = os.path.join(os.path.dirname(__file__), 'data', 'test.sac')
        with NamedTemporaryFile() as tf:
            tempfile = tf.name
            with open(tfile, "rb") as fh:
                t = SacIO(fh)
            t.set_header_value('evla', 48.15)
            t.set_header_value('evlo', 11.58333)
            t.set_header_value('stla', -41.2869)
            t.set_header_value('stlo', 174.7746)
            t.set_header_value('lcalda', 1)
            with open(tempfile, "wb") as fh:
                t.write_sac_binary(fh)
            with open(tempfile, "rb") as fh:
                t2 = SacIO(fh)
        b = np.array([18486532.5788 / 1000., 65.654154562, 305.975459869],
                     dtype=native_str('>f4'))
        self.assertEqual(t2.get_header_value('dist'), b[0])
        self.assertEqual(t2.get_header_value('az'), b[1])
        self.assertEqual(t2.get_header_value('baz'), b[2])

    def test_is_sac(self):
        """
        Assertion is raised if file is not a SAC file
        """
        t = SacIO()
        with open(__file__, "rb") as fh:
            self.assertRaises(SacError, t.read_sac_file, fh)

    def test_getattr(self):
        tfile = os.path.join(os.path.dirname(__file__), 'data', 'test.sac')
        with open(tfile, "rb") as fh:
            tr = SacIO(fh)
        self.assertEqual(tr.npts, tr.get_header_value('npts'))
        self.assertEqual(tr.kstnm, tr.get_header_value('kstnm'))

    # def test_raiseOnGetDist(self):
    #     """
    #     Test case to check that SACError is raised if obspy.signal is not
    #     installed. SACError must be raised as it is caught by various
    #     methods. The import of setuptools introduces a function
    #     findall, which recursively searches directories for pth files.
    #     Could not get obspy.signal out of the path so far...
    #     """
    #     t = SacIO()
    #     t.set_header_value('evla',48.15)
    #     t.set_header_value('evlo',11.58333)
    #     t.set_header_value('stla',-41.2869)
    #     t.set_header_value('stlo',174.7746)
    #     delete obspy.signal from system path list
    #     signal_path = [sys.path.pop(sys.path.index(j)) for j in \
    #             [i for i in sys.path if 'obspy.signal' in i]]
    #     # delete obspy.signal from all imported modules dict
    #     #[sys.modules.pop(i) for i in \
    #     #        sys.modules.keys() if 'obspy.signal' in i]
    #     self.assertRaises(SacError, t._get_dist_)
    #     sys.path.extend(signal_path)

    def test_attach_paz(self):
        fvelhz = io.StringIO("""ZEROS 3
        -5.032 0.0
        POLES 6
        -0.02365 0.02365
        -0.02365 -0.02365
        -39.3011 0.
        -7.74904 0.
        -53.5979 21.7494
        -53.5979 -21.7494
        CONSTANT 2.16e18""")
        tr = Trace()
        attach_paz(tr, fvelhz, torad=True, todisp=True)
        np.testing.assert_array_almost_equal(tr.stats.paz['zeros'][0],
                                             - 31.616988, decimal=6)
        self.assertEqual(len(tr.stats.paz['zeros']), 4)

    def test_attach_paz_diff_order(self):
        pazfile = os.path.join(os.path.dirname(__file__),
                               'data', 'NZCRLZ_HHZ10.pz')
        tr = Trace()
        attach_paz(tr, pazfile)
        np.testing.assert_array_almost_equal(tr.stats.paz['gain'],
                                             7.4592e-2, decimal=6)
        self.assertEqual(len(tr.stats.paz['zeros']), 5)
        self.assertEqual(len(tr.stats.paz['poles']), 4)

    def test_sacpaz_from_dataless(self):
        # The following dictionary is extracted from a datalessSEED
        # file
        pazdict = {'sensitivity': 2516580000.0,
                   'digitizer_gain': 1677720.0, 'seismometer_gain': 1500.0,
                   'zeros': [0j, 0j], 'gain': 59198800.0,
                   'poles': [(-0.037010000000000001 + 0.037010000000000001j),
                             (-0.037010000000000001 - 0.037010000000000001j),
                             (-131 + 467.30000000000001j),
                             (-131 - 467.30000000000001j),
                             (-251.30000000000001 + 0j)]}
        tr = Trace()
        # This file was extracted from the datalessSEED file using rdseed
        pazfile = os.path.join(os.path.dirname(__file__),
                               'data', 'SAC_PZs_NZ_HHZ_10')
        attach_paz(tr, pazfile, todisp=False)
        sacconstant = pazdict['digitizer_gain'] * \
            pazdict['seismometer_gain'] * pazdict['gain']
        np.testing.assert_almost_equal(tr.stats.paz['gain'] / 1e17,
                                       sacconstant / 1e17, decimal=6)
        # pole-zero files according to the SAC convention are in displacement
        self.assertEqual(len(tr.stats.paz['zeros']), 3)

    def test_sacpaz_from_resp(self):
        # The following two files were both extracted from a dataless
        # seed file using rdseed
        respfile = os.path.join(os.path.dirname(__file__),
                                'data', 'RESP.NZ.CRLZ.10.HHZ')
        sacpzfile = os.path.join(os.path.dirname(__file__),
                                 'data', 'SAC_PZs_NZ_CRLZ_HHZ')
        # This is a rather lengthy test, in which the
        # poles, zeros and the gain of each instrument response file
        # are converted into the corresponding velocity frequency response
        # function which have to be sufficiently close. Possibly due to
        # different truncations in the RESP-formatted and SAC-formatted
        # response files the frequency response functions are not identical.
        tr1 = Trace()
        tr2 = Trace()
        attach_resp(tr1, respfile, torad=True, todisp=False)
        attach_paz(tr2, sacpzfile, torad=False, tovel=True)
        p1 = tr1.stats.paz.poles
        z1 = tr1.stats.paz.zeros
        g1 = tr1.stats.paz.gain
        t_samp = 0.01
        n = 32768
        fy = 1 / (t_samp * 2.0)
        # start at zero to get zero for offset/ DC of fft
        f = np.arange(0, fy + fy / n, fy / n)  # arange should includes fy
        w = f * 2 * np.pi
        s = 1j * w
        a1 = np.poly(p1)
        b1 = g1 * np.poly(z1)
        h1 = np.polyval(b1, s) / np.polyval(a1, s)
        h1 = np.conj(h1)
        h1[-1] = h1[-1].real + 0.0j
        p2 = tr2.stats.paz.poles
        z2 = tr2.stats.paz.zeros
        g2 = tr2.stats.paz.gain
        a2 = np.poly(p2)
        b2 = g2 * np.poly(z2)
        h2 = np.polyval(b2, s) / np.polyval(a2, s)
        h2 = np.conj(h2)
        h2[-1] = h2[-1].real + 0.0j
        amp1 = abs(h1)
        amp2 = abs(h2)
        phase1 = np.unwrap(np.arctan2(-h1.imag, h1.real))
        phase2 = np.unwrap(np.arctan2(-h2.imag, h2.real))
        np.testing.assert_almost_equal(phase1, phase2, decimal=4)
        rms = np.sqrt(np.sum((amp1 - amp2) ** 2) /
                      np.sum(amp2 ** 2))
        self.assertTrue(rms < 2.02e-06)
        self.assertTrue(tr1.stats.paz.t_shift, 0.4022344)
        # The following plots the comparison between the
        # two frequency response functions.
        # import pylab as plt
        # plt.subplot(1,2,1)
        # plt.loglog(f,amp1)
        # plt.loglog(f,amp2,'k--')
        # plt.subplot(1,2,2)
        # plt.semilogx(f,phase1)
        # plt.semilogx(f,phase2,'k--')
        # plt.show()

    def test_issue171(self):
        """
        Test for issue #171.
        """
        tr = read()[0]
        with NamedTemporaryFile() as tf:
            tempfile = tf.name
            tr.write(tempfile, format="SAC")
            with open(tempfile, "rb") as fh:
                trace = SacIO(fh)
            trace.set_header_value('stel', 91.0)
            # Open with modification!
            with open(tempfile, "rb+") as fh:
                trace.write_sac_header(fh)
            with open(tempfile, "rb") as fh:
                SacIO(fh)

    def test_read_with_fsize(self):
        """
        Testing fsize option on SacIO.read_sac_file()
        """
        # reading sac file with wrong file size should raise error
        longer_file = os.path.join(self.path, 'seism-longer.sac')
        shorter_file = os.path.join(self.path, 'seism-shorter.sac')
        t = SacIO()
        # default
        with open(longer_file, "rb") as fh:
            self.assertRaises(SacError, t.read_sac_file, fh)
        with open(shorter_file, "rb") as fh:
            self.assertRaises(SacError, t.read_sac_file, fh)
        # fsize=True
        with open(longer_file, "rb") as fh:
            self.assertRaises(SacError, t.read_sac_file, fh, fsize=True)
        with open(shorter_file, "rb") as fh:
            self.assertRaises(SacError, t.read_sac_file, fh, fsize=True)
        # using fsize=False should not work for shorter file
        # (this is not supported by SAC) ...
        with open(shorter_file, "rb") as fh:
            self.assertRaises(SacIOError, t.read_sac_file, fh, fsize=False)
        # ...but it should work for longer file
        with open(longer_file, "rb") as fh:
            t.read_sac_file(fh, fsize=False)
        # checking trace
        self.assertEqual(t.get_header_value('nzyear'), 1981)
        self.assertEqual(t.get_header_value('nzjday'), 88)
        self.assertEqual(t.get_header_value('nzhour'), 10)
        self.assertEqual(t.get_header_value('nzmin'), 38)
        self.assertEqual(t.get_header_value('nzsec'), 14)
        self.assertEqual(t.get_header_value('nzmsec'), 0)
        # we should never test equality for float values:
        self.assertLessEqual(abs(t.get_header_value('delta') - 0.01), 1e-9)
        self.assertEqual(t.get_header_value('scale'), -12345.0)
        self.assertEqual(t.get_header_value('npts'), 998)
        self.assertEqual(t.get_header_value('knetwk'), '-12345  ')
        self.assertEqual(t.get_header_value('kstnm'), 'CDV     ')
        self.assertEqual(t.get_header_value('kcmpnm'), 'Q       ')


def suite():
    return unittest.makeSuite(SacIOTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
