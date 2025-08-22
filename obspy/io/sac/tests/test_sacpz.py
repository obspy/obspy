# -*- coding: utf-8 -*-
"""
The sac.sacpz test suite.
"""
import io
import warnings

import numpy as np
import pytest

from obspy import read_inventory, Trace
from obspy.core.inventory.util import Equipment
from obspy.core.util import NamedTemporaryFile
from obspy.core.util.base import CatchAndAssertWarnings
from obspy.io.sac import attach_paz, attach_resp


class TestSACPZ:
    """
    """
    @pytest.fixture(autouse=True, scope="function")
    def setup(self, testdata):
        # these files were checked against data given by IRIS SACPZ web service
        # http://service.iris.edu/irisws/sacpz/1/
        #                                query?net=IU&loc=*&cha=BH?&sta=ANMO
        # DIP seems to be systematically different in SACPZ output compared to
        # StationXML served by IRIS...
        self.file1 = testdata['IU_ANMO_00_BHZ.sacpz']
        self.file2 = testdata['IU_ANMO_BH.sacpz']

    @pytest.fixture(scope="class")
    def sacpz_with_no_sensors(self, testdata):
        expected = []
        with open(testdata['IU_ANMO_00_BHZ.sacpz']) as fh:
            for line in fh:
                if "INSTTYPE" in line:
                    line = "* INSTTYPE    : "
                if "CREATED" not in line:
                    expected.append(line.rstrip("\n"))

        return expected

    def test_write_sacpz_single_channel(self, root):
        """
        """
        path = root / 'core' / 'tests' / 'data' / 'IU_ANMO_00_BHZ.xml'
        inv = read_inventory(path)
        with NamedTemporaryFile() as tf:
            tempfile = tf.name
            inv.write(tempfile, format='SACPZ')
            with open(tempfile) as fh:
                got = fh.read()
        with open(self.file1) as fh:
            expected = fh.read()
        # remove CREATED line that changes
        got = [line for line in got.split("\n") if "CREATED" not in line]
        expected = [line for line in expected.split("\n")
                    if "CREATED" not in line]
        assert got == expected

    def test_write_sacpz_multiple_channels(self, root):
        """
        """
        path = root / 'core' / 'tests' / 'data' / 'IU_ANMO_BH.xml'
        inv = read_inventory(path)
        with NamedTemporaryFile() as tf:
            tempfile = tf.name
            inv.write(tempfile, format='SACPZ')
            with open(tempfile) as fh:
                got = fh.read()
        with open(self.file2) as fh:
            expected = fh.read()
        # remove CREATED line that changes
        got = [line for line in got.split("\n") if "CREATED" not in line]
        expected = [line for line in expected.split("\n")
                    if "CREATED" not in line]
        assert got == expected

    def test_write_sacpz_soh(self, datapath):
        path = (datapath.parent.parent.parent /
                'stationxml' / 'tests' / 'data' / 'only_soh.xml')
        inv = read_inventory(path)
        f = io.StringIO()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            inv.write(f, format='SACPZ')
        # Testxml has 2 channels: 1 no paz, 2 unrecognized units.
        assert len(w) == 2
        # Assert warning messages contain correct warnings
        assert any('has no paz' in str(x.message) for x in w)
        assert any('has unrecognized input units' in str(x.message) for x in w)
        # Only 2 newlines are written.
        assert 2 == f.tell()

    @pytest.mark.parametrize("sensor", [None, Equipment(type=None)])
    def test_write_sacpz_no_sensor(self, sensor, sacpz_with_no_sensors):
        """
        Test sacpz writer when no sensor or sensor type are specified
        """
        inv = read_inventory("/path/to/IU_ANMO_00_BHZ.xml")
        inv.networks[0].stations[0].channels[0].sensor = sensor
        with NamedTemporaryFile() as tf:
            tempfile = tf.name
            inv.write(tempfile, format='SACPZ')
            with open(tempfile) as fh:
                got = [line.rstrip("\n") for line in fh
                       if "CREATED" not in line]

        assert got == sacpz_with_no_sensors

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
        assert len(tr.stats.paz['zeros']) == 4

    def test_attach_paz_diff_order(self, testdata):
        pazfile = testdata['NZCRLZ_HHZ10.pz']
        tr = Trace()
        attach_paz(tr, pazfile)
        np.testing.assert_array_almost_equal(tr.stats.paz['gain'],
                                             7.4592e-2, decimal=6)
        assert len(tr.stats.paz['zeros']) == 5
        assert len(tr.stats.paz['poles']) == 4

    def test_sacpaz_from_dataless(self, testdata):
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
        pazfile = testdata['SAC_PZs_NZ_HHZ_10']
        attach_paz(tr, pazfile, todisp=False)
        sacconstant = pazdict['digitizer_gain'] * \
            pazdict['seismometer_gain'] * pazdict['gain']
        np.testing.assert_almost_equal(tr.stats.paz['gain'] / 1e17,
                                       sacconstant / 1e17, decimal=6)
        # pole-zero files according to the SAC convention are in displacement
        assert len(tr.stats.paz['zeros']) == 3

    def test_sacpaz_from_resp(self, testdata):
        # The following two files were both extracted from a dataless
        # seed file using rdseed
        respfile = testdata['RESP.NZ.CRLZ.10.HHZ']
        sacpzfile = testdata['SAC_PZs_NZ_CRLZ_HHZ']
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
        assert rms < 2.02e-06
        assert tr1.stats.paz.t_shift == 0.4022344
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

    def test_writing_sacpz_hertz_to_radians(self, root):
        """
        Tests writing out a response with poles and zeros described in Hertz as
        a SACPZ file, which implicitely expects the given data to be RADIANS/S,
        so that a conversion is needed.

        See #3334
        """
        path = root / 'core' / 'tests' / 'data' / 'G_CAN__LHZ.xml'
        inv = read_inventory(path, 'STATIONXML')

        expected_a0 = 3.959488e+03
        expected_constant = 7.304622e+12
        expected_zeros = [0j, 0j, 0j]
        expected_poles = [-1.233948e-02+1.234319e-02j,
                          -1.233948e-02-1.234319e-02j,
                          -3.917566e+01+4.912339e+01j,
                          -3.917566e+01-4.912339e+01j]

        sio = io.StringIO()
        # ignore a warning because there is an additional unity PAZ stage
        with CatchAndAssertWarnings():
            inv.write(sio, format='SACPZ')
        sio.seek(0)
        lines = sio.readlines()
        # make sure we find the lines we are looking for
        for expected_start in ('* A0  ', 'ZEROS', 'POLES', 'CONSTANT'):
            for line in lines:
                if line.startswith(expected_start):
                    break
            else:
                msg = f"No line starting with '{expected_start}' found."
                pytest.fail(msg)
        # now test values
        for i, line in enumerate(lines):
            if line.startswith('* A0  '):
                value = float(line.split()[-1])
                assert round(value, 3) == expected_a0
            elif line.startswith('ZEROS'):
                num_zeros = int(line.split()[-1])
                assert num_zeros == 3
                zeros = [complex(*map(float, lines[i+1+j].split()))
                         for j in range(num_zeros)]
                np.testing.assert_allclose(zeros, expected_zeros)
            elif line.startswith('POLES'):
                num_poles = int(line.split()[-1])
                assert num_poles == 4
                poles = [complex(*map(float, lines[i+1+j].split()))
                         for j in range(num_poles)]
                np.testing.assert_allclose(poles, expected_poles)
            elif line.startswith('CONSTANT'):
                value = float(line.split()[-1])
                assert round(value, 3) == expected_constant
