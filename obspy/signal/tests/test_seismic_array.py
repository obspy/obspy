#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The SeismicArray test suite.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import io
import os
import unittest
import numpy as np

from obspy import Stream, Trace, UTCDateTime
from obspy.signal.array_analysis import SeismicArray
from obspy.signal.util import util_lon_lat
from obspy import read
from obspy.core.inventory import read_inventory
from obspy.core.inventory.channel import Channel
from obspy.core.inventory.station import Station
from obspy.core.inventory.network import Network
from obspy.core.inventory.inventory import Inventory


class SeismicArrayTestCase(unittest.TestCase):
    """
    Test cases for array and array analysis functions.
    """

    def setUp(self):
        self.path = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                 'data'))

        def create_simple_array(coords, sys='xy'):
            """
            Set up a legal array more easily from x-y or long-lat coordinates.
            Note it's usually lat-lon in other applications.
            """
            if sys == 'xy':
                coords_lonlat = [util_lon_lat(0, 0, stn[0], stn[1])
                                 for stn in coords]
            else:
                coords_lonlat = coords
            stns = [Station(str(_i), coords_lonlat[_i][1],
                            coords_lonlat[_i][0], 0)  # flat!
                    for _i in range(len(coords_lonlat))]
            testinv = Inventory([Network("testnetwork", stations=stns)],
                                'testsender')
            return SeismicArray('testarray', inventory=testinv)

        # Set up an array for geometry tests.
        geometry_coords = [[0, 0], [2, 0], [1, 1], [0, 2], [2, 2]]
        self.geometry_array = create_simple_array(geometry_coords, 'longlat')

        # Set up the test array for the _covariance_array_processing,
        # stream_offset and array_rotation_strain tests.
        self.fk_testcoords = np.array([[0.0, 0.0, 0.0],
                                       [-5.0, 7.0, 0.0],
                                       [5.0, 7.0, 0.0],
                                       [10.0, 0.0, 0.0],
                                       [5.0, -7.0, 0.0],
                                       [-5.0, -7.0, 0.0],
                                       [-10.0, 0.0, 0.0]])
        self.fk_testcoords /= 100
        self.fk_array = create_simple_array(self.fk_testcoords)

        # Set up test array for transff tests.
        transff_testcoords = np.array([[10., 60., 0.],
                                       [200., 50., 0.],
                                       [-120., 170., 0.],
                                       [-100., -150., 0.],
                                       [30., -220., 0.]])
        transff_testcoords /= 1000.
        self.transff_array = create_simple_array(transff_testcoords)

    def test__get_geometry(self):
        geo = self.geometry_array.geometry
        geo_exp = {'testnetwork.0..': {'absolute_height_in_km': 0.0,
                                       'latitude': 0.0, 'longitude': 0.0},
                   'testnetwork.1..': {'absolute_height_in_km': 0.0,
                                       'latitude': 0.0, 'longitude': 2.0},
                   'testnetwork.2..': {'absolute_height_in_km': 0.0,
                                       'latitude': 1.0, 'longitude': 1.0},
                   'testnetwork.3..': {'absolute_height_in_km': 0.0,
                                       'latitude': 2.0, 'longitude': 0.0},
                   'testnetwork.4..': {'absolute_height_in_km': 0.0,
                                       'latitude': 2.0, 'longitude': 2.0}}
        self.assertEqual(geo, geo_exp)
        # todo test for inventories with and w/o channels

    def test_get_geometry_xyz(self):
        """
        Test get_geometry_xyz and, implicitly, _get_geometry (necessary because
        self.geometry is a property and can't be set).
        """
        geox_exp = {'testnetwork.0..': {'x': -111.31564682647114,
                                        'y': -110.5751633754653, 'z': 0.0},
                    'testnetwork.1..': {'x': 111.31564682647114,
                                        'y': -110.5751633754653, 'z': 0.0},
                    'testnetwork.2..': {'x': 0.0, 'y': 0.0, 'z': 0.0},
                    'testnetwork.3..': {'x': -111.28219117308639, 'y':
                                        110.5751633754653, 'z': 0.0},
                    'testnetwork.4..': {'x': 111.28219117308639, 'y':
                                        110.5751633754653, 'z': 0.0}}
        geox = self.geometry_array.get_geometry_xyz(1, 1, 0,
                                                    correct_3dplane=False)
        self.assertEqual(geox, geox_exp)
        geox = self.geometry_array.get_geometry_xyz(1, 1, 0,
                                                    correct_3dplane=True)
        self.assertEqual(geox_exp, geox)

    def test_center_of_gravity(self):
        self.assertEqual(self.geometry_array.center_of_gravity,
                         {'absolute_height_in_km': 0.0,
                          'latitude': 1.0, 'longitude': 1.0})

    def test_geometrical_center(self):
        self.assertEqual(self.geometry_array.geometrical_center,
                         {'absolute_height_in_km': 0.0,
                          'latitude': 1.0, 'longitude': 1.0})

    def test_inventory_cull(self):
        time = UTCDateTime('2016-04-05T06:44:0.0Z')
        # Method should work even when traces do not cover same times.
        st = Stream([
            Trace(data=np.empty(20), header={'network': 'BP', 'station':
                  'CCRB', 'location': '1', 'channel': 'BP1',
                    'starttime': time}),
            Trace(data=np.empty(20), header={'network': 'BP', 'station':
                  'EADB', 'channel': 'BPE', 'starttime': time-60})])
        # Set up channels, correct ones first. The eadb channel should also be
        # selected despite no given time.
        kwargs = {'latitude': 0, 'longitude': 0, 'elevation': 0, 'depth': 0}
        ch_ccrb = Channel(code='BP1', start_date=time-10, end_date=time+60,
                          location_code='1', **kwargs)
        wrong = [Channel(code='BP1', start_date=time-60, end_date=time-10,
                         location_code='1', **kwargs),
                 Channel(code='BP2', location_code='1', **kwargs)]
        ccrb = Station('CCRB', 0, 0, 0, channels=[ch_ccrb] + wrong)

        ch_eadb = Channel(code='BPE', location_code='', **kwargs)
        wrong = Channel(code='BPE', location_code='2', **kwargs)
        eadb = Station('EADB', 0, 0, 0, channels=[ch_eadb, wrong])
        wrong_stn = Station('VCAB', 0, 0, 0, channels=[ch_eadb, wrong])

        array = SeismicArray('testarray', Inventory([Network('BP',
                             stations=[ccrb, eadb, wrong_stn])], 'testinv'))

        array.inventory_cull(st)
        self.assertEqual(array.inventory[0][0][0], ch_ccrb)
        self.assertEqual(array.inventory[0][1][0], ch_eadb)
        tbc = [array.inventory.networks, array.inventory[0].stations,
               array.inventory[0][0].channels, array.inventory[0][1].channels]
        self.assertEqual([len(item) for item in tbc], [1, 2, 1, 1])

    def test_covariance_array_processing(self):
        # Generate some synthetic data for the FK/Capon tests
        np.random.seed(2348)
        slowness = 1.3       # in s/km
        baz_degree = 20.0    # 0.0 > source in x direction
        baz = baz_degree * np.pi / 180.
        df = 100             # samplerate
        # SNR = 100.         # signal to noise ratio
        amp = .00001         # amplitude of coherent wave
        length = 500         # signal length in samples
        coherent_wave = amp * np.random.randn(length)
        # time offsets in samples
        dt = np.round(df * slowness * (np.cos(baz) * self.fk_testcoords[:, 1] +
                                       np.sin(baz) * self.fk_testcoords[:, 0]))
        trl = []
        for i in range(len(self.fk_testcoords)):
            tr = Trace(coherent_wave[-(np.min(dt) - 1) +
                                     dt[i]:-(np.max(dt) + 1) + dt[i]].copy())
            tr.stats.sampling_rate = df
            # lowpass random signal to f_nyquist / 2
            tr.filter("lowpass", freq=df / 4.)
            trl.append(tr)
        st = Stream(trl)
        stime = UTCDateTime(1970, 1, 1, 0, 0)
        fk_args = (st, 2, 0.2, -3, 3, -3, 3, 0.1,
                   -1e99, -1e99, 1, 8, stime, stime + 4)
        # Tests for FK analysis:
        out = self.fk_array._covariance_array_processing(*fk_args,
                                                         prewhiten=0, method=0)
        raw = """
        9.68742255e-01 1.95739086e-05 1.84349488e+01 1.26491106e+00
        9.60822403e-01 1.70468277e-05 1.84349488e+01 1.26491106e+00
        9.61689241e-01 1.35971034e-05 1.84349488e+01 1.26491106e+00
        9.64670470e-01 1.35565806e-05 1.84349488e+01 1.26491106e+00
        9.56880885e-01 1.16028992e-05 1.84349488e+01 1.26491106e+00
        9.49584782e-01 9.67131311e-06 1.84349488e+01 1.26491106e+00
        """
        ref = np.loadtxt(io.StringIO(raw), dtype=np.float32)
        self.assertTrue(np.allclose(ref, out[:, 1:], rtol=1e-6))

        out = self.fk_array._covariance_array_processing(*fk_args,
                                                         prewhiten=1, method=0)
        raw = """
        1.40997967e-01 1.95739086e-05 1.84349488e+01 1.26491106e+00
        1.28566503e-01 1.70468277e-05 1.84349488e+01 1.26491106e+00
        1.30517975e-01 1.35971034e-05 1.84349488e+01 1.26491106e+00
        1.34614854e-01 1.35565806e-05 1.84349488e+01 1.26491106e+00
        1.33609938e-01 1.16028992e-05 1.84349488e+01 1.26491106e+00
        1.32638966e-01 9.67131311e-06 1.84349488e+01 1.26491106e+00
        """
        ref = np.loadtxt(io.StringIO(raw), dtype=np.float32)
        self.assertTrue(np.allclose(ref, out[:, 1:]))

        # Tests for Capon
        out = self.fk_array._covariance_array_processing(*fk_args,
                                                         prewhiten=0, method=1)
        raw = """
        9.06938200e-01 9.06938200e-01  1.49314172e+01  1.55241747e+00
        8.90494375e+02 8.90494375e+02 -9.46232221e+00  1.21655251e+00
        3.07129784e+03 3.07129784e+03 -4.95739213e+01  3.54682957e+00
        5.00019137e+03 5.00019137e+03 -1.35000000e+02  1.41421356e-01
        7.94530414e+02 7.94530414e+02 -1.65963757e+02  2.06155281e+00
        6.08349575e+03 6.08349575e+03  1.77709390e+02  2.50199920e+00
        """
        ref = np.loadtxt(io.StringIO(raw), dtype=np.float32)
        self.assertTrue(np.allclose(ref, out[:, 1:], rtol=1e-6))

        out = self.fk_array._covariance_array_processing(*fk_args,
                                                         prewhiten=1, method=1)
        raw = """
        1.30482688e-01 9.06938200e-01  1.49314172e+01  1.55241747e+00
        8.93029978e-03 8.90494375e+02 -9.46232221e+00  1.21655251e+00
        9.55393634e-03 1.50655072e+01  1.42594643e+02  2.14009346e+00
        8.85762420e-03 7.27883670e+01  1.84349488e+01  1.26491106e+00
        1.51510617e-02 6.54541771e-01  6.81985905e+01  2.15406592e+00
        3.10761699e-02 7.38667657e+00  1.13099325e+01  1.52970585e+00
        """
        ref = np.loadtxt(io.StringIO(raw), dtype=np.float32)
        self.assertTrue(np.allclose(ref, out[:, 1:], rtol=1e-6))

    def test_get_stream_offset(self):
        """
        Test case for #682
        """
        stime = UTCDateTime(1970, 1, 1, 0, 0)
        etime = UTCDateTime(1970, 1, 1, 0, 0) + 10
        data = np.empty(20)
        # sampling rate defaults to 1 Hz
        st = Stream([
            Trace(data, {'starttime': stime - 1}),
            Trace(data, {'starttime': stime - 4}),
            Trace(data, {'starttime': stime - 2}),
        ])
        spoint, epoint = self.fk_array.get_stream_offsets(st, stime, etime)
        self.assertTrue(np.allclose([1, 4, 2], spoint))
        self.assertTrue(np.allclose([8, 5, 7], epoint))

    def test_fk_array_transff_freqslowness(self):
        transff = self.transff_array.array_transff_freqslowness(40, 20, 1,
                                                                10, 1)
        transffth = np.array(
            [[0.41915119, 0.33333333, 0.32339525, 0.24751548, 0.67660475],
             [0.25248452, 0.41418215, 0.34327141, 0.65672859, 0.33333333],
             [0.24751548, 0.25248452, 1.00000000, 0.25248452, 0.24751548],
             [0.33333333, 0.65672859, 0.34327141, 0.41418215, 0.25248452],
             [0.67660475, 0.24751548, 0.32339525, 0.33333333, 0.41915119]])
        np.testing.assert_array_almost_equal(transff, transffth, decimal=6)

    def test_fk_array_transff_wavenumber(self):
        transff = self.transff_array.array_transff_wavenumber(40, 20)
        transffth = np.array(
            [[3.13360360e-01, 4.23775796e-02, 6.73650243e-01,
              4.80470652e-01, 8.16891615e-04],
             [2.98941684e-01, 2.47377842e-01, 9.96352135e-02,
              6.84732871e-02, 5.57078203e-01],
             [1.26523678e-01, 2.91010683e-01, 1.00000000e+00,
              2.91010683e-01, 1.26523678e-01],
             [5.57078203e-01, 6.84732871e-02, 9.96352135e-02,
              2.47377842e-01, 2.98941684e-01],
             [8.16891615e-04, 4.80470652e-01, 6.73650243e-01,
              4.23775796e-02, 3.13360360e-01]])
        np.testing.assert_array_almost_equal(transff, transffth, decimal=6)

    def test_array_rotation_strain(self):
        """
        Test function array_rotation_strain with for pure rotation, pure
        dilation and pure shear strain.
        """
        # Test function array_rotation_strain with synthetic data with pure
        # rotation and no strain.
        self.ts1 = np.empty((1000, 7))
        self.ts2 = np.empty((1000, 7))
        self.ts3 = np.empty((1000, 7))
        self.ts1.fill(np.NaN)
        self.ts2.fill(np.NaN)
        self.ts3.fill(np.NaN)
        self.subarray = np.array([0, 1, 2, 3, 4, 5, 6])
        self.Vp = 1.93
        self.Vs = 0.326
        self.sigmau = 0.0001
        rotx = 0.00001 * np.exp(-1 * np.square(np.linspace(-2, 2, 1000))) * \
            np.sin(np.linspace(-30 * np.pi, 30 * np.pi, 1000))
        roty = 0.00001 * np.exp(-1 * np.square(np.linspace(-2, 2, 1000))) * \
            np.sin(np.linspace(-20 * np.pi, 20 * np.pi, 1000))
        rotz = 0.00001 * np.exp(-1 * np.square(np.linspace(-2, 2, 1000))) * \
            np.sin(np.linspace(-10 * np.pi, 10 * np.pi, 1000))

        for stat in range(7):
            for t in range(1000):
                self.ts1[t, stat] = -1. * self.fk_testcoords[stat, 1] * rotz[t]
                self.ts2[t, stat] = self.fk_testcoords[stat, 0] * rotz[t]
                self.ts3[t, stat] = self.fk_testcoords[stat, 1] * rotx[t] - \
                    self.fk_testcoords[stat, 0] * roty[t]

        out = SeismicArray.array_rotation_strain(self.subarray, self.ts1,
                                                 self.ts2, self.ts3, self.Vp,
                                                 self.Vs, self.fk_testcoords,
                                                 self.sigmau)

        np.testing.assert_array_almost_equal(rotx, out['ts_w1'], decimal=12)
        np.testing.assert_array_almost_equal(roty, out['ts_w2'], decimal=12)
        np.testing.assert_array_almost_equal(rotz, out['ts_w3'], decimal=12)
        np.testing.assert_array_almost_equal(np.zeros(1000), out['ts_s'],
                                             decimal=15)
        np.testing.assert_array_almost_equal(np.zeros(1000), out['ts_d'],
                                             decimal=15)
        np.testing.assert_array_almost_equal(np.zeros(1000), out['ts_m'],
                                             decimal=12)

        # Test function array_rotation_strain with synthetic data with pure
        # dilation and no rotation or shear strain.
        eta = 1 - 2 * self.Vs ** 2 / self.Vp ** 2
        dilation = .00001 * np.exp(
            -1 * np.square(np.linspace(-2, 2, 1000))) * \
            np.sin(np.linspace(-40 * np.pi, 40 * np.pi, 1000))
        for stat in range(7):
            for t in range(1000):
                self.ts1[t, stat] = self.fk_testcoords[stat, 0] * dilation[t]
                self.ts2[t, stat] = self.fk_testcoords[stat, 1] * dilation[t]
                self.ts3[t, stat] = self.fk_testcoords[stat, 2] * dilation[t]

        out = SeismicArray.array_rotation_strain(self.subarray, self.ts1,
                                                 self.ts2, self.ts3, self.Vp,
                                                 self.Vs, self.fk_testcoords,
                                                 self.sigmau)
        # remember free surface boundary conditions!
        # see Spudich et al, 1995, (A2)
        np.testing.assert_array_almost_equal(dilation * (2 - 2 * eta),
                                             out['ts_d'], decimal=12)
        np.testing.assert_array_almost_equal(dilation * 2, out['ts_dh'],
                                             decimal=12)
        np.testing.assert_array_almost_equal(
            abs(dilation * .5 * (1 + 2 * eta)), out['ts_s'], decimal=12)
        np.testing.assert_array_almost_equal(np.zeros(1000), out['ts_sh'],
                                             decimal=12)
        np.testing.assert_array_almost_equal(np.zeros(1000), out['ts_w1'],
                                             decimal=15)
        np.testing.assert_array_almost_equal(np.zeros(1000), out['ts_w2'],
                                             decimal=15)
        np.testing.assert_array_almost_equal(np.zeros(1000), out['ts_w3'],
                                             decimal=15)
        np.testing.assert_array_almost_equal(np.zeros(1000), out['ts_m'],
                                             decimal=12)

        # Test function array_rotation_strain with synthetic data with pure
        # horizontal shear strain, no rotation or dilation.
        shear_strainh = .00001 * np.exp(
            -1 * np.square(np.linspace(-2, 2, 1000))) * \
            np.sin(np.linspace(-10 * np.pi, 10 * np.pi, 1000))
        ts3 = np.zeros((1000, 7))
        for stat in range(7):
            for t in range(1000):
                self.ts1[t, stat] = self.fk_testcoords[stat, 1] * \
                    shear_strainh[t]
                self.ts2[t, stat] = self.fk_testcoords[stat, 0] * \
                    shear_strainh[t]

        out = SeismicArray.array_rotation_strain(self.subarray, self.ts1,
                                                 self.ts2, ts3, self.Vp,
                                                 self.Vs, self.fk_testcoords,
                                                 self.sigmau)
        np.testing.assert_array_almost_equal(np.zeros(1000), out['ts_d'],
                                             decimal=12)
        np.testing.assert_array_almost_equal(np.zeros(1000), out['ts_dh'],
                                             decimal=12)
        np.testing.assert_array_almost_equal(abs(shear_strainh), out['ts_s'],
                                             decimal=12)
        np.testing.assert_array_almost_equal(abs(shear_strainh), out['ts_sh'],
                                             decimal=12)
        np.testing.assert_array_almost_equal(np.zeros(1000), out['ts_w1'],
                                             decimal=12)
        np.testing.assert_array_almost_equal(np.zeros(1000), out['ts_w2'],
                                             decimal=12)
        np.testing.assert_array_almost_equal(np.zeros(1000), out['ts_w3'],
                                             decimal=12)
        np.testing.assert_array_almost_equal(np.zeros(1000), out['ts_m'],
                                             decimal=12)

    def test_three_component_beamforming(self):
        """
        Integration test for three-component beamforming with instaseis data
        and the real Parkfield array. Parameter values are fairly arbitrary.
        """
        pfield = SeismicArray('pfield', inventory=read_inventory(
                os.path.join(self.path, 'pfield_inv_for_instaseis.xml'),
                format='stationxml'))
        vel = read(os.path.join(self.path, 'pfield_instaseis.mseed'))
        out = pfield.three_component_beamforming(
            vel.select(channel='BXN'), vel.select(channel='BXE'),
            vel.select(channel='BXZ'), 64, 0, 0.6, 0.03, wavetype='P',
            freq_range=[0.1, .3], whiten=True, coherency=False)
        self.assertEqual(out.max_pow_baz, 246)
        self.assertEqual(out.max_pow_slow, 0.3)
        np.testing.assert_array_almost_equal(out.max_rel_power, 1.22923997,
                                             decimal=8)


def suite():
    return unittest.makeSuite(SeismicArrayTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
