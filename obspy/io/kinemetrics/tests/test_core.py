#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The obspy.io.kinemetrics.core test suite.
"""
import io
import os
import unittest

import numpy as np

from obspy import read
from obspy.core.utcdatetime import UTCDateTime
from obspy.io.kinemetrics.core import is_evt, read_evt


class CoreTestCase(unittest.TestCase):
    """
    Test cases for kinemetrics core interface
    """
    def setUp(self):
        # directory where the test files are located
        self.path = os.path.join(os.path.dirname(__file__), 'data')

    def test_is_evt(self):
        """
        Test for the is_evt() function.
        """
        valid_files = [os.path.join(self.path, "BI008_MEMA-04823.evt"),
                       os.path.join(self.path, "BX456_MOLA-02351.evt")]
        invalid_files = [os.path.join(self.path, "NOUTF8.evt")]
        py_dir = os.path.join(self.path, os.pardir, os.pardir)
        for filename in os.listdir(py_dir):
            if filename.endswith(".py"):
                invalid_files.append(
                    os.path.abspath(os.path.join(py_dir, filename)))
        self.assertGreater(len(invalid_files), 0)

        for filename in valid_files:
            self.assertTrue(is_evt(filename))
        for filename in invalid_files:
            self.assertFalse(is_evt(filename))

    def test_is_evt_from_bytesio(self):
        """
        Test for the is_evt() function from BytesIO objects.
        """
        valid_files = [os.path.join(self.path, "BI008_MEMA-04823.evt"),
                       os.path.join(self.path, "BX456_MOLA-02351.evt")]
        invalid_files = [os.path.join(self.path, "NOUTF8.evt")]
        py_dir = os.path.join(self.path, os.pardir, os.pardir)
        for filename in os.listdir(py_dir):
            if filename.endswith(".py"):
                invalid_files.append(
                    os.path.abspath(os.path.join(py_dir, filename)))

        for filename in valid_files:
            with open(filename, "rb") as fh:
                buf = io.BytesIO(fh.read())
            buf.seek(0, 0)
            self.assertTrue(is_evt(buf))
            # The is_evt() method should not change the file pointer.
            self.assertEqual(buf.tell(), 0)
        for filename in invalid_files:
            with open(filename, "rb") as fh:
                buf = io.BytesIO(fh.read())
            buf.seek(0, 0)
            self.assertFalse(is_evt(buf))
            # The is_evt() method should not change the file pointer.
            self.assertEqual(buf.tell(), 0)

    def test_read_via_obspy(self):
        """
        Read files via obspy.core.stream.read function.
        """
        filename = os.path.join(self.path, 'BI008_MEMA-04823.evt')
        # 1
        st = read(filename, apply_calib=True)
        st.verify()
        self.assertEqual(len(st), 3)
        self.assertEqual(st[0].stats.starttime,
                         UTCDateTime('2013-08-15T09:20:28.000000Z'))
        self.assertEqual(st[1].stats.starttime,
                         UTCDateTime('2013-08-15T09:20:28.000000Z'))
        self.assertEqual(st[2].stats.starttime,
                         UTCDateTime('2013-08-15T09:20:28.000000Z'))
        self.assertEqual(len(st[0]), 230 * 25)
        self.assertAlmostEqual(st[0].stats.sampling_rate, 250.0)
        self.assertEqual(st[0].stats.channel, '0')
        self.assertEqual(st[0].stats.station, 'MEMA')

        self.verify_stats_evt(st[0].stats.kinemetrics_evt)
        self.verify_data_evt0(st[0].data)
        self.verify_data_evt2(st[2].data)

        # 2
        filename = os.path.join(self.path, 'BX456_MOLA-02351.evt')
        st = read(filename, apply_calib=True)
        st.verify()
        self.assertEqual(len(st), 6)
        self.assertEqual(st[0].stats.starttime,
                         UTCDateTime('2012-01-17T09:54:36.000000Z'))
        self.assertEqual(st[1].stats.starttime,
                         UTCDateTime('2012-01-17T09:54:36.000000Z'))
        self.assertEqual(st[2].stats.starttime,
                         UTCDateTime('2012-01-17T09:54:36.000000Z'))
        self.assertEqual(st[3].stats.starttime,
                         UTCDateTime('2012-01-17T09:54:36.000000Z'))
        self.assertEqual(st[4].stats.starttime,
                         UTCDateTime('2012-01-17T09:54:36.000000Z'))
        self.assertEqual(st[5].stats.starttime,
                         UTCDateTime('2012-01-17T09:54:36.000000Z'))
        self.assertEqual(len(st[0]), 390 * 25)
        self.assertAlmostEqual(st[0].stats.sampling_rate, 250.0)
        self.assertEqual(st[0].stats.channel, '0')
        self.assertEqual(st[0].stats.station, 'MOLA')

    def test_reading_via_obspy_and_bytesio(self):
        """
        Test the reading of Evt files from BytesIO objects.
        """
        # 1
        filename = os.path.join(self.path, 'BI008_MEMA-04823.evt')
        with open(filename, "rb") as fh:
            buf = io.BytesIO(fh.read())
        buf.seek(0, 0)
        st = read(buf, apply_calib=True)
        st.verify()
        self.assertEqual(len(st), 3)
        self.assertEqual(st[0].stats.starttime,
                         UTCDateTime('2013-08-15T09:20:28.000000Z'))
        self.assertEqual(st[1].stats.starttime,
                         UTCDateTime('2013-08-15T09:20:28.000000Z'))
        self.assertEqual(st[2].stats.starttime,
                         UTCDateTime('2013-08-15T09:20:28.000000Z'))
        self.assertEqual(len(st[0]), 230 * 25)
        self.assertAlmostEqual(st[0].stats.sampling_rate, 250.0)
        self.assertEqual(st[0].stats.channel, '0')
        self.assertEqual(st[0].stats.station, 'MEMA')

        self.verify_stats_evt(st[0].stats.kinemetrics_evt)
        self.verify_data_evt0(st[0].data)
        self.verify_data_evt2(st[2].data)

        # 2
        filename = os.path.join(self.path, 'BX456_MOLA-02351.evt')
        with open(filename, "rb") as fh:
            buf = io.BytesIO(fh.read())
        buf.seek(0, 0)
        st = read(buf, apply_calib=True)
        st.verify()
        self.assertEqual(len(st), 6)
        self.assertEqual(st[0].stats.starttime,
                         UTCDateTime('2012-01-17T09:54:36.000000Z'))
        self.assertEqual(st[1].stats.starttime,
                         UTCDateTime('2012-01-17T09:54:36.000000Z'))
        self.assertEqual(st[2].stats.starttime,
                         UTCDateTime('2012-01-17T09:54:36.000000Z'))
        self.assertEqual(st[3].stats.starttime,
                         UTCDateTime('2012-01-17T09:54:36.000000Z'))
        self.assertEqual(st[4].stats.starttime,
                         UTCDateTime('2012-01-17T09:54:36.000000Z'))
        self.assertEqual(st[5].stats.starttime,
                         UTCDateTime('2012-01-17T09:54:36.000000Z'))
        self.assertEqual(len(st[0]), 390 * 25)
        self.assertAlmostEqual(st[0].stats.sampling_rate, 250.0)
        self.assertEqual(st[0].stats.channel, '0')
        self.assertEqual(st[0].stats.station, 'MOLA')

    def test_read_via_module(self):
        """
        Read files via obspy.io.kinemetrics.core.read_evt function.
        """
        filename = os.path.join(self.path, 'BI008_MEMA-04823.evt')
        # 1
        st = read_evt(filename, apply_calib=True)
        st.verify()
        self.assertEqual(len(st), 3)
        self.assertEqual(st[0].stats.starttime,
                         UTCDateTime('2013-08-15T09:20:28.000000Z'))
        self.assertEqual(st[1].stats.starttime,
                         UTCDateTime('2013-08-15T09:20:28.000000Z'))
        self.assertEqual(st[2].stats.starttime,
                         UTCDateTime('2013-08-15T09:20:28.000000Z'))
        self.assertEqual(len(st[0]), 230 * 25)
        self.assertAlmostEqual(st[0].stats.sampling_rate, 250.0)
        self.assertEqual(st[0].stats.channel, '0')
        self.assertEqual(st[0].stats.station, 'MEMA')

        self.verify_stats_evt(st[0].stats.kinemetrics_evt)
        self.verify_data_evt0(st[0].data)
        self.verify_data_evt2(st[2].data)

        # 2
        filename = os.path.join(self.path, 'BX456_MOLA-02351.evt')
        st = read_evt(filename, apply_calib=True)
        st.verify()
        self.assertEqual(len(st), 6)
        self.assertEqual(st[0].stats.starttime,
                         UTCDateTime('2012-01-17T09:54:36.000000Z'))
        self.assertEqual(st[1].stats.starttime,
                         UTCDateTime('2012-01-17T09:54:36.000000Z'))
        self.assertEqual(st[2].stats.starttime,
                         UTCDateTime('2012-01-17T09:54:36.000000Z'))
        self.assertEqual(st[3].stats.starttime,
                         UTCDateTime('2012-01-17T09:54:36.000000Z'))
        self.assertEqual(st[4].stats.starttime,
                         UTCDateTime('2012-01-17T09:54:36.000000Z'))
        self.assertEqual(st[5].stats.starttime,
                         UTCDateTime('2012-01-17T09:54:36.000000Z'))
        self.assertEqual(len(st[0]), 390 * 25)
        self.assertAlmostEqual(st[0].stats.sampling_rate, 250.0)
        self.assertEqual(st[0].stats.channel, '0')
        self.assertEqual(st[0].stats.station, 'MOLA')

    def test_read_via_module_and_bytesio(self):
        """
        Read files via obspy.io.kinemetrics.core.read_evt function from BytesIO
        objects.
        """
        # 1
        filename = os.path.join(self.path, 'BI008_MEMA-04823.evt')
        with open(filename, "rb") as fh:
            buf = io.BytesIO(fh.read())
        buf.seek(0, 0)
        st = read_evt(buf, apply_calib=True)
        st.verify()
        self.assertEqual(len(st), 3)
        self.assertEqual(st[0].stats.starttime,
                         UTCDateTime('2013-08-15T09:20:28.000000Z'))
        self.assertEqual(st[1].stats.starttime,
                         UTCDateTime('2013-08-15T09:20:28.000000Z'))
        self.assertEqual(st[2].stats.starttime,
                         UTCDateTime('2013-08-15T09:20:28.000000Z'))
        self.assertEqual(len(st[0]), 230 * 25)
        self.assertAlmostEqual(st[0].stats.sampling_rate, 250.0)
        self.assertEqual(st[0].stats.channel, '0')
        self.assertEqual(st[0].stats.station, 'MEMA')

        self.verify_stats_evt(st[0].stats.kinemetrics_evt)
        self.verify_data_evt0(st[0].data)
        self.verify_data_evt2(st[2].data)

        # 2
        filename = os.path.join(self.path, 'BX456_MOLA-02351.evt')
        with open(filename, "rb") as fh:
            buf = io.BytesIO(fh.read())
        buf.seek(0, 0)
        st = read_evt(buf, apply_calib=True)
        st.verify()
        self.assertEqual(len(st), 6)
        self.assertEqual(st[0].stats.starttime,
                         UTCDateTime('2012-01-17T09:54:36.000000Z'))
        self.assertEqual(st[1].stats.starttime,
                         UTCDateTime('2012-01-17T09:54:36.000000Z'))
        self.assertEqual(st[2].stats.starttime,
                         UTCDateTime('2012-01-17T09:54:36.000000Z'))
        self.assertEqual(st[3].stats.starttime,
                         UTCDateTime('2012-01-17T09:54:36.000000Z'))
        self.assertEqual(st[4].stats.starttime,
                         UTCDateTime('2012-01-17T09:54:36.000000Z'))
        self.assertEqual(st[5].stats.starttime,
                         UTCDateTime('2012-01-17T09:54:36.000000Z'))
        self.assertEqual(len(st[0]), 390 * 25)
        self.assertAlmostEqual(st[0].stats.sampling_rate, 250.0)
        self.assertEqual(st[0].stats.channel, '0')
        self.assertEqual(st[0].stats.station, 'MOLA')

    def verify_stats_evt(self, evt_stats):
        dico = {'chan_fullscale': 2.5, 'chan_sensorgain': 1,
                'chan_calcoil': 0.0500, 'chan_damping': 0.7070,
                'chan_natfreq': 196.00,
                'latitude': 50.609795, 'longitude': 6.009250,
                'elevation': 298}
        # Values from Kinemetrics QLWin program for BI008_MEMA-04823.evt

        for key in dico:
            self.assertAlmostEqual(dico[key], evt_stats[key], 6)

        self.assertEqual(UTCDateTime(2013, 8, 15, 9, 20, 28),
                         evt_stats['starttime'])

    def verify_data_evt0(self, data):
        valuesdeb = np.array([-2.4464752525e-002, -2.4534918368e-002,
                              -2.4467090145e-002, -2.4511529133e-002,
                              -2.4478785694e-002, -2.4483462796e-002,
                              -2.4434346706e-002, -2.4504512548e-002,
                              -2.4527901784e-002, -2.4455396459e-002,
                              -2.4509189650e-002, -2.4478785694e-002,
                              -2.4541934952e-002, -2.4497495964e-002,
                              -2.4448379874e-002, -2.4502173066e-002,
                              -2.4420313537e-002, -2.4455396459e-002,
                              -2.4546612054e-002, -2.4509189650e-002])
        valuesend = np.array([-2.4488139898e-002, -2.4530241266e-002,
                              -2.4525562301e-002, -2.4506852031e-002,
                              -2.4424990639e-002])
        # Data values from Tsoft Program

        self.assertEqual(len(data), 5750)
        self.assertTrue(np.allclose(valuesdeb, data[:len(valuesdeb)]))
        self.assertTrue(np.allclose(valuesend, data[-len(valuesend):]))

    def verify_data_evt2(self, data):
        valuesdeb = np.array([-4.4351171702e-002, -4.4479820877e-002,
                              -4.4447075576e-002, -4.4367544353e-002,
                              -4.4402632862e-002, -4.4386260211e-002,
                              -4.4360529631e-002, -4.4440057129e-002,
                              -4.4411987066e-002, -4.4395614415e-002,
                              -4.4421344995e-002, -4.4433038682e-002,
                              -4.4442396611e-002, -4.4423684478e-002,
                              -4.4428363442e-002, -4.4419005513e-002,
                              -4.4388595968e-002, -4.4360529631e-002,
                              -4.4358190149e-002, -4.4362869114e-002])
        valuesend = np.array([-4.4538296759e-002, -4.4549994171e-002,
                              -4.4493857771e-002, -4.4451754540e-002,
                              -4.4409647584e-002])

        # Data values from Tsoft Program
        # length is 5750

        self.assertEqual(len(data), 5750)
        self.assertTrue(np.allclose(valuesdeb, data[:len(valuesdeb)]))
        self.assertTrue(np.allclose(valuesend, data[-len(valuesend):]))

    def test_read_via_module_raw(self):
        """
        Read files via obspy.io.kinemetrics.core.read_evt function.
        """
        filename = os.path.join(self.path, 'BI008_MEMA-04823.evt')
        # 1
        st = read_evt(filename, apply_calib=False)
        st.verify()
        self.assertEqual(len(st), 3)
        self.assertEqual(st[0].stats.starttime,
                         UTCDateTime('2013-08-15T09:20:28.000000Z'))
        self.assertEqual(st[1].stats.starttime,
                         UTCDateTime('2013-08-15T09:20:28.000000Z'))
        self.assertEqual(st[2].stats.starttime,
                         UTCDateTime('2013-08-15T09:20:28.000000Z'))
        self.assertEqual(len(st[0]), 230 * 25)
        self.assertAlmostEqual(st[0].stats.sampling_rate, 250.0)
        self.assertEqual(st[0].stats.channel, '0')
        self.assertEqual(st[0].stats.station, 'MEMA')

        self.verify_stats_evt(st[0].stats.kinemetrics_evt)
        self.verify_data_evt0_raw(st[0].data)
        self.verify_data_evt2_raw(st[2].data)

    def verify_data_evt0_raw(self, data):
        valuesdeb = np.array([-20920., -20980., -20922., -20960.,
                             -20932., -20936., -20894., -20954.,
                             -20974., -20912., -20958., -20932.,
                             -20986., -20948., -20906., -20952.,
                             -20882., -20912., -20990., -20958.])
        valuesend = np.array([-21070., -20962., -20930., -20918.,
                              -20964., -20910., -20934., -21026.,
                              -20968., -20956., -20976., -20954.,
                              -20954., -21000., -20966., -20940.,
                              -20976., -20972., -20956., -20886.])
        # Data values from Tsoft Program

        self.assertEqual(len(data), 5750)
        self.assertTrue(np.allclose(valuesdeb, data[:len(valuesdeb)]))
        self.assertTrue(np.allclose(valuesend, data[-len(valuesend):]))

    def verify_data_evt2_raw(self, data):
        valuesdeb = np.array([-37922., -38032., -38004., -37936.,
                              -37966., -37952., -37930., -37998.,
                              -37974., -37960., -37982., -37992.,
                              -38000., -37984., -37988., -37980.,
                              -37954., -37930., -37928., -37932.])
        valuesend = np.array([-37996., -38010., -38024., -38048.,
                              -37960., -37878., -37842., -37864.,
                              -37902., -37908., -38038., -38070.,
                              -38126., -38170., -38070., -38082.,
                              -38092., -38044., -38008., -37972.])

        # Data values from Tsoft Program
        # length is 5750

        self.assertEqual(len(data), 5750)
        self.assertTrue(np.allclose(valuesdeb, data[:len(valuesdeb)]))
        self.assertTrue(np.allclose(valuesend, data[-len(valuesend):]))
