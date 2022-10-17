# -*- coding: utf-8 -*-
"""
The obspy.clients.iris.client test suite.
"""
import os
import unittest

import numpy as np
import pytest

from obspy.core.utcdatetime import UTCDateTime
from obspy.core.util import NamedTemporaryFile
from obspy.clients.iris import Client

pytestmark = pytest.mark.network


class ClientTestCase(unittest.TestCase):
    """
    Test cases for obspy.clients.iris.client.Client.
    """

    def setUp(self):
        # directory where the test files are located
        self.path = os.path.dirname(__file__)

    def test_sacpz(self):
        """
        Fetches SAC poles and zeros information.
        """
        client = Client()
        # 1
        t1 = UTCDateTime("2005-01-01")
        t2 = UTCDateTime("2008-01-01")
        result = client.sacpz("IU", "ANMO", "00", "BHZ", t1, t2)
        # drop lines with creation date (current time during request)
        result = result.splitlines()
        sacpz_file = os.path.join(self.path, 'data', 'IU.ANMO.00.BHZ.sacpz')
        with open(sacpz_file, 'rb') as fp:
            expected = fp.read().splitlines()
        result.pop(5)
        expected.pop(5)
        self.assertEqual(result, expected)
        # 2 - empty location code
        dt = UTCDateTime("2002-11-01")
        result = client.sacpz('UW', 'LON', '', 'BHZ', dt)
        self.assertIn(b"* STATION    (KSTNM): LON", result)
        self.assertIn(b"* LOCATION   (KHOLE):", result)
        # 3 - empty location code via '--'
        result = client.sacpz('UW', 'LON', '--', 'BHZ', dt)
        self.assertIn(b"* STATION    (KSTNM): LON", result)
        self.assertIn(b"* LOCATION   (KHOLE):", result)

    def test_distaz(self):
        """
        Tests distance and azimuth calculation between two points on a sphere.
        """
        client = Client()
        # normal request
        result = client.distaz(stalat=1.1, stalon=1.2, evtlat=3.2, evtlon=1.4)
        self.assertAlmostEqual(result['distance'], 2.10256)
        self.assertAlmostEqual(result['distancemeters'], 233272.79028)
        self.assertAlmostEqual(result['backazimuth'], 5.46944)
        self.assertAlmostEqual(result['azimuth'], 185.47695)
        self.assertEqual(result['ellipsoidname'], 'WGS84')
        self.assertTrue(isinstance(result['distance'], float))
        self.assertTrue(isinstance(result['distancemeters'], float))
        self.assertTrue(isinstance(result['backazimuth'], float))
        self.assertTrue(isinstance(result['azimuth'], float))
        self.assertTrue(isinstance(result['ellipsoidname'], str))
        # w/o kwargs
        result = client.distaz(1.1, 1.2, 3.2, 1.4)
        self.assertAlmostEqual(result['distance'], 2.10256)
        self.assertAlmostEqual(result['distancemeters'], 233272.79028)
        self.assertAlmostEqual(result['backazimuth'], 5.46944)
        self.assertAlmostEqual(result['azimuth'], 185.47695)
        self.assertEqual(result['ellipsoidname'], 'WGS84')
        # missing parameters
        self.assertRaises(Exception, client.distaz, stalat=1.1)
        self.assertRaises(Exception, client.distaz, 1.1)
        self.assertRaises(Exception, client.distaz, stalat=1.1, stalon=1.2)
        self.assertRaises(Exception, client.distaz, 1.1, 1.2)

    def test_flinnengdahl(self):
        """
        Tests calculation of Flinn-Engdahl region code or name.
        """
        client = Client()
        # code
        result = client.flinnengdahl(lat=-20.5, lon=-100.6, rtype="code")
        self.assertEqual(result, 683)
        self.assertTrue(isinstance(result, int))
        # w/o kwargs
        result = client.flinnengdahl(-20.5, -100.6, "code")
        self.assertEqual(result, 683)
        # region
        result = client.flinnengdahl(lat=42, lon=-122.24, rtype="region")
        self.assertEqual(result, 'OREGON')
        self.assertTrue(isinstance(result, str))
        # w/o kwargs
        result = client.flinnengdahl(42, -122.24, "region")
        self.assertEqual(result, 'OREGON')
        # both
        result = client.flinnengdahl(lat=-20.5, lon=-100.6, rtype="both")
        self.assertEqual(result, (683, 'SOUTHEAST CENTRAL PACIFIC OCEAN'))
        self.assertTrue(isinstance(result[0], int))
        self.assertTrue(isinstance(result[1], str))
        # w/o kwargs
        result = client.flinnengdahl(-20.5, -100.6, "both")
        self.assertEqual(result, (683, 'SOUTHEAST CENTRAL PACIFIC OCEAN'))
        # default rtype
        result = client.flinnengdahl(lat=42, lon=-122.24)
        self.assertEqual(result, (32, 'OREGON'))
        # w/o kwargs
        # outside boundaries
        self.assertRaises(Exception, client.flinnengdahl, lat=-90.1, lon=0)
        self.assertRaises(Exception, client.flinnengdahl, lat=90.1, lon=0)
        self.assertRaises(Exception, client.flinnengdahl, lat=0, lon=-180.1)
        self.assertRaises(Exception, client.flinnengdahl, lat=0, lon=180.1)

    def test_traveltime(self):
        """
        Tests calculation of travel-times for seismic phases.
        """
        client = Client()
        result = client.traveltime(
            evloc=(-36.122, -72.898), evdepth=22.9,
            staloc=[(-33.45, -70.67), (47.61, -122.33), (35.69, 139.69)])
        self.assertTrue(result.startswith(b'Model: iasp91'))

    def test_evalresp(self):
        """
        Tests evaluating instrument response information.
        """
        client = Client()
        dt = UTCDateTime("2005-01-01")
        # plot as PNG file
        with NamedTemporaryFile() as tf:
            tempfile = tf.name
            client.evalresp(network="IU", station="ANMO", location="00",
                            channel="BHZ", time=dt, output='plot',
                            filename=tempfile)
            with open(tempfile, 'rb') as fp:
                self.assertEqual(fp.read(4)[1:4], b'PNG')
        # plot-amp as PNG file
        with NamedTemporaryFile() as tf:
            tempfile = tf.name
            client.evalresp(network="IU", station="ANMO", location="00",
                            channel="BHZ", time=dt, output='plot-amp',
                            filename=tempfile)
            with open(tempfile, 'rb') as fp:
                self.assertEqual(fp.read(4)[1:4], b'PNG')
        # plot-phase as PNG file
        with NamedTemporaryFile() as tf:
            tempfile = tf.name
            client.evalresp(network="IU", station="ANMO", location="00",
                            channel="BHZ", time=dt, output='plot-phase',
                            filename=tempfile)
            with open(tempfile, 'rb') as fp:
                self.assertEqual(fp.read(4)[1:4], b'PNG')
        # fap as ASCII file
        with NamedTemporaryFile() as tf:
            tempfile = tf.name
            client.evalresp(network="IU", station="ANMO", location="00",
                            channel="BHZ", time=dt, output='fap',
                            filename=tempfile)
            with open(tempfile, 'rt') as fp:
                self.assertEqual(fp.readline(),
                                 '1.000000E-05 1.055934E+04 1.792007E+02\n')
        # cs as ASCII file
        with NamedTemporaryFile() as tf:
            tempfile = tf.name
            client.evalresp(network="IU", station="ANMO", location="00",
                            channel="BHZ", time=dt, output='cs',
                            filename=tempfile)
            with open(tempfile, 'rt') as fp:
                self.assertEqual(fp.readline(),
                                 '1.000000E-05  -1.055831E+04  1.472963E+02\n')
        # fap & def as ASCII file
        with NamedTemporaryFile() as tf:
            tempfile = tf.name
            client.evalresp(network="IU", station="ANMO", location="00",
                            channel="BHZ", time=dt, output='fap', units='def',
                            filename=tempfile)
            with open(tempfile, 'rt') as fp:
                self.assertEqual(fp.readline(),
                                 '1.000000E-05 1.055934E+04 1.792007E+02\n')
        # fap & dis as ASCII file
        with NamedTemporaryFile() as tf:
            tempfile = tf.name
            client.evalresp(network="IU", station="ANMO", location="00",
                            channel="BHZ", time=dt, output='fap', units='dis',
                            filename=tempfile)
            with open(tempfile, 'rt') as fp:
                self.assertEqual(fp.readline(),
                                 '1.000000E-05 6.634627E-01 2.692007E+02\n')
        # fap & vel as ASCII file
        with NamedTemporaryFile() as tf:
            tempfile = tf.name
            client.evalresp(network="IU", station="ANMO", location="00",
                            channel="BHZ", time=dt, output='fap', units='vel',
                            filename=tempfile)
            with open(tempfile, 'rt') as fp:
                self.assertEqual(fp.readline(),
                                 '1.000000E-05 1.055934E+04 1.792007E+02\n')
        # fap & acc as ASCII file
        with NamedTemporaryFile() as tf:
            tempfile = tf.name
            client.evalresp(network="IU", station="ANMO", location="00",
                            channel="BHZ", time=dt, output='fap', units='acc',
                            filename=tempfile)
            with open(tempfile, 'rt') as fp:
                self.assertEqual(fp.readline(),
                                 '1.000000E-05 1.680571E+08 8.920073E+01\n')
        # fap as NumPy ndarray
        data = client.evalresp(network="IU", station="ANMO", location="00",
                               channel="BHZ", time=dt, output='fap')
        np.testing.assert_array_equal(
            data[0], [1.00000000e-05, 1.05593400e+04, 1.79200700e+02])
        # cs as NumPy ndarray
        data = client.evalresp(network="IU", station="ANMO", location="00",
                               channel="BHZ", time=dt, output='cs')
        np.testing.assert_array_equal(
            data[0], [1.00000000e-05, -1.05583100e+04, 1.472963e+02])

    def test_resp(self):
        """
        Tests resp Web service interface.

        Examples are inspired by https://service.iris.edu/irisws/resp/1/.
        """
        client = Client()
        # 1
        t1 = UTCDateTime("2005-001T00:00:00")
        t2 = UTCDateTime("2008-001T00:00:00")
        result = client.resp("IU", "ANMO", "00", "BHZ", t1, t2)
        self.assertIn(b'B050F03     Station:     ANMO', result)
        # Exception: No response data available
        # 2 - empty location code
        # result = client.resp("UW", "LON", "", "EHZ")
        # self.assertIn(b'B050F03     Station:     LON', result)
        # self.assertIn(b'B052F03     Location:    ??', result)
        # 3 - empty location code via '--'
        # result = client.resp("UW", "LON", "--", "EHZ")
        # self.assertIn(b'B050F03     Station:     LON', result)
        # self.assertIn(b'B052F03     Location:    ??', result)
        # 4
        dt = UTCDateTime("2010-02-27T06:30:00.000")
        result = client.resp("IU", "ANMO", "*", "*", dt)
        self.assertIn(b'B050F03     Station:     ANMO', result)

        dt = UTCDateTime("2005-001T00:00:00")
        result = client.resp("AK", "RIDG", "--", "LH?", dt)
        self.assertIn(b'B050F03     Station:     RIDG', result)

    def test_timeseries(self):
        """
        Tests timeseries Web service interface.

        Examples are inspired by https://service.iris.edu/irisws/timeseries/1/.
        """
        client = Client()
        # 1
        t1 = UTCDateTime("2005-001T00:00:00")
        t2 = UTCDateTime("2005-001T00:01:00")
        # no filter
        st1 = client.timeseries("IU", "ANMO", "00", "BHZ", t1, t2)
        # instrument corrected
        st2 = client.timeseries("IU", "ANMO", "00", "BHZ", t1, t2,
                                filter=["correct"])
        # compare results
        self.assertEqual(st1[0].stats.starttime, st2[0].stats.starttime)
        self.assertEqual(st1[0].stats.endtime, st2[0].stats.endtime)
        self.assertEqual(st1[0].data[0], 24)
        self.assertAlmostEqual(st2[0].data[0], -2.8373747e-06)
