# -*- coding: utf-8 -*-
"""
The obspy.clients.iris.client test suite.
"""
import numpy as np
import pytest

from obspy.core.utcdatetime import UTCDateTime
from obspy.core.util import NamedTemporaryFile
from obspy.core.util.deprecation_helpers import ObsPyDeprecationWarning
from obspy.clients.iris import Client


MSG = "EarthScope has announced the retirement"


@pytest.mark.network
class TestClient():
    """
    Test cases for obspy.clients.iris.client.Client.
    """
    def test_sacpz(self, testdata):
        """
        Fetches SAC poles and zeros information.
        """
        client = Client()
        # 1
        t1 = UTCDateTime("2005-01-01")
        t2 = UTCDateTime("2008-01-01")
        with pytest.warns(ObsPyDeprecationWarning, match=MSG):
            try:
                client.sacpz("IU", "ANMO", "00", "BHZ", t1, t2)
            except Exception as e:
                # expected to start failing soon
                assert 'HTTP Error 410: Gone' in str(e)
        # drop lines with creation date (current time during request)
        # 2 - empty location code
        dt = UTCDateTime("2002-11-01")
        with pytest.warns(ObsPyDeprecationWarning, match=MSG):
            try:
                client.sacpz('UW', 'LON', '', 'BHZ', dt)
            except Exception as e:
                # expected to start failing soon
                assert 'HTTP Error 410: Gone' in str(e)
        # 3 - empty location code via '--'
        with pytest.warns(ObsPyDeprecationWarning, match=MSG):
            try:
                client.sacpz('UW', 'LON', '--', 'BHZ', dt)
            except Exception as e:
                # expected to start failing soon
                assert 'HTTP Error 410: Gone' in str(e)

    def test_distaz(self):
        """
        Tests distance and azimuth calculation between two points on a sphere.
        """
        client = Client()
        # normal request
        with pytest.warns(ObsPyDeprecationWarning, match=MSG):
            try:
                client.distaz(stalat=1.1, stalon=1.2, evtlat=3.2, evtlon=1.4)
            except Exception as e:
                # expected to start failing soon
                assert 'HTTP Error 410: Gone' in str(e)
        # w/o kwargs
        with pytest.warns(ObsPyDeprecationWarning, match=MSG):
            try:
                client.distaz(1.1, 1.2, 3.2, 1.4)
            except Exception as e:
                # expected to start failing soon
                assert 'HTTP Error 410: Gone' in str(e)

    def test_traveltime(self):
        """
        Tests calculation of travel-times for seismic phases.
        """
        client = Client()
        with pytest.warns(ObsPyDeprecationWarning, match=MSG):
            try:
                client.traveltime(
                    evloc=(-36.122, -72.898), evdepth=22.9,
                    staloc=[(-33.45, -70.67), (47.61, -122.33),
                            (35.69, 139.69)])
            except Exception as e:
                # expected to start failing soon
                assert 'HTTP Error 410: Gone' in str(e)

    def test_evalresp(self):
        """
        Tests evaluating instrument response information.

        This is the only custom irisws endpoint that will stay for now.
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
                assert fp.read(4)[1:4] == b'PNG'
        # plot-amp as PNG file
        with NamedTemporaryFile() as tf:
            tempfile = tf.name
            client.evalresp(network="IU", station="ANMO", location="00",
                            channel="BHZ", time=dt, output='plot-amp',
                            filename=tempfile)
            with open(tempfile, 'rb') as fp:
                assert fp.read(4)[1:4] == b'PNG'
        # plot-phase as PNG file
        with NamedTemporaryFile() as tf:
            tempfile = tf.name
            client.evalresp(network="IU", station="ANMO", location="00",
                            channel="BHZ", time=dt, output='plot-phase',
                            filename=tempfile)
            with open(tempfile, 'rb') as fp:
                assert fp.read(4)[1:4] == b'PNG'
        # fap as ASCII file
        with NamedTemporaryFile() as tf:
            tempfile = tf.name
            client.evalresp(network="IU", station="ANMO", location="00",
                            channel="BHZ", time=dt, output='fap',
                            filename=tempfile)
            with open(tempfile, 'rt') as fp:
                assert fp.readline() == \
                                 '1.000000E-05 1.055934E+04 1.792007E+02\n'
        # cs as ASCII file
        with NamedTemporaryFile() as tf:
            tempfile = tf.name
            client.evalresp(network="IU", station="ANMO", location="00",
                            channel="BHZ", time=dt, output='cs',
                            filename=tempfile)
            with open(tempfile, 'rt') as fp:
                assert fp.readline() == \
                                 '1.000000E-05  -1.055831E+04  1.472963E+02\n'
        # fap & def as ASCII file
        with NamedTemporaryFile() as tf:
            tempfile = tf.name
            client.evalresp(network="IU", station="ANMO", location="00",
                            channel="BHZ", time=dt, output='fap', units='def',
                            filename=tempfile)
            with open(tempfile, 'rt') as fp:
                assert fp.readline() == \
                                 '1.000000E-05 1.055934E+04 1.792007E+02\n'
        # fap & dis as ASCII file
        with NamedTemporaryFile() as tf:
            tempfile = tf.name
            client.evalresp(network="IU", station="ANMO", location="00",
                            channel="BHZ", time=dt, output='fap', units='dis',
                            filename=tempfile)
            with open(tempfile, 'rt') as fp:
                assert fp.readline() == \
                                 '1.000000E-05 6.634627E-01 2.692007E+02\n'
        # fap & vel as ASCII file
        with NamedTemporaryFile() as tf:
            tempfile = tf.name
            client.evalresp(network="IU", station="ANMO", location="00",
                            channel="BHZ", time=dt, output='fap', units='vel',
                            filename=tempfile)
            with open(tempfile, 'rt') as fp:
                assert fp.readline() == \
                                 '1.000000E-05 1.055934E+04 1.792007E+02\n'
        # fap & acc as ASCII file
        with NamedTemporaryFile() as tf:
            tempfile = tf.name
            client.evalresp(network="IU", station="ANMO", location="00",
                            channel="BHZ", time=dt, output='fap', units='acc',
                            filename=tempfile)
            with open(tempfile, 'rt') as fp:
                assert fp.readline() == \
                                 '1.000000E-05 1.680571E+08 8.920073E+01\n'
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

        Examples are inspired by
        https://service.earthscope.org/irisws/resp/1/.
        """
        client = Client()
        # 1
        t1 = UTCDateTime("2005-001T00:00:00")
        t2 = UTCDateTime("2008-001T00:00:00")
        with pytest.warns(ObsPyDeprecationWarning, match=MSG):
            try:
                client.resp("IU", "ANMO", "00", "BHZ", t1, t2)
            except Exception as e:
                # expected to start failing soon
                assert 'HTTP Error 410: Gone' in str(e)

        dt = UTCDateTime("2010-02-27T06:30:00.000")
        with pytest.warns(ObsPyDeprecationWarning, match=MSG):
            try:
                client.resp("IU", "ANMO", "*", "*", dt)
            except Exception as e:
                # expected to start failing soon
                assert 'HTTP Error 410: Gone' in str(e)

        dt = UTCDateTime("2005-001T00:00:00")
        with pytest.warns(ObsPyDeprecationWarning, match=MSG):
            try:
                client.resp("AK", "RIDG", "--", "LH?", dt)
            except Exception as e:
                # expected to start failing soon
                assert 'HTTP Error 410: Gone' in str(e)

    def test_timeseries(self):
        """
        Tests timeseries Web service interface.

        Examples are inspired by
        https://service.earthscope.org/irisws/timeseries/1/.
        """
        client = Client()
        # 1
        t1 = UTCDateTime("2005-001T00:00:00")
        t2 = UTCDateTime("2005-001T00:01:00")
        # no filter
        with pytest.warns(ObsPyDeprecationWarning, match=MSG):
            with pytest.raises(Exception, match="410: Gone"):
                client.timeseries("IU", "ANMO", "00", "BHZ", t1, t2)
        # instrument corrected
        with pytest.warns(ObsPyDeprecationWarning, match=MSG):
            with pytest.raises(Exception, match="410: Gone"):
                client.timeseries("IU", "ANMO", "00", "BHZ", t1, t2,
                                  filter=["correct"])

    def test_flinnengdahl_deprecation_warning(self):
        client = Client()
        msg = ('EarthScope has announced the retirement')
        with pytest.warns(ObsPyDeprecationWarning, match=msg):
            with pytest.raises(Exception, match="410: Gone"):
                client.flinnengdahl(lat=-20.5, lon=-100.6, rtype="code")
