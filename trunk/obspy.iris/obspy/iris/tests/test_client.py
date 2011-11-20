# -*- coding: utf-8 -*-
"""
The obspy.iris.client test suite.
"""

from obspy.core.utcdatetime import UTCDateTime
from obspy.core.util import NamedTemporaryFile
from obspy.iris import Client
import filecmp
import numpy as np
import os
import unittest
import urllib


class ClientTestCase(unittest.TestCase):
    """
    Test cases for obspy.iris.client.Client.
    """
    def setUp(self):
        # directory where the test files are located
        self.path = os.path.dirname(__file__)

    def test_getWaveform(self):
        """
        Testing simple waveform request method.
        """
        # simple example
        client = Client()
        start = UTCDateTime("2010-02-27T06:30:00.019538Z")
        end = start + 20
        stream = client.getWaveform("IU", "ANMO", "00", "BHZ", start, end)
        self.assertEquals(len(stream), 1)
        self.assertEquals(stream[0].stats.starttime, start)
        self.assertEquals(stream[0].stats.endtime, end)
        self.assertEquals(stream[0].stats.network, 'IU')
        self.assertEquals(stream[0].stats.station, 'ANMO')
        self.assertEquals(stream[0].stats.location, '00')
        self.assertEquals(stream[0].stats.channel, 'BHZ')
        # no data raises an exception
        self.assertRaises(Exception, client.getWaveform, "YY", "XXXX", "00",
                          "BHZ", start, end)

    def test_saveWaveform(self):
        """
        Testing simple waveform file save method.
        """
        # file identical to file retrieved via web interface
        client = Client()
        start = UTCDateTime("2010-02-27T06:30:00")
        end = UTCDateTime("2010-02-27T06:31:00")
        origfile = os.path.join(self.path, 'data', 'IU.ANMO.00.BHZ.mseed')
        tempfile = NamedTemporaryFile().name
        client.saveWaveform(tempfile, "IU", "ANMO", "00", "BHZ", start, end)
        self.assertTrue(filecmp.cmp(origfile, tempfile))
        os.remove(tempfile)
        # no data raises an exception
        self.assertRaises(Exception, client.saveWaveform, "YY", "XXXX", "00",
                          "BHZ", start, end)

    def test_saveResponse(self):
        """
        Fetches and stores response information as SEED RESP file.
        """
        client = Client()
        start = UTCDateTime("2005-001T00:00:00")
        end = UTCDateTime("2008-001T00:00:00")
        # RESP, single channel
        origfile = os.path.join(self.path, 'data', 'RESP.ANMO.IU.00.BHZ')
        tempfile = NamedTemporaryFile().name
        client.saveResponse(tempfile, "IU", "ANMO", "00", "BHZ", start, end)
        self.assertTrue(filecmp.cmp(origfile, tempfile))
        os.remove(tempfile)
        # RESP, multiple channels
        origfile = os.path.join(self.path, 'data', 'RESP.ANMO.IU._.BH_')
        tempfile = NamedTemporaryFile().name
        client.saveResponse(tempfile, "IU", "ANMO", "*", "BH?", start, end)
        self.assertTrue(filecmp.cmp(origfile, tempfile))
        os.remove(tempfile)
        # StationXML, single channel
        tempfile = NamedTemporaryFile().name
        client.saveResponse(tempfile, "IU", "ANMO", "00", "BHZ", start, end,
                            format="StationXML")
        data = open(tempfile).read()
        self.assertTrue('<Station net_code="IU" sta_code="ANMO">' in data)
        os.remove(tempfile)
        # SACPZ, single channel
        tempfile = NamedTemporaryFile().name
        client.saveResponse(tempfile, "IU", "ANMO", "00", "BHZ", start, end,
                            format="SACPZ")
        data = open(tempfile).read()
        self.assertTrue('NETWORK   (KNETWK): IU' in data)
        self.assertTrue('STATION    (KSTNM): ANMO' in data)
        os.remove(tempfile)

    def test_sacpz(self):
        """
        Fetches SAC poles and zeros information.
        Can not be tested in docstring because creation date on server is
        included in resulting text.
        """
        client = Client()
        t1 = UTCDateTime("2005-01-01")
        t2 = UTCDateTime("2008-01-01")
        got = client.sacpz(network="IU", station="ANMO", location="00",
                           channel="BHZ", starttime=t1, endtime=t2)
        got = got.splitlines()
        sacpz_file = os.path.join(self.path, 'data', 'IU.ANMO.00.BHZ.sacpz')
        expected = open(sacpz_file, 'rt').read().splitlines()
        # drop lines with creation date (current time during request)
        got.pop(5)
        expected.pop(5)
        self.assertEquals(got, expected)

    def test_distaz(self):
        """
        Tests distance and azimuth calculation between two points on a sphere.
        """
        client = Client()
        # normal request
        result = client.distaz(stalat=1.1, stalon=1.2, evtlat=3.2, evtlon=1.4)
        self.assertAlmostEquals(result['distance'], 2.09554)
        self.assertAlmostEquals(result['backazimuth'], 5.46946)
        self.assertAlmostEquals(result['azimuth'], 185.47692)
        # missing parameters
        self.assertRaises(Exception, client.distaz, stalat=1.1)
        self.assertRaises(Exception, client.distaz, stalat=1.1, stalon=1.2)

    def test_flinnengdahl(self):
        """
        Tests calculation of Flinn-Engdahl region code or name.
        """
        client = Client()
        # code
        result = client.flinnengdahl(lat=-20.5, lon=-100.6, rtype="code")
        self.assertEquals(result, 683)
        # region
        result = client.flinnengdahl(lat=42, lon=-122.24, rtype="region")
        self.assertEquals(result, 'OREGON')
        # both
        result = client.flinnengdahl(lat=-20.5, lon=-100.6, rtype="both")
        self.assertEquals(result, (683, 'SOUTHEAST CENTRAL PACIFIC OCEAN'))
        # default rtype
        result = client.flinnengdahl(lat=42, lon=-122.24)
        self.assertEquals(result, (32, 'OREGON'))
        # outside boundaries
        self.assertRaises(Exception, client.flinnengdahl, lat=-90.1, lon=0)
        self.assertRaises(Exception, client.flinnengdahl, lat=90.1, lon=0)
        self.assertRaises(Exception, client.flinnengdahl, lat=0, lon=-180.1)
        self.assertRaises(Exception, client.flinnengdahl, lat=0, lon=180.1)

    def test_evalresp(self):
        """
        Tests evaluating instrument response information.
        """
        client = Client()
        dt = UTCDateTime("2005-01-01")
        # plot as PNG file
        tempfile = NamedTemporaryFile().name
        client.evalresp(network="IU", station="ANMO", location="00",
                        channel="BHZ", time=dt, output='plot',
                        filename=tempfile)
        self.assertEqual(open(tempfile, 'rb').read(4)[1:4], 'PNG')
        os.remove(tempfile)
        # plot-amp as PNG file
        tempfile = NamedTemporaryFile().name
        client.evalresp(network="IU", station="ANMO", location="00",
                        channel="BHZ", time=dt, output='plot-amp',
                        filename=tempfile)
        self.assertEqual(open(tempfile, 'rb').read(4)[1:4], 'PNG')
        os.remove(tempfile)
        # plot-phase as PNG file
        tempfile = NamedTemporaryFile().name
        client.evalresp(network="IU", station="ANMO", location="00",
                        channel="BHZ", time=dt, output='plot-phase',
                        filename=tempfile)
        self.assertEqual(open(tempfile, 'rb').read(4)[1:4], 'PNG')
        os.remove(tempfile)
        # fap as ASCII file
        tempfile = NamedTemporaryFile().name
        client.evalresp(network="IU", station="ANMO", location="00",
                        channel="BHZ", time=dt, output='fap',
                        filename=tempfile)
        self.assertEqual(open(tempfile, 'rt').readline(),
                         '1.000000E-05  1.202802E+04  1.792007E+02\n')
        os.remove(tempfile)
        # cs as ASCII file
        tempfile = NamedTemporaryFile().name
        client.evalresp(network="IU", station="ANMO", location="00",
                        channel="BHZ", time=dt, output='cs',
                        filename=tempfile)
        self.assertEqual(open(tempfile, 'rt').readline(),
                         '1.000000E-05 -1.202685E+04 1.677835E+02\n')
        os.remove(tempfile)
        # fap & def as ASCII file
        tempfile = NamedTemporaryFile().name
        client.evalresp(network="IU", station="ANMO", location="00",
                        channel="BHZ", time=dt, output='fap', units='def',
                        filename=tempfile)
        self.assertEqual(open(tempfile, 'rt').readline(),
                         '1.000000E-05  1.202802E+04  1.792007E+02\n')
        os.remove(tempfile)
        # fap & dis as ASCII file
        tempfile = NamedTemporaryFile().name
        client.evalresp(network="IU", station="ANMO", location="00",
                        channel="BHZ", time=dt, output='fap', units='dis',
                        filename=tempfile)
        self.assertEqual(open(tempfile, 'rt').readline(),
                         '1.000000E-05  7.557425E-01  2.692007E+02\n')
        os.remove(tempfile)
        # fap & vel as ASCII file
        tempfile = NamedTemporaryFile().name
        client.evalresp(network="IU", station="ANMO", location="00",
                        channel="BHZ", time=dt, output='fap', units='vel',
                        filename=tempfile)
        self.assertEqual(open(tempfile, 'rt').readline(),
                         '1.000000E-05  1.202802E+04  1.792007E+02\n')
        os.remove(tempfile)
        # fap & acc as ASCII file
        tempfile = NamedTemporaryFile().name
        client.evalresp(network="IU", station="ANMO", location="00",
                        channel="BHZ", time=dt, output='fap', units='acc',
                        filename=tempfile)
        self.assertEqual(open(tempfile, 'rt').readline(),
                         '1.000000E-05  1.914318E+08  8.920073E+01\n')
        os.remove(tempfile)
        # fap as NumPy ndarray
        data = client.evalresp(network="IU", station="ANMO", location="00",
                               channel="BHZ", time=dt, output='fap')
        np.testing.assert_array_equal(data[0],
            [1.00000000e-05, 1.20280200e+04, 1.79200700e+02])
        # cs as NumPy ndarray
        data = client.evalresp(network="IU", station="ANMO", location="00",
                               channel="BHZ", time=dt, output='cs')
        np.testing.assert_array_equal(data[0],
            [1.00000000e-05, -1.20268500e+04, 1.67783500e+02])

    def test_event(self):
        """
        Tests event Web service interface.

        Examples are inspired by http://www.iris.edu/ws/event/.
        """
        client = Client()
        # 1
        url = "http://www.iris.edu/ws/event/query?mindepth=34.9&" + \
            "maxdepth=35.1&catalog=NEIC%20PDE&contributor=NEIC%20PDE-Q&" + \
            "magtype=MB&lat=-56.1&lon=-26.7&maxradius=.1"
        # direct call
        doc1 = urllib.urlopen(url).read()
        # using client
        doc2 = client.event(mindepth=34.9, maxdepth=35.1, catalog="NEIC PDE",
                            contributor="NEIC PDE-Q", magtype="MB", lat=-56.1,
                            lon=-26.7, maxradius=0.1)
        self.assertEqual(doc1, doc2)
        client = Client()
        # 2
        url = "http://www.iris.edu/ws/event/query?eventid=3316989"
        # direct call
        doc1 = urllib.urlopen(url).read()
        # using client
        doc2 = client.event(eventid=3316989)
        self.assertEqual(doc1, doc2)
        # 2
        url = "http://www.iris.edu/ws/event/query?eventid=3316989"
        # direct call
        doc1 = urllib.urlopen(url).read()
        # using client
        doc2 = client.event(eventid=3316989)
        self.assertEqual(doc1, doc2)
        # 3
        url = "http://www.iris.edu/ws/event/query?minmag=8.5"
        # direct call
        doc1 = urllib.urlopen(url).read()
        # using client
        doc2 = client.event(minmag=8.5)
        self.assertEqual(doc1, doc2)
        # 4
        url = "http://www.iris.edu/ws/event/query?starttime=2011-01-07T" + \
            "14%3A00%3A00&endtime=2011-02-07&minlat=15&maxlat=40&" + \
            "minlon=-170&maxlon=170&preferredonly=yes&" + \
            "includeallmagnitudes=yes&orderby=magnitude"
        # direct call
        doc1 = urllib.urlopen(url).read()
        # using client
        doc2 = client.event(starttime=UTCDateTime(2011, 1, 7, 14),
                            endtime=UTCDateTime('2011-02-07'), minlat=15.0,
                            maxlat=40.0, minlon=-170, maxlon=170,
                            preferredonly=True, includeallmagnitudes=True,
                            orderby='magnitude')
        self.assertEqual(doc1, doc2)

    def test_availability(self):
        """
        Tests availability of waveform data at the DMC.
        """
        client = Client()
        # GE network
        dt = UTCDateTime("2011-11-13T07:00:00")
        result = client.availability(network='GE')
        self.assertTrue(isinstance(result, basestring))
        self.assertTrue('GE DAG -- BHE' in result)
        # unknown network results in empty string
        dt = UTCDateTime(2011, 11, 16)
        result = client.availability(network='XX', starttime=dt,
                                     endtime=dt + 10)


def suite():
    return unittest.makeSuite(ClientTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
