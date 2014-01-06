# -*- coding: utf-8 -*-
"""
The obspy.neries.client test suite.
"""

from obspy import UTCDateTime, read
from obspy.core.event import Catalog
from obspy.core.util import NamedTemporaryFile
from obspy.neries import Client
import unittest


class ClientTestCase(unittest.TestCase):
    """
    Test cases for obspy.neries.client.Client.
    """
    def test_getEventsList(self):
        """
        Testing event request method.
        """
        client = Client()
        # 1
        results = client.getEvents(format="list", min_depth=-700,
                                   max_datetime="2005-01-01")
        expected = [{'author': u'EMSC', 'event_id': u'20040312_0000026',
                     'origin_id': 1347097, 'longitude': 57.143,
                     'datetime': UTCDateTime('2004-03-12T22:48:05Z'),
                     'depth': -700.0, 'magnitude': 4.4, 'magnitude_type':
                     u'mb', 'latitude': 26.303,
                     'flynn_region': u'SOUTHERN IRAN'}]
        self.assertEqual(results, expected)
        # 2
        results = client.getEvents(format="list", min_latitude=-95,
                                   max_latitude=-1, min_longitude=20,
                                   max_longitude=90, max_datetime="2005-01-01")
        expected = [{'author': u'NEIR', 'event_id': u'20041016_0000009',
                     'origin_id': 120690, 'longitude': 33.682,
                     'datetime': UTCDateTime('2004-10-16T01:29:14Z'),
                     'depth': -10.0, 'magnitude': 5.0, 'magnitude_type': u'm ',
                     'latitude': -46.394,
                     'flynn_region': u'PRINCE EDWARD ISLANDS REGION'}]
        self.assertEqual(results, expected)
        # 3
        results = client.getEvents(format="list", min_depth=-11,
                                   max_depth=-22.33, min_magnitude=6.6,
                                   max_magnitude=7, max_datetime="2005-01-01")
        expected = [{'author': u'EMSC', 'event_id': u'20001206_0000014',
                     'origin_id': 1441886, 'longitude': 54.843,
                     'datetime': UTCDateTime('2000-12-06T17:11:05Z'),
                     'depth': -11.4, 'magnitude': 6.7, 'magnitude_type': u'mb',
                     'latitude': 39.604},
                    {'author': u'EMSC', 'event_id': u'20010210_0000010',
                     'origin_id': 1438991, 'longitude': 43.784,
                     'datetime': UTCDateTime('2001-02-10T18:21:57Z'),
                     'depth': -17.0, 'magnitude': 6.6,
                     'magnitude_type': u'mb', 'latitude': 12.045,
                     'flynn_region': u'NEAR THE COAST OF YEMEN'}]
        self.assertEqual(results, expected)
        # 4
        results = client.getEvents(format="list", author="EMSC", max_results=3,
                                   magnitude_type="mw", min_magnitude=4,
                                   max_datetime="2005-01-01")
        expected = [{'author': u'EMSC', 'event_id': u'19980110_0000006',
                     'origin_id': 1500183, 'longitude': 20.816,
                     'datetime': UTCDateTime('1998-01-10T19:21:55Z'),
                     'depth': -10.0, 'magnitude': 5.5, 'magnitude_type': u'mw',
                     'latitude': 37.243, 'flynn_region': u'IONIAN SEA'},
                    {'author': u'EMSC', 'event_id': u'19980128_0000006',
                     'origin_id': 1500249, 'longitude': 32.204,
                     'datetime': UTCDateTime('1998-01-28T22:38:57Z'),
                     'depth': -41.6, 'magnitude': 4.3, 'magnitude_type': u'mw',
                     'latitude': 34.429},
                    {'author': u'EMSC', 'event_id': u'19980213_0000004',
                     'origin_id': 1500135, 'longitude': 28.459,
                     'datetime': UTCDateTime('1998-02-13T07:18:50Z'),
                     'depth': -69.2, 'magnitude': 4.8, 'magnitude_type': u'mw',
                     'latitude': 36.284}]
        self.assertEqual(results, expected)

    def test_getEventsWithUTCDateTimes(self):
        """
        Testing event request method with UTCDateTimes as input parameters.
        """
        client = Client()
        # 1
        results = client.getEvents(format="list", min_depth=-700,
                                   max_datetime=UTCDateTime("2005-01-01"))
        expected = [{'author': u'EMSC', 'event_id': u'20040312_0000026',
                     'origin_id': 1347097, 'longitude': 57.143,
                     'datetime': UTCDateTime('2004-03-12T22:48:05Z'),
                     'depth': -700.0, 'magnitude': 4.4,
                     'magnitude_type': u'mb',
                     'latitude': 26.303, 'flynn_region': u'SOUTHERN IRAN'}]
        self.assertEqual(results, expected)
        # 2
        results = client.getEvents(format="list", min_depth=-700,
                                   min_datetime=UTCDateTime("2004-01-01"),
                                   max_datetime=UTCDateTime("2005-01-01"))
        expected = [{'author': u'EMSC', 'event_id': u'20040312_0000026',
                     'origin_id': 1347097, 'longitude': 57.143,
                     'datetime': UTCDateTime('2004-03-12T22:48:05Z'),
                     'depth': -700.0, 'magnitude': 4.4,
                     'magnitude_type': u'mb',
                     'latitude': 26.303, 'flynn_region': u'SOUTHERN IRAN'}]
        self.assertEqual(results, expected)

    def test_getEventsAsQuakeML(self):
        """
        Testing event request with QuakeML as output format.
        """
        client = Client()
        results = client.getEvents(format="xml", min_depth=-700,
                                   max_datetime=UTCDateTime("2005-01-01"))
        self.assertTrue(isinstance(results, basestring))
        # check for origin id
        self.assertTrue('1347097' in results)

    def test_getEventsAsCatalog(self):
        """
        Testing event request with Catalog as output format.
        """
        client = Client()
        cat = client.getEvents(format="catalog", min_depth=-700,
                               max_datetime=UTCDateTime("2005-01-01"))
        self.assertTrue(isinstance(cat, Catalog))
        # check for origin id
        self.assertTrue(cat[0].preferred_origin_id.endswith('1347097'))

    def test_getEventDetail(self):
        """
        Testing event detail request method.
        """
        client = Client()
        # EMSC identifier
        # xml
        data = client.getEventDetail("19990817_0000001", format='xml')
        self.assertTrue(isinstance(data, basestring))
        self.assertTrue(data.startswith('<?xml'))
        # list
        data = client.getEventDetail("19990817_0000001", format='list')
        self.assertTrue(isinstance(data, list))
        # catalog
        data = client.getEventDetail("19990817_0000001", format='catalog')
        self.assertTrue(isinstance(data, Catalog))
        # QuakeML identifier
        # xml
        data = client.getEventDetail("quakeml:eu.emsc/event#19990817_0000001",
                                     format='xml')
        self.assertTrue(data.startswith('<?xml'))
        # list
        data = client.getEventDetail("quakeml:eu.emsc/event#19990817_0000001",
                                     format='list')
        self.assertTrue(isinstance(data, list))
        # catalog
        data = client.getEventDetail("quakeml:eu.emsc/event#19990817_0000001",
                                     format='catalog')
        self.assertTrue(isinstance(data, Catalog))

    def test_getLatestEvents(self):
        """
        Testing request method for latest events.

        XXX: Currently we can not rely on the length of the returned list due
            to a bug in Web Service implementation.
        """
        client = Client()
        # xml
        data = client.getLatestEvents(5, format='xml')
        self.assertTrue(isinstance(data, basestring))
        self.assertTrue(data.startswith('<?xml'))
        # list
        data = client.getLatestEvents(5, format='list')
        self.assertTrue(isinstance(data, list))
        # catalog
        data = client.getLatestEvents(5, format='catalog')
        self.assertTrue(isinstance(data, Catalog))
        # no given number of events should default to 10
        data = client.getLatestEvents(format='list')
        self.assertTrue(isinstance(data, list))
        # invalid number of events should default to 10
        data = client.getLatestEvents(num='blah', format='list')
        self.assertTrue(isinstance(data, list))

    def test_getTravelTimes(self):
        """
        Testing request method for calculating travel times.
        """
        client = Client()
        # 1
        result = client.getTravelTimes(20, 20, 10, [(48, 12)], 'ak135')
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result[0]['P'], 356988.24732429383)
        self.assertAlmostEqual(result[0]['S'], 645775.5623471631)
        # 2
        result = client.getTravelTimes(0, 0, 10,
                                       [(120, 0), (150, 0), (180, 0)])
        self.assertEqual(len(result), 3)
        self.assertAlmostEqual(result[0]['P'], 605519.0321213702)
        self.assertAlmostEqual(result[0]['S'], 1097834.6352750373)
        self.assertAlmostEqual(result[1]['P'], 367256.0587305712)
        self.assertAlmostEqual(result[1]['S'], 665027.0583152708)
        self.assertEqual(result[2], {})

    def test_saveWaveform(self):
        """
        """
        # initialize client
        client = Client(user='test@obspy.org')
        start = UTCDateTime(2012, 1, 1)
        end = start + 10
        with NamedTemporaryFile() as tf:
            mseedfile = tf.name
            # MiniSEED
            client.saveWaveform(mseedfile, 'BW', 'MANZ', '', 'EHZ', start, end)
            st = read(mseedfile)
            # MiniSEED may not start with Volume Index Control Headers (V)
            self.assertNotEqual(open(mseedfile).read(8)[6], "V")
        # ArcLink cuts on record base
        self.assertTrue(st[0].stats.starttime <= start)
        self.assertTrue(st[0].stats.endtime >= end)
        self.assertEqual(st[0].stats.network, 'BW')
        self.assertEqual(st[0].stats.station, 'MANZ')
        self.assertEqual(st[0].stats.location, '')
        self.assertEqual(st[0].stats.channel, 'EHZ')
        # Full SEED
        with NamedTemporaryFile() as tf:
            fseedfile = tf.name
            client.saveWaveform(fseedfile, 'BW', 'MANZ', '', 'EHZ', start, end,
                                format='FSEED')
            st = read(fseedfile)
            # Full SEED must start with Volume Index Control Headers (V)
            self.assertEqual(open(fseedfile).read(8)[6], "V")
        # ArcLink cuts on record base
        self.assertTrue(st[0].stats.starttime <= start)
        self.assertTrue(st[0].stats.endtime >= end)
        self.assertEqual(st[0].stats.network, 'BW')
        self.assertEqual(st[0].stats.station, 'MANZ')
        self.assertEqual(st[0].stats.location, '')
        self.assertEqual(st[0].stats.channel, 'EHZ')

    def test_getInventory(self):
        """
        Testing inventory requests.
        """
        client = Client(user='test@obspy.org')
        dt1 = UTCDateTime("1974-01-01T00:00:00")
        dt2 = UTCDateTime("2011-01-01T00:00:00")
        # 1 - XML w/ instruments
        result = client.getInventory('GE', 'SNAA', '', 'BHZ', dt1, dt2,
                                     format='XML')
        self.assertTrue(result.startswith('<?xml'))
        self.assertTrue('code="GE"' in result)
        # 2 - SUDS object w/o instruments
        result = client.getInventory('GE', 'SNAA', '', 'BHZ', dt1, dt2,
                                     instruments=False)
        self.assertTrue(isinstance(result, object))
        self.assertEqual(result.ArclinkInventory.inventory.network._code, 'GE')
        # 3 - SUDS object w/ instruments
        result = client.getInventory('GE', 'SNAA', '', 'BHZ', dt1, dt2,
                                     instruments=True)
        self.assertTrue(isinstance(result, object))
        self.assertEqual(result.ArclinkInventory.inventory.network._code, 'GE')
        self.assertTrue('sensor' in result.ArclinkInventory.inventory)
        self.assertTrue('responsePAZ' in result.ArclinkInventory.inventory)
        # 4 - SUDS object with spatial filters
        client = Client(user='test@obspy.org')
        result = client.getInventory('GE', 'SNAA', '', 'BHZ', dt1, dt2,
                                     min_latitude=-72.0, max_latitude=-71.0,
                                     min_longitude=-3, max_longitude=-2)
        self.assertTrue(isinstance(result, object))
        self.assertEqual(result.ArclinkInventory.inventory.network._code, 'GE')
        # 5 - SUDS object with spatial filters with incorrect coordinates
        client = Client(user='test@obspy.org')
        result = client.getInventory('GE', 'SNAA', '', 'BHZ', dt1, dt2,
                                     min_latitude=-71.0, max_latitude=-72.0,
                                     min_longitude=-2, max_longitude=-3)
        self.assertTrue(isinstance(result, object))
        self.assertEqual(result.ArclinkInventory.inventory.network._code, 'GE')

    def test_issue531(self):
        """
        Event_type "other" has been replaced by "other event" in recent
        QuakeML version
        """
        client = Client(user='test@obspy.org')
        events = client.getEvents(
            minlon=-30, maxlon=40, minlat=30, maxlat=90,
            min_datetime=UTCDateTime(2000, 4, 11, 11, 24, 31),
            max_datetime=UTCDateTime(2000, 4, 11, 11, 24, 32),
            minmag=5.5, format='catalog')
        self.assertEquals(len(events), 1)
        self.assertEquals(events[0].event_type, 'other event')


def suite():
    return unittest.makeSuite(ClientTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
