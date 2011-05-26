# -*- coding: utf-8 -*-
"""
The obspy.neries.client test suite.
"""

from obspy.core import UTCDateTime
from obspy.neries import Client
import unittest


class ClientTestCase(unittest.TestCase):
    """
    Test cases for obspy.neries.client.Client.
    """
    def test_getEvents(self):
        """
        Testing event request method.
        """
        client = Client()
        #1
        results = client.getEvents(format="list", min_depth= -700,
                                   max_datetime="2005-01-01")
        expected = [{'author': u'EMSC', 'event_id': u'20040312_0000026',
                     'origin_id': 1347097, 'longitude': 57.143,
                     'datetime': UTCDateTime('2004-03-12T22:48:05Z'),
                     'depth':-700.0, 'magnitude': 4.4, 'magnitude_type': u'mb',
                     'latitude': 26.303, 'flynn_region': u'SOUTHERN IRAN'}]
        self.assertEquals(results, expected)
        #2
        results = client.getEvents(format="list", min_latitude= -95,
                                   max_latitude= -1, min_longitude=20,
                                   max_longitude=90, max_datetime="2005-01-01")
        expected = [{'author': u'NEIR', 'event_id': u'20041016_0000009',
                     'origin_id': 120690, 'longitude': 33.682,
                     'datetime': UTCDateTime('2004-10-16T01:29:14Z'),
                     'depth':-10.0, 'magnitude': 5.0, 'magnitude_type': u'm ',
                     'latitude':-46.394,
                     'flynn_region': u'PRINCE EDWARD ISLANDS REGION'}]
        self.assertEquals(results, expected)
        #3
        results = client.getEvents(format="list", min_depth= -11,
                                   max_depth= -22.33, min_magnitude=6.6,
                                   max_magnitude=7, max_datetime="2005-01-01")
        expected = [{'author': u'EMSC', 'event_id': u'20001206_0000014',
                     'origin_id': 1441886, 'longitude': 54.843,
                     'datetime': UTCDateTime('2000-12-06T17:11:03Z'),
                     'depth':-11.4, 'magnitude': 6.7, 'magnitude_type': u'mb',
                     'latitude': 39.604},
                    {'author': u'EMSC', 'event_id': u'20010210_0000010',
                     'origin_id': 1438991, 'longitude': 43.784,
                     'datetime': UTCDateTime('2001-02-10T18:21:57Z'),
                     'depth':-17.0, 'magnitude': 6.6,
                     'magnitude_type': u'mb', 'latitude': 12.045,
                     'flynn_region': u'NEAR THE COAST OF YEMEN'}]
        self.assertEquals(results, expected)
        #4
        results = client.getEvents(format="list", author="EMSC", max_results=3,
                                   magnitude_type="mw", min_magnitude=4,
                                   max_datetime="2005-01-01")
        expected = [{'author': u'EMSC', 'event_id': u'19980110_0000006',
                     'origin_id': 1500183, 'longitude': 20.816,
                     'datetime': UTCDateTime('1998-01-10T19:21:54Z'),
                     'depth':-10.0, 'magnitude': 5.5, 'magnitude_type': u'mw',
                     'latitude': 37.243, 'flynn_region': u'IONIAN SEA'},
                    {'author': u'EMSC', 'event_id': u'19980128_0000006',
                     'origin_id': 1500249, 'longitude': 32.204,
                     'datetime': UTCDateTime('1998-01-28T22:38:55Z'),
                     'depth':-41.6, 'magnitude': 4.3, 'magnitude_type': u'mw',
                     'latitude': 34.429},
                    {'author': u'EMSC', 'event_id': u'19980213_0000004',
                     'origin_id': 1500135, 'longitude': 28.459,
                     'datetime': UTCDateTime('1998-02-13T07:18:49Z'),
                     'depth':-69.2, 'magnitude': 4.8, 'magnitude_type': u'mw',
                     'latitude': 36.284}]
        self.assertEquals(results, expected)

    def test_getEventsWithUTCDateTimes(self):
        """
        Testing event request method with UTCDateTimes as input parameters.
        """
        client = Client()
        #1
        results = client.getEvents(format="list", min_depth= -700,
                                   max_datetime=UTCDateTime("2005-01-01"))
        expected = [{'author': u'EMSC', 'event_id': u'20040312_0000026',
                     'origin_id': 1347097, 'longitude': 57.143,
                     'datetime': UTCDateTime('2004-03-12T22:48:05Z'),
                     'depth':-700.0, 'magnitude': 4.4, 'magnitude_type': u'mb',
                     'latitude': 26.303, 'flynn_region': u'SOUTHERN IRAN'}]
        self.assertEquals(results, expected)
        #2
        results = client.getEvents(format="list", min_depth= -700,
                                   min_datetime=UTCDateTime("2004-01-01"),
                                   max_datetime=UTCDateTime("2005-01-01"))
        expected = [{'author': u'EMSC', 'event_id': u'20040312_0000026',
                     'origin_id': 1347097, 'longitude': 57.143,
                     'datetime': UTCDateTime('2004-03-12T22:48:05Z'),
                     'depth':-700.0, 'magnitude': 4.4, 'magnitude_type': u'mb',
                     'latitude': 26.303, 'flynn_region': u'SOUTHERN IRAN'}]
        self.assertEquals(results, expected)

    def test_getEventDetail(self):
        """
        Testing event detail request method.
        """
        client = Client()
        # default format & EMSC identifier
        data = client.getEventDetail("19990817_0000001")
        self.assertTrue(isinstance(data, basestring))
        self.assertTrue(data.startswith('<?xml'))
        # list format
        data = client.getEventDetail("19990817_0000001", format='list')
        self.assertTrue(isinstance(data, list))
        # XML format
        data = client.getEventDetail("19990817_0000001", format='xml')
        self.assertTrue(isinstance(data, basestring))
        self.assertTrue(data.startswith('<?xml'))
        # default format & QuakeML identifier
        data = client.getEventDetail("quakeml:eu.emsc/event#19990817_0000001")
        self.assertTrue(data.startswith('<?xml'))
        # list format
        data = client.getEventDetail("quakeml:eu.emsc/event#19990817_0000001",
                                     format='list')
        self.assertTrue(isinstance(data, list))

    def test_getLatestEvents(self):
        """
        Testing request method for latest events.
        """
        client = Client()
        # default format
        data = client.getLatestEvents(5)
        self.assertTrue(isinstance(data, basestring))
        self.assertTrue(data.startswith('<?xml'))
        # list format
        data = client.getLatestEvents(5, format='list')
        self.assertTrue(isinstance(data, list))
        self.assertEquals(len(data), 5)
        # XML format
        data = client.getLatestEvents(5, format='xml')
        self.assertTrue(isinstance(data, basestring))
        self.assertTrue(data.startswith('<?xml'))

    def test_getTravelTimes(self):
        """
        Testing request method for calculating travel times.
        """
        client = Client()
        #1
        result = client.getTravelTimes(20, 20, 10, [(48, 12)], 'test', 'ak135')
        self.assertEquals(len(result), 1)
        self.assertEquals(result[0]['event_id'], 'test')
        self.assertEquals(result[0]['arrival_times'],
                          [('P', 356988.24732429383),
                           ('S', 645775.5623471631)])


def suite():
    return unittest.makeSuite(ClientTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
