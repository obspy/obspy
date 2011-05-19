# -*- coding: utf-8 -*-
"""
The obspy.neries.client test suite.
"""

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
        # testing requests with format="list"
        request_params = [
                {"min_depth":-700},
                # datetimes not tested, this is buggy for json requests at EMSC
                #{"min_datetime": "2010-07-01",
                # "max_datetime": "2010-07-01T01:00:00Z"},
                {"min_latitude":-95, "max_latitude":-1,
                 "min_longitude": 20, "max_longitude": 90,
                 "max_datetime": "2005-01-01"},
                {"min_depth":-11, "max_depth":-22.33,
                 "min_magnitude": 6.6, "max_magnitude": 7,
                 "max_datetime": "2005-01-01"},
                {"author": "EMSC", "max_results": 3,
                 "magnitude_type": "mw", "min_magnitude": 4,
                 "max_datetime": "2005-01-01"},
        ]
        request_results = [
                [{'author': 'EMSC',
                  'datetime': '2004-03-12T22:48:05Z',
                  'depth':-700.0,
                  'event_id': '20040312_0000026',
                  'flynn_region': 'SOUTHERN IRAN',
                  'latitude': 26.303,
                  'longitude': 57.143,
                  'magnitude': 4.4,
                  'magnitude_type': 'mb',
                  'origin_id': 1347097},
                 {'author': 'NEIR',
                  'datetime': '2007-05-06T21:22:20Z',
                  'depth':-700.9,
                  'event_id': '20070506_0000099',
                  'flynn_region': 'FIJI REGION',
                  'latitude':-19.503,
                  'longitude':-179.437,
                  'magnitude': 4.6,
                  'magnitude_type': 'mb',
                  'origin_id': 268538},
                 {'author': 'INFO',
                  'datetime': '2007-05-06T22:01:09Z',
                  'depth':-703.0,
                  'event_id': '20070506_0000100',
                  'flynn_region': 'FIJI REGION',
                  'latitude':-19.25,
                  'longitude':-179.390,
                  'magnitude': 6.1,
                  'magnitude_type': 'mw',
                  'origin_id': 266417}],
                [{'author': 'NEIR',
                  'datetime': '2004-10-16T01:29:14Z',
                  'depth':-10.0,
                  'event_id': '20041016_0000009',
                  'flynn_region': 'PRINCE EDWARD ISLANDS REGION',
                  'latitude':-46.394,
                  'longitude': 33.682,
                  'magnitude': 5.0,
                  'magnitude_type': 'm ',
                  'origin_id': 120690}],
                [{'author': 'EMSC',
                  'datetime': '2000-06-21T00:51:47Z',
                  'depth':-17.1,
                  'event_id': '20000621_0000002',
                  'latitude': 64.041,
                  'longitude':-20.717,
                  'magnitude': 6.6,
                  'magnitude_type': 'ms',
                  'origin_id': 322839},
                 {'author': 'EMSC',
                  'datetime': '2000-12-06T17:11:03Z',
                  'depth':-11.4,
                  'event_id': '20001206_0000014',
                  'latitude': 39.604,
                  'longitude': 54.843,
                  'magnitude': 6.7,
                  'magnitude_type': 'mb',
                  'origin_id': 1441886},
                 {'author': 'EMSC',
                  'datetime': '2001-02-10T18:21:57Z',
                  'depth':-17.0,
                  'event_id': '20010210_0000010',
                  'flynn_region': 'NEAR THE COAST OF YEMEN',
                  'latitude': 12.045,
                  'longitude': 43.784,
                  'magnitude': 6.6,
                  'magnitude_type': 'mb',
                  'origin_id': 1438991}],
                [{'author': 'EMSC',
                  'datetime': '1998-01-10T19:21:54Z',
                  'depth':-10.0,
                  'event_id': '19980110_0000006',
                  'flynn_region': 'IONIAN SEA',
                  'latitude': 37.273,
                  'longitude': 20.753,
                  'magnitude': 5.5,
                  'magnitude_type': 'mw',
                  'origin_id': 582},
                 {'author': 'EMSC',
                  'datetime': '1998-01-28T22:38:55Z',
                  'depth':-41.6,
                  'event_id': '19980128_0000006',
                  'latitude': 34.444,
                  'longitude': 32.167,
                  'magnitude': 4.3,
                  'magnitude_type': 'mw',
                  'origin_id': 318330},
                 {'author': 'EMSC',
                  'datetime': '1998-02-13T07:18:49Z',
                  'depth':-70.0,
                  'event_id': '19980213_0000004',
                  'latitude': 36.343,
                  'longitude': 28.501,
                  'magnitude': 4.8,
                  'magnitude_type': 'mw',
                  'origin_id': 327440}],
        ]
        client = Client()
        for params, result in zip(request_params, request_results):
            events = client.getEvents(format="list", **params)
            self.assertEquals(result, events)

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


def suite():
    return unittest.makeSuite(ClientTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
