# -*- coding: utf-8 -*-
"""
The obspy.orfeus.client test suite.
"""

from obspy.core.utcdatetime import UTCDateTime
from obspy.orfeus import Client
import unittest


class ClientTestCase(unittest.TestCase):
    """
    Test cases for obspy.orfeus.client.Client.
    """
    def setUp(self):
        self.client = Client()

    def test_getEvents(self):
        """
        Testing event request method.
        """
        # testing requests with format="list"
        request_params = [
                {"min_depth": -700},
                # datetimes not tested, this is buggy for json requests at EMSC
                #{"min_datetime": "2010-07-01",
                # "max_datetime": "2010-07-01T01:00:00Z"},
                {"min_latitude": -95, "max_latitude": -1,
                 "min_longitude": 20, "max_longitude": 90,
                 "max_datetime": "2005-01-01"},
                {"min_depth": -11, "max_depth": -22.33,
                 "min_magnitude": 6.6, "max_magnitude": 7,
                 "max_datetime": "2005-01-01"},
                {"author": "EMSC", "max_results": 3,
                 "magnitude_type": "mw", "min_magnitude": 4,
                 "max_datetime": "2005-01-01"},
        ]
        request_results = [
                [{'author': 'EMSC',
                  'datetime': '2004-03-12T22:48:05Z',
                  'depth': -700.0,
                  'event_id': '20040312_0000026',
                  'flynn_region': 'SOUTHERN IRAN',
                  'latitude': 26.303000000000001,
                  'longitude': 57.143000000000001,
                  'magnitude': 4.4000000000000004,
                  'magnitude_type': 'mb',
                  'origin_id': 1347097},
                 {'author': 'NEIR',
                  'datetime': '2007-05-06T21:22:20Z',
                  'depth': -700.89999999999998,
                  'event_id': '20070506_0000099',
                  'flynn_region': 'FIJI REGION',
                  'latitude': -19.503,
                  'longitude': -179.43700000000001,
                  'magnitude': 4.5999999999999996,
                  'magnitude_type': 'mb',
                  'origin_id': 268538},
                 {'author': 'INFO',
                  'datetime': '2007-05-06T22:01:09Z',
                  'depth': -703.0,
                  'event_id': '20070506_0000100',
                  'flynn_region': 'FIJI REGION',
                  'latitude': -19.25,
                  'longitude': -179.38999999999999,
                  'magnitude': 6.0999999999999996,
                  'magnitude_type': 'mw',
                  'origin_id': 266417}],
                [{'author': 'NEIR',
                  'datetime': '2004-10-16T01:29:14Z',
                  'depth': -10.0,
                  'event_id': '20041016_0000009',
                  'flynn_region': 'PRINCE EDWARD ISLANDS REGION',
                  'latitude': -46.393999999999998,
                  'longitude': 33.682000000000002,
                  'magnitude': 5.0,
                  'magnitude_type': 'm ',
                  'origin_id': 120690}],
                [{'author': 'EMSC',
                  'datetime': '2000-06-21T00:51:47Z',
                  'depth': -17.100000000000001,
                  'event_id': '20000621_0000002',
                  'latitude': 64.040999999999997,
                  'longitude': -20.716999999999999,
                  'magnitude': 6.5999999999999996,
                  'magnitude_type': 'ms',
                  'origin_id': 322839},
                 {'author': 'EMSC',
                  'datetime': '2000-12-06T17:11:03Z',
                  'depth': -11.4,
                  'event_id': '20001206_0000014',
                  'latitude': 39.659999999999997,
                  'longitude': 54.851999999999997,
                  'magnitude': 6.7000000000000002,
                  'magnitude_type': 'mb',
                  'origin_id': 323110},
                 {'author': 'EMSC',
                  'datetime': '2001-02-10T18:21:57Z',
                  'depth': -17.0,
                  'event_id': '20010210_0000010',
                  'flynn_region': 'NEAR THE COAST OF YEMEN',
                  'latitude': 12.045,
                  'longitude': 43.783999999999999,
                  'magnitude': 6.5999999999999996,
                  'magnitude_type': 'mb',
                  'origin_id': 1438991}],
                [{'author': 'EMSC',
                  'datetime': '1998-01-10T19:21:54Z',
                  'depth': -10.0,
                  'event_id': '19980110_0000006',
                  'flynn_region': 'IONIAN SEA',
                  'latitude': 37.273000000000003,
                  'longitude': 20.753,
                  'magnitude': 5.5,
                  'magnitude_type': 'mw',
                  'origin_id': 582},
                 {'author': 'EMSC',
                  'datetime': '1998-01-28T22:38:55Z',
                  'depth': -41.600000000000001,
                  'event_id': '19980128_0000006',
                  'latitude': 34.444000000000003,
                  'longitude': 32.167000000000002,
                  'magnitude': 4.2999999999999998,
                  'magnitude_type': 'mw',
                  'origin_id': 318330},
                 {'author': 'EMSC',
                  'datetime': '1998-02-13T07:18:49Z',
                  'depth': -70.0,
                  'event_id': '19980213_0000004',
                  'latitude': 36.343000000000004,
                  'longitude': 28.501000000000001,
                  'magnitude': 4.7999999999999998,
                  'magnitude_type': 'mw',
                  'origin_id': 327440}],
        ]
        for params, result in zip(request_params, request_results):
            events = self.client.getEvents(format="list", **params)
            self.assertEquals(result, events)

def suite():
    return unittest.makeSuite(ClientTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
