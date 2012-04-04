# -*- coding: utf-8 -*-
from obspy.core.event import readEvents
import os
import unittest


class EventTestCase(unittest.TestCase):
    """
    Test suite for obspy.core.event
    """
    def setUp(self):
        # directory where the test files are located
        path = os.path.join(os.path.dirname(__file__), 'data')
        self.iris_xml = os.path.join(path, 'iris_events.xml')
        self.neries_xml = os.path.join(path, 'neries_events.xml')

    def test_readEventsWithoutParameters(self):
        """
        Calling readEvents w/o any parameter will create an example catalog.
        """
        catalog = readEvents()
        self.assertEquals(len(catalog), 3)

    def test_str(self):
        """
        Testing __str__ method of catalog and event object.
        """
        # catalog
        catalog = readEvents()
        self.assertTrue(catalog.__str__().startswith("3 Event(s) in Catalog:"))
        self.assertTrue(catalog.__str__().endswith("37.736 | 3.0 ML | manual"))
        # event
        event = catalog[1]
        s = event.__str__()
        self.assertEquals("2012-04-04T14:18:37.000000Z | +39.342,  +41.044" + \
                          " | 4.3 ML | manual", s)

    def test_readEvents(self):
        """
        Tests readEvents function using entry points.
        """
        # iris
        catalog = readEvents(self.iris_xml)
        self.assertEquals(len(catalog), 2)
        self.assertEquals(catalog[0]._format, 'QUAKEML')
        self.assertEquals(catalog[1]._format, 'QUAKEML')
        # neries
        catalog = readEvents(self.neries_xml)
        self.assertEquals(len(catalog), 3)
        self.assertEquals(catalog[0]._format, 'QUAKEML')
        self.assertEquals(catalog[1]._format, 'QUAKEML')
        self.assertEquals(catalog[2]._format, 'QUAKEML')


def suite():
    return unittest.makeSuite(EventTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
