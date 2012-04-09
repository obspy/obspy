# -*- coding: utf-8 -*-

from obspy.core.event import readEvents, Catalog, Event, Origin, CreationInfo
from obspy.core.utcdatetime import UTCDateTime
import os
import unittest


class EventTestCase(unittest.TestCase):
    """
    Test suite for obspy.core.event.Event
    """
    def test_str(self):
        """
        Testing the __str__ method of the Event object.
        """
        event = readEvents()[1]
        s = event.__str__()
        self.assertEquals("2012-04-04T14:18:37.000000Z | +39.342,  +41.044" + \
                          " | 4.3 ML | manual", s)

    def test_eq(self):
        """
        Testing the __eq__ method of the Event object.
        """
        # events are equal if the have the same public_id
        ev1 = Event('id1')
        ev2 = Event('id1')
        ev3 = Event('id2')
        self.assertTrue(ev1 == ev2)
        self.assertTrue(ev2 == ev1)
        self.assertFalse(ev1 == ev3)
        self.assertFalse(ev3 == ev1)
        # comparing with other objects fails
        self.assertFalse(ev1 == 1)
        self.assertFalse(ev2 == "id1")


class OriginTestCase(unittest.TestCase):
    """
    Test suite for obspy.core.event.Origin
    """
    def test_creationInfo(self):
        # 1 - empty Origin class must create a correct creation_info attribute
        orig = Origin()
        self.assertTrue(isinstance(orig.creation_info, CreationInfo))
        # 2 - preset via dict or existing CreationInfo object
        orig = Origin(creation_info={})
        self.assertTrue(isinstance(orig.creation_info, CreationInfo))
        orig = Origin(creation_info=CreationInfo({'author': 'test2'}))
        self.assertTrue(isinstance(orig.creation_info, CreationInfo))
        self.assertEquals(orig.creation_info.author, 'test2')
        # 3 - setting to anything except dict and CreationInfo fails
        self.assertRaises(TypeError, Origin, creation_info=None)
        self.assertRaises(TypeError, Origin, creation_info='assasas')
        # 4 - check set values
        orig = Origin(creation_info={'author': 'test'})
        self.assertEquals(orig.creation_info, orig['creation_info'])
        self.assertEquals(orig.creation_info.author, 'test')
        self.assertEquals(orig['creation_info']['author'], 'test')
        orig.creation_info.agency_id = "muh"
        self.assertEquals(orig.creation_info, orig['creation_info'])
        self.assertEquals(orig.creation_info.agency_id, 'muh')
        self.assertEquals(orig['creation_info']['agency_id'], 'muh')

    def test_multipleOrigins(self):
        """
        Parameters of multiple origins should not interfere with each other.
        """
        origin = Origin()
        origin.public_id = 'smi:ch.ethz.sed/origin/37465'
        origin.time.value = UTCDateTime(0)
        origin.latitude.value = 12
        origin.latitude.confidence_level = 95
        origin.longitude.value = 42
        origin.depth_type = 'from location'
        origin2 = Origin()
        origin2.latitude.value = 13.4
        self.assertEquals(origin2.depth_type, None)
        self.assertEquals(origin2.public_id, '')
        self.assertEquals(origin2.latitude.value, 13.4)
        self.assertEquals(origin2.latitude.confidence_level, None)
        self.assertEquals(origin2.longitude.value, None)


class CatalogTestCase(unittest.TestCase):
    """
    Test suite for obspy.core.event.Catalog
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
        Testing the __str__ method of the Catalog object.
        """
        catalog = readEvents()
        self.assertTrue(catalog.__str__().startswith("3 Event(s) in Catalog:"))
        self.assertTrue(catalog.__str__().endswith("37.736 | 3.0 ML | manual"))

    def test_readEvents(self):
        """
        Tests the readEvents function using entry points.
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

    def test_append(self):
        """
        Tests the append method of the Catalog object.
        """
        # 1 - create catalog and add a few events
        catalog = Catalog()
        event1 = Event()
        event2 = Event()
        self.assertEquals(len(catalog), 0)
        catalog.append(event1)
        self.assertEquals(len(catalog), 1)
        self.assertEquals(catalog.events, [event1])
        catalog.append(event2)
        self.assertEquals(len(catalog), 2)
        self.assertEquals(catalog.events, [event1, event2])
        # 2 - adding objects other as Event should fails
        self.assertRaises(TypeError, catalog.append, str)
        self.assertRaises(TypeError, catalog.append, Catalog)
        self.assertRaises(TypeError, catalog.append, [event1])

    def test_extend(self):
        """
        Tests the extend method of the Catalog object.
        """
        # 1 - create catalog and extend it with list of events
        catalog = Catalog()
        event1 = Event()
        event2 = Event()
        self.assertEquals(len(catalog), 0)
        catalog.extend([event1, event2])
        self.assertEquals(len(catalog), 2)
        self.assertEquals(catalog.events, [event1, event2])
        # 2 - extend it with other catalog
        event3 = Event()
        event4 = Event()
        catalog2 = Catalog([event3, event4])
        self.assertEquals(len(catalog), 2)
        catalog.extend(catalog2)
        self.assertEquals(len(catalog), 4)
        self.assertEquals(catalog.events, [event1, event2, event3, event4])
        # adding objects other as Catalog or list should fails
        self.assertRaises(TypeError, catalog.extend, str)
        self.assertRaises(TypeError, catalog.extend, event1)
        self.assertRaises(TypeError, catalog.extend, (event1, event2))

    def test_iadd(self):
        """
        Tests the __iadd__ method of the Catalog object.
        """
        # 1 - create catalog and add it with another catalog
        event1 = Event()
        event2 = Event()
        event3 = Event()
        catalog = Catalog([event1])
        catalog2 = Catalog([event2, event3])
        self.assertEquals(len(catalog), 1)
        catalog += catalog2
        self.assertEquals(len(catalog), 3)
        self.assertEquals(catalog.events, [event1, event2, event3])
        # 3 - extend it with another Event
        event4 = Event()
        self.assertEquals(len(catalog), 3)
        catalog += event4
        self.assertEquals(len(catalog), 4)
        self.assertEquals(catalog.events, [event1, event2, event3, event4])
        # adding objects other as Catalog or Event should fails
        self.assertRaises(TypeError, catalog.__iadd__, str)
        self.assertRaises(TypeError, catalog.__iadd__, (event1, event2))
        self.assertRaises(TypeError, catalog.__iadd__, [event1, event2])

    def test_countAndLen(self):
        """
        Tests the count and __len__ methods of the Catalog object.
        """
        # empty catalog without events
        catalog = Catalog()
        self.assertEqual(len(catalog), 0)
        self.assertEqual(catalog.count(), 0)
        # catalog with events
        catalog = readEvents()
        self.assertEqual(len(catalog), 3)
        self.assertEqual(catalog.count(), 3)

    def test_getitem(self):
        """
        Tests the __getitem__ method of the Catalog object.
        """
        catalog = readEvents()
        self.assertEqual(catalog[0], catalog.events[0])
        self.assertEqual(catalog[-1], catalog.events[-1])
        self.assertEqual(catalog[2], catalog.events[2])
        # out of index should fail
        self.assertRaises(IndexError, catalog.__getitem__, 3)
        self.assertRaises(IndexError, catalog.__getitem__, -99)

    def test_slicing(self):
        """
        Tests the __getslice__ method of the Catalog object.
        """
        catalog = readEvents()
        self.assertEqual(catalog[0:], catalog[0:])
        self.assertEqual(catalog[:2], catalog[:2])
        self.assertEqual(catalog[:], catalog[:])
        self.assertEqual(len(catalog), 3)
        new_catalog = catalog[1:3]
        self.assertTrue(isinstance(new_catalog, Catalog))
        self.assertEqual(len(new_catalog), 2)

    def test_slicingWithStep(self):
        """
        Tests the __getslice__ method of the Catalog object with step.
        """
        ev1 = Event()
        ev2 = Event()
        ev3 = Event()
        ev4 = Event()
        ev5 = Event()
        catalog = Catalog([ev1, ev2, ev3, ev4, ev5])
        self.assertEqual(catalog[0:6].events, [ev1, ev2, ev3, ev4, ev5])
        self.assertEqual(catalog[0:6:1].events, [ev1, ev2, ev3, ev4, ev5])
        self.assertEqual(catalog[0:6:2].events, [ev1, ev3, ev5])
        self.assertEqual(catalog[1:6:2].events, [ev2, ev4])
        self.assertEqual(catalog[1:6:6].events, [ev2])

    def test_copy(self):
        """
        Testing the copy method of the Catalog object.
        """
        cat = readEvents()
        cat2 = cat.copy()
        self.assertTrue(cat == cat2)
        self.assertTrue(cat2 == cat)
        self.assertFalse(cat is cat2)
        self.assertFalse(cat2 is cat)
        self.assertTrue(cat.events[0] == cat2.events[0])
        self.assertFalse(cat.events[0] is cat2.events[0])


def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(CatalogTestCase, 'test'))
    suite.addTest(unittest.makeSuite(EventTestCase, 'test'))
    suite.addTest(unittest.makeSuite(OriginTestCase, 'test'))
    return suite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
