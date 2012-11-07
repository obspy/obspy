# -*- coding: utf-8 -*-

from obspy.core.event import readEvents, Catalog, Event, WaveformStreamID, \
    Origin, CreationInfo, ResourceIdentifier, Comment, Pick
from obspy.core.utcdatetime import UTCDateTime
import os
import sys
import unittest
import warnings


class EventTestCase(unittest.TestCase):
    """
    Test suite for obspy.core.event.Event
    """
    def test_str(self):
        """
        Testing the __str__ method of the Event object.
        """
        event = readEvents()[1]
        s = event.short_str()
        self.assertEquals("2012-04-04T14:18:37.000000Z | +39.342,  +41.044" +
                          " | 4.3 ML | manual", s)

    def test_eq(self):
        """
        Testing the __eq__ method of the Event object.
        """
        # events are equal if the have the same public_id
        # Catch warnings about the same different objects with the same
        # resource id so they do not clutter the test output.
        with warnings.catch_warnings() as _:
            warnings.simplefilter("ignore")
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

    def test_clear_method_resets_objects(self):
        """
        Tests that the clear() method properly resets all objects. Test for
        #449.
        """
        # Test with basic event object.
        e = Event()
        e.comments.append(Comment("test"))
        e.event_type = "explosion"
        self.assertEqual(len(e.comments), 1)
        self.assertEqual(e.event_type, "explosion")
        e.clear()
        self.assertTrue(e == Event())
        self.assertEqual(len(e.comments), 0)
        self.assertEqual(e.event_type, None)
        # Test with pick object. Does not really fit in the event test case but
        # it tests the same thing...
        p = Pick()
        p.comments.append(Comment("test"))
        p.phase_hint = "p"
        self.assertEqual(len(p.comments), 1)
        self.assertEqual(p.phase_hint, "p")
        # Add some more random attributes. These should disappear upon
        # cleaning.
        p.test_1 = "a"
        p.test_2 = "b"
        self.assertEqual(p.test_1, "a")
        self.assertEqual(p.test_2, "b")
        p.clear()
        self.assertEqual(len(p.comments), 0)
        self.assertEqual(p.phase_hint, None)
        self.assertFalse(hasattr(p, "test_1"))
        self.assertFalse(hasattr(p, "test_2"))


class OriginTestCase(unittest.TestCase):
    """
    Test suite for obspy.core.event.Origin
    """
    def test_creationInfo(self):
        # 1 - empty Origin class will set creation_info to None
        orig = Origin()
        self.assertEquals(orig.creation_info, None)
        # 2 - preset via dict or existing CreationInfo object
        orig = Origin(creation_info={})
        self.assertTrue(isinstance(orig.creation_info, CreationInfo))
        orig = Origin(creation_info=CreationInfo(author='test2'))
        self.assertTrue(isinstance(orig.creation_info, CreationInfo))
        self.assertEquals(orig.creation_info.author, 'test2')
        # 3 - check set values
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
        origin.time = UTCDateTime(0)
        origin.latitude = 12
        origin.latitude_errors.confidence_level = 95
        origin.longitude = 42
        origin.depth_type = 'from location'
        self.assertEquals(origin.latitude, 12)
        self.assertEquals(origin.latitude_errors.confidence_level, 95)
        self.assertEquals(origin.latitude_errors.uncertainty, None)
        self.assertEquals(origin.longitude, 42)
        origin2 = Origin()
        origin2.latitude = 13.4
        self.assertEquals(origin2.depth_type, None)
        self.assertEquals(origin2.resource_id, None)
        self.assertEquals(origin2.latitude, 13.4)
        self.assertEquals(origin2.latitude_errors.confidence_level, None)
        self.assertEquals(origin2.longitude, None)


class CatalogTestCase(unittest.TestCase):
    """
    Test suite for obspy.core.event.Catalog
    """
    def setUp(self):
        # directory where the test files are located
        path = os.path.join(os.path.dirname(__file__), 'data')
        self.iris_xml = os.path.join(path, 'iris_events.xml')
        self.neries_xml = os.path.join(path, 'neries_events.xml')

    def test_creationInfo(self):
        cat = Catalog()
        cat.creation_info = CreationInfo(author='test2')
        self.assertTrue(isinstance(cat.creation_info, CreationInfo))
        self.assertEquals(cat.creation_info.author, 'test2')

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

    def test_filter(self):
        """
        Testing the filter method of the Catalog object.
        """
        def getattrs(event, attr):
            if attr == 'magnitude':
                obj = event.magnitudes[0]
                attr = 'mag'
            else:
                obj = event.origins[0]
            for a in attr.split('.'):
                obj = getattr(obj, a)
            return obj
        cat = readEvents()
        self.assertTrue(all(event.magnitudes[0].mag < 4.
                            for event in cat.filter('magnitude < 4.')))
        attrs = ('magnitude', 'latitude', 'longitude', 'depth', 'time',
                 'quality.standard_error', 'quality.azimuthal_gap',
                 'quality.used_station_count', 'quality.used_phase_count')
        values = (4., 40., 50., 10., UTCDateTime('2012-04-04 14:20:00'),
                  1., 50, 40, 20)
        for attr, value in zip(attrs, values):
            attr_filter = attr.split('.')[-1]
            cat_smaller = cat.filter('%s < %s' % (attr_filter, value))
            cat_bigger = cat.filter('%s >= %s' % (attr_filter, value))
            self.assertTrue(all(getattrs(event, attr) < value
                                for event in cat_smaller))
            self.assertTrue(all(getattrs(event, attr) >= value
                                for event in cat_bigger))
            self.assertTrue(all(event in cat
                                for event in (cat_smaller + cat_bigger)))


class WaveformStreamIDTestCase(unittest.TestCase):
    """
    Test suite for obspy.core.event.WaveformStreamID.
    """
    def test_initialization(self):
        """
        Test the different initialization methods.
        """
        # Default init.
        waveform_id = WaveformStreamID()
        self.assertEqual(waveform_id.network_code, None)
        self.assertEqual(waveform_id.station_code, None)
        self.assertEqual(waveform_id.location_code, None)
        self.assertEqual(waveform_id.channel_code, None)
        # With seed string.
        waveform_id = WaveformStreamID(seed_string="BW.FUR.01.EHZ")
        self.assertEqual(waveform_id.network_code, "BW")
        self.assertEqual(waveform_id.station_code, "FUR")
        self.assertEqual(waveform_id.location_code, "01")
        self.assertEqual(waveform_id.channel_code, "EHZ")
        # As soon as any other argument is set, the seed_string will not be
        # used and the default values will be used for any unset arguments.
        waveform_id = WaveformStreamID(location_code="02",
                                       seed_string="BW.FUR.01.EHZ")
        self.assertEqual(waveform_id.network_code, None)
        self.assertEqual(waveform_id.station_code, None)
        self.assertEqual(waveform_id.location_code, "02")
        self.assertEqual(waveform_id.channel_code, None)

    def test_initialization_with_invalid_seed_string(self):
        """
        Test initialization with an invalid seed string. Should raise a
        warning.

        Skipped for Python 2.5 because it does not have the catch_warnings
        context manager.
        """
        # An invalid SEED string will issue a warning and fill the object with
        # the default values.
        with warnings.catch_warnings(record=True):
            warnings.simplefilter('error', UserWarning)
            self.assertRaises(UserWarning, WaveformStreamID,
                              seed_string="Invalid SEED string")
            # Now ignore the warnings and test the default values.
            warnings.simplefilter('ignore', UserWarning)
            waveform_id = WaveformStreamID(seed_string="Invalid Seed String")
            self.assertEqual(waveform_id.network_code, None)
            self.assertEqual(waveform_id.station_code, None)
            self.assertEqual(waveform_id.location_code, None)
            self.assertEqual(waveform_id.channel_code, None)


class ResourceIdentifierTestCase(unittest.TestCase):
    """
    Test suite for obspy.core.event.ResourceIdentifier.
    """
    def setUp(self):
        # Clear the Resource Identifier dict for the tests. NEVER do this
        # otherwise.
        ResourceIdentifier._ResourceIdentifier__resource_id_weak_dict.clear()

    def test_same_resource_id_different_referred_object(self):
        """
        Tests the handling of the case that different ResourceIdentifier
        instances are created that have the same resource id but different
        objects. This should not happen and thus a warning should be emitted.

        Skipped for Python 2.5 because it does not have the catch_warnings
        context manager.
        """
        object_a = UTCDateTime()
        object_b = UTCDateTime()
        self.assertEqual(object_a is object_b, False)
        resource_id = 'obspy.org/tests/test_resource'
        res_a = ResourceIdentifier(resource_id=resource_id,
                                   referred_object=object_a)
        # Now create a new resource with the same id but a different object.
        # This will raise a warning.
        with warnings.catch_warnings(record=True):
            warnings.simplefilter('error', UserWarning)
            self.assertRaises(UserWarning, ResourceIdentifier,
                      resource_id=resource_id, referred_object=object_b)
            # Now ignore the warning and actually create the new
            # ResourceIdentifier.
            warnings.simplefilter('ignore', UserWarning)
            res_b = ResourceIdentifier(resource_id=resource_id,
                                       referred_object=object_b)
        # Object b was the last to added, thus all resource identifiers will
        # now point to it.
        self.assertEqual(object_b is res_a.getReferredObject(), True)
        self.assertEqual(object_b is res_b.getReferredObject(), True)

    def test_objects_garbage_collection(self):
        """
        Test that the ResourceIdentifier class does not mess with the garbage
        collection of the attached objects.
        """
        object_a = UTCDateTime()
        ref_count = sys.getrefcount(object_a)
        _res_id = ResourceIdentifier(referred_object=object_a)
        self.assertEqual(sys.getrefcount(object_a), ref_count)

    def test_id_without_reference_not_in_global_list(self):
        """
        This tests some internal workings of the ResourceIdentifier class.
        NEVER modify the __resource_id_weak_dict!

        Only those ResourceIdentifiers that have a reference to an object that
        is refered to somewhere else should stay in the dictionary.
        """
        r_dict = ResourceIdentifier._ResourceIdentifier__resource_id_weak_dict
        ResourceIdentifier()
        self.assertEqual(len(r_dict.keys()), 0)
        # Adding a ResourceIdentifier with an object that has a reference
        # somewhere will have no effect because it gets garbage collected
        # pretty much immediately.
        ResourceIdentifier(referred_object=UTCDateTime())
        self.assertEqual(len(r_dict.keys()), 0)
        # Give it a reference and it will stick around.
        obj = UTCDateTime()
        ResourceIdentifier(referred_object=obj)
        self.assertEqual(len(r_dict.keys()), 1)

    def test_adding_a_referred_object_after_creation(self):
        """
        Check that the referred objects can also be made available after the
        ResourceIdentifier instances have been created.
        """
        obj = UTCDateTime()
        obj_id = id(obj)
        res_id = "obspy.org/time/test"
        ref_a = ResourceIdentifier(res_id)
        ref_b = ResourceIdentifier(res_id)
        ref_c = ResourceIdentifier(res_id)
        # All three will have no resource attached.
        self.assertEqual(ref_a.getReferredObject(), None)
        self.assertEqual(ref_b.getReferredObject(), None)
        self.assertEqual(ref_c.getReferredObject(), None)
        # Setting the object for one will make it available to all other
        # instances.
        ref_b.setReferredObject(obj)
        self.assertEqual(id(ref_a.getReferredObject()), obj_id)
        self.assertEqual(id(ref_b.getReferredObject()), obj_id)
        self.assertEqual(id(ref_c.getReferredObject()), obj_id)

    def test_resources_in_global_dict_get_garbage_colleted(self):
        """
        Tests that the ResourceIdentifiers in the class level resource dict get
        deleted if they have no other reference and the object they refer to
        goes out of scope.
        """
        obj_a = UTCDateTime()
        obj_b = UTCDateTime()
        ResourceIdentifier(referred_object=obj_a)
        ResourceIdentifier(referred_object=obj_b)
        # Now two keys should be in the global dict.
        rdict = ResourceIdentifier._ResourceIdentifier__resource_id_weak_dict
        self.assertEqual(len(rdict.keys()), 2)
        # Deleting the objects should also remove the from the dictionary.
        del obj_a, obj_b
        self.assertEqual(len(rdict.keys()), 0)


def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(CatalogTestCase, 'test'))
    suite.addTest(unittest.makeSuite(EventTestCase, 'test'))
    suite.addTest(unittest.makeSuite(OriginTestCase, 'test'))
    suite.addTest(unittest.makeSuite(WaveformStreamIDTestCase, 'test'))
    suite.addTest(unittest.makeSuite(ResourceIdentifierTestCase, 'test'))
    return suite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
